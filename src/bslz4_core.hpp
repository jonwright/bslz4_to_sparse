#pragma once
/*
 * Bitshuffle-LZ4/zstd decode, generic over pixel type T and the
 * untranspose backend. Each (T, Untranspose) combination compiles to its
 * own ordinary function, instantiated explicitly (see
 * bslz4_to_sparse.cpp) behind a plain extern "C" name.
 *
 * codec (lz4 vs zstd, see bslz4_codec.hpp) is a runtime int parameter
 * rather than a template axis: it is HDF5 filter metadata the Python
 * caller already has, and can differ between two datasets read by the
 * same compiled function in one process. The untranspose backend is a
 * session-wide choice, so it stays a template axis.
 *
 * No malloc, no VLA: the only scratch space used is a caller-owned
 * "workspace" buffer, carved by pointer arithmetic.
 *
 * There is one implementation of each decode family (plain sparse, CSC),
 * taking nframes frames per call. A single frame is nframes == 1:
 * src/__init__.py's chunk2sparse/chunk2sparseCSC/bslz4_to_sparse() build
 * a 1-element compressed_ptrs/compressed_lengths/npx_out/cursors set and
 * call bslz4_decode_multi/bslz4_csc_decode_multi directly.
 *
 * Both decoders are block-outer / frame-inner (decode the current ~8KB
 * block for every frame before moving to the next block position), so
 * the workspace's raw/scratch/block regions are one shared set reused
 * per frame -- 3*blocksize total, independent of nframes. The CSC
 * indptr/indices/data lookup is not shared across frames: how much CSC
 * work a block deserves is decided per (frame, block), see
 * bslz4_csc_decode_multi below.
 *
 * The mask+threshold collect step in both decoders (scan a block,
 * compact matching (index, value) pairs) goes through
 * bslz4_collect_gt<T>/bslz4_collect_nz<T> (bslz4_collect_simd.hpp),
 * which are the scalar loops seen here for every T except uint16_t/
 * uint32_t, where an explicit template specialization takes a SIMD path
 * instead (AVX-512, AVX2, SSE2 or VSX, tried in that order). The best
 * tier the machine actually supports is enabled by default; see that
 * header for how that is decided and how to override it.
 */

#include "bslz4_codec.hpp"
#include "bslz4_common.hpp"
#include "bslz4_collect_simd.hpp"

#include <cstring>
#include <cstdint>

namespace bslz4 {

/*
 * Batched plain sparse decode over "nframes" frames from the same
 * dataset (same detector shape/dtype/block size). One caller-owned
 * workspace (3*blocksize: raw/scratch/block, shared and reused per
 * frame). Batching amortises the Python/C call boundary over nframes
 * and keeps the mask slice for the current block hot across frames.
 *
 * compressed_ptrs/compressed_lengths are address+length pairs (Python
 * ints from e.g. a numpy array's .ctypes.data) rather than one buffer
 * per frame, since c2py23's buffer parameters are one Python object
 * each and frames compress to different lengths -- this keeps every
 * frame's bytes zero-copy in Python's own per-frame arrays instead of
 * requiring them to be concatenated first. See bslz4_to_sparse.cpp.
 *
 * All frames are assumed to share total_output_length and block size
 * (read from frame 0's header, checked against every other frame's
 * header). Per-frame compressed sizes differ, so each frame's read
 * cursor into its own compressed stream persists across the block loop
 * in the caller-owned "cursors" array (one int64_t per frame).
 *
 * output/output_adr are (nframes, NIJ) arrays (row per frame, caller-
 * sized to the worst case of every pixel passing threshold); npx_out is
 * one count per frame.
 */
template<typename T,
         int64_t (*Untranspose)(void *out, const void *in, void *scratch,
                                 size_t size, size_t elem_size)>
int bslz4_decode_multi(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                        const int32_t *BSLZ4_RESTRICT compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                        T *BSLZ4_RESTRICT output, uint32_t *BSLZ4_RESTRICT output_adr,
                        int32_t *BSLZ4_RESTRICT npx_out,
                        int threshold,
                        uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                        int64_t *BSLZ4_RESTRICT cursors) {
    constexpr size_t NB = sizeof(T);

    if (nframes <= 0) return ERR_BAD_NFRAMES;
    if (threshold < 0) return ERR_BAD_THRESHOLD;
    /* TODO(threshold API, needs cleanup): threshold is a plain C int for
     * every dtype, including f32/f64 -- (T) threshold can only ever
     * express a whole-number cut for float data, never a fractional
     * one, and for u8/u16 there is no check that the value actually
     * fits the destination type (threshold=300 for u8 data silently
     * becomes (uint8_t)300 == 44 here, no error, unlike the existing
     * hard ValueError c2py23 already raises for threshold > INT32_MAX).
     * Not fixed here -- deliberately deferred; a real fix likely means
     * validating threshold against each dtype's representable range at
     * the API boundary, and/or taking a wider (e.g. double) threshold
     * for the float dtype family specifically. */
    const T cut = (T) threshold;

    const char *compressed0 = (const char *) (intptr_t) compressed_ptrs[0];
    const uint64_t total_output_length = read_be64((const uint8_t *) compressed0);
    if (total_output_length / NB > (uint64_t) NIJ) return ERR_TOO_MANY_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return ERR_TOO_LARGE;

    size_t blocksize = read_be32((const uint8_t *) compressed0 + 8);
    if (blocksize == 0) blocksize = DEFAULT_BLOCK_BYTES;

    for (int f = 1; f < nframes; f++) {
        const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
        if (BSLZ4_UNLIKELY(read_be64((const uint8_t *) cf) != total_output_length))
            return ERR_FRAME_MISMATCH;
        size_t bsf = read_be32((const uint8_t *) cf + 8);
        if (bsf == 0) bsf = DEFAULT_BLOCK_BYTES;
        if (BSLZ4_UNLIKELY(bsf != blocksize)) return ERR_FRAME_MISMATCH;
    }

    if (workspace_len < 3 * blocksize) return ERR_WORKSPACE_TOO_SMALL;

    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + blocksize;
    T *BSLZ4_RESTRICT block = (T *) (workspace + 2 * blocksize);
    const size_t block_elems = blocksize / NB;

    for (int f = 0; f < nframes; f++) {
        npx_out[f] = 0;
        cursors[f] = 12;
    }

    int i0 = 0;
    int64_t remaining = (int64_t) total_output_length;

    for (; remaining >= (int64_t) blocksize; remaining -= (int64_t) blocksize) {
        for (int f = 0; f < nframes; f++) {
            const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
            int64_t p = cursors[f];
            uint32_t nbytes = read_be32((const uint8_t *) cf + p);
            int ret = bslz4_decompress(codec, cf + p + 4, (int) nbytes,
                                        (char *) raw, (int) blocksize);
            cursors[f] = p + (int64_t) nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) blocksize)) return ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, block_elems, NB) < 0))
                return ERR_UNTRANSPOSE;

            T *outf = output + (size_t) f * NIJ;
            uint32_t *outadrf = output_adr + (size_t) f * NIJ;
            int32_t npx = npx_out[f];
            npx += bslz4_collect_gt<T>(block, mask, (size_t) i0, block_elems, cut,
                                        outf + npx, outadrf + npx);
            npx_out[f] = npx;
        }
        i0 += (int) block_elems;
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    for (int f = 0; f < nframes; f++) {
        const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
        if (tail_block > 0) {
            int64_t p = cursors[f];
            uint32_t nbytes = read_be32((const uint8_t *) cf + p);
            int ret = bslz4_decompress(codec, cf + p + 4, (int) nbytes,
                                        (char *) raw, (int) tail_block);
            cursors[f] = p + (int64_t) nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, tail_block / NB, NB) < 0))
                return ERR_UNTRANSPOSE;
        }
        int64_t rem_f = remaining - (int64_t) tail_block;
        if (rem_f > 0) {
            memcpy(&block[tail_block / NB], cf + compressed_lengths[f] - rem_f, (size_t) rem_f);
        }
        T *outf = output + (size_t) f * NIJ;
        uint32_t *outadrf = output_adr + (size_t) f * NIJ;
        int32_t npx = npx_out[f];
        size_t ntail = (size_t(rem_f) + tail_block) / NB;
        npx += bslz4_collect_gt<T>(block, mask, (size_t) i0, ntail, cut,
                                    outf + npx, outadrf + npx);
        npx_out[f] = npx;
    }
    return 0;
}

/* Dense-vs-sparse CSC routing threshold: a chunk's compression factor
 * (blocksize / compressed_bytes_for_that_block) above this picks the
 * sparse route, at or below it picks dense -- see bslz4_csc_decode_multi.
 * PLACEHOLDER default, not calibrated against real data (synthetic i.i.d.
 * Poisson noise lacks the frame-to-frame pixel correlation real detector
 * data has, so a synthetic-data crossover isn't trustworthy here). Real
 * detector frames often correlate strongly frame-to-frame (same active
 * pixels -- same powder rings, same hot pixels), which synthetic i.i.d.
 * noise does not reproduce, so recalibrate against real data before
 * relying on the default. */
inline double &bslz4_csc_dense_sparse_threshold() {
    static double x = 8.0;
    return x;
}

/*
 * Batched CSC decode over "nframes" frames from the same dataset.
 * Workspace is 3*blocksize (raw/scratch/block, shared/reused per frame)
 * plus a compaction scratch pair sized to one block's worth of pixels
 * (tidx: block_elems uint32_t, tval: block_elems T) -- also shared and
 * reused per frame, since routing and compaction are both per-frame, per-
 * block decisions with nothing to share across frames. Total workspace
 * need is therefore independent of nframes.
 *
 * Per (frame, block), after decoding: compare that block's own
 * compressed byte count (already known -- it is read off the block
 * header before the LZ4/zstd call) against blocksize via
 * bslz4_csc_dense_sparse_threshold(). Above threshold (well compressed,
 * likely sparse): pass 1 compacts (index, value) for every masked,
 * non-zero pixel (compared in T's own domain -- val != 0, not a cast
 * through an unsigned type, which would silently drop legitimate
 * negative values for signed dtypes); pass 2 accumulates the CSC product
 * over just that compacted list, and a separate pass 3 walks the same
 * compacted list for the threshold/sparse-output list, preserving raster
 * order. At or below threshold (dense): pass 1 accumulates the CSC
 * product over every masked pixel unconditionally (no compaction, no
 * per-pixel branch beyond the mask check); a separate pass 2 does the
 * threshold/collect over the same already-decoded block. Either way,
 * sum(csc output) == sum(masked image) (the CSC accumulate never looks
 * at threshold) and sum(sparse output values) == sum(masked image
 * values > threshold) -- both routes, every dtype.
 *
 * TODO(signed T, issue #9): the sparse route's val != 0 gate (and the
 * dense route's unconditional accumulate) include negative pixel values
 * in the CSC powder sum. That's correct for background-subtracted
 * float/signed data, where negatives are real signal. But for old
 * Pilatus-style signed-integer sentinels (-1 dead pixel, -2 overload,
 * ...) those negatives are error markers, not signal, and arguably
 * should be dropped from the sum instead -- a genuine per-dataset
 * choice that isn't resolved or parameterized here yet.
 */
template<typename T,
         int64_t (*Untranspose)(void *out, const void *in, void *scratch,
                                 size_t size, size_t elem_size)>
int bslz4_csc_decode_multi(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                            const int32_t *BSLZ4_RESTRICT compressed_lengths,
                            int nframes, int codec,
                            const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                            T *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                            int32_t *BSLZ4_RESTRICT npx_out,
                            int threshold,
                            double *BSLZ4_RESTRICT output, int NOUT,
                            const float *BSLZ4_RESTRICT data,
                            const uint32_t *BSLZ4_RESTRICT indices,
                            const uint32_t *BSLZ4_RESTRICT indptr,
                            uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                            int64_t *BSLZ4_RESTRICT cursors) {
    constexpr size_t NB = sizeof(T);

    if (nframes <= 0) return ERR_BAD_NFRAMES;
    if (threshold < 0) return ERR_BAD_THRESHOLD;
    /* TODO(threshold API, needs cleanup): see the identical note in
     * bslz4_decode_multi above -- same open issue, same cast. */
    const T cut = (T) threshold;
    const double dense_sparse_x = bslz4_csc_dense_sparse_threshold();

    const char *compressed0 = (const char *) (intptr_t) compressed_ptrs[0];
    const uint64_t total_output_length = read_be64((const uint8_t *) compressed0);
    if (total_output_length / NB > (uint64_t) NIJ) return ERR_TOO_MANY_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return ERR_TOO_LARGE;

    size_t blocksize = read_be32((const uint8_t *) compressed0 + 8);
    if (blocksize == 0) blocksize = DEFAULT_BLOCK_BYTES;

    for (int f = 1; f < nframes; f++) {
        const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
        if (BSLZ4_UNLIKELY(read_be64((const uint8_t *) cf) != total_output_length))
            return ERR_FRAME_MISMATCH;
        size_t bsf = read_be32((const uint8_t *) cf + 8);
        if (bsf == 0) bsf = DEFAULT_BLOCK_BYTES;
        if (BSLZ4_UNLIKELY(bsf != blocksize)) return ERR_FRAME_MISMATCH;
    }

    const size_t block_elems = blocksize / NB;
    if (workspace_len < 3 * blocksize + block_elems * (sizeof(uint32_t) + NB))
        return ERR_WORKSPACE_TOO_SMALL;

    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + blocksize;
    T *BSLZ4_RESTRICT block = (T *) (workspace + 2 * blocksize);
    uint32_t *BSLZ4_RESTRICT tidx = (uint32_t *) (workspace + 3 * blocksize);
    T *BSLZ4_RESTRICT tval = (T *) ((uint8_t *) tidx + block_elems * sizeof(uint32_t));

    for (int f = 0; f < nframes; f++) {
        double *outf = output + (size_t) f * NOUT;
        for (int j = 0; j < NOUT; j++) outf[j] = 0.0;
        npx_out[f] = 0;
        cursors[f] = 12;
    }

    int i0 = 0;
    int64_t remaining = (int64_t) total_output_length;

    for (; remaining >= (int64_t) blocksize; remaining -= (int64_t) blocksize) {
        for (int f = 0; f < nframes; f++) {
            const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
            int64_t p = cursors[f];
            uint32_t nbytes = read_be32((const uint8_t *) cf + p);
            int ret = bslz4_decompress(codec, cf + p + 4, (int) nbytes,
                                        (char *) raw, (int) blocksize);
            cursors[f] = p + (int64_t) nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) blocksize)) return ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, block_elems, NB) < 0))
                return ERR_UNTRANSPOSE;

            double *outf = output + (size_t) f * NOUT;
            T *outpxf = outpx + (size_t) f * NIJ;
            uint32_t *outadrf = output_adr + (size_t) f * NIJ;
            int32_t npx = npx_out[f];

            if ((double) blocksize > dense_sparse_x * (double) nbytes) {
                /* sparse route */
                int nz = bslz4_collect_nz<T>(block, mask, (size_t) i0, block_elems, tval, tidx);
                for (int kk = 0; kk < nz; kk++) {
                    uint32_t addr = tidx[kk];
                    T val = tval[kk];
                    uint32_t k0 = indptr[addr], k1 = indptr[addr + 1];
                    for (uint32_t k = k0; k < k1; k++)
                        outf[indices[k]] += (double) data[k] * (double) val;
                }
                for (int kk = 0; kk < nz; kk++) {
                    T val = tval[kk];
                    if (BSLZ4_UNLIKELY(val > cut)) {
                        outpxf[npx] = val;
                        outadrf[npx] = tidx[kk];
                        npx++;
                    }
                }
            } else {
                /* dense route */
                for (size_t j = 0; j < block_elems; j++) {
                    if (BSLZ4_UNLIKELY(mask[j + i0] > 0)) {
                        uint32_t k0 = indptr[j + i0], k1 = indptr[j + i0 + 1];
                        T px = block[j];
                        for (uint32_t k = k0; k < k1; k++)
                            outf[indices[k]] += (double) data[k] * (double) px;
                    }
                }
                for (size_t j = 0; j < block_elems; j++) {
                    if (BSLZ4_UNLIKELY(mask[j + i0] > 0)) {
                        T px = block[j];
                        if (BSLZ4_UNLIKELY(px > cut)) {
                            outpxf[npx] = px;
                            outadrf[npx] = (uint32_t) (j + i0);
                            npx++;
                        }
                    }
                }
            }
            npx_out[f] = npx;
        }
        i0 += (int) block_elems;
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    for (int f = 0; f < nframes; f++) {
        const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
        uint32_t tail_nbytes = 0;
        if (tail_block > 0) {
            int64_t p = cursors[f];
            tail_nbytes = read_be32((const uint8_t *) cf + p);
            int ret = bslz4_decompress(codec, cf + p + 4, (int) tail_nbytes,
                                        (char *) raw, (int) tail_block);
            cursors[f] = p + (int64_t) tail_nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, tail_block / NB, NB) < 0))
                return ERR_UNTRANSPOSE;
        }
        int64_t rem_f = remaining - (int64_t) tail_block;
        if (rem_f > 0) {
            memcpy(&block[tail_block / NB], cf + compressed_lengths[f] - rem_f, (size_t) rem_f);
        }
        size_t ntail = (size_t(rem_f) + tail_block) / NB;

        double *outf = output + (size_t) f * NOUT;
        T *outpxf = outpx + (size_t) f * NIJ;
        uint32_t *outadrf = output_adr + (size_t) f * NIJ;
        int32_t npx = npx_out[f];

        bool use_sparse = tail_block > 0
            ? (double) tail_block > dense_sparse_x * (double) tail_nbytes
            : true; /* pure literal remainder (no compressed tail block): trivially cheap either way */
        if (use_sparse) {
            int nz = bslz4_collect_nz<T>(block, mask, (size_t) i0, ntail, tval, tidx);
            for (int kk = 0; kk < nz; kk++) {
                uint32_t addr = tidx[kk];
                T val = tval[kk];
                uint32_t k0 = indptr[addr], k1 = indptr[addr + 1];
                for (uint32_t k = k0; k < k1; k++)
                    outf[indices[k]] += (double) data[k] * (double) val;
            }
            for (int kk = 0; kk < nz; kk++) {
                T val = tval[kk];
                if (BSLZ4_UNLIKELY(val > cut)) {
                    outpxf[npx] = val;
                    outadrf[npx] = tidx[kk];
                    npx++;
                }
            }
        } else {
            for (size_t j = 0; j < ntail; j++) {
                if (BSLZ4_UNLIKELY(mask[j + i0] > 0)) {
                    uint32_t k0 = indptr[j + i0], k1 = indptr[j + i0 + 1];
                    T px = block[j];
                    for (uint32_t k = k0; k < k1; k++)
                        outf[indices[k]] += (double) data[k] * (double) px;
                }
            }
            for (size_t j = 0; j < ntail; j++) {
                if (BSLZ4_UNLIKELY(mask[j + i0] > 0)) {
                    T px = block[j];
                    if (BSLZ4_UNLIKELY(px > cut)) {
                        outpxf[npx] = px;
                        outadrf[npx] = (uint32_t) (j + i0);
                        npx++;
                    }
                }
            }
        }
        npx_out[f] = npx;
    }
    return 0;
}

} /* namespace bslz4 */
