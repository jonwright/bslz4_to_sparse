#pragma once
/*
 * Bitshuffle-LZ4/zstd decode, generic over pixel type T and the
 * untranspose backend. Templates replace the previous #define DATATYPE /
 * #include macro-repetition; each (T, Untranspose) combination compiles
 * to its own ordinary function, instantiated explicitly (see
 * bslz4_to_sparse.cpp) behind a plain extern "C" name.
 *
 * codec (lz4 vs zstd, see bslz4_codec.hpp) is an ordinary runtime int
 * parameter rather than a template axis: which codec a dataset used is
 * HDF5 filter metadata the Python caller already has, and can differ
 * between two datasets read with the very same compiled function in the
 * same process, unlike the untranspose backend (a session-wide choice
 * for benchmarking kernels against each other).
 *
 * No malloc, no VLA: the only scratch space used is a caller-owned
 * "workspace" buffer, carved by pointer arithmetic into three
 * same-sized regions (one block each): the decompressed-but-still
 * -bitshuffled data, the untranspose backend's own scratch, and the
 * final untransposed block. Its required size (3 * blocksize, where
 * blocksize is read from the compressed stream header) is not known
 * until runtime, since HDF5 datasets may use any block size -- so it
 * cannot be a template constant or a fixed stack array.
 */

#include "bslz4_codec.hpp"
#include "bslz4_common.hpp"

#include <cstring>
#include <cstdint>

namespace bslz4 {

template<typename T,
         int64_t (*Untranspose)(void *out, const void *in, void *scratch,
                                 size_t size, size_t elem_size)>
int bslz4_decode(const char *BSLZ4_RESTRICT compressed, int compressed_length, int codec,
                  const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                  T *BSLZ4_RESTRICT output, uint32_t *BSLZ4_RESTRICT output_adr,
                  int threshold,
                  uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len) {
    constexpr size_t NB = sizeof(T);

    if (threshold < 0) return ERR_BAD_THRESHOLD;
    const T cut = (T) threshold;

    const uint64_t total_output_length = read_be64((const uint8_t *) compressed);
    if (total_output_length / NB > (uint64_t) NIJ) return ERR_TOO_MANY_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return ERR_TOO_LARGE;

    size_t blocksize = read_be32((const uint8_t *) compressed + 8);
    if (blocksize == 0) blocksize = DEFAULT_BLOCK_BYTES;
    if (workspace_len < 3 * blocksize) return ERR_WORKSPACE_TOO_SMALL;

    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + blocksize;
    T *BSLZ4_RESTRICT block = (T *) (workspace + 2 * blocksize);

    int npx = 0;
    int i0 = 0;
    int p = 12;
    int64_t remaining = (int64_t) total_output_length;

    for (; remaining >= (int64_t) blocksize; remaining -= (int64_t) blocksize) {
        uint32_t nbytes = read_be32((const uint8_t *) compressed + p);
        int ret = bslz4_decompress(codec, compressed + p + 4, (int) nbytes,
                                    (char *) raw, (int) blocksize);
        p += (int) nbytes + 4;
        if (BSLZ4_UNLIKELY(ret != (int) blocksize)) return ERR_DECOMPRESS;

        if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, blocksize / NB, NB) < 0))
            return ERR_UNTRANSPOSE;

        for (size_t j = 0; j < blocksize / NB; j++) {
            /* Plain "&" (not "&&"): both sides are always evaluated, so this
             * stays branchless -- unlike the old mask*value>cut trick, it
             * compares in T's own signed/float domain instead of forcing
             * everything through uint32_t (wrong for negative or
             * fractional values). */
            if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (block[j] > cut))) {
                *(output++) = block[j];
                *(output_adr++) = (uint32_t) (j + i0);
                npx++;
            }
        }
        i0 += (int) (blocksize / NB);
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    if (tail_block > 0) {
        uint32_t nbytes = read_be32((const uint8_t *) compressed + p);
        int ret = bslz4_decompress(codec, compressed + p + 4, (int) nbytes,
                                    (char *) raw, (int) tail_block);
        p += (int) nbytes + 4;
        if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return ERR_DECOMPRESS;
        if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, tail_block / NB, NB) < 0))
            return ERR_UNTRANSPOSE;
    }
    remaining -= (int64_t) tail_block;
    if (remaining > 0) {
        memcpy(&block[tail_block / NB], compressed + compressed_length - remaining, (size_t) remaining);
    }
    for (size_t j = 0; j < (size_t(remaining) + tail_block) / NB; j++) {
        if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (block[j] > cut))) {
            *(output++) = block[j];
            *(output_adr++) = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/*
 * Batched CSC decode over a series of "nframes" frames from the same
 * dataset (same detector shape/dtype/block size), issue #12: decoding
 * a series of frames and integrating them in one call gets better cache
 * locality on the CSC matrix (data/indices/indptr) than one call per
 * frame, because the indptr/indices/data lookup for pixel j is done
 * ONCE and applied across every frame's value for that pixel, instead
 * of being re-walked from scratch for each frame separately. This means
 * the loop order is pixel-outer / frame-inner, which in turn means every
 * frame's untransposed data for the current block must be resident at
 * once -- unlike the single-frame path there is one "block" region per
 * frame in the workspace, not a single shared one.
 *
 * compressed_ptrs/compressed_lengths are address+length pairs (Python
 * ints written by note_chunk(), see bslz4_to_sparse.cpp and
 * _gather_chunks() in __init__.py) rather than one buffer per frame,
 * since c2py23's buffer parameters are one Python object each and
 * frames compress to different lengths -- this keeps every frame's
 * bytes zero-copy in Python's own per-frame arrays instead of requiring
 * them to be concatenated first.
 *
 * All frames are assumed to share total_output_length and block size
 * (read from frame 0's header, and checked against every other frame's
 * header -- a mismatch usually means frames from different datasets got
 * batched together by mistake). Per-frame compressed sizes do differ
 * (that's the whole point of compression), so each frame's read cursor
 * into its own compressed stream must persist across the block loop;
 * since nframes is a runtime value this can't be a stack array without
 * either a malloc or a VLA, so it is caller-owned scratch too (the
 * "cursors" parameter, one int64_t per frame) -- the single-frame
 * wrapper below just uses a 1-element local array instead, since there
 * nframes==1 is a compile-time constant.
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

    if (workspace_len < (size_t) (nframes + 2) * blocksize) return ERR_WORKSPACE_TOO_SMALL;

    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + blocksize;
    T *BSLZ4_RESTRICT blockbuf = (T *) (workspace + 2 * blocksize);
    const size_t block_elems = blocksize / NB;

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
            T *blk_f = blockbuf + (size_t) f * block_elems;
            if (BSLZ4_UNLIKELY(Untranspose(blk_f, raw, scratch, block_elems, NB) < 0))
                return ERR_UNTRANSPOSE;
        }

        for (size_t j = 0; j < block_elems; j++) {
            if (BSLZ4_UNLIKELY(mask[j + i0] > 0)) {
                uint32_t k0 = indptr[j + i0];
                uint32_t k1 = indptr[j + i0 + 1];
                for (int f = 0; f < nframes; f++) {
                    T px = blockbuf[(size_t) f * block_elems + j];
                    /* TODO(signed T): px != 0 includes negative values in the
                     * powder sum -- correct for background-subtracted float/
                     * signed data, where negatives are real signal. But for
                     * old Pilatus-style signed-integer sentinels (-1 dead
                     * pixel, -2 overload, ...) those negatives are error
                     * markers, not signal, and arguably should be dropped
                     * instead. Genuine per-dataset choice, not resolved or
                     * parameterized here yet. */
                    if (BSLZ4_UNLIKELY(px != 0)) {
                        double *outf = output + (size_t) f * NOUT;
                        for (uint32_t k = k0; k < k1; k++)
                            outf[indices[k]] += (double) data[k] * (double) px;
                        if (BSLZ4_UNLIKELY(px > cut)) {
                            int32_t n = npx_out[f]++;
                            outpx[(size_t) f * NIJ + n] = px;
                            output_adr[(size_t) f * NIJ + n] = (uint32_t) (j + i0);
                        }
                    }
                }
            }
        }
        i0 += (int) block_elems;
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    if (tail_block > 0) {
        for (int f = 0; f < nframes; f++) {
            const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
            int64_t p = cursors[f];
            uint32_t nbytes = read_be32((const uint8_t *) cf + p);
            int ret = bslz4_decompress(codec, cf + p + 4, (int) nbytes,
                                        (char *) raw, (int) tail_block);
            cursors[f] = p + (int64_t) nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return ERR_DECOMPRESS;
            T *blk_f = blockbuf + (size_t) f * block_elems;
            if (BSLZ4_UNLIKELY(Untranspose(blk_f, raw, scratch, tail_block / NB, NB) < 0))
                return ERR_UNTRANSPOSE;
        }
    }
    remaining -= (int64_t) tail_block;
    if (remaining > 0) {
        for (int f = 0; f < nframes; f++) {
            const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
            T *blk_f = blockbuf + (size_t) f * block_elems;
            memcpy(&blk_f[tail_block / NB], cf + compressed_lengths[f] - remaining, (size_t) remaining);
        }
    }
    for (size_t j = 0; j < (size_t(remaining) + tail_block) / NB; j++) {
        if (BSLZ4_UNLIKELY(mask[j + i0] > 0)) {
            uint32_t k0 = indptr[j + i0];
            uint32_t k1 = indptr[j + i0 + 1];
            for (int f = 0; f < nframes; f++) {
                T px = blockbuf[(size_t) f * block_elems + j];
                /* TODO(signed T): see the identical note in the main block
                 * loop above -- same open question, same code path. */
                if (BSLZ4_UNLIKELY(px != 0)) {
                    double *outf = output + (size_t) f * NOUT;
                    for (uint32_t k = k0; k < k1; k++)
                        outf[indices[k]] += (double) data[k] * (double) px;
                    if (BSLZ4_UNLIKELY(px > cut)) {
                        int32_t n = npx_out[f]++;
                        outpx[(size_t) f * NIJ + n] = px;
                        output_adr[(size_t) f * NIJ + n] = (uint32_t) (j + i0);
                    }
                }
            }
        }
    }
    return 0;
}

/* Single-frame CSC decode, sitting on top of bslz4_csc_decode_multi with
 * nframes == 1 (a compile-time constant here, so the per-frame cursor is
 * an ordinary 1-element local array rather than caller-supplied scratch --
 * existing single-frame callers need no new buffer argument). */
template<typename T,
         int64_t (*Untranspose)(void *out, const void *in, void *scratch,
                                 size_t size, size_t elem_size)>
int bslz4_csc_decode(const char *BSLZ4_RESTRICT compressed, int compressed_length, int codec,
                      const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                      T *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                      int threshold,
                      double *BSLZ4_RESTRICT output, int NOUT,
                      const float *BSLZ4_RESTRICT data,
                      const uint32_t *BSLZ4_RESTRICT indices,
                      const uint32_t *BSLZ4_RESTRICT indptr,
                      uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len) {
    int64_t ptr = (int64_t) (intptr_t) compressed;
    int32_t len = (int32_t) compressed_length;
    int64_t cursor[1] = {12};
    int32_t npx = 0;
    int ret = bslz4_csc_decode_multi<T, Untranspose>(
        &ptr, &len, 1, codec, mask, NIJ, outpx, output_adr, &npx, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len, cursor);
    if (ret < 0) return ret;
    return npx;
}

} /* namespace bslz4 */
