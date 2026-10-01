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
 * instead (AVX-512, AVX2, SSE2, VSX or NEON, tried in that order). The best
 * tier the machine actually supports is enabled by default; see that
 * header for how that is decided and how to override it.
 */

#include "bslz4_codec.hpp"
#include "bslz4_common.hpp"
#include "bslz4_collect_simd.hpp"
#include "bslz4_registry.h"

#include <string.h>
#include <stdint.h>

namespace bslz4 {

/* Compile-time dtype index (0..9) into bslz4_inner_table, in the order
 * u8,u16,u32,u64,i8,i16,i32,i64,f32,f64 (see plan.md section 5). */
template<typename T> struct bslz4_dtype_index;
template<> struct bslz4_dtype_index<uint8_t>  { enum { value = 0 }; };
template<> struct bslz4_dtype_index<uint16_t> { enum { value = 1 }; };
template<> struct bslz4_dtype_index<uint32_t> { enum { value = 2 }; };
template<> struct bslz4_dtype_index<uint64_t> { enum { value = 3 }; };
template<> struct bslz4_dtype_index<int8_t>   { enum { value = 4 }; };
template<> struct bslz4_dtype_index<int16_t>  { enum { value = 5 }; };
template<> struct bslz4_dtype_index<int32_t>  { enum { value = 6 }; };
template<> struct bslz4_dtype_index<int64_t>  { enum { value = 7 }; };
template<> struct bslz4_dtype_index<float>    { enum { value = 8 }; };
template<> struct bslz4_dtype_index<double>   { enum { value = 9 }; };

/* bslz4_decompress takes BSLZ4_RESTRICT parameters, which is part of the
 * function type in GCC; the driver's bslz4_decompress_fn typedef does not,
 * so route through this thin non-restrict wrapper. */
static int bslz4_decompress_call(int codec, const char *src, int compressed_size,
                                 char *dst, int dst_capacity) {
    return bslz4_decompress(codec, src, compressed_size, dst, dst_capacity);
}

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
    /* Phase 1: the block/tail loop is owned by the C driver
     * (bslz4_driver.c); this template only builds the stage table for
     * dtype T and untranspose backend and hands off.  The logic is
     * identical to the previous inline loop (same checks, same order,
     * same workspace layout), so results are bit-identical. */
    const bslz4_inner_entry *ie = &bslz4_inner_table[bslz4_dtype_index<T>::value];
    bslz4_stage st;
    st.elem_size = ie->elem_size;
    st.decompress = &bslz4_decompress_call;
    st.untranspose = Untranspose;
    st.sparse = ie->sparse;
    st.sparse_dot = ie->sparse_dot;
    return bslz4_driver_sparsify(compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
                                 (void *) output, output_adr, npx_out, threshold,
                                 workspace, workspace_len, cursors, &st);
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
    /* Phase 1: the block/tail loop is owned by the C driver; this
     * template builds the stage table for dtype T and untranspose backend
     * and hands off.  Logic (and thus results) is identical to the old
     * inline loop. */
    const double dense_sparse_x = bslz4_csc_dense_sparse_threshold();
    const bslz4_inner_entry *ie = &bslz4_inner_table[bslz4_dtype_index<T>::value];
    bslz4_stage st;
    st.elem_size = ie->elem_size;
    st.decompress = &bslz4_decompress_call;
    st.untranspose = Untranspose;
    st.sparse = ie->sparse;
    st.sparse_dot = ie->sparse_dot;
    return bslz4_driver_sparsify_and_dot(compressed_ptrs, compressed_lengths, nframes, codec,
                                         mask, NIJ, (void *) outpx, output_adr, npx_out, threshold,
                                         output, NOUT, data, indices, indptr, dense_sparse_x,
                                         workspace, workspace_len, cursors, &st);
}

} /* namespace bslz4 */
