/*
 * bslz4_driver.c -- the single decode driver for bslz4_to_sparse.
 *
 * This is the dtype-agnostic C port of the block/tail loop.  The only
 * dtype-generic work (the per-block sparse / sparse_dot call) is a direct
 * call into the C++ dtype switch (kernels_generic.cpp) with the resolved
 * collect/dot ids in the bslz4_stage; decompress and untranspose are plain
 * non-generic function pointers.  The per-block instrumentation counters
 * (decompress/untranspose/collect/dot) are bumped here.
 *
 * The block/tail loop is written once (bslz4_driver_run).  The two public
 * entry points only differ in which inner dispatch they request -- the plain
 * sparse versus the CSC-sparse (powder) one -- and are thin wrappers.
 */

#include "bslz4_common.h"
#include "bslz4_registry.h"

#include <string.h>

static int bslz4_driver_check_frames(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                      const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                      int nframes,
                                      uint64_t *BSLZ4_RESTRICT out_total,
                                      size_t *BSLZ4_RESTRICT out_blocksize) {
    if (nframes <= 0) return BSLZ4_ERR_BAD_NFRAMES;
    for (int f = 0; f < nframes; f++) {
        if (BSLZ4_UNLIKELY(compressed_lengths[f] < 12)) return BSLZ4_ERR_CORRUPT_CHUNK;
    }

    const char *BSLZ4_RESTRICT compressed0 = (const char *) (intptr_t) compressed_ptrs[0];
    const uint64_t total_output_length = bslz4_read_be64((const uint8_t *) compressed0);

    size_t blocksize = (size_t) bslz4_read_be32((const uint8_t *) compressed0 + 8);
    if (blocksize == 0) blocksize = BSLZ4_DEFAULT_BLOCK_BYTES;

    for (int f = 1; f < nframes; f++) {
        const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
        if (BSLZ4_UNLIKELY(bslz4_read_be64((const uint8_t *) cf) != total_output_length))
            return BSLZ4_ERR_FRAME_MISMATCH;
        size_t bsf = (size_t) bslz4_read_be32((const uint8_t *) cf + 8);
        if (bsf == 0) bsf = BSLZ4_DEFAULT_BLOCK_BYTES;
        if (BSLZ4_UNLIKELY(bsf != blocksize)) return BSLZ4_ERR_FRAME_MISMATCH;
    }

    *out_total = total_output_length;
    *out_blocksize = blocksize;
    return 0;
}

/* Shared driver core.  kind: 0 = plain sparse, 1 = dot (powder).  For
 * kind 1, `mat` is the layout descriptor (interpreted by the kernel per
 * dot_id), `powder` the output buffer (double* for the float layouts;
 * element size from the dot table), `nout` the number of output bins and
 * `dense_sparse_x` the dense/sparse route threshold.  For kind 0 these are
 * ignored. */
static int bslz4_driver_run(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                            const int32_t *BSLZ4_RESTRICT compressed_lengths,
                            int nframes, int codec,
                            const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                            void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                            int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                            void *BSLZ4_RESTRICT powder, int nout,
                            const void *BSLZ4_RESTRICT mat,
                            double dense_sparse_x,
                            uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                            int64_t *BSLZ4_RESTRICT cursors,
                            const bslz4_stage *BSLZ4_RESTRICT st, int kind) {
    const size_t NB = st->elem_size;
    const int is_dot = (kind != 0);
    const int out_size = is_dot ? bslz4_dot_out_size(st->dot_id) : 0;

    if (threshold < 0) return BSLZ4_ERR_BAD_THRESHOLD;

    uint64_t total_output_length;
    size_t blocksize;
    int rc = bslz4_driver_check_frames(compressed_ptrs, compressed_lengths, nframes,
                                       &total_output_length, &blocksize);
    if (rc) return rc;

    if (total_output_length / NB > (uint64_t) NIJ) return BSLZ4_ERR_TOO_MANY_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return BSLZ4_ERR_TOO_LARGE;

    const size_t block_elems = blocksize / NB;
    if (is_dot) {
        const int layout = bslz4_dot_layout(st->dot_id);
        if (layout == BSLZ4_LAYOUT_PADDED) {
            if (((const bslz4_mat_padded *) mat)->block_elems != block_elems)
                return BSLZ4_ERR_BAD_LAYOUT;
        } else if (layout == BSLZ4_LAYOUT_BSBCSR) {
            if (((const bslz4_mat_bsbcsr *) mat)->block_elems != block_elems)
                return BSLZ4_ERR_BAD_LAYOUT;
        }
    }
    if (workspace_len < 3 * blocksize + (is_dot ? block_elems * (sizeof(uint32_t) + NB) : 0))
        return BSLZ4_ERR_WORKSPACE_TOO_SMALL;

    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + blocksize;
    uint8_t *BSLZ4_RESTRICT block = workspace + 2 * blocksize;
    uint32_t *BSLZ4_RESTRICT tidx = NULL;
    void *BSLZ4_RESTRICT tval = NULL;
    if (is_dot) {
        tidx = (uint32_t *) (workspace + 3 * blocksize);
        tval = (void *) ((uint8_t *) tidx + block_elems * sizeof(uint32_t));
    }

    for (int f = 0; f < nframes; f++) {
        if (is_dot) {
            uint8_t *outf = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
            memset(outf, 0, (size_t) nout * (size_t) out_size);
        }
        npx_out[f] = 0;
        cursors[f] = 12;
    }

    /* Filled once, then n/i0/out_vals/out_adr/route/powder are updated per
     * block or tail.  A new scheme adds a dot row + kernel; the template is
     * never widened. */
    bslz4_work wbase = {0};
    wbase.dtype = st->dtype;
    wbase.collect_id = st->collect_id;
    wbase.dot_id = st->dot_id;
    wbase.no_mask = (st->options & BSLZ4_OPT_NO_MASK) != 0;
    wbase.threshold = (int64_t) threshold;
    wbase.block = block;
    wbase.mask = mask;
    wbase.mat = mat;
    wbase.tidx = tidx;
    wbase.tval = tval;
    wbase.nout = nout;

    int i0 = 0;
    int64_t remaining = (int64_t) total_output_length;

    for (; remaining >= (int64_t) blocksize; remaining -= (int64_t) blocksize) {
        for (int f = 0; f < nframes; f++) {
            const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
            const int64_t clen = compressed_lengths[f];
            int64_t p = cursors[f];
            if (BSLZ4_UNLIKELY(clen - p < 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            uint32_t nbytes = bslz4_read_be32((const uint8_t *) cf + p);
            if (BSLZ4_UNLIKELY((int64_t) nbytes > clen - p - 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            int ret = st->decompress(codec, cf + p + 4, (int) nbytes, (char *) raw, (int) blocksize);
            cursors[f] = p + (int64_t) nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) blocksize)) return BSLZ4_ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(st->untranspose(block, raw, scratch, block_elems, NB) < 0))
                return BSLZ4_ERR_UNTRANSPOSE;
            bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, codec);
            bslz4_counters_bump(BSLZ4_STAGE_UNTRANSPOSE, st->untranspose_id);

            uint8_t *BSLZ4_RESTRICT outf = (uint8_t *) outpx + (size_t) f * NIJ * NB;
            uint32_t *BSLZ4_RESTRICT outadrf = output_adr + (size_t) f * NIJ;
            int32_t npx = npx_out[f];
            bslz4_counters_bump(BSLZ4_STAGE_COLLECT, st->collect_id);
            bslz4_work w = wbase;
            w.n = block_elems;
            w.i0 = (size_t) i0;
            w.out_vals = outf + (size_t) npx * NB;
            w.out_adr = outadrf + npx;
            if (is_dot) {
                w.route = (double) blocksize > dense_sparse_x * (double) nbytes;
                w.powder = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
                bslz4_counters_bump(BSLZ4_STAGE_DOT, st->dot_id);
                npx += bslz4_sparse_dot_dispatch(&w);
            } else {
                npx += bslz4_sparse_dispatch(&w);
            }
            npx_out[f] = npx;
        }
        i0 += (int) block_elems;
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    for (int f = 0; f < nframes; f++) {
        const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
        uint32_t tail_nbytes = 0;
        if (tail_block > 0) {
            const int64_t clen = compressed_lengths[f];
            int64_t p = cursors[f];
            if (BSLZ4_UNLIKELY(clen - p < 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            tail_nbytes = bslz4_read_be32((const uint8_t *) cf + p);
            if (BSLZ4_UNLIKELY((int64_t) tail_nbytes > clen - p - 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            int ret = st->decompress(codec, cf + p + 4, (int) tail_nbytes, (char *) raw, (int) tail_block);
            cursors[f] = p + (int64_t) tail_nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return BSLZ4_ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(st->untranspose(block, raw, scratch, tail_block / NB, NB) < 0))
                return BSLZ4_ERR_UNTRANSPOSE;
            bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, codec);
            bslz4_counters_bump(BSLZ4_STAGE_UNTRANSPOSE, st->untranspose_id);
        }
        int64_t rem_f = remaining - (int64_t) tail_block;
        if (rem_f > 0) {
            if (BSLZ4_UNLIKELY(compressed_lengths[f] < rem_f ||
                               cursors[f] > (int64_t) compressed_lengths[f] - rem_f))
                return BSLZ4_ERR_CORRUPT_CHUNK;
            memcpy(block + tail_block, cf + compressed_lengths[f] - rem_f, (size_t) rem_f);
        }
        uint8_t *BSLZ4_RESTRICT outf = (uint8_t *) outpx + (size_t) f * NIJ * NB;
        uint32_t *BSLZ4_RESTRICT outadrf = output_adr + (size_t) f * NIJ;
        int32_t npx = npx_out[f];
        size_t ntail = ((size_t) rem_f + tail_block) / NB;
        bslz4_counters_bump(BSLZ4_STAGE_COLLECT, st->collect_id);
        bslz4_work w = wbase;
        w.n = ntail;
        w.i0 = (size_t) i0;
        w.out_vals = outf + (size_t) npx * NB;
        w.out_adr = outadrf + npx;
        if (is_dot) {
            w.route = tail_block > 0
                ? (double) tail_block > dense_sparse_x * (double) tail_nbytes
                : 1; /* pure literal remainder: trivially cheap either way */
            w.powder = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
            bslz4_counters_bump(BSLZ4_STAGE_DOT, st->dot_id);
            npx += bslz4_sparse_dot_dispatch(&w);
        } else {
            npx += bslz4_sparse_dispatch(&w);
        }
        npx_out[f] = npx;
    }
    return 0;
}

int bslz4_driver_sparsify(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                          const int32_t *BSLZ4_RESTRICT compressed_lengths,
                          int nframes, int codec,
                          const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                          void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                          int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                          uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                          int64_t *BSLZ4_RESTRICT cursors,
                          const bslz4_stage *BSLZ4_RESTRICT st) {
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, codec,
                            mask, NIJ, outpx, output_adr, npx_out, threshold,
                            NULL, 0, NULL, 0.0,
                            workspace, workspace_len, cursors, st, 0);
}

int bslz4_driver_sparsify_and_dot(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                  const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                  int nframes, int codec,
                                  const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                  void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                  int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                  double *BSLZ4_RESTRICT powder, int nout,
                                  const float *BSLZ4_RESTRICT data,
                                  const uint32_t *BSLZ4_RESTRICT indices,
                                  const uint32_t *BSLZ4_RESTRICT indptr,
                                  double dense_sparse_x,
                                  uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                  int64_t *BSLZ4_RESTRICT cursors,
                                  const bslz4_stage *BSLZ4_RESTRICT st) {
    bslz4_mat_csc csc;
    csc.data = data;
    csc.indices = indices;
    csc.indptr = indptr;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, codec,
                            mask, NIJ, outpx, output_adr, npx_out, threshold,
                            powder, nout, &csc, dense_sparse_x,
                            workspace, workspace_len, cursors, st, 1);
}

int bslz4_driver_sparsify_and_dot_padded(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                         const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                         int nframes, int codec,
                                         const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                         void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                         int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                         double *BSLZ4_RESTRICT powder, int nout,
                                         const int32_t *BSLZ4_RESTRICT base,
                                         const float *BSLZ4_RESTRICT weights,
                                         const int32_t *BSLZ4_RESTRICT pixels,
                                         const int32_t *BSLZ4_RESTRICT rowmap,
                                         const int32_t *BSLZ4_RESTRICT row_ptr,
                                         int width, int listed, size_t block_elems,
                                         double dense_sparse_x,
                                         uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                         int64_t *BSLZ4_RESTRICT cursors,
                                         const bslz4_stage *BSLZ4_RESTRICT st) {
    bslz4_mat_padded m;
    m.base = base;
    m.weights = weights;
    m.pixels = pixels;
    m.rowmap = rowmap;
    m.row_ptr = row_ptr;
    m.width = width;
    m.listed = listed;
    m.block_elems = block_elems;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, codec,
                            mask, NIJ, outpx, output_adr, npx_out, threshold,
                            powder, nout, &m, dense_sparse_x,
                            workspace, workspace_len, cursors, st, 1);
}

int bslz4_driver_sparsify_and_dot_bsbcsr(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                         const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                         int nframes, int codec,
                                         const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                         void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                         int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                         double *BSLZ4_RESTRICT powder, int nout,
                                         const uint32_t *BSLZ4_RESTRICT blk_ptr,
                                         const uint32_t *BSLZ4_RESTRICT bins,
                                         const uint32_t *BSLZ4_RESTRICT bin_ptr,
                                         const uint16_t *BSLZ4_RESTRICT idx,
                                         const float *BSLZ4_RESTRICT data,
                                         const void *BSLZ4_RESTRICT csc_data,
                                         const uint32_t *BSLZ4_RESTRICT csc_indices,
                                         const uint32_t *BSLZ4_RESTRICT csc_indptr,
                                         size_t block_elems,
                                         double dense_sparse_x,
                                         uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                         int64_t *BSLZ4_RESTRICT cursors,
                                         const bslz4_stage *BSLZ4_RESTRICT st) {
    bslz4_mat_bsbcsr m;
    m.blk_ptr = blk_ptr;
    m.bins = bins;
    m.bin_ptr = bin_ptr;
    m.idx = idx;
    m.data = data;
    m.csc.data = csc_data;
    m.csc.indices = csc_indices;
    m.csc.indptr = csc_indptr;
    m.block_elems = block_elems;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, codec,
                            mask, NIJ, outpx, output_adr, npx_out, threshold,
                            powder, nout, &m, dense_sparse_x,
                            workspace, workspace_len, cursors, st, 1);
}
