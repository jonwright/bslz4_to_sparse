/*
 * driver.c -- the frame / block loop: it calls the steps one after another
 * for every block of every frame of a batch.
 *
 *   decode       src/steps/decode.h       compressed block -> bit-planes (raw)
 *   mask         src/steps/mask.h         (BSLZ4_MASK_PLANES: ANDed into raw)
 *   untranspose  src/steps/lowplanes.h    u16 < 256: raw -> pixel list (fused) or block
 *                src/steps/untranspose.h  raw -> pixel block (kcb, bitshuffle)
 *   collect      src/steps/collect.hpp    block -> pixel list    } src/steps/kernels.cpp,
 *   dot          src/steps/dot_*.hpp      block or list -> powder } one call per block
 *
 * Per block, route decides whether a dot object takes the dense route (dot
 * over the whole block) or the sparse one (dot over the list of non-zero
 * pixels).  The per-block counters record which step values ran.
 *
 * Step "route" (dot objects only; a plain sparsify always takes the list):
 *
 *   BSLZ4_ROUTE_RATIO_RULE  per block: the sparse route when the block is
 *                      compressed more than BSLZ4_ROUTE_RATIO (8) times
 *                      (blocksize > 8 * compressed bytes), the dense route
 *                      otherwise.  Well-compressed blocks are mostly zero,
 *                      so a list of the few non-zero pixels is cheaper than
 *                      a pass over the block.
 *   BSLZ4_ROUTE_DENSE  every block untransposed to the pixel block and the
 *                      dot taken over every pixel of it.
 *   BSLZ4_ROUTE_SPARSE  every block reduced to the list of its non-zero
 *                      pixels (mask applied), the dot taken over the list;
 *                      for u16 blocks < 256 the untranspose is fused with
 *                      the list's collect.
 */

#include "pipeline.h"
#include "../steps/decode.h"
#include "../steps/mask.h"
#include "../steps/untranspose.h"
#include "../steps/lowplanes.h"

#include <string.h>

int bslz4_lowplanes_available(int value) { return bslz4_lowplanes_capable(value); }

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

/* raw -> pixel block for one full block (raw past nz_end zero but maybe
 * unwritten).  Returns 0 or a negative value. */
static inline int bslz4_untranspose_full(const bslz4_pipe *BSLZ4_RESTRICT p, uint8_t *BSLZ4_RESTRICT block,
                                         uint8_t *BSLZ4_RESTRICT raw, uint8_t *BSLZ4_RESTRICT scratch,
                                         size_t ne, size_t nz_end) {
    const int u = p->step[BSLZ4_STEP_UNTRANSPOSE];
    const size_t NB = p->elem_size;
    if (u <= BSLZ4_UNTRANSPOSE_LOWPLANES_C) {          /* u16 only (bslz4_resolve) */
        if (bslz4_lowplanes_block(u, (uint16_t *) (void *) block, raw, scratch, ne, nz_end)) return 0;
    } else if (nz_end < ne * NB) {
        memset(raw + nz_end, 0, ne * NB - nz_end);
    }
    return bslz4_untranspose_backend(u, block, raw, scratch, ne, NB) < 0 ? -1 : 0;
}

/* top_pixels (see bslz4_top in pipeline.h): a frame's current threshold, and the move to the next
 * level(s) while its list is over the limit.  The first level, the call's cut, is level -1. */
static inline int64_t top_thr(const bslz4_top *BSLZ4_RESTRICT top, int f, int64_t cut) {
    const int l = top->level[f];
    return l < 0 ? cut : top->thr[l];
}

static int32_t top_raise(const bslz4_top *BSLZ4_RESTRICT top, int dtype, int f, int32_t n,
                         void *vals, uint32_t *adr) {
    int lev = top->level[f];
    for (;;) {
        const int over = lev < 0 ? n > top->howmany : n >= top->howmany;
        if (!over || lev + 1 >= top->nthr) break;
        lev++;
        n = bslz4_filter_gt_dispatch(dtype, vals, adr, n, top->thr[lev]);
    }
    top->level[f] = lev;
    return n;
}

int bslz4_driver_run(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                     const int32_t *BSLZ4_RESTRICT compressed_lengths, int nframes,
                     const uint8_t *BSLZ4_RESTRICT mask, int NIJ, void *BSLZ4_RESTRICT outpx,
                     uint32_t *BSLZ4_RESTRICT output_adr, int32_t *BSLZ4_RESTRICT npx_out,
                     int threshold, void *BSLZ4_RESTRICT powder, int nout,
                     const void *BSLZ4_RESTRICT mat, uint8_t *BSLZ4_RESTRICT workspace,
                     size_t workspace_len, int64_t *BSLZ4_RESTRICT cursors,
                     const bslz4_pipe *BSLZ4_RESTRICT p, const bslz4_top *BSLZ4_RESTRICT top) {
    const size_t NB = p->elem_size;
    const int decode = p->step[BSLZ4_STEP_DECODE];
    const int untranspose = p->step[BSLZ4_STEP_UNTRANSPOSE];
    const int collect = p->step[BSLZ4_STEP_COLLECT];
    const int route = p->step[BSLZ4_STEP_ROUTE];
    const int dot = p->step[BSLZ4_STEP_DOT];
    const int is_dot = dot != 0;
    int out_size = is_dot ? bslz4_dot_out_size(dot) : 0;
    if (is_dot && out_size == 0) out_size = (int) NB;     /* packed output: the pixel dtype */

    if (threshold < 0) return BSLZ4_ERR_BAD_THRESHOLD;
    if (top && (top->howmany < 0 || top->nthr < 0)) return BSLZ4_ERR_BAD_THRESHOLD;

    /* decode == PIXELS: the input frames are uncompressed pixels (NIJ * NB bytes each), so there is
     * no header, no decode, no untranspose and no plane mask: the collect and dot kernels read each
     * frame's blocks where they lie. The block size is the default one, which is also the one the
     * matrix layouts are built at. */
    const int pixels = decode == BSLZ4_DECODE_PIXELS;
    uint64_t total_output_length;
    size_t blocksize;
    int rc = 0;
    if (pixels) {
        if (nframes <= 0) return BSLZ4_ERR_BAD_NFRAMES;
        for (int f = 0; f < nframes; f++)
            if (compressed_lengths[f] < 0 ||
                (uint64_t) compressed_lengths[f] != (uint64_t) NIJ * NB)
                return (uint64_t) compressed_lengths[f] > (uint64_t) NIJ * NB
                           ? BSLZ4_ERR_TOO_MANY_PIXELS
                           : BSLZ4_ERR_TOO_FEW_PIXELS;
        total_output_length = (uint64_t) NIJ * NB;
        blocksize = BSLZ4_DEFAULT_BLOCK_BYTES;
    } else {
        rc = bslz4_driver_check_frames(compressed_ptrs, compressed_lengths, nframes,
                                       &total_output_length, &blocksize);
    }
    if (rc) return rc;

    /* A frame must hold exactly NIJ pixels (frame shape == mask shape): the
     * derived layouts (padded rows, bsb-csr entries) cover every mask pixel
     * and would read past a shorter decoded tail. */
    if (total_output_length / NB > (uint64_t) NIJ) return BSLZ4_ERR_TOO_MANY_PIXELS;
    if (total_output_length != (uint64_t) NIJ * NB) return BSLZ4_ERR_TOO_FEW_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return BSLZ4_ERR_TOO_LARGE;

    const size_t block_elems = blocksize / NB;
    const uint8_t *BSLZ4_RESTRICT emask = p->step[BSLZ4_STEP_MASK] == BSLZ4_MASK_NONE ? NULL : mask;
    /* mask in the bit-plane domain: needs blocks of <= 8192 pixels */
    const int maskplanes = !pixels && p->step[BSLZ4_STEP_MASK] == BSLZ4_MASK_PLANES &&
                           emask != NULL && block_elems <= BSLZ4_MASKPLANE_MAX_ELEMS &&
                           block_elems % 64 == 0;
    uint64_t maskbits[BSLZ4_MASKPLANE_MAX_ELEMS / 64];
    const int lowplanes = untranspose <= BSLZ4_UNTRANSPOSE_LOWPLANES_C;

    if (is_dot) {
        /* The derived layouts index a per-block pointer array by
         * i0 / block_elems, so they must be built at this block size and
         * cover every block of the frame (nblocks + 1 entries). */
        const int layout = bslz4_dot_layout(dot);
        const size_t nblocks = block_elems ? ((size_t) NIJ + block_elems - 1) / block_elems : 0;
        if (layout == BSLZ4_LAYOUT_PADDED) {
            const bslz4_mat_padded *m = (const bslz4_mat_padded *) mat;
            if (block_elems == 0 || m->block_elems != block_elems || m->row_ptr_n < nblocks + 1)
                return BSLZ4_ERR_BAD_LAYOUT;
        } else if (layout == BSLZ4_LAYOUT_BSBCSR) {
            const bslz4_mat_bsbcsr *m = (const bslz4_mat_bsbcsr *) mat;
            if (block_elems == 0 || m->block_elems != block_elems || m->blk_ptr_n < nblocks + 1)
                return BSLZ4_ERR_BAD_LAYOUT;
        }
    }
    if (workspace_len < 3 * blocksize + (is_dot ? block_elems * (sizeof(uint32_t) + NB) : 0))
        return BSLZ4_ERR_WORKSPACE_TOO_SMALL;

    /* raw: the decoded planes (scratch follows it: the zero-aware decoder's
     * slack); scratch: the untranspose's; block: the pixels */
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
        cursors[f] = pixels ? 0 : 12;
        if (top) {
            top->level[f] = -1;
            top->nabove[f] = 0;
        }
    }

    /* Filled once; n, i0, out_vals, out_adr, route and powder are set per
     * block or tail. */
    bslz4_work wbase = {0};
    wbase.dtype = p->dtype;
    wbase.collect = collect;
    wbase.dot = dot;
    wbase.no_mask = emask == NULL;
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
        /* the block's mask, packed once for all frames; 1 = nothing masked */
        const int block_unmasked = maskplanes ? bslz4_pack_mask(maskbits, emask + i0, block_elems) : 0;
        for (int f = 0; f < nframes; f++) {
            const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
            const int64_t clen = compressed_lengths[f];
            uint32_t nbytes = 0;
            size_t nz_end = blocksize;
            if (!pixels) {
                int64_t pos = cursors[f];
                if (BSLZ4_UNLIKELY(clen - pos < 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
                nbytes = bslz4_read_be32((const uint8_t *) cf + pos);
                if (BSLZ4_UNLIKELY((int64_t) nbytes > clen - pos - 4))
                    return BSLZ4_ERR_CORRUPT_CHUNK;
                cursors[f] = pos + (int64_t) nbytes + 4;

                /* decode: raw[nz_end..blocksize) is zero, maybe unwritten */
                if (BSLZ4_UNLIKELY(bslz4_decode_block(decode, (const uint8_t *) cf + pos + 4,
                                                      nbytes, raw, blocksize,
                                                      &nz_end) != (int) blocksize))
                    return BSLZ4_ERR_DECOMPRESS;
                bslz4_counters_bump(BSLZ4_STEP_DECODE, decode);
            }

            /* mask: masked pixels are 0 from here, no per-pixel tests */
            if (maskplanes && !block_unmasked)
                bslz4_mask_planes(raw, maskbits, block_elems / 8, nz_end);
            const uint8_t *BSLZ4_RESTRICT bmask = maskplanes ? NULL : emask;

            /* pixels: nothing says how sparse a block is, so only an explicit sparse route takes
             * the list */
            const int sparse = !is_dot || route == BSLZ4_ROUTE_SPARSE ||
                               (!pixels && route == BSLZ4_ROUTE_RATIO_RULE &&
                                blocksize > BSLZ4_ROUTE_RATIO * (size_t) nbytes);
            if (top) /* the previous block may have taken this frame's list over the limit */
                npx_out[f] = top_raise(top, p->dtype, f, npx_out[f],
                                       (uint8_t *) outpx + (size_t) f * NIJ * NB,
                                       output_adr + (size_t) f * NIJ);
            const int32_t npx0 = npx_out[f];
            const int64_t thr_f = top ? top_thr(top, f, (int64_t) threshold) : (int64_t) threshold;
            uint8_t *BSLZ4_RESTRICT out_vals = (uint8_t *) outpx + ((size_t) f * NIJ + (size_t) npx0) * NB;
            uint32_t *BSLZ4_RESTRICT out_adr = output_adr + (size_t) f * NIJ + npx0;
            bslz4_work w = wbase;
            w.threshold = thr_f;
            if (maskplanes) w.no_mask = 1;
            w.n = block_elems;
            w.i0 = (size_t) i0;
            if (pixels) w.block = (const uint8_t *) cf + (size_t) i0 * NB;
            w.out_vals = out_vals;
            w.out_adr = out_adr;
            if (is_dot) {
                w.route = sparse;
                w.powder = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
            }

            /* untranspose + collect fused, for u16 blocks < 256 that need
             * only the list: the plain sparsify's > cut pixels, or the sparse
             * route's non-zero ones */
            if (lowplanes && sparse && !pixels) {
                const int r =
                    is_dot ? bslz4_lowplanes_list(untranspose, raw, block_elems, nz_end, nbytes,
                                                  blocksize, scratch, bmask, (size_t) i0, 0u,
                                                  (uint16_t *) tval, tidx)
                           : bslz4_lowplanes_list(untranspose, raw, block_elems, nz_end, nbytes,
                                                  blocksize, scratch, bmask, (size_t) i0,
                                                  thr_f > 65535 ? 65535u : (unsigned) thr_f,
                                                  (uint16_t *) (void *) out_vals, out_adr);
                if (r >= 0) {
                    bslz4_counters_bump(BSLZ4_STEP_UNTRANSPOSE, untranspose);
                    bslz4_counters_bump(BSLZ4_STEP_COLLECT, collect);
                    if (!is_dot) {
                        npx_out[f] = npx0 + r;
                        continue;
                    }
                    w.precompacted = 1;
                    w.pre_nz = r;
                    bslz4_counters_bump(BSLZ4_STEP_ROUTE, BSLZ4_ROUTE_SPARSE);
                    bslz4_counters_bump(BSLZ4_STEP_DOT, dot);
                    npx_out[f] = npx0 + bslz4_sparse_dot_dispatch(&w);
                    continue;
                }
            }

            /* untranspose to the block, then collect (and dot) over it */
            if (!pixels) {
                if (BSLZ4_UNLIKELY(
                        bslz4_untranspose_full(p, block, raw, scratch, block_elems, nz_end) < 0))
                    return BSLZ4_ERR_UNTRANSPOSE;
                bslz4_counters_bump(BSLZ4_STEP_UNTRANSPOSE, untranspose);
            }
            bslz4_counters_bump(BSLZ4_STEP_COLLECT, collect);
            if (is_dot) {
                bslz4_counters_bump(BSLZ4_STEP_ROUTE, sparse ? BSLZ4_ROUTE_SPARSE : BSLZ4_ROUTE_DENSE);
                bslz4_counters_bump(BSLZ4_STEP_DOT, dot);
                npx_out[f] = npx0 + bslz4_sparse_dot_dispatch(&w);
            } else {
                npx_out[f] = npx0 + bslz4_sparse_dispatch(&w);
            }
        }
        i0 += (int) block_elems;
    }

    /* The tail: the last partial block (a multiple of 8 elements, compressed
     * like the others) and any remainder stored raw at the chunk's end. */
    const int tail_decode = decode == BSLZ4_DECODE_ZSTD ? BSLZ4_DECODE_ZSTD : BSLZ4_DECODE_LZ4_STOCK;
    const int tail_untranspose = lowplanes ? BSLZ4_UNTRANSPOSE_KCB : untranspose;
    size_t tail_block =
        pixels
            ? 0
            : (8 * NB) * ((size_t) remaining / (8 * NB)); /* pixels: all raw, at the frame's end */
    for (int f = 0; f < nframes; f++) {
        const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
        uint32_t tail_nbytes = 0;
        if (tail_block > 0) {
            const int64_t clen = compressed_lengths[f];
            int64_t pos = cursors[f];
            if (BSLZ4_UNLIKELY(clen - pos < 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            tail_nbytes = bslz4_read_be32((const uint8_t *) cf + pos);
            if (BSLZ4_UNLIKELY((int64_t) tail_nbytes > clen - pos - 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            int ret = bslz4_decode_plain(tail_decode, cf + pos + 4, (int) tail_nbytes, (char *) raw,
                                         (int) tail_block);
            cursors[f] = pos + (int64_t) tail_nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return BSLZ4_ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(bslz4_untranspose_backend(tail_untranspose, block, raw, scratch,
                                                         tail_block / NB, NB) < 0))
                return BSLZ4_ERR_UNTRANSPOSE;
            bslz4_counters_bump(BSLZ4_STEP_DECODE, decode);
            bslz4_counters_bump(BSLZ4_STEP_UNTRANSPOSE, untranspose);
        }
        int64_t rem_f = remaining - (int64_t) tail_block;
        if (rem_f > 0 && !pixels) {
            if (BSLZ4_UNLIKELY(compressed_lengths[f] < rem_f ||
                               cursors[f] > (int64_t) compressed_lengths[f] - rem_f))
                return BSLZ4_ERR_CORRUPT_CHUNK;
            memcpy(block + tail_block, cf + compressed_lengths[f] - rem_f, (size_t) rem_f);
        }
        if (top)
            npx_out[f] =
                top_raise(top, p->dtype, f, npx_out[f], (uint8_t *) outpx + (size_t) f * NIJ * NB,
                          output_adr + (size_t) f * NIJ);
        const int32_t npx0 = npx_out[f];
        bslz4_work w = wbase;
        w.threshold = top ? top_thr(top, f, (int64_t) threshold) : (int64_t) threshold;
        if (pixels)
            w.block =
                (const uint8_t *) cf + (size_t) i0 * NB; /* the frame's last pixels, in place */
        w.n = ((size_t) rem_f + tail_block) / NB;
        w.i0 = (size_t) i0;
        w.out_vals = (uint8_t *) outpx + ((size_t) f * NIJ + (size_t) npx0) * NB;
        w.out_adr = output_adr + (size_t) f * NIJ + npx0;
        bslz4_counters_bump(BSLZ4_STEP_COLLECT, collect);
        if (is_dot) {
            /* a pure raw remainder is trivially cheap either way */
            w.route = route == BSLZ4_ROUTE_SPARSE ||
                      (!pixels && route == BSLZ4_ROUTE_RATIO_RULE &&
                       (tail_block == 0 || tail_block > BSLZ4_ROUTE_RATIO * (size_t) tail_nbytes));
            w.powder = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
            bslz4_counters_bump(BSLZ4_STEP_ROUTE, w.route ? BSLZ4_ROUTE_SPARSE : BSLZ4_ROUTE_DENSE);
            bslz4_counters_bump(BSLZ4_STEP_DOT, dot);
            npx_out[f] = npx0 + bslz4_sparse_dot_dispatch(&w);
        } else {
            npx_out[f] = npx0 + bslz4_sparse_dispatch(&w);
        }
        if (top) { /* the last move to a threshold, then at most howmany pixels, the first ones */
            npx_out[f] =
                top_raise(top, p->dtype, f, npx_out[f], (uint8_t *) outpx + (size_t) f * NIJ * NB,
                          output_adr + (size_t) f * NIJ);
            top->nabove[f] = npx_out[f];
            if (npx_out[f] > top->howmany) npx_out[f] = top->howmany;
        }
    }
    return 0;
}

/* ---- the front half alone ---- */

int bslz4_driver_decompress(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                            const int32_t *BSLZ4_RESTRICT compressed_lengths, int nframes,
                            uint8_t *BSLZ4_RESTRICT out, size_t out_len,
                            uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len, int elem_size,
                            int decode, int untranspose, int unshuffle) {
    const size_t NB = (size_t) elem_size;
    if (NB != 1 && NB != 2 && NB != 4 && NB != 8) return BSLZ4_ERR_DTYPE;
    if (decode != BSLZ4_DECODE_LZ4_STOCK && decode != BSLZ4_DECODE_ZSTD)
        return BSLZ4_ERR_BAD_PIPELINE;
    uint64_t total;
    size_t blocksize;
    int rc =
        bslz4_driver_check_frames(compressed_ptrs, compressed_lengths, nframes, &total, &blocksize);
    if (rc) return rc;
    if (total > (uint64_t) INT32_MAX) return BSLZ4_ERR_TOO_LARGE;
    if ((uint64_t) nframes * total > (uint64_t) out_len) return BSLZ4_ERR_WORKSPACE_TOO_SMALL;
    const size_t wsz =
        blocksize < total ? blocksize : (size_t) total; /* no block is larger than the data */
    if (unshuffle && blocksize % (8 * NB))
        return BSLZ4_ERR_CORRUPT_CHUNK; /* the bit transpose needs whole groups of 8 */
    if (unshuffle && workspace_len < 3 * wsz) return BSLZ4_ERR_WORKSPACE_TOO_SMALL;
    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + wsz;
    uint8_t *BSLZ4_RESTRICT block = workspace + 2 * wsz;

    ZSTD_DCtx *zctx =
        decode == BSLZ4_DECODE_ZSTD ? ZSTD_createDCtx() : NULL; /* one context for every block */
    if (decode == BSLZ4_DECODE_ZSTD && !zctx) return BSLZ4_ERR_DECOMPRESS;
    const size_t full = (size_t) (total / blocksize);
    const size_t remaining = (size_t) (total - (uint64_t) full * blocksize);
    const size_t tail_block = (8 * NB) * (remaining / (8 * NB));
    rc = 0;
    for (int f = 0; f < nframes && rc == 0; f++) {
        const uint8_t *BSLZ4_RESTRICT cf = (const uint8_t *) (intptr_t) compressed_ptrs[f];
        const int64_t clen = compressed_lengths[f];
        uint8_t *BSLZ4_RESTRICT dst = out + (size_t) f * (size_t) total;
        int64_t pos = 12;
        size_t off = 0;
        for (size_t b = 0; b < full + (tail_block > 0) && rc == 0; b++) {
            const size_t size = b < full ? blocksize : tail_block;
            if (BSLZ4_UNLIKELY(clen - pos < 4)) {
                rc = BSLZ4_ERR_CORRUPT_CHUNK;
                break;
            }
            const uint32_t nbytes = bslz4_read_be32(cf + pos);
            if (BSLZ4_UNLIKELY((int64_t) nbytes > clen - pos - 4)) {
                rc = BSLZ4_ERR_CORRUPT_CHUNK;
                break;
            }
            uint8_t *into = unshuffle ? raw : dst + off;
            const int64_t got =
                zctx ? (int64_t) ZSTD_decompressDCtx(zctx, into, size, cf + pos + 4, nbytes)
                     : (int64_t) LZ4_decompress_safe((const char *) cf + pos + 4, (char *) into,
                                                     (int) nbytes, (int) size);
            if (BSLZ4_UNLIKELY(got != (int64_t) size)) {
                rc = BSLZ4_ERR_DECOMPRESS;
                break;
            }
            if (unshuffle) {
                if (BSLZ4_UNLIKELY(bslz4_untranspose_backend(untranspose, block, raw, scratch,
                                                             size / NB, NB) < 0)) {
                    rc = BSLZ4_ERR_UNTRANSPOSE;
                    break;
                }
                memcpy(dst + off, block, size);
            }
            pos += (int64_t) nbytes + 4;
            off += size;
        }
        if (rc) break;
        const int64_t rem_f =
            (int64_t) (remaining - tail_block); /* stored raw at the chunk's end */
        if (rem_f > 0) {
            if (BSLZ4_UNLIKELY(clen < rem_f || pos > clen - rem_f)) {
                rc = BSLZ4_ERR_CORRUPT_CHUNK;
                break;
            }
            memcpy(dst + off, cf + clen - rem_f, (size_t) rem_f);
        }
    }
    if (zctx) ZSTD_freeDCtx(zctx);
    return rc;
}
