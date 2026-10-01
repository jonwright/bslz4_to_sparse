/*
 * bslz4_driver.c -- the single decode driver for bslz4_to_sparse.
 *
 * This is the dtype-agnostic C port of the block/tail loop that used to
 * live, four times over, in the bslz4_core.hpp templates.  The only
 * dtype-generic work (the per-block sparse / sparse_dot call) is reached
 * through the function pointers in a bslz4_stage, built by the caller.
 *
 * Phase 1: the old C++ templates construct a bslz4_stage and call these;
 * the generated layer and native API are unchanged.  The loop logic is
 * identical to the previous templates (same checks, same order, same
 * workspace layout), so results are bit-identical.
 */

#include "bslz4_common.h"
#include "bslz4_registry.h"

#include <string.h>

static int bslz4_driver_check_frames(const int64_t *compressed_ptrs,
                                      const int32_t *compressed_lengths,
                                      int nframes, const bslz4_stage *st,
                                      uint64_t *out_total, size_t *out_blocksize) {
    if (nframes <= 0) return BSLZ4_ERR_BAD_NFRAMES;
    for (int f = 0; f < nframes; f++) {
        if (BSLZ4_UNLIKELY(compressed_lengths[f] < 12)) return BSLZ4_ERR_CORRUPT_CHUNK;
    }

    const char *compressed0 = (const char *) (intptr_t) compressed_ptrs[0];
    const uint64_t total_output_length = bslz4_read_be64((const uint8_t *) compressed0);

    size_t blocksize = (size_t) bslz4_read_be32((const uint8_t *) compressed0 + 8);
    if (blocksize == 0) blocksize = BSLZ4_DEFAULT_BLOCK_BYTES;

    for (int f = 1; f < nframes; f++) {
        const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
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

int bslz4_driver_sparsify(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                          int nframes, int codec, const uint8_t *mask, int NIJ,
                          void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                          uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                          const bslz4_stage *st) {
    const size_t NB = st->elem_size;

    if (threshold < 0) return BSLZ4_ERR_BAD_THRESHOLD;

    uint64_t total_output_length;
    size_t blocksize;
    int rc = bslz4_driver_check_frames(compressed_ptrs, compressed_lengths, nframes, st,
                                       &total_output_length, &blocksize);
    if (rc) return rc;

    if (total_output_length / NB > (uint64_t) NIJ) return BSLZ4_ERR_TOO_MANY_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return BSLZ4_ERR_TOO_LARGE;

    if (workspace_len < 3 * blocksize) return BSLZ4_ERR_WORKSPACE_TOO_SMALL;

    uint8_t *raw = workspace;
    uint8_t *scratch = workspace + blocksize;
    uint8_t *block = workspace + 2 * blocksize;
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

            uint8_t *outf = (uint8_t *) outpx + (size_t) f * NIJ * NB;
            uint32_t *outadrf = output_adr + (size_t) f * NIJ;
            int32_t npx = npx_out[f];
            npx += st->sparse(block, block_elems, mask, (size_t) i0, (int64_t) threshold,
                              outf + (size_t) npx * NB, outadrf + npx);
            npx_out[f] = npx;
        }
        i0 += (int) block_elems;
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    for (int f = 0; f < nframes; f++) {
        const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
        if (tail_block > 0) {
            const int64_t clen = compressed_lengths[f];
            int64_t p = cursors[f];
            if (BSLZ4_UNLIKELY(clen - p < 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            uint32_t nbytes = bslz4_read_be32((const uint8_t *) cf + p);
            if (BSLZ4_UNLIKELY((int64_t) nbytes > clen - p - 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            int ret = st->decompress(codec, cf + p + 4, (int) nbytes, (char *) raw, (int) tail_block);
            cursors[f] = p + (int64_t) nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return BSLZ4_ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(st->untranspose(block, raw, scratch, tail_block / NB, NB) < 0))
                return BSLZ4_ERR_UNTRANSPOSE;
        }
        int64_t rem_f = remaining - (int64_t) tail_block;
        if (rem_f > 0) {
            if (BSLZ4_UNLIKELY(compressed_lengths[f] < rem_f ||
                               cursors[f] > (int64_t) compressed_lengths[f] - rem_f))
                return BSLZ4_ERR_CORRUPT_CHUNK;
            memcpy(block + tail_block, cf + compressed_lengths[f] - rem_f, (size_t) rem_f);
        }
        uint8_t *outf = (uint8_t *) outpx + (size_t) f * NIJ * NB;
        uint32_t *outadrf = output_adr + (size_t) f * NIJ;
        int32_t npx = npx_out[f];
        size_t ntail = ((size_t) rem_f + tail_block) / NB;
        npx += st->sparse(block, ntail, mask, (size_t) i0, (int64_t) threshold,
                          outf + (size_t) npx * NB, outadrf + npx);
        npx_out[f] = npx;
    }
    return 0;
}

int bslz4_driver_sparsify_and_dot(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                                  int nframes, int codec, const uint8_t *mask, int NIJ,
                                  void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                                  double *powder, int nout, const float *data,
                                  const uint32_t *indices, const uint32_t *indptr,
                                  double dense_sparse_x,
                                  uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                                  const bslz4_stage *st) {
    const size_t NB = st->elem_size;

    if (threshold < 0) return BSLZ4_ERR_BAD_THRESHOLD;

    uint64_t total_output_length;
    size_t blocksize;
    int rc = bslz4_driver_check_frames(compressed_ptrs, compressed_lengths, nframes, st,
                                       &total_output_length, &blocksize);
    if (rc) return rc;

    if (total_output_length / NB > (uint64_t) NIJ) return BSLZ4_ERR_TOO_MANY_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return BSLZ4_ERR_TOO_LARGE;

    const size_t block_elems = blocksize / NB;
    if (workspace_len < 3 * blocksize + block_elems * (sizeof(uint32_t) + NB))
        return BSLZ4_ERR_WORKSPACE_TOO_SMALL;

    uint8_t *raw = workspace;
    uint8_t *scratch = workspace + blocksize;
    uint8_t *block = workspace + 2 * blocksize;
    uint32_t *tidx = (uint32_t *) (workspace + 3 * blocksize);
    void *tval = (void *) ((uint8_t *) tidx + block_elems * sizeof(uint32_t));

    for (int f = 0; f < nframes; f++) {
        double *outf = powder + (size_t) f * nout;
        for (int j = 0; j < nout; j++) outf[j] = 0.0;
        npx_out[f] = 0;
        cursors[f] = 12;
    }

    int i0 = 0;
    int64_t remaining = (int64_t) total_output_length;

    for (; remaining >= (int64_t) blocksize; remaining -= (int64_t) blocksize) {
        for (int f = 0; f < nframes; f++) {
            const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
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

            double *outf = powder + (size_t) f * nout;
            uint8_t *outpxf = (uint8_t *) outpx + (size_t) f * NIJ * NB;
            uint32_t *outadrf = output_adr + (size_t) f * NIJ;
            int32_t npx = npx_out[f];
            npx += st->sparse_dot(block, block_elems, blocksize, (size_t) nbytes,
                                  mask, (size_t) i0, (int64_t) threshold,
                                  outpxf + (size_t) npx * NB, outadrf + npx,
                                  outf, nout, data, indices, indptr,
                                  dense_sparse_x, tidx, tval);
            npx_out[f] = npx;
        }
        i0 += (int) block_elems;
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    for (int f = 0; f < nframes; f++) {
        const char *cf = (const char *) (intptr_t) compressed_ptrs[f];
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
        }
        int64_t rem_f = remaining - (int64_t) tail_block;
        if (rem_f > 0) {
            if (BSLZ4_UNLIKELY(compressed_lengths[f] < rem_f ||
                               cursors[f] > (int64_t) compressed_lengths[f] - rem_f))
                return BSLZ4_ERR_CORRUPT_CHUNK;
            memcpy(block + tail_block, cf + compressed_lengths[f] - rem_f, (size_t) rem_f);
        }
        size_t ntail = ((size_t) rem_f + tail_block) / NB;

        double *outf = powder + (size_t) f * nout;
        uint8_t *outpxf = (uint8_t *) outpx + (size_t) f * NIJ * NB;
        uint32_t *outadrf = output_adr + (size_t) f * NIJ;
        int32_t npx = npx_out[f];
        npx += st->sparse_dot(block, ntail, tail_block, (size_t) tail_nbytes,
                              mask, (size_t) i0, (int64_t) threshold,
                              outpxf + (size_t) npx * NB, outadrf + npx,
                              outf, nout, data, indices, indptr,
                              dense_sparse_x, tidx, tval);
        npx_out[f] = npx;
    }
    return 0;
}
