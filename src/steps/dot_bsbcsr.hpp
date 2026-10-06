#pragma once
/*
 * Step "dot", bsb-csr: the matrix transposed per bitshuffle block (CSR of
 * the bins each block touches), for matrices where most pixels have no
 * entry (rings): the dense route walks only the touched bins.  The sparse
 * route uses the nested CSC.  bsb-csr-nosplit: the same for a histogram.
 */

#include "dot_csc.hpp"

/* dense bsb-csr route: for each active bin, register-accumulate and write
 * once.  The matrix is mask-folded at build time (masked pixels' entries
 * dropped), so the gather needs no mask test throughout.  Cost is O(nnz), the
 * pixels the block's bins actually reference (the ring pixels); empty
 * bins/blocks are skipped because blk_ptr only lists active bins.
 *
 * The FMA always writes a LOCAL register (a0..a3), never memory; out[bin] is
 * written once at the end of each bin.  The four partial accumulators break
 * the serial FMA dependency chain within a bin (the dominant cost: `idx` is
 * data-dependent, so the loads are gathers the compiler cannot vectorize).
 *
 * out[bin] is data-dependent (bins[bi]); the bins of a block are distinct so
 * there is no write-after-read hazard, but the compiler cannot prove it and
 * treats the stores as non-aliasing-unknown.  The register accumulate above
 * already avoids any per-entry RMW.
 *
 * An empty tail (n == 0, when the frame is a whole number of blocks) has no
 * blk_ptr entry of its own and is skipped. */
template<typename T>
static inline
int bsbcsr_dense(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_bsbcsr *BSLZ4_RESTRICT m = (const bslz4_mat_bsbcsr *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    if (n == 0) return 0;
    const size_t block_idx = i0 / m->block_elems;
    const uint32_t b0 = m->blk_ptr[block_idx], b1 = m->blk_ptr[block_idx + 1];
    const float *BSLZ4_RESTRICT data = (const float *) m->data;
    const uint16_t *BSLZ4_RESTRICT idx = (const uint16_t *) m->idx;
    for (uint32_t bi = b0; bi < b1; bi++) {
        const uint32_t bin = m->bins[bi];
        const uint32_t k0 = m->bin_ptr[bi], k1 = m->bin_ptr[bi + 1];
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        uint32_t k = k0;
        for (; k + 4 <= k1; k += 4) {
            a0 += (double) data[k]     * (double) px[idx[k]];
            a1 += (double) data[k + 1] * (double) px[idx[k + 1]];
            a2 += (double) data[k + 2] * (double) px[idx[k + 2]];
            a3 += (double) data[k + 3] * (double) px[idx[k + 3]];
        }
        for (; k < k1; k++)
            a0 += (double) data[k] * (double) px[idx[k]];
        out[bin] += (a0 + a1) + (a2 + a3);
    }
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>(px, w->no_mask ? NULL : w->mask, i0, n, cut,
                               w->collect, (T *) w->out_vals, w->out_adr);
}

/* bsb-csr sparse route: compaction over the nested csc (powder + >cut). */
template<typename T>
static inline
int bsbcsr_sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    const bslz4_mat_bsbcsr *BSLZ4_RESTRICT m = (const bslz4_mat_bsbcsr *) w->mat;
    return sparse_dot_core<T>(w, &m->csc);
}

/* bsb-csr-nosplit: the bsb-csr layout of a histogram (every weight 1).
 * Dense route: per active bin, the sum of its pixels' values; data is not
 * read and nothing is multiplied.  Sparse route: the nested csc carries one
 * bin per pixel (csc-nosplit's indices, BSLZ4_NO_BIN for none), so it is
 * csc-nosplit's loop; csc.data and csc.indptr are not read. */
template<typename T>
static inline
int bsbcsr_nosplit_dense(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_bsbcsr *BSLZ4_RESTRICT m = (const bslz4_mat_bsbcsr *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    if (n == 0) return 0;
    const size_t block_idx = i0 / m->block_elems;
    const uint32_t b0 = m->blk_ptr[block_idx], b1 = m->blk_ptr[block_idx + 1];
    const uint16_t *BSLZ4_RESTRICT idx = (const uint16_t *) m->idx;
    for (uint32_t bi = b0; bi < b1; bi++) {
        const uint32_t bin = m->bins[bi];
        const uint32_t k0 = m->bin_ptr[bi], k1 = m->bin_ptr[bi + 1];
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        uint32_t k = k0;
        for (; k + 4 <= k1; k += 4) {
            a0 += (double) px[idx[k]];
            a1 += (double) px[idx[k + 1]];
            a2 += (double) px[idx[k + 2]];
            a3 += (double) px[idx[k + 3]];
        }
        for (; k < k1; k++)
            a0 += (double) px[idx[k]];
        out[bin] += (a0 + a1) + (a2 + a3);
    }
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>(px, w->no_mask ? NULL : w->mask, i0, n, cut,
                               w->collect, (T *) w->out_vals, w->out_adr);
}

template<typename T>
static inline
int bsbcsr_nosplit_sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    const bslz4_mat_bsbcsr *BSLZ4_RESTRICT m = (const bslz4_mat_bsbcsr *) w->mat;
    return nosplit_sparse_core<T>(w, m->csc.indices);
}

