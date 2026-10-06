#pragma once
/*
 * Step "dot", the CSC-entry layouts: csc (one bin per entry), csc-run
 * (start bin + consecutive weights per pixel), csc-nosplit (one bin per
 * pixel, weight 1: a histogram) and csc-nosplit-moment (a histogram with the
 * first moment beside it).  Each has a dense route (every pixel of the
 * block) and a sparse route (the list of non-zero pixels).  The matrix is
 * mask-folded in Python, so the dense loops make no mask test; the mask
 * still selects the > cut sparse output.
 */

#include "../pipeline/common.h"
#include "../pipeline/pipeline.h"
#include "collect.hpp"

#include <limits>

#if defined(_MSC_VER) && defined(_M_X64)
#include <xmmintrin.h>
#endif

using namespace bslz4;

/* A threshold above the dtype's maximum means no pixel can be > cut: the
 * sparse output is empty, so its scan is skipped (and the wrapped (T) cut
 * never used).  uint64_t and the floating types are never above an int64
 * threshold. */
template<typename T> inline bool bslz4_cut_above_max(const bslz4_work *w) {
    return w->threshold > (int64_t) std::numeric_limits<T>::max();
}
template<> inline bool bslz4_cut_above_max<uint64_t>(const bslz4_work *) { return false; }
template<> inline bool bslz4_cut_above_max<float>(const bslz4_work *) { return false; }
template<> inline bool bslz4_cut_above_max<double>(const bslz4_work *) { return false; }

/* dense CSC "dot then threshold": dot.dense over every pixel, then a
 * separate >cut collect.  The matrix is mask-folded at build time, so masked
 * pixels have empty columns and contribute nothing -- no mask test is needed
 * here (there is no per-pixel `if(mask)`); the >cut collect still uses the
 * mask.  No compaction: tidx/tval are not needed. */
template<typename T>
static inline
int dense_dot(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_csc *BSLZ4_RESTRICT m = (const bslz4_mat_csc *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const float *BSLZ4_RESTRICT data = (const float *) m->data;
    const uint32_t *BSLZ4_RESTRICT indices = m->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = m->indptr;
    for (size_t j = 0; j < n; j++) {
        uint32_t k0 = indptr[j + i0], k1 = indptr[j + i0 + 1];
        T pv = px[j];
        for (uint32_t k = k0; k < k1; k++)
            out[indices[k]] += (double) data[k] * (double) pv;
    }
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect,
                               (T *) w->out_vals, w->out_adr);
}

/* a read prefetch hint, gcc/clang or MSVC x64 (else nothing) */
static inline void bslz4_prefetch(const void *p) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p);
#elif defined(_MSC_VER) && defined(_M_X64)
    _mm_prefetch((const char *) p, _MM_HINT_T0);
#else
    (void) p;
#endif
}

#ifndef BSLZ4_DOT_PREFETCH
/* sparse CSC route: software prefetch distance in pixels (0 = off) */
#define BSLZ4_DOT_PREFETCH 8
#endif

#ifndef BSLZ4_DOT_STAGE
/* sparse CSC route: pixels per staging chunk (0 = walk each pixel's entries) */
#define BSLZ4_DOT_STAGE 128
#endif

/* The sparse-route compaction (non-zero, unmasked pixels into tv/tidx), or
 * the list the driver already built for this block (w->precompacted). */
template<typename T>
static inline
int bslz4_collect_nz_w(const bslz4_work *BSLZ4_RESTRICT w, const T *BSLZ4_RESTRICT px,
                       const uint8_t *BSLZ4_RESTRICT mask, size_t i0, size_t n,
                       T *BSLZ4_RESTRICT tv, uint32_t *BSLZ4_RESTRICT tidx) {
    if (w->precompacted) return w->pre_nz;
    return bslz4_collect_nz<T>(px, mask, i0, n, w->collect, tv, tidx);
}

/* sparse CSC route body over an explicit csc: compact non-zeros into
 * (tval,tidx), dot.sparse over the list, then the >cut collect over the same
 * list.  Shared by the CSC dots (which read the descriptor from w->mat) and
 * bsb-csr (which reads the nested csc). */
template<typename T>
static inline
int sparse_dot_core(const bslz4_work *BSLZ4_RESTRICT w, const bslz4_mat_csc *BSLZ4_RESTRICT m) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    T *tv = (T *) w->tval;
    uint32_t *BSLZ4_RESTRICT tidx = w->tidx;
    int npx = 0;
    int nz = bslz4_collect_nz_w<T>(w, px, mask, i0, n, tv, tidx);
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const float *BSLZ4_RESTRICT data = (const float *) m->data;
    const uint32_t *BSLZ4_RESTRICT indices = m->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = m->indptr;
#if BSLZ4_DOT_STAGE
    /* Staged: for a chunk of pixels, each writes 8 (entry, value) slots with
     * fixed stores and advances by its entry count; then one branch-free
     * loop applies all staged entries.  The per-pixel loop over 2 or 3
     * entries (pyFAI bbox, at random) mispredicted about once per pixel.
     * Only for blocks averaging 1.5-4 entries per pixel: at 1 (no split) the
     * staging is pure overhead, at ~7 (5 bins per pixel) it lost too
     * (real WAu and synthetic frames, 2026-10-04). */
    const size_t eblk = (size_t) (indptr[i0 + n] - indptr[i0]);
    const int stage = 2 * eblk > 3 * n && eblk < 4 * n;
    uint32_t sk[BSLZ4_DOT_STAGE * 8 + 8];
    float sv[BSLZ4_DOT_STAGE * 8 + 8];
    for (int base = 0; stage && base < nz; base += BSLZ4_DOT_STAGE) {
        const int end = nz - base < BSLZ4_DOT_STAGE ? nz : base + BSLZ4_DOT_STAGE;
        int pos = 0;
        for (int kk = base; kk < end; kk++) {
#if BSLZ4_DOT_PREFETCH
            if (kk + 2 * BSLZ4_DOT_PREFETCH < nz)
                bslz4_prefetch(&indptr[tidx[kk + 2 * BSLZ4_DOT_PREFETCH]]);
            if (kk + BSLZ4_DOT_PREFETCH < nz) {
                const uint32_t kp = indptr[tidx[kk + BSLZ4_DOT_PREFETCH]];
                bslz4_prefetch(&indices[kp]);
                bslz4_prefetch(&data[kp]);
            }
#endif
            const uint32_t addr = tidx[kk];
            const uint32_t k0 = indptr[addr], len = indptr[addr + 1] - k0;
            const float v = (float) tv[kk];
            if (!BSLZ4_UNLIKELY(len > 8)) {
                for (int j = 0; j < 8; j++) { sk[pos + j] = k0 + (uint32_t) j; sv[pos + j] = v; }
                pos += (int) len;
            } else {                    /* rare: flush, then this pixel directly */
                for (int e = 0; e < pos; e++)
                    out[indices[sk[e]]] += (double) data[sk[e]] * (double) sv[e];
                pos = 0;
                for (uint32_t k = k0; k < k0 + len; k++)
                    out[indices[k]] += (double) data[k] * (double) tv[kk];
            }
        }
        for (int e = 0; e < pos; e++)
            out[indices[sk[e]]] += (double) data[sk[e]] * (double) sv[e];
    }
    for (int kk = 0; !stage && kk < nz; kk++) {
#else
    for (int kk = 0; kk < nz; kk++) {
#endif
#if BSLZ4_DOT_PREFETCH
        /* the pixel list is known: fetch indptr two strides ahead, then the
         * entries one stride ahead (each active pixel lands at a random
         * place in the multi-MB matrix; latency, not arithmetic) */
        if (kk + 2 * BSLZ4_DOT_PREFETCH < nz)
            bslz4_prefetch(&indptr[tidx[kk + 2 * BSLZ4_DOT_PREFETCH]]);
        if (kk + BSLZ4_DOT_PREFETCH < nz) {
            const uint32_t kp = indptr[tidx[kk + BSLZ4_DOT_PREFETCH]];
            bslz4_prefetch(&indices[kp]);
            bslz4_prefetch(&data[kp]);
        }
#endif
        uint32_t addr = tidx[kk];
        T val = tv[kk];
        uint32_t k0 = indptr[addr], k1 = indptr[addr + 1];
        for (uint32_t k = k0; k < k1; k++)
            out[indices[k]] += (double) data[k] * (double) val;
    }
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT out_adr = w->out_adr;
    const int skip = bslz4_cut_above_max<T>(w);
    for (int kk = 0; kk < nz; kk++) {
        T val = tv[kk];
        if (!skip && BSLZ4_UNLIKELY(val > cut)) {
            ov[npx] = val;
            out_adr[npx] = tidx[kk];
            npx++;
        }
    }
    return npx;
}

/* sparse CSC route over the CSC descriptor in w->mat. */
template<typename T>
static inline
int sparse_dot(const bslz4_work *BSLZ4_RESTRICT w) {
    return sparse_dot_core<T>(w, (const bslz4_mat_csc *) w->mat);
}

/* ---- csc-run (start + length) and csc-nosplit (histogram) ----------
 *
 * New dots on the one CSC entry; the arrays mean something else per dot id:
 *
 *   csc-run     indices[p] = first bin of pixel p (n = npix); its entries
 *               are data[indptr[p]..indptr[p+1]] for the consecutive bins
 *               indices[p], indices[p]+1, ...  One bin read per pixel instead
 *               of one per entry.
 *   csc-nosplit indices[p] = the one bin of pixel p, or BSLZ4_NO_BIN; the
 *               weight is 1, so neither data nor indptr is read.
 *
 * Kept as separate leaves (not shared with the csc kernels above) so the
 * existing kernels are unchanged while these are measured. */
#define BSLZ4_NO_BIN 0xFFFFFFFFu

template<typename T>
static inline
int run_dense(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_csc *BSLZ4_RESTRICT m = (const bslz4_mat_csc *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const float *BSLZ4_RESTRICT data = (const float *) m->data;
    const uint32_t *BSLZ4_RESTRICT start = m->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = m->indptr;
    for (size_t j = 0; j < n; j++) {
        uint32_t k0 = indptr[j + i0], k1 = indptr[j + i0 + 1];
        double *BSLZ4_RESTRICT o = out + start[j + i0];
        const float *BSLZ4_RESTRICT d = data + k0;
        double pv = (double) px[j];
        for (uint32_t k = 0; k < k1 - k0; k++)
            o[k] += (double) d[k] * pv;
    }
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect,
                               (T *) w->out_vals, w->out_adr);
}

template<typename T>
static inline
int run_sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    T *tv = (T *) w->tval;
    uint32_t *BSLZ4_RESTRICT tidx = w->tidx;
    int npx = 0;
    int nz = bslz4_collect_nz_w<T>(w, px, mask, i0, n, tv, tidx);
    const bslz4_mat_csc *BSLZ4_RESTRICT m = (const bslz4_mat_csc *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const float *BSLZ4_RESTRICT data = (const float *) m->data;
    const uint32_t *BSLZ4_RESTRICT start = m->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = m->indptr;
    for (int kk = 0; kk < nz; kk++) {
        uint32_t addr = tidx[kk];
        uint32_t k0 = indptr[addr], k1 = indptr[addr + 1];
        double *BSLZ4_RESTRICT o = out + start[addr];
        const float *BSLZ4_RESTRICT d = data + k0;
        double val = (double) tv[kk];
        for (uint32_t k = 0; k < k1 - k0; k++)
            o[k] += (double) d[k] * val;
    }
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT out_adr = w->out_adr;
    const int skip = bslz4_cut_above_max<T>(w);
    for (int kk = 0; kk < nz; kk++) {
        T val = tv[kk];
        if (!skip && BSLZ4_UNLIKELY(val > cut)) {
            ov[npx] = val;
            out_adr[npx] = tidx[kk];
            npx++;
        }
    }
    return npx;
}

template<typename T>
static inline
int nosplit_dense(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_csc *BSLZ4_RESTRICT m = (const bslz4_mat_csc *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const uint32_t *BSLZ4_RESTRICT bin = m->indices + i0;
    for (size_t j = 0; j < n; j++) {
        uint32_t b = bin[j];
        if (b != BSLZ4_NO_BIN) out[b] += (double) px[j];
    }
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect,
                               (T *) w->out_vals, w->out_adr);
}

template<typename T>
static inline
int nosplit_sparse_core(const bslz4_work *BSLZ4_RESTRICT w, const uint32_t *BSLZ4_RESTRICT bin) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    T *tv = (T *) w->tval;
    uint32_t *BSLZ4_RESTRICT tidx = w->tidx;
    int npx = 0;
    int nz = bslz4_collect_nz_w<T>(w, px, mask, i0, n, tv, tidx);
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    for (int kk = 0; kk < nz; kk++) {
        uint32_t b = bin[tidx[kk]];
        if (b != BSLZ4_NO_BIN) out[b] += (double) tv[kk];
    }
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT out_adr = w->out_adr;
    const int skip = bslz4_cut_above_max<T>(w);
    for (int kk = 0; kk < nz; kk++) {
        T val = tv[kk];
        if (!skip && BSLZ4_UNLIKELY(val > cut)) {
            ov[npx] = val;
            out_adr[npx] = tidx[kk];
            npx++;
        }
    }
    return npx;
}

template<typename T>
static inline
int nosplit_sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    return nosplit_sparse_core<T>(w, ((const bslz4_mat_csc *) w->mat)->indices);
}

/* csc-nosplit-moment: a histogram with a first moment beside each output.
 * Pixel p reaches the pair of bins b = indices[p] (sum I) and b+1 (sum qI),
 * or none (BSLZ4_NO_BIN); data[p] is its q (n = npix).  The sum I bin gets
 * the value with no multiply, the sum qI bin gets value * q.  indptr is not
 * read.  This is the interleaved [I, qI, I, qI, ...] output order. */
template<typename T>
static inline
int nosplit_moment_dense(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_csc *BSLZ4_RESTRICT m = (const bslz4_mat_csc *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const uint32_t *BSLZ4_RESTRICT bin = m->indices + i0;
    const float *BSLZ4_RESTRICT q = (const float *) m->data + i0;
    for (size_t j = 0; j < n; j++) {
        uint32_t b = bin[j];
        if (b != BSLZ4_NO_BIN) {
            double v = (double) px[j];
            out[b] += v;
            out[b + 1] += v * (double) q[j];
        }
    }
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect,
                               (T *) w->out_vals, w->out_adr);
}

template<typename T>
static inline
int nosplit_moment_sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    T *tv = (T *) w->tval;
    uint32_t *BSLZ4_RESTRICT tidx = w->tidx;
    int npx = 0;
    int nz = bslz4_collect_nz_w<T>(w, px, mask, i0, n, tv, tidx);
    const bslz4_mat_csc *BSLZ4_RESTRICT m = (const bslz4_mat_csc *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const uint32_t *BSLZ4_RESTRICT bin = m->indices;
    const float *BSLZ4_RESTRICT q = (const float *) m->data;
    for (int kk = 0; kk < nz; kk++) {
        uint32_t p = tidx[kk];
        uint32_t b = bin[p];
        if (b != BSLZ4_NO_BIN) {
            double v = (double) tv[kk];
            out[b] += v;
            out[b + 1] += v * (double) q[p];
        }
    }
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT out_adr = w->out_adr;
    const int skip = bslz4_cut_above_max<T>(w);
    for (int kk = 0; kk < nz; kk++) {
        T val = tv[kk];
        if (!skip && BSLZ4_UNLIKELY(val > cut)) {
            ov[npx] = val;
            out_adr[npx] = tidx[kk];
            npx++;
        }
    }
    return npx;
}

