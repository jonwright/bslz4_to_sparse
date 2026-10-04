/*
 * kernels_generic.cpp -- the only genuinely dtype-generic code.
 *
 * The dtype-generic work is an inner per-block/tail call, never the outer
 * multi-frame loop (which lives in bslz4_driver.c).  This TU holds:
 *   * sparse<T>          -- plain route
 *   * sparse_dot<T>      -- sparse CSC route
 *   * dense_dot<T>       -- dense CSC route (dot, then a separate >cut)
 *   * dense_dot_fused<T> -- dense route with the >cut sparsify fused into
 *                           the CSC loop (one pass over the block)
 * for the 10 pixel dtypes, plus the scalar/nz/dot loops they need.
 *
 * Each dtype gets a small *noinline* kernel (bslz4_sparse_u8, ...,
 * bslz4_sparse_dot_f64) that inlines its own template, so each kernel is
 * register-allocated independently; the dispatch is a thin switch of direct
 * calls.  A new route strategy is just another leaf kernel selected by
 * dot_id.
 *
 * The mask is folded into the matrix once, in Python (_fold_mask drops the
 * entries of masked pixels, so their columns are empty).  The dense matvec
 * loops therefore need no mask test; the mask itself is still passed in and
 * used by the sparse-output (>cut) collect and by the sparse-route
 * compaction (bslz4_collect_nz).  Every route gives the same powder for a
 * folded matrix; a caller of the C entries must pass one.
 *
 * The SIMD collect tiers are the bslz4_collect_gt<T>/bslz4_collect_nz<T>
 * entry points (bslz4_collect_simd.hpp); the collect tier id chosen by the
 * pipeline is dispatched with a predicted if-chain.
 *
 * The per-block instrumentation counters are bumped by the driver
 * (bslz4_driver.c), one bump per block for the collect/dot ids, so a test
 * can show the selected impl ran in every route and in the tail and that
 * the scalar collect tier was not picked for a SIMD dtype.
 *
 * Compiled with -fno-exceptions -fno-rtti -fno-threadsafe-statics (C++11)
 * only for this TU.
 */

#include "bslz4_common.h"
#include "bslz4_collect_simd.hpp"
#include "bslz4_padded.hpp"
#include "bslz4_registry.h"
#include "bslz4_collect_caps.h"

#include <limits>

using namespace bslz4;

/* A threshold above the dtype's maximum means no pixel can ever satisfy
 * `> cut`, so the >cut sparse output is empty.  Skip the O(n) scan in that
 * case (and avoid the wrapped `(T)threshold` cast that would otherwise emit
 * garbage against a small wrapped cut); the matvec/powder is unaffected.
 * The default treats T as an integral type whose maximum fits in int64;
 * uint64_t and the floating dtypes have a maximum above any int64 threshold,
 * so they never skip (an int64 threshold can never exceed them). */
template<typename T> struct bslz4_cutmax {
    static inline bool above(int64_t th) {
        return th > (int64_t) std::numeric_limits<T>::max();
    }
};
template<> struct bslz4_cutmax<uint64_t> { static inline bool above(int64_t) { return false; } };
template<> struct bslz4_cutmax<float>    { static inline bool above(int64_t) { return false; } };
template<> struct bslz4_cutmax<double>   { static inline bool above(int64_t) { return false; } };

#define BSLZ4_CUT_ABOVE_MAX(w) \
    (bslz4_cutmax<T>::above((int64_t)(w)->threshold))

/* C-linkage availability for the SIMD collect tiers (bslz4_collect_caps.h);
 * used by bslz4_impl_available(COLLECT, id).  The pick itself is by collect id
 * in the decode stages, so there is no set_/get_ state here anymore. */
extern "C" int bslz4_available_avx512_collect(void) { return bslz4_avx512_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_avx512cs_collect(void) { return bslz4_avx512cs_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_avx2_collect(void)   { return bslz4_avx2_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_sse2_collect(void)   { return bslz4_sse2_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_vsx_collect(void)    { return bslz4_vsx_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_neon_collect(void)   { return bslz4_neon_collect_capable() ? 1 : 0; }

extern "C" int bslz4_available_avx2_padded(void)    { return bslz4_padded_avx2_capable() ? 1 : 0; }
extern "C" int bslz4_available_sse2_padded(void)    { return bslz4_padded_sse2_capable() ? 1 : 0; }
extern "C" int bslz4_available_avx512_padded(void)  { return bslz4_padded_avx512_capable() ? 1 : 0; }

/* ---- per-dtype generic kernels (all take a single bslz4_work*) ----
 *
 * Each is a self-contained function that inlines its own template and
 * collect/dot selection.  They are *noinline* so each kernel is
 * register-allocated independently; the dispatchers are thin switches that
 * call them.  The collect tier and dot implementation are runtime ids in the
 * work struct, so a new route strategy is a leaf function + a switch case.
 */

/* plain sparse, generic */
template<typename T>
static inline
int sparse_plain(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>((const T *) w->block, w->no_mask ? NULL : w->mask, w->i0,
                               w->n, cut, w->collect_id, (T *) w->out_vals, w->out_adr);
}

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
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect_id,
                               (T *) w->out_vals, w->out_adr);
}

/* dense CSC "fused": dot.dense and the >cut sparse-output collect run in one
 * pass over the block.  The matrix is mask-folded (masked columns empty), so
 * the matvec alone would need no mask test; the >cut emission must gate on
 * the image mask, and since that test is made anyway it also skips the
 * (empty) column walk of a masked pixel. */
template<typename T>
static inline
int dense_dot_fused(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_csc *BSLZ4_RESTRICT m = (const bslz4_mat_csc *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const float *BSLZ4_RESTRICT data = (const float *) m->data;
    const uint32_t *BSLZ4_RESTRICT indices = m->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = m->indptr;
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT oadr = w->out_adr;
    int npx = 0;
    const int skip = BSLZ4_CUT_ABOVE_MAX(w);
    for (size_t j = 0; j < n; j++) {
        if (bslz4_mask_ok(mask, j + i0)) {
            uint32_t k0 = indptr[j + i0], k1 = indptr[j + i0 + 1];
            T pv = px[j];
            for (uint32_t k = k0; k < k1; k++)
                out[indices[k]] += (double) data[k] * (double) pv;
            if (!skip && BSLZ4_UNLIKELY(pv > cut)) {
                ov[npx] = pv;
                oadr[npx] = (uint32_t) (j + i0);
                npx++;
            }
        }
    }
    return npx;
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
    return bslz4_collect_nz<T>(px, mask, i0, n, w->collect_id, tv, tidx);
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
                __builtin_prefetch(&indptr[tidx[kk + 2 * BSLZ4_DOT_PREFETCH]]);
            if (kk + BSLZ4_DOT_PREFETCH < nz) {
                const uint32_t kp = indptr[tidx[kk + BSLZ4_DOT_PREFETCH]];
                __builtin_prefetch(&indices[kp]);
                __builtin_prefetch(&data[kp]);
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
            __builtin_prefetch(&indptr[tidx[kk + 2 * BSLZ4_DOT_PREFETCH]]);
        if (kk + BSLZ4_DOT_PREFETCH < nz) {
            const uint32_t kp = indptr[tidx[kk + BSLZ4_DOT_PREFETCH]];
            __builtin_prefetch(&indices[kp]);
            __builtin_prefetch(&data[kp]);
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
    const int skip = BSLZ4_CUT_ABOVE_MAX(w);
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
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect_id,
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
    const int skip = BSLZ4_CUT_ABOVE_MAX(w);
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
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect_id,
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
    const int skip = BSLZ4_CUT_ABOVE_MAX(w);
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
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect_id,
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
    const int skip = BSLZ4_CUT_ABOVE_MAX(w);
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

/* padded route tier for a padded dot id: 2 scalar (0), 3 sse2 (1), 4 avx2 (2),
 * 5 avx512 (3) */
static inline int padded_tier(int id) {
    return id == 2 ? 0 : (id == 3 ? 1 : (id == 4 ? 2 : 3));
}

/* dense padded route: accumulate the powder over this block's rows (scalar or
 * a SIMD tier), then the >cut sparse output in ascending pixel order.  Rows of
 * masked pixels carry all-zero weights (the matrix is mask-folded), so no row
 * tests the mask.  An empty tail (n == 0, when the frame is a whole number of
 * blocks) has no row_ptr entry of its own and is skipped. */
template<typename T>
static inline
int padded_dense(const bslz4_work *BSLZ4_RESTRICT w, int tier) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_padded *BSLZ4_RESTRICT m = (const bslz4_mat_padded *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    if (n == 0) return 0;
    bslz4_padded_dense_tier<T>(px, i0, n, m, mask, tier, out);
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect_id,
                               (T *) w->out_vals, w->out_adr);
}

/* padded sparse route: compact non-zeros, then accumulate each row with a
 * scalar row walk.  Measured: on the sparse path a strided SIMD walk is
 * *slower* than this -- the compacted non-zero pixels are already scattered
 * (not adjacent-raster), so a stride hurts cache locality, and the SIMD row
 * barely helps for the small widths here.  The padded layout's sparse-route
 * edge over CSC is index-free accumulation, captured by the scalar walk. */
template<typename T>
static inline
int padded_sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_padded *BSLZ4_RESTRICT m = (const bslz4_mat_padded *) w->mat;
    T *tv = (T *) w->tval;
    uint32_t *BSLZ4_RESTRICT tidx = w->tidx;
    int npx = 0;
    int nz = bslz4_collect_nz_w<T>(w, px, mask, i0, n, tv, tidx);
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    const size_t W = (size_t) m->width;
    const int32_t *BSLZ4_RESTRICT base = m->base;
    const float *BSLZ4_RESTRICT weights = (const float *) m->weights;
    for (int kk = 0; kk < nz; kk++) {
        const uint32_t pixel = tidx[kk];
        const int row = m->listed ? (int) m->rowmap[pixel] : (int) pixel;
        if (row < 0) continue;
        const double v = (double) tv[kk];
        const float *wr = weights + (size_t) row * W;
        double *o = out + base[row];
        for (size_t k = 0; k < W; k++) o[k] += (double) wr[k] * v;
    }
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT out_adr = w->out_adr;
    const int skip = BSLZ4_CUT_ABOVE_MAX(w);
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
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, w->no_mask ? NULL : w->mask, i0, n, cut,
                               w->collect_id, (T *) w->out_vals, w->out_adr);
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
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, w->no_mask ? NULL : w->mask, i0, n, cut,
                               w->collect_id, (T *) w->out_vals, w->out_adr);
}

template<typename T>
static inline
int bsbcsr_nosplit_sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    const bslz4_mat_bsbcsr *BSLZ4_RESTRICT m = (const bslz4_mat_bsbcsr *) w->mat;
    return nosplit_sparse_core<T>(w, m->csc.indices);
}

#include "bslz4_csc_variants.hpp"

/* ---- per-dtype noinline kernels ------------------------------------ */

#define BSLZ4_DTYPE_KERNELS(T, name)                                      \
    extern "C" BSLZ4_NOINLINE int bslz4_sparse_##name(const bslz4_work *w) { \
        return sparse_plain<T>(w);                                        \
    }                                                                     \
    extern "C" BSLZ4_NOINLINE int bslz4_sparse_dot_##name(const bslz4_work *w) { \
        const int id = w->dot_id;                                         \
        if (id <= 1) {                                                    \
            if (w->route) return sparse_dot<T>(w);                        \
            return id == 1 ? dense_dot_fused<T>(w) : dense_dot<T>(w);     \
        } else if (id == 2 || id == 3 || id == 4 || id == 5) {            \
            const int tier = padded_tier(id);                             \
            if (w->route) return padded_sparse<T>(w);                     \
            return padded_dense<T>(w, tier);                              \
        } else if (id == 6) {                                             \
            if (w->route) return bsbcsr_sparse<T>(w);                     \
            return bsbcsr_dense<T>(w);                                    \
        } else if (id == 7) {                                             \
            if (w->route) return run_sparse<T>(w);                        \
            return run_dense<T>(w);                                       \
        } else if (id == 8) {                                             \
            if (w->route) return nosplit_sparse<T>(w);                    \
            return nosplit_dense<T>(w);                                   \
        } else if (id == 9) {                                             \
            if (w->route) return bsbcsr_nosplit_sparse<T>(w);             \
            return bsbcsr_nosplit_dense<T>(w);                            \
        } else if (id == 10) {                                            \
            if (w->route) return nosplit_moment_sparse<T>(w);             \
            return nosplit_moment_dense<T>(w);                            \
        } else if (id >= 11) {                                            \
            return bslz4v::dispatch<T>(w);                                \
        }                                                                 \
        return BSLZ4_ERR_BAD_LAYOUT;                                      \
    }

BSLZ4_DTYPE_KERNELS(uint8_t,  u8)
BSLZ4_DTYPE_KERNELS(uint16_t, u16)
BSLZ4_DTYPE_KERNELS(uint32_t, u32)
BSLZ4_DTYPE_KERNELS(uint64_t, u64)
BSLZ4_DTYPE_KERNELS(int8_t,   i8)
BSLZ4_DTYPE_KERNELS(int16_t,  i16)
BSLZ4_DTYPE_KERNELS(int32_t,  i32)
BSLZ4_DTYPE_KERNELS(int64_t,  i64)
BSLZ4_DTYPE_KERNELS(float,    f32)
BSLZ4_DTYPE_KERNELS(double,   f64)

#undef BSLZ4_DTYPE_KERNELS

/* ---- thin dtype-dispatching switches -------------------------------- */

extern "C" int bslz4_sparse_dispatch(const bslz4_work *BSLZ4_RESTRICT w) {
    switch (w->dtype) {
    case 0:  return bslz4_sparse_u8(w);
    case 1:  return bslz4_sparse_u16(w);
    case 2:  return bslz4_sparse_u32(w);
    case 3:  return bslz4_sparse_u64(w);
    case 4:  return bslz4_sparse_i8(w);
    case 5:  return bslz4_sparse_i16(w);
    case 6:  return bslz4_sparse_i32(w);
    case 7:  return bslz4_sparse_i64(w);
    case 8:  return bslz4_sparse_f32(w);
    case 9:  return bslz4_sparse_f64(w);
    default: return BSLZ4_ERR_DTYPE;
    }
}

extern "C" int bslz4_sparse_dot_dispatch(const bslz4_work *BSLZ4_RESTRICT w) {
    switch (w->dtype) {
    case 0:  return bslz4_sparse_dot_u8(w);
    case 1:  return bslz4_sparse_dot_u16(w);
    case 2:  return bslz4_sparse_dot_u32(w);
    case 3:  return bslz4_sparse_dot_u64(w);
    case 4:  return bslz4_sparse_dot_i8(w);
    case 5:  return bslz4_sparse_dot_i16(w);
    case 6:  return bslz4_sparse_dot_i32(w);
    case 7:  return bslz4_sparse_dot_i64(w);
    case 8:  return bslz4_sparse_dot_f32(w);
    case 9:  return bslz4_sparse_dot_f64(w);
    default: return BSLZ4_ERR_DTYPE;
    }
}
