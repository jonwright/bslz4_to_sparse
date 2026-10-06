#pragma once
/*
 * Step "dot", the layouts read through a body per pixel value on the CSC
 * entry:
 *
 *   csc-permute     a permutation (FAZIT: each pixel to its own output slot,
 *                   b = indices[p]); the value is stored, not added, in the
 *                   pixel dtype, so the output is the image re-ordered.
 *                   4M dense 8.2 ms vs 27.7 for the double powder of csc.
 *   csc-int         fixed-point integer weights (u32) per CSC entry,
 *   csc-run-int     ... start bin per pixel + u32 weights (csc-run's layout),
 *   csc-run-int16   ... start bin per pixel + u16 weights:
 *                   exact int64 sums of w * v, so sparse pixels can later be
 *                   subtracted from the integration without loss.
 *
 * Each is a "body" (what one pixel value adds); the dense route calls it for
 * every pixel of the block, the sparse route for the listed non-zero ones.
 */

#include "dot_csc.hpp"

#include <string.h>

namespace bslz4v {

/* start bin of pixel p: indices[p] */
struct Start32 {
    const uint32_t *BSLZ4_RESTRICT s;
    inline void init(const bslz4_work *w) { s = ((const bslz4_mat_csc *) w->mat)->indices; }
    inline uint32_t at(size_t p) { return s[p]; }
};

/* ---- bodies: what one pixel value adds ------------------------------- */

/* split, start + length, fixed-point integer weights W (u32 / u16), int64
 * powder: exact integer sums of w * v. */
template<typename S, typename W>
struct RunInt {
    S src;
    const W *BSLZ4_RESTRICT data;
    const uint32_t *BSLZ4_RESTRICT indptr;
    int64_t *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
        src.init(w);
        data = (const W *) m->data;
        indptr = m->indptr;
        out = (int64_t *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t s = src.at(p);
        const uint32_t k0 = indptr[p], k1 = indptr[p + 1];
        int64_t *BSLZ4_RESTRICT o = out + s;
        const W *BSLZ4_RESTRICT d = data + k0;
        const int64_t pv = (int64_t) v;
        for (uint32_t k = 0; k < k1 - k0; k++) o[k] += (int64_t) d[k] * pv;
    }
};

/* general csc, one u32 bin per entry, u32 fixed-point weights, int64 powder */
struct CscInt {
    const uint32_t *BSLZ4_RESTRICT data;
    const uint32_t *BSLZ4_RESTRICT indices;
    const uint32_t *BSLZ4_RESTRICT indptr;
    int64_t *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
        data = (const uint32_t *) m->data;
        indices = m->indices;
        indptr = m->indptr;
        out = (int64_t *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t k0 = indptr[p], k1 = indptr[p + 1];
        const int64_t pv = (int64_t) v;
        for (uint32_t k = k0; k < k1; k++) out[indices[k]] += (int64_t) data[k] * pv;
    }
};

/* permutation (FAZIT): every pixel has its own output slot, b = indices[p]
 * (BSLZ4_NO_BIN: none); the value is stored, not added, in the pixel dtype,
 * so the output is the image itself in (radius, azimuth) order.  Slots of
 * pixels that are zero (sparse route) keep the driver's zero fill. */
template<typename TP>
struct Permute {
    const uint32_t *BSLZ4_RESTRICT bin;
    TP *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        bin = ((const bslz4_mat_csc *) w->mat)->indices;
        out = (TP *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t b = bin[p];
        if (b != 0xFFFFFFFFu) out[b] = (TP) v;
    }
};

/* ---- the two loops ---------------------------------------------------- */

template<typename T, typename Body>
static inline int dense(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    if (n == 0) return 0;
    Body body;
    body.init(w);
    for (size_t j = 0; j < n; j++) body.add(j + i0, px[j]);
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect,
                               (T *) w->out_vals, w->out_adr);
}

template<typename T, typename Body>
static inline int sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    if (n == 0) return 0;
    T *tv = (T *) w->tval;
    uint32_t *BSLZ4_RESTRICT tidx = w->tidx;
    int nz = bslz4_collect_nz_w<T>(w, px, mask, i0, n, tv, tidx);
    if (nz > 0) {
        Body body;
        body.init(w);
        for (int kk = 0; kk < nz; kk++) body.add(tidx[kk], tv[kk]);
    }
    int npx = 0;
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

template<typename T, typename Body>
static inline int run(const bslz4_work *BSLZ4_RESTRICT w) {
    return w->route ? sparse<T, Body>(w) : dense<T, Body>(w);
}

template<typename T>
static inline int dispatch(const bslz4_work *BSLZ4_RESTRICT w) {
    switch (w->dot) {
    case BSLZ4_DOT_CSC_PERMUTE:   return run<T, Permute<T> >(w);
    case BSLZ4_DOT_CSC_INT:       return run<T, CscInt>(w);
    case BSLZ4_DOT_CSC_RUN_INT:   return run<T, RunInt<Start32, uint32_t> >(w);
    case BSLZ4_DOT_CSC_RUN_INT16: return run<T, RunInt<Start32, uint16_t> >(w);
    default:                      return BSLZ4_ERR_BAD_LAYOUT;
    }
}

} /* namespace bslz4v */
