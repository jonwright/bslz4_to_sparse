#pragma once
/*
 * Step "dot", padded: for matrices whose pixels each hit one run of
 * consecutive bins (1D bbox / full splitting; ring sums with azimuth as the
 * fast axis).  Each row (one per pixel, or one per listed pixel) holds a
 * first bin "base" and "width" float32 weights for bins base .. base +
 * width - 1, zero padded where the pixel touches fewer bins.  A pixel value
 * v adds v * weights[k] to out[base + k]: one destination and a small vector
 * multiply-add, no index read per entry.  _padded_from_csc (Python) checks
 * the matrix once.  The best 1D split layout measured (2026-10-06, Zen 4,
 * default route): 9 % frames 3.1 ms vs 3.9 csc, WAu0012 0.79 vs 0.99.
 *
 * The mask is folded into the matrix first, so a masked pixel's row has
 * zero weights (implicit layout) or no row (listed layout): no mask tests.
 *
 * Values: padded (scalar) and padded-avx2 (AVX2 + FMA rows on the dense
 * route; the sparse route is the same scalar walk for both).  Only widths <=
 * PADDED_MAX_SIMD_WIDTH are vectorised; the row kernels are templated on W
 * so the guards fold away.  The dense route walks a block's rows in tiles,
 * STRIDE rows apart per pass, so consecutive rows' output windows (usually
 * one bin apart) are not still in flight -- overlapping FMA stores defeat
 * store-to-load forwarding.
 */

#include "dot_csc.hpp"

#if BSLZ4_X86_SIMD
#include <immintrin.h>
#include "c2py_amd64.h"
#endif

namespace bslz4 {

constexpr int PADDED_MAX_WIDTH = 64;       /* widest layout accepted */
constexpr int PADDED_MAX_SIMD_WIDTH = 8;   /* widest with a vectorised kernel */
constexpr size_t PADDED_STRIDE = 16;
constexpr size_t PADDED_TILE = 128;

inline bool bslz4_padded_avx2_capable() {
#if BSLZ4_X86_SIMD
    return (c2py_amd64_avx2 && c2py_amd64_fma) != 0;
#else
    return false;
#endif
}

/* the pixel of row r */
inline size_t bslz4_padded_pixel(const bslz4_mat_padded *m, size_t r) {
    return m->listed ? (size_t) m->pixels[r] : r;
}

template<typename T>
inline void bslz4_padded_row_scalar(const T *block, size_t i0, const bslz4_mat_padded *m,
                                    size_t r, double *out) {
    const double px = (double) block[bslz4_padded_pixel(m, r) - i0];
    const float *w = (const float *) m->weights + r * (size_t) m->width;
    double *o = (double *) out + m->base[r];
    for (int k = 0; k < m->width; k++) o[k] += (double) w[k] * px;
}

#if BSLZ4_X86_SIMD
/* A guard keeps each full-width store inside the row, so neighbouring rows'
 * bins are never clobbered. */
template<typename T, int W>
__attribute__((target("avx2,fma")))
inline void bslz4_padded_row_avx2(const T *block, size_t i0, const bslz4_mat_padded *m,
                                  size_t r, double *out) {
    const double p = (double) block[bslz4_padded_pixel(m, r) - i0];
    const float *w = (const float *) m->weights + r * (size_t) W;
    double *o = (double *) out + m->base[r];
    int c = 0;
    if (W - c >= 4) {
        const __m256d px = _mm256_set1_pd(p);
        _mm256_storeu_pd(o + c, _mm256_fmadd_pd(px, _mm256_cvtps_pd(_mm_loadu_ps(w + c)),
                                                _mm256_loadu_pd(o + c)));
        c += 4;
    }
    if (W - c >= 2) {
        const __m128d px = _mm_set1_pd(p);
        const __m128d wd = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) (w + c))));
        _mm_storeu_pd(o + c, _mm_fmadd_pd(px, wd, _mm_loadu_pd(o + c)));
        c += 2;
    }
    for (; c < W; c++) o[c] += p * (double) w[c];
}

/* rows r0..r1 in tiles, STRIDE apart */
template<typename T, int W>
__attribute__((target("avx2,fma")))
inline void bslz4_padded_rows_avx2(const T *block, size_t i0, const bslz4_mat_padded *m,
                                   size_t r0, size_t r1, double *out) {
    for (size_t t0 = r0; t0 < r1; t0 += PADDED_TILE) {
        const size_t t1 = (t0 + PADDED_TILE < r1) ? t0 + PADDED_TILE : r1;
        for (size_t s = 0; s < PADDED_STRIDE; s++)
            for (size_t r = t0 + s; r < t1; r += PADDED_STRIDE)
                bslz4_padded_row_avx2<T, W>(block, i0, m, r, out);
    }
}
#endif

/* Accumulate the rows of the block holding pixel i0 into this frame's bins
 * (block = i0 / block_elems).  The width dispatch happens once per block,
 * keeping the dense loop free of switches.  The tail block is bounded by the
 * matrix's own npix, which the driver checks equals the frame's pixel count.
 * Called only with n > 0. */
template<typename T>
inline void bslz4_padded_dense_block(const T *block, size_t i0, const bslz4_mat_padded *m,
                                     bool avx2, double *out) {
    const size_t block_idx = i0 / m->block_elems;
    const size_t r0 = (size_t) m->row_ptr[block_idx];
    const size_t r1 = (size_t) m->row_ptr[block_idx + 1];
#if BSLZ4_X86_SIMD
    if (avx2) {
        switch (m->width) {
        case 1: bslz4_padded_rows_avx2<T, 1>(block, i0, m, r0, r1, out); return;
        case 2: bslz4_padded_rows_avx2<T, 2>(block, i0, m, r0, r1, out); return;
        case 3: bslz4_padded_rows_avx2<T, 3>(block, i0, m, r0, r1, out); return;
        case 4: bslz4_padded_rows_avx2<T, 4>(block, i0, m, r0, r1, out); return;
        case 5: bslz4_padded_rows_avx2<T, 5>(block, i0, m, r0, r1, out); return;
        case 6: bslz4_padded_rows_avx2<T, 6>(block, i0, m, r0, r1, out); return;
        case 7: bslz4_padded_rows_avx2<T, 7>(block, i0, m, r0, r1, out); return;
        case 8: bslz4_padded_rows_avx2<T, 8>(block, i0, m, r0, r1, out); return;
        default: break;                         /* wider: the scalar loop */
        }
    }
#else
    (void) avx2;
#endif
    for (size_t r = r0; r < r1; r++)
        bslz4_padded_row_scalar<T>(block, i0, m, r, out);
}

} /* namespace bslz4 */

/* dense padded route: accumulate the powder over this block's rows (scalar or
 * AVX2), then the >cut sparse output in ascending pixel order.  Rows of
 * masked pixels carry all-zero weights (the matrix is mask-folded), so no row
 * tests the mask.  An empty tail (n == 0, when the frame is a whole number of
 * blocks) has no row_ptr entry of its own and is skipped. */
template<typename T>
static inline
int padded_dense(const bslz4_work *BSLZ4_RESTRICT w, bool avx2) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    const bslz4_mat_padded *BSLZ4_RESTRICT m = (const bslz4_mat_padded *) w->mat;
    double *BSLZ4_RESTRICT out = (double *) w->powder;
    if (n == 0) return 0;
    bslz4_padded_dense_block<T>(px, i0, m, avx2, out);
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect,
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

