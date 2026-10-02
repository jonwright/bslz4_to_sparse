#pragma once
/*
 * Padded-CSC accumulate kernels, integrated into the refactored
 * bslz4_driver.c / kernels_generic.cpp design.
 *
 * A padded CSC matrix stores, for each row (one per pixel, or one per listed
 * pixel), a single first output bin "base" and "width" float32 weights for the
 * consecutive bins base, base+1, ..., base+width-1, zero padded where the
 * pixel touches fewer bins.  A pixel value v adds v*weights[k] to
 * output[base+k]: one destination and one small vector multiply-add, no
 * per-entry index reads.  This only represents matrices whose pixels each hit
 * a run of consecutive bins (1D bbox/no-split integrations, and ring sums
 * with azimuth as the fast axis); _padded_from_csc (Python) checks that once.
 *
 * The mask is NOT folded into the weights: the kernels test mask[pixel] per
 * row, exactly as the CSC dense route does, so the powder is identical to the
 * CSC route and the weights stay untouched.
 *
 * Tiers are chosen by the dot id (scalar / SSE2 / AVX2+FMA / AVX-512+FMA),
 * mirroring the no-flags design of the collect tiers.  Only widths <=
 * PADDED_MAX_SIMD_WIDTH are vectorized; the row/block kernels are template<
 * T,W>-specialized so W is a compile-time constant and the vector guard
 * conditions fold away.  Wider matrices use the scalar loop.
 *
 * The dense route walks a block's rows in tiles, STRIDE rows apart per pass,
 * so consecutive rows' output windows (usually one bin apart) do not overlap
 * in still-in-flight registers -- overlapping FMA stores defeat store-to-load
 * forwarding.  The sparse route compacts non-zeros and applies the same
 * per-row tier to each non-zero pixel (via bslz4_padded_row_tier).
 *
 * The >cut sparse output is produced separately, pixel-major and ascending,
 * by bslz4_collect_gt (see kernels_generic.cpp), so the sparse output stays
 * bitwise-identical to the CSC route.
 */

#include "bslz4_common.h"
#include "bslz4_registry.h"

#include <stddef.h>
#include <stdint.h>

#include "c2py_amd64.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define BSLZ4_HAVE_PADDED_X86 1
#include <immintrin.h>
#else
#define BSLZ4_HAVE_PADDED_X86 0
#endif

namespace bslz4 {

constexpr int PADDED_MAX_WIDTH = 64;       /* widest layout any tier accepts */
constexpr int PADDED_MAX_SIMD_WIDTH = 8;   /* widest with a vectorized kernel */
constexpr size_t PADDED_STRIDE = 16;
constexpr size_t PADDED_TILE = 128;

inline bool bslz4_padded_avx512_capable() {
#if BSLZ4_HAVE_PADDED_X86
    return (c2py_amd64_avx512f && c2py_amd64_fma) != 0;
#else
    return false;
#endif
}

inline bool bslz4_padded_avx2_capable() {
#if BSLZ4_HAVE_PADDED_X86
    return (c2py_amd64_avx2 && c2py_amd64_fma) != 0;
#else
    return false;
#endif
}

inline bool bslz4_padded_sse2_capable() {
#if BSLZ4_HAVE_PADDED_X86
    return true; /* x86-64 ABI baseline */
#else
    return false;
#endif
}

#define BSLZ4_PADDED_PIXEL(M, r) ((M)->listed ? (size_t) (M)->pixels[r] : (size_t) (r))

/* ---------------------------------------------------------------- scalar */

template<typename T>
inline void bslz4_padded_row_scalar(const T *block, size_t i0, const bslz4_mat_padded *m,
                                    const uint8_t *mask, size_t r, double *out) {
    (void) mask;
    const size_t pixel = BSLZ4_PADDED_PIXEL(m, r);
    const double px = (double) block[pixel - i0];
    const float *w = (const float *) m->weights + r * (size_t) m->width;
    double *o = (double *) out + m->base[r];
    for (int k = 0; k < m->width; k++) o[k] += (double) w[k] * px;
}

#if BSLZ4_HAVE_PADDED_X86

/* Per-row, template<W> so W is a compile-time constant and the `W-c>=N`
 * guards fold away.  A guard ensures a full N-wide store stays within the row,
 * so neighbouring rows' bins are never clobbered. */

template<typename T, int W>
__attribute__((target("avx512f,fma")))
inline void bslz4_padded_row_avx512(const T *block, size_t i0, const bslz4_mat_padded *m,
                                    const uint8_t *mask, size_t r, double *out) {
    (void) mask;
    const size_t pixel = BSLZ4_PADDED_PIXEL(m, r);
    const double p = (double) block[pixel - i0];
    const float *w = (const float *) m->weights + r * (size_t) W;
    double *o = (double *) out + m->base[r];
    int c = 0;
    if (W - c >= 8) {
        const __m512d px = _mm512_set1_pd(p);
        _mm512_storeu_pd(o + c, _mm512_fmadd_pd(px, _mm512_cvtps_pd(_mm256_loadu_ps(w + c)),
                                                _mm512_loadu_pd(o + c)));
        c += 8;
    }
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

template<typename T, int W>
__attribute__((target("avx2,fma")))
inline void bslz4_padded_row_avx2(const T *block, size_t i0, const bslz4_mat_padded *m,
                                  const uint8_t *mask, size_t r, double *out) {
    (void) mask;
    const size_t pixel = BSLZ4_PADDED_PIXEL(m, r);
    const double p = (double) block[pixel - i0];
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

template<typename T, int W>
inline void bslz4_padded_row_sse2(const T *block, size_t i0, const bslz4_mat_padded *m,
                                  const uint8_t *mask, size_t r, double *out) {
    (void) mask;
    const size_t pixel = BSLZ4_PADDED_PIXEL(m, r);
    const double p = (double) block[pixel - i0];
    const float *w = (const float *) m->weights + r * (size_t) W;
    double *o = (double *) out + m->base[r];
    int c = 0;
    if (W - c >= 2) {
        const __m128d px = _mm_set1_pd(p);
        const __m128d wd = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) (w + c))));
        _mm_storeu_pd(o + c, _mm_add_pd(_mm_mul_pd(px, wd), _mm_loadu_pd(o + c)));
        c += 2;
    }
    for (; c < W; c++) o[c] += p * (double) w[c];
}

/* ---- per-W tile/stride block walks (one dispatch per block) ---- */

template<typename T, int W>
__attribute__((target("avx512f,fma")))
inline void bslz4_padded_dense_avx512(const T *block, size_t i0, const bslz4_mat_padded *m,
                                      const uint8_t *mask, size_t r0, size_t r1, double *out) {
    for (size_t t0 = r0; t0 < r1; t0 += PADDED_TILE) {
        const size_t t1 = (t0 + PADDED_TILE < r1) ? t0 + PADDED_TILE : r1;
        for (size_t s = 0; s < PADDED_STRIDE; s++)
            for (size_t r = t0 + s; r < t1; r += PADDED_STRIDE)
                bslz4_padded_row_avx512<T, W>(block, i0, m, mask, r, out);
    }
}

template<typename T, int W>
__attribute__((target("avx2,fma")))
inline void bslz4_padded_dense_avx2(const T *block, size_t i0, const bslz4_mat_padded *m,
                                    const uint8_t *mask, size_t r0, size_t r1, double *out) {
    for (size_t t0 = r0; t0 < r1; t0 += PADDED_TILE) {
        const size_t t1 = (t0 + PADDED_TILE < r1) ? t0 + PADDED_TILE : r1;
        for (size_t s = 0; s < PADDED_STRIDE; s++)
            for (size_t r = t0 + s; r < t1; r += PADDED_STRIDE)
                bslz4_padded_row_avx2<T, W>(block, i0, m, mask, r, out);
    }
}

template<typename T, int W>
inline void bslz4_padded_dense_sse2(const T *block, size_t i0, const bslz4_mat_padded *m,
                                    const uint8_t *mask, size_t r0, size_t r1, double *out) {
    for (size_t t0 = r0; t0 < r1; t0 += PADDED_TILE) {
        const size_t t1 = (t0 + PADDED_TILE < r1) ? t0 + PADDED_TILE : r1;
        for (size_t s = 0; s < PADDED_STRIDE; s++)
            for (size_t r = t0 + s; r < t1; r += PADDED_STRIDE)
                bslz4_padded_row_sse2<T, W>(block, i0, m, mask, r, out);
    }
}

#endif /* BSLZ4_HAVE_PADDED_X86 */

/* ------------------------------------------------------------- block walk */

/* Accumulate the rows of the block containing pixel i0 (data covers
 * [i0, i0+n)) into this frame's bins.  block = i0 / block_elems.  The
 * per-W/tier dispatch happens once per block (not per row), which keeps the
 * hot dense loop free of switches.  Widths above PADDED_MAX_SIMD_WIDTH use
 * the scalar loop.  The last (tail) block is bounded by the matrix's own
 * npix, so no row references a pixel outside the frame. */
template<typename T>
inline void bslz4_padded_dense_tier(const T *block, size_t i0, size_t n,
                                    const bslz4_mat_padded *m, const uint8_t *mask,
                                    int tier, double *out) {
    const size_t block_idx = i0 / m->block_elems;
    const size_t r0 = (size_t) m->row_ptr[block_idx];
    const size_t r1 = (size_t) m->row_ptr[block_idx + 1];
    const int W = m->width;
    if (W <= PADDED_MAX_SIMD_WIDTH) {
#if BSLZ4_HAVE_PADDED_X86
        switch (tier) {
        case 3:
            switch (W) {
#define BSLZ4_PADDED_DENSE_CASE(TI) case 1: bslz4_padded_dense_avx512<T, 1>(block, i0, m, mask, r0, r1, out); return; \
            case 2: bslz4_padded_dense_avx512<T, 2>(block, i0, m, mask, r0, r1, out); return; \
            case 3: bslz4_padded_dense_avx512<T, 3>(block, i0, m, mask, r0, r1, out); return; \
            case 4: bslz4_padded_dense_avx512<T, 4>(block, i0, m, mask, r0, r1, out); return; \
            case 5: bslz4_padded_dense_avx512<T, 5>(block, i0, m, mask, r0, r1, out); return; \
            case 6: bslz4_padded_dense_avx512<T, 6>(block, i0, m, mask, r0, r1, out); return; \
            case 7: bslz4_padded_dense_avx512<T, 7>(block, i0, m, mask, r0, r1, out); return; \
            case 8: bslz4_padded_dense_avx512<T, 8>(block, i0, m, mask, r0, r1, out); return; }
            BSLZ4_PADDED_DENSE_CASE(avx512)
#undef BSLZ4_PADDED_DENSE_CASE
            break;
        case 2:
            switch (W) {
            case 1: bslz4_padded_dense_avx2<T, 1>(block, i0, m, mask, r0, r1, out); return;
            case 2: bslz4_padded_dense_avx2<T, 2>(block, i0, m, mask, r0, r1, out); return;
            case 3: bslz4_padded_dense_avx2<T, 3>(block, i0, m, mask, r0, r1, out); return;
            case 4: bslz4_padded_dense_avx2<T, 4>(block, i0, m, mask, r0, r1, out); return;
            case 5: bslz4_padded_dense_avx2<T, 5>(block, i0, m, mask, r0, r1, out); return;
            case 6: bslz4_padded_dense_avx2<T, 6>(block, i0, m, mask, r0, r1, out); return;
            case 7: bslz4_padded_dense_avx2<T, 7>(block, i0, m, mask, r0, r1, out); return;
            case 8: bslz4_padded_dense_avx2<T, 8>(block, i0, m, mask, r0, r1, out); return;
            }
            break;
        case 1:
            switch (W) {
            case 1: bslz4_padded_dense_sse2<T, 1>(block, i0, m, mask, r0, r1, out); return;
            case 2: bslz4_padded_dense_sse2<T, 2>(block, i0, m, mask, r0, r1, out); return;
            case 3: bslz4_padded_dense_sse2<T, 3>(block, i0, m, mask, r0, r1, out); return;
            case 4: bslz4_padded_dense_sse2<T, 4>(block, i0, m, mask, r0, r1, out); return;
            case 5: bslz4_padded_dense_sse2<T, 5>(block, i0, m, mask, r0, r1, out); return;
            case 6: bslz4_padded_dense_sse2<T, 6>(block, i0, m, mask, r0, r1, out); return;
            case 7: bslz4_padded_dense_sse2<T, 7>(block, i0, m, mask, r0, r1, out); return;
            case 8: bslz4_padded_dense_sse2<T, 8>(block, i0, m, mask, r0, r1, out); return;
            }
            break;
        }
#endif
    }
    for (size_t r = r0; r < r1; r++)
        bslz4_padded_row_scalar<T>(block, i0, m, mask, r, out);
}

/* ------------------------------------------------------- per-row dispatch */

#undef BSLZ4_PADDED_PIXEL

} /* namespace bslz4 */
