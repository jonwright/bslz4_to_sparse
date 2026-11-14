#pragma once
/*
 * SIMD mask+threshold collection for the u8/u16/u32 decode hot path:
 * the plain threshold-collect loop (mask>0 & value>cut) and the CSC
 * sparse-route compaction (mask>0 & value!=0, called here with cut==0 --
 * identical to !=0 for unsigned T).
 *
 * Six SIMD collect values, for u8/u16/u32 and i8/i16/i32 pixels (signed ones: the
 * same kernels on the same bits with a signed compare, template argument S): avx512cs (AVX-512 F+BW+VL+VBMI2,
 * compress-store), avx512f (AVX-512 F+BW+VL without VBMI2: dword compress),
 * avx2cs (AVX2, table-driven compaction), sse2 (x86-64 ABI
 * baseline, always present), VSX (POWER8+, probe in
 * tools/bslz4_power9_collect_probe.c) and NEON (AArch64, ASIMD is
 * ARMv8-A baseline, probe in tools/bslz4_neon_collect_probe.c). The x86
 * tiers carry __attribute__((target(...))), so this file compiles under
 * the project's ordinary -O2; VSX needs tools/build_extension.py to pass -maltivec -mvsx
 * globally on ppc64le, NEON needs nothing extra on aarch64 (ASIMD is
 * baseline, like SSE2 on x86-64). MSVC has none of these, and the
 * BSLZ4_HAVE_*_COLLECT guards below make this file inert there.
 *
 *   BSLZ4_COLLECT_AVX512CS  AVX-512 F+BW+VL+VBMI2: compress-store of the
 *                           selected pixels and their addresses; u8/u16/u32 (and signed).
 *   BSLZ4_COLLECT_AVX512F   AVX-512 F+BW+VL (Skylake-SP, Cascade Lake: no
 *                           VBMI2): compare at the pixel width, compress 16
 *                           pixels at a time as dwords; u8/u16/u32.
 *   BSLZ4_COLLECT_AVX2CS    AVX2: u8/u16 compact the selected lanes with a
 *                           table, u32 walks the set bits; u8/u16/u32.
 *   BSLZ4_COLLECT_VSX       POWER8+ (needs -maltivec -mvsx); u8/u16/u32.
 *   BSLZ4_COLLECT_NEON      AArch64 ASIMD (ARMv8-A baseline); u8/u16/u32.
 *   BSLZ4_COLLECT_SSE2      x86-64 baseline, always present; u8/u16/u32.
 *   BSLZ4_COLLECT_SCALAR    the plain scalar loop, any dtype; u64, i64 and the
 *                           floats take it whatever the value.
 *
 * The value actually used is pipeline[BSLZ4_STEP_COLLECT], validated by
 * bslz4_resolve (src/pipeline/registry.c).  Availability is decided by
 * bslz4_<tier>_collect_capable(), which reads c2py23's cpuid-equivalent
 * globals (c2py_amd64_avx512f/bw/vl/vbmi2, c2py_amd64_avx2/popcnt, c2py_ppc64_vsx,
 * c2py_arm64_asimd; SSE2 needs no check); capable() is the single source of
 * truth, shared with bslz4_step_available (registry.c), which Python sees as
 * step_available / _pipeline.available().  AVX-512 can
 * trigger frequency throttling on some chips that outweighs the wider vector
 * for this workload, so it is selectable but not always a win.
 *
 * bslz4_collect_gt<T>/bslz4_collect_nz<T> (bottom of this file) take the
 * collect value from the pipeline (bslz4_resolve already validated it) and
 * route straight to that tier; any value that is not a SIMD tier
 * (BSLZ4_COLLECT_SCALAR) runs the plain scalar loop.
 */

#include "../pipeline/common.h"
#include "../pipeline/pipeline.h"

#include <string.h>
#include <type_traits>

#include "c2py_amd64.h"
#include "c2py_arm64.h"
#include "c2py_ppc64.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define BSLZ4_HAVE_AVX512_COLLECT 1
#define BSLZ4_HAVE_AVX2_COLLECT 1
#define BSLZ4_HAVE_SSE2_COLLECT 1
#include <immintrin.h>
#else
#define BSLZ4_HAVE_AVX512_COLLECT 0
#define BSLZ4_HAVE_AVX2_COLLECT 0
#define BSLZ4_HAVE_SSE2_COLLECT 0
#endif

#if defined(__VSX__) && defined(__ALTIVEC__)
#define BSLZ4_HAVE_VSX_COLLECT 1
#include <altivec.h>
#else
#define BSLZ4_HAVE_VSX_COLLECT 0
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
#define BSLZ4_HAVE_NEON_COLLECT 1
#include <arm_neon.h>
#else
#define BSLZ4_HAVE_NEON_COLLECT 0
#endif

namespace bslz4 {

/* v > cut on the raw (unsigned) bits of a pixel, as a signed comparison when S: the SIMD
 * collects take S for signed pixels (i8/i16/i32), working on the same bits as the unsigned
 * ones (the values they store are the pixels' own). */
template<bool S, typename U>
inline bool bslz4_gt(U a, U b) {
    typedef typename std::make_signed<U>::type I;
    return S ? (I) a > (I) b : a > b;
}

#if BSLZ4_HAVE_VSX_COLLECT
/* the VSX compares, unsigned or (S) signed */
template<bool S>
inline __vector __bool char bslz4_vsx_gt(__vector unsigned char v, __vector unsigned char c) {
    return S ? vec_cmpgt((__vector signed char) v, (__vector signed char) c) : vec_cmpgt(v, c);
}
template<bool S>
inline int bslz4_vsx_any_gt(__vector unsigned char v, __vector unsigned char c) {
    return S ? vec_any_gt((__vector signed char) v, (__vector signed char) c) : vec_any_gt(v, c);
}
template<bool S>
inline __vector __bool short bslz4_vsx_gt(__vector unsigned short v, __vector unsigned short c) {
    return S ? vec_cmpgt((__vector signed short) v, (__vector signed short) c) : vec_cmpgt(v, c);
}
template<bool S>
inline int bslz4_vsx_any_gt(__vector unsigned short v, __vector unsigned short c) {
    return S ? vec_any_gt((__vector signed short) v, (__vector signed short) c) : vec_any_gt(v, c);
}
template<bool S>
inline __vector __bool int bslz4_vsx_gt(__vector unsigned int v, __vector unsigned int c) {
    return S ? vec_cmpgt((__vector signed int) v, (__vector signed int) c) : vec_cmpgt(v, c);
}
template<bool S>
inline int bslz4_vsx_any_gt(__vector unsigned int v, __vector unsigned int c) {
    return S ? vec_any_gt((__vector signed int) v, (__vector signed int) c) : vec_any_gt(v, c);
}
#endif

/* A NULL mask means the mask step is none, or the mask was already applied
 * to the planes (every pixel is valid; src/pipeline/driver.c, and
 * src/steps/kernels.cpp passes `w->no_mask ? NULL : w->mask`), so the
 * per-pixel mask check is skipped. */
static inline bool bslz4_mask_ok(const uint8_t *mask, size_t idx) {
    return mask == NULL || mask[idx] > 0;
}

/* Single source of truth for "is this tier actually usable" -- shared
 * by the pipeline's validation and the Python-visible step_available query
 * (the bslz4_available_*_collect delegates in src/steps/kernels.cpp call
 * these rather than duplicating the cpuid-check logic). */

inline bool bslz4_avx512_collect_capable() {
#if BSLZ4_HAVE_AVX512_COLLECT
    return (c2py_amd64_avx512f && c2py_amd64_avx512bw && c2py_amd64_avx512vl) != 0;
#else
    return false;
#endif
}

/* The compress-store tier (collect avx512cs, BSLZ4_COLLECT_AVX512CS) also
 * needs AVX-512 VBMI2 for the 16-bit compress. */
inline bool bslz4_avx512cs_collect_capable() {
#if BSLZ4_HAVE_AVX512_COLLECT && BSLZ4_HAVE_VBMI_GFNI
    return bslz4_avx512_collect_capable() && c2py_amd64_avx512vbmi2 != 0 && c2py_amd64_popcnt != 0;
#else
    return false;
#endif
}

/* collect avx512f: AVX-512 F/BW and popcnt (no VBMI2) */
inline bool bslz4_avx512f_collect_capable() {
#if BSLZ4_HAVE_AVX512_COLLECT
    return bslz4_avx512_collect_capable() && c2py_amd64_avx512bw != 0 && c2py_amd64_popcnt != 0;
#else
    return false;
#endif
}

inline bool bslz4_avx2cs_collect_capable() {
#if BSLZ4_HAVE_AVX2_COLLECT
    return c2py_amd64_avx2 != 0 && c2py_amd64_popcnt != 0;
#else
    return false;
#endif
}

inline bool bslz4_sse2_collect_capable() {
#if BSLZ4_HAVE_SSE2_COLLECT
    return true; /* x86-64 ABI baseline -- always present, no cpuid check needed */
#else
    return false;
#endif
}

inline bool bslz4_vsx_collect_capable() {
#if BSLZ4_HAVE_VSX_COLLECT
    return c2py_ppc64_vsx != 0;
#else
    return false;
#endif
}

inline bool bslz4_neon_collect_capable() {
#if BSLZ4_HAVE_NEON_COLLECT
    return c2py_arm64_asimd != 0;
#else
    return false;
#endif
}

/* The tiers are chosen at run time by the pipeline's collect value, not by
 * mutable flags here. */

#if BSLZ4_HAVE_AVX512_COLLECT

/* Collect avx512cs (BSLZ4_COLLECT_AVX512CS): mask & (value > cut) per 32
 * pixels, but the selected values and their pixel indices are compressed into registers
 * (vpcompressw / vpcompressd), stored with plain full-width stores, and the
 * output advanced by popcnt -- no per-pixel loop, so no data-dependent
 * branch per selected pixel.  (Compress-to-register then store: the
 * compress-to-memory form is slow on AMD CPUs.)
 *
 * The full-width stores write up to 32 entries past the last selected one.
 * That stays inside the caller's buffers: before group j, at most j of this
 * block's pixels have been selected, and the caller's offset counts at most
 * i0 earlier pixels, so a store ends before i0 + j + 32 <= i0 + block_elems,
 * which the per-frame output (NIJ entries) and the per-block compaction
 * scratch (block_elems entries) both cover.  The entries past the returned
 * count are scratch the caller never reads. */
#if BSLZ4_HAVE_VBMI_GFNI
template<bool S = false>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi2,popcnt")))
inline int bslz4_collect_avx512cs_u16(const uint16_t *BSLZ4_RESTRICT block,
                                       const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t block_elems, uint16_t cut,
                                       uint16_t *BSLZ4_RESTRICT out_vals,
                                       uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m512i vcut = _mm512_set1_epi16((short) cut);
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i sixteen = _mm512_set1_epi32(16);
    for (; j + 32 <= block_elems; j += 32) {
        const __m512i v = _mm512_loadu_si512((const void *) &block[j]);
        __mmask32 m = (S ? _mm512_cmpgt_epi16_mask(v, vcut) : _mm512_cmpgt_epu16_mask(v, vcut));
        if (mask)
            m &= _mm256_cmpneq_epu8_mask(_mm256_loadu_si256((const __m256i *) &mask[j + i0]),
                                         _mm256_setzero_si256());
        if (!m) continue;                     /* the common case for sparse data */
        _mm512_storeu_si512((void *) &out_vals[npx], _mm512_maskz_compress_epi16(m, v));
        const __m512i lo = _mm512_add_epi32(_mm512_set1_epi32((int) (j + i0)), iota);
        const __mmask16 mlo = (__mmask16) m, mhi = (__mmask16) (m >> 16);
        const int nlo = __builtin_popcount((unsigned) mlo);
        _mm512_storeu_si512((void *) &out_adr[npx], _mm512_maskz_compress_epi32(mlo, lo));
        _mm512_storeu_si512((void *) &out_adr[npx + nlo],
                            _mm512_maskz_compress_epi32(mhi, _mm512_add_epi32(lo, sixteen)));
        npx += nlo + __builtin_popcount((unsigned) mhi);
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/* u32 version of collect avx512cs: vpcompressd is plain AVX-512F.  Same store
 * bound as the u16 one, with 16 entries per group. */
template<bool S = false>
__attribute__((target("avx512f,avx512bw,avx512vl,popcnt")))
inline int bslz4_collect_avx512cs_u32(const uint32_t *BSLZ4_RESTRICT block,
                                       const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t block_elems, uint32_t cut,
                                       uint32_t *BSLZ4_RESTRICT out_vals,
                                       uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m512i vcut = _mm512_set1_epi32((int) cut);
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (; j + 16 <= block_elems; j += 16) {
        const __m512i v = _mm512_loadu_si512((const void *) &block[j]);
        __mmask16 m = (S ? _mm512_cmpgt_epi32_mask(v, vcut) : _mm512_cmpgt_epu32_mask(v, vcut));
        if (mask)
            m &= _mm_cmpneq_epu8_mask(_mm_loadu_si128((const __m128i *) &mask[j + i0]),
                                      _mm_setzero_si128());
        if (!m) continue;
        _mm512_storeu_si512((void *) &out_vals[npx], _mm512_maskz_compress_epi32(m, v));
        _mm512_storeu_si512((void *) &out_adr[npx],
                            _mm512_maskz_compress_epi32(m, _mm512_add_epi32(_mm512_set1_epi32((int) (j + i0)), iota)));
        npx += __builtin_popcount((unsigned) m);
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/* u8 version of collect avx512cs: 64 pixels per step, the values by one byte
 * compress (VBMI2), the indices by four dword compresses.  Full-width stores
 * reach at most 64 entries past the count: before group j at most j of this
 * block's pixels were selected, so a store ends before i0 + j + 64 <=
 * i0 + block_elems. */
template<bool S = false>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi2,popcnt")))
inline int bslz4_collect_avx512cs_u8(const uint8_t *BSLZ4_RESTRICT block,
                                      const uint8_t *BSLZ4_RESTRICT mask,
                                      size_t i0, size_t block_elems, uint8_t cut,
                                      uint8_t *BSLZ4_RESTRICT out_vals,
                                      uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m512i vcut = _mm512_set1_epi8((char) cut);
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i sixteen = _mm512_set1_epi32(16);
    for (; j + 64 <= block_elems; j += 64) {
        const __m512i v = _mm512_loadu_si512((const void *) &block[j]);
        __mmask64 m = (S ? _mm512_cmpgt_epi8_mask(v, vcut) : _mm512_cmpgt_epu8_mask(v, vcut));
        if (mask) {
            const __m512i mv = _mm512_loadu_si512((const void *) &mask[j + i0]);
            m &= _mm512_test_epi8_mask(mv, mv);
        }
        if (!m) continue;                     /* the common case for sparse data */
        _mm512_storeu_si512((void *) &out_vals[npx], _mm512_maskz_compress_epi8(m, v));
        __m512i ix = _mm512_add_epi32(_mm512_set1_epi32((int) (j + i0)), iota);
        for (int q = 0; q < 4; q++) {
            const __mmask16 mq = (__mmask16) (m >> (16 * q));
            _mm512_storeu_si512((void *) &out_adr[npx], _mm512_maskz_compress_epi32(mq, ix));
            npx += __builtin_popcount((unsigned) mq);
            ix = _mm512_add_epi32(ix, sixteen);
        }
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}
#endif

/* groups (64 u8, 32 u16 pixels) with at most this many selected pixels: a bit loop */
#ifndef BSLZ4_AVX512F_SCALAR_MAX
#define BSLZ4_AVX512F_SCALAR_MAX 2
#endif

/* Collect avx512f (BSLZ4_COLLECT_AVX512F): AVX-512 F/BW without VBMI2 (Skylake-SP, Cascade
 * Lake), which has no byte or word compress.  The compare runs at the pixel width (BW), then 16
 * pixels at a time are widened to dwords, compressed with vpcompressd (F) together with their
 * indices, and narrowed back.  Full-width stores reach at most 16 entries past the count. */
__attribute__((target("avx512f,avx512bw"), always_inline))
inline void bslz4_avx512f_store16(uint8_t *o, __m512i c) { _mm_storeu_si128((__m128i *) (void *) o, _mm512_cvtepi32_epi8(c)); }
__attribute__((target("avx512f,avx512bw"), always_inline))
inline void bslz4_avx512f_store16(uint16_t *o, __m512i c) { _mm256_storeu_si256((__m256i *) (void *) o, _mm512_cvtepi32_epi16(c)); }
__attribute__((target("avx512f,avx512bw"), always_inline))
inline void bslz4_avx512f_store16(uint32_t *o, __m512i c) { _mm512_storeu_si512((void *) o, c); }

/* 16 pixels (as dwords v32, selection mq) at index base: returns the new count */
template<typename T>
__attribute__((target("avx512f,avx512bw,popcnt"), always_inline))
inline int bslz4_avx512f_emit16(__m512i v32, __mmask16 mq, size_t base, int npx,
                                T *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    if (!mq) return npx;
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    bslz4_avx512f_store16(&out_vals[npx], _mm512_maskz_compress_epi32(mq, v32));
    _mm512_storeu_si512((void *) &out_adr[npx],
                        _mm512_maskz_compress_epi32(mq, _mm512_add_epi32(_mm512_set1_epi32((int) base), iota)));
    return npx + __builtin_popcount((unsigned) mq);
}

template<bool S, typename T>
inline int bslz4_collect_tail(const T *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                              size_t j, size_t n, T cut, int npx, T *BSLZ4_RESTRICT out_vals,
                              uint32_t *BSLZ4_RESTRICT out_adr) {
    for (; j < n; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<bool S = false>
__attribute__((target("avx512f,avx512bw,popcnt")))
inline int bslz4_collect_avx512f_u8(const uint8_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                     size_t i0, size_t n, uint8_t cut,
                                     uint8_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m512i vcut = _mm512_set1_epi8((char) cut);
    for (; j + 64 <= n; j += 64) {
        const __m512i v = _mm512_loadu_si512((const void *) &block[j]);
        __mmask64 m = (S ? _mm512_cmpgt_epi8_mask(v, vcut) : _mm512_cmpgt_epu8_mask(v, vcut));
        if (mask) {
            const __m512i mv = _mm512_loadu_si512((const void *) &mask[j + i0]);
            m &= _mm512_test_epi8_mask(mv, mv);
        }
        if (!m) continue;                     /* the common case for sparse data */
        if (__builtin_popcountll((unsigned long long) m) <= BSLZ4_AVX512F_SCALAR_MAX) {
            for (uint64_t mb = (uint64_t) m; mb; mb &= mb - 1) {
                const int b = __builtin_ctzll(mb);
                out_vals[npx] = block[j + b];
                out_adr[npx++] = (uint32_t) (j + i0 + b);
            }
            continue;
        }
        /* extract lanes are immediates: the four quarters spelled out */
        npx = bslz4_avx512f_emit16(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v, 0)), (__mmask16) m,
                                   j + i0, npx, out_vals, out_adr);
        npx = bslz4_avx512f_emit16(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v, 1)), (__mmask16) (m >> 16),
                                   j + i0 + 16, npx, out_vals, out_adr);
        npx = bslz4_avx512f_emit16(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v, 2)), (__mmask16) (m >> 32),
                                   j + i0 + 32, npx, out_vals, out_adr);
        npx = bslz4_avx512f_emit16(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v, 3)), (__mmask16) (m >> 48),
                                   j + i0 + 48, npx, out_vals, out_adr);
    }
    return bslz4_collect_tail<S>(block, mask, i0, j, n, cut, npx, out_vals, out_adr);
}

template<bool S = false>
__attribute__((target("avx512f,avx512bw,popcnt")))
inline int bslz4_collect_avx512f_u16(const uint16_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                      size_t i0, size_t n, uint16_t cut,
                                      uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m512i vcut = _mm512_set1_epi16((short) cut);
    for (; j + 32 <= n; j += 32) {
        const __m512i v = _mm512_loadu_si512((const void *) &block[j]);
        __mmask32 m = (S ? _mm512_cmpgt_epi16_mask(v, vcut) : _mm512_cmpgt_epu16_mask(v, vcut));
        if (mask) {
            const __m256i mv = _mm256_loadu_si256((const __m256i *) (const void *) &mask[j + i0]);
            m &= ~(uint32_t) _mm256_movemask_epi8(_mm256_cmpeq_epi8(mv, _mm256_setzero_si256()));
        }
        if (!m) continue;
        if (__builtin_popcount((unsigned) m) <= BSLZ4_AVX512F_SCALAR_MAX) {
            for (uint32_t mb = (uint32_t) m; mb; mb &= mb - 1) {
                const int b = __builtin_ctz(mb);
                out_vals[npx] = block[j + b];
                out_adr[npx++] = (uint32_t) (j + i0 + b);
            }
            continue;
        }
        npx = bslz4_avx512f_emit16(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 0)), (__mmask16) m,
                                   j + i0, npx, out_vals, out_adr);
        npx = bslz4_avx512f_emit16(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1)), (__mmask16) (m >> 16),
                                   j + i0 + 16, npx, out_vals, out_adr);
    }
    return bslz4_collect_tail<S>(block, mask, i0, j, n, cut, npx, out_vals, out_adr);
}

template<bool S = false>
__attribute__((target("avx512f,avx512bw,popcnt")))
inline int bslz4_collect_avx512f_u32(const uint32_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                      size_t i0, size_t n, uint32_t cut,
                                      uint32_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m512i vcut = _mm512_set1_epi32((int) cut);
    for (; j + 16 <= n; j += 16) {
        const __m512i v = _mm512_loadu_si512((const void *) &block[j]);
        __mmask16 m = (S ? _mm512_cmpgt_epi32_mask(v, vcut) : _mm512_cmpgt_epu32_mask(v, vcut));
        if (mask)
            m &= (__mmask16) ~(uint32_t) _mm_movemask_epi8(
                _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *) (const void *) &mask[j + i0]), _mm_setzero_si128()));
        if (!m) continue;
        npx = bslz4_avx512f_emit16(v, m, j + i0, npx, out_vals, out_adr);
    }
    return bslz4_collect_tail<S>(block, mask, i0, j, n, cut, npx, out_vals, out_adr);
}

#endif /* BSLZ4_HAVE_AVX512_COLLECT */

#if BSLZ4_HAVE_AVX2_COLLECT

/*
 * AVX2 has no mask registers and no unsigned compare-greater -- both
 * AVX-512-only conveniences. Unsigned "greater than" is built the
 * standard way (XOR both operands with the sign bit, then a signed
 * compare on the biased values gives the same ordering as an unsigned
 * compare on the originals). Getting a per-lane scalar bitmask out of a
 * 256-bit compare result needs an extra step AVX-512's native k-mask
 * compare doesn't: for u32 (8 lanes/vector), bitcast to __m256 and use
 * _mm256_movemask_ps directly (8 bits, one per lane, no packing). For
 * u16 (16 lanes/vector), a 16-bit-element compare's true/false result
 * fills BOTH bytes of each lane identically, so _mm256_movemask_epi8
 * gives 32 bits in duplicated pairs, not the 16 bits we want -- packed
 * down via the standard _mm256_packs_epi16(cmp, cmp) trick: AVX2's pack
 * operates per-128-bit-lane, so packing cmp with itself produces
 * [pack(lanes0-7)][pack(lanes0-7) dup][pack(lanes8-15)][pack(lanes8-15)
 * dup] -- movemask of that is a 32-bit value whose low byte is exactly
 * lanes 0-7 and whose byte 2 (bits 16-23) is exactly lanes 8-15,
 * recombined below into one 16-bit mask.
 *
 * Once there's a scalar bitmask, the avx2cs kernels below extract the
 * selected pixels from it (see the comment on avx2cs).
 */

/* Collect avx2cs (BSLZ4_COLLECT_AVX2CS): the AVX2 selection, extracted
 * without a loop per pixel (the avx512cs scheme on AVX2).  For each 8 pixels the selection byte
 * m indexes a table of 8 lane numbers (the set bits of m, packed): vpermd
 * packs the 8 pixel indices with it, and pshufb the 8 u16 values (lane k ->
 * bytes 2k, 2k+1).  Both are stored full width and the output advances by
 * popcount(m).  The full-width stores reach 8 entries past the count: as
 * for avx512cs, before group j at most j of this block's pixels were
 * selected, so a store ends before i0 + j + 16 <= i0 + block_elems.  Without
 * AVX-512 a bit loop per selected pixel is a large part of sparsify. */
#ifndef BSLZ4_AVX2CS_SCALAR_MAX
#define BSLZ4_AVX2CS_SCALAR_MAX 1
#endif

struct bslz4_avx2cs_lut {
    uint64_t lane[256];
    bslz4_avx2cs_lut() {
        for (int m = 0; m < 256; m++) {
            uint64_t v = 0;
            int k = 0;
            for (int b = 0; b < 8; b++)
                if (m >> b & 1) v |= (uint64_t) b << (8 * k++);
            lane[m] = v;
        }
    }
};

inline const uint64_t *bslz4_avx2cs_table() {
    static const bslz4_avx2cs_lut t;      /* -fno-threadsafe-statics: no guard */
    return t.lane;
}

template<bool S = false>
__attribute__((target("avx2,popcnt")))
inline int bslz4_collect_avx2cs_u16(const uint16_t *BSLZ4_RESTRICT block,
                                     const uint8_t *BSLZ4_RESTRICT mask,
                                     size_t i0, size_t block_elems, uint16_t cut,
                                     uint16_t *BSLZ4_RESTRICT out_vals,
                                     uint32_t *BSLZ4_RESTRICT out_adr) {
    const uint64_t *lut = bslz4_avx2cs_table();
    int npx = 0;
    size_t j = 0;
    const __m256i sign_bias = _mm256_set1_epi16((short) (S ? 0 : 0x8000));
    const __m256i vcut_b = _mm256_xor_si256(_mm256_set1_epi16((short) cut), sign_bias);
    const __m256i iota8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m128i one = _mm_set1_epi8(1);
    for (; j + 16 <= block_elems; j += 16) {
        uint32_t mnz16 = mask
            ? (~(uint32_t) _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *) &mask[j + i0]), _mm_setzero_si128()))) & 0xFFFFu
            : 0xFFFFu;
        __m256i v = _mm256_loadu_si256((const __m256i *) &block[j]);
        __m256i gt = _mm256_cmpgt_epi16(_mm256_xor_si256(v, sign_bias), vcut_b);
        uint32_t packed = (uint32_t) _mm256_movemask_epi8(_mm256_packs_epi16(gt, gt));
        uint32_t bits = mnz16 & ((packed & 0xFFu) | ((packed >> 8) & 0xFF00u));
        if (!bits) continue;                  /* the common case for sparse data */
        if (__builtin_popcount(bits) <= BSLZ4_AVX2CS_SCALAR_MAX) {
            /* one pixel (photon noise): the bit loop is cheaper than
             * packing both halves; the test is predictable per dataset */
            do {
                const int b = __builtin_ctz(bits);
                out_vals[npx] = block[j + b];
                out_adr[npx] = (uint32_t) (j + i0 + b);
                npx++;
                bits &= bits - 1;
            } while (bits);
            continue;
        }
        for (int h = 0; h < 2; h++) {
            const uint32_t m = (bits >> (8 * h)) & 0xFFu;
            const __m128i sel = _mm_loadl_epi64((const __m128i *) (const void *) &lut[m]);
            const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32((int) (j + i0 + 8 * h)), iota8);
            _mm256_storeu_si256((__m256i *) (void *) &out_adr[npx],
                                _mm256_permutevar8x32_epi32(idx, _mm256_cvtepu8_epi32(sel)));
            const __m128i s2 = _mm_add_epi8(sel, sel);
            const __m128i shuf = _mm_unpacklo_epi8(s2, _mm_add_epi8(s2, one));
            const __m128i v8 = _mm_loadu_si128((const __m128i *) (const void *) &block[j + 8 * h]);
            _mm_storeu_si128((__m128i *) (void *) &out_vals[npx], _mm_shuffle_epi8(v8, shuf));
            npx += __builtin_popcount(m);
        }
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<bool S = false>
__attribute__((target("avx2")))
inline int bslz4_collect_avx2cs_u32(const uint32_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t block_elems, uint32_t cut,
                                   uint32_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m256i sign_bias = _mm256_set1_epi32((int) (S ? 0 : 0x80000000u));
    const __m256i vcut_b = _mm256_xor_si256(_mm256_set1_epi32((int) cut), sign_bias);
    for (; j + 8 <= block_elems; j += 8) {
        uint32_t mnz8 = mask
            ? (~(uint32_t) _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadl_epi64((const __m128i *) &mask[j + i0]), _mm_setzero_si128()))) & 0xFFu
            : 0xFFu;

        __m256i v = _mm256_loadu_si256((const __m256i *) &block[j]);
        __m256i vb = _mm256_xor_si256(v, sign_bias);
        __m256i gt = _mm256_cmpgt_epi32(vb, vcut_b);
        uint32_t mgt8 = (uint32_t) _mm256_movemask_ps(_mm256_castsi256_ps(gt));

        uint32_t bits = mnz8 & mgt8;
        while (bits) {
            int b = __builtin_ctz(bits);
            out_vals[npx] = block[j + b];
            out_adr[npx] = (uint32_t) (j + i0 + b);
            npx++;
            bits &= bits - 1;
        }
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/* u8 version of collect avx2cs: 32 pixels per step, the selection byte of
 * each 8 indexing the lane table as for u16 (vpermd the indices, pshufb the
 * bytes; stored full width, 8 entries past the count at most). */
template<bool S = false>
__attribute__((target("avx2,popcnt")))
inline int bslz4_collect_avx2cs_u8(const uint8_t *BSLZ4_RESTRICT block,
                                    const uint8_t *BSLZ4_RESTRICT mask,
                                    size_t i0, size_t block_elems, uint8_t cut,
                                    uint8_t *BSLZ4_RESTRICT out_vals,
                                    uint32_t *BSLZ4_RESTRICT out_adr) {
    const uint64_t *lut = bslz4_avx2cs_table();
    int npx = 0;
    size_t j = 0;
    const __m256i bias = _mm256_set1_epi8((char) (S ? 0 : 0x80));
    const __m256i vcut_b = _mm256_xor_si256(_mm256_set1_epi8((char) cut), bias);
    const __m256i iota8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (; j + 32 <= block_elems; j += 32) {
        uint32_t mnz = mask
            ? ~(uint32_t) _mm256_movemask_epi8(_mm256_cmpeq_epi8(
                  _mm256_loadu_si256((const __m256i *) (const void *) &mask[j + i0]), _mm256_setzero_si256()))
            : 0xFFFFFFFFu;
        const __m256i v = _mm256_loadu_si256((const __m256i *) (const void *) &block[j]);
        uint32_t bits = mnz & (uint32_t) _mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_xor_si256(v, bias), vcut_b));
        if (!bits) continue;                  /* the common case for sparse data */
        if (__builtin_popcount(bits) <= BSLZ4_AVX2CS_SCALAR_MAX) {
            do {
                const int b = __builtin_ctz(bits);
                out_vals[npx] = block[j + b];
                out_adr[npx] = (uint32_t) (j + i0 + b);
                npx++;
                bits &= bits - 1;
            } while (bits);
            continue;
        }
        for (int h = 0; h < 4; h++) {
            const uint32_t m = (bits >> (8 * h)) & 0xFFu;
            if (!m) continue;
            const __m128i sel = _mm_loadl_epi64((const __m128i *) (const void *) &lut[m]);
            const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32((int) (j + i0 + 8 * h)), iota8);
            _mm256_storeu_si256((__m256i *) (void *) &out_adr[npx],
                                _mm256_permutevar8x32_epi32(idx, _mm256_cvtepu8_epi32(sel)));
            const __m128i v8 = _mm_loadl_epi64((const __m128i *) (const void *) &block[j + 8 * h]);
            _mm_storel_epi64((__m128i *) (void *) &out_vals[npx], _mm_shuffle_epi8(v8, sel));
            npx += __builtin_popcount(m);
        }
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

#endif /* BSLZ4_HAVE_AVX2_COLLECT */

#if BSLZ4_HAVE_SSE2_COLLECT

/*
 * SSE2 is the x86-64 ABI baseline -- always present, no cpuid check
 * needed to compile OR to call this (unlike the AVX-512/AVX2 kernels
 * above, both genuinely optional ISA extensions). Same unsigned-via-
 * sign-bias trick as AVX2; u16 needs the same pack-based mask
 * recombination (here within a single 128-bit lane, so no cross-lane
 * duplication to undo: _mm_movemask_epi8(_mm_packs_epi16(cmp, cmp))'s
 * low 8 bits are already exactly the 8 lane results). u32 uses
 * _mm_movemask_ps directly, 4 lanes, no packing.
 */

template<bool S = false>
__attribute__((target("sse2")))
inline int bslz4_collect_sse2_u16(const uint16_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t block_elems, uint16_t cut,
                                   uint16_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m128i sign_bias = _mm_set1_epi16((short) (S ? 0 : 0x8000));
    const __m128i vcut_b = _mm_xor_si128(_mm_set1_epi16((short) cut), sign_bias);
    for (; j + 8 <= block_elems; j += 8) {
        uint32_t mnz8 = mask
            ? (~(uint32_t) _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadl_epi64((const __m128i *) &mask[j + i0]), _mm_setzero_si128()))) & 0xFFu
            : 0xFFu;

        __m128i v = _mm_loadu_si128((const __m128i *) &block[j]);
        __m128i vb = _mm_xor_si128(v, sign_bias);
        __m128i gt = _mm_cmpgt_epi16(vb, vcut_b);
        uint32_t mgt8 = (uint32_t) _mm_movemask_epi8(_mm_packs_epi16(gt, gt)) & 0xFFu;

        uint32_t bits = mnz8 & mgt8;
        while (bits) {
            int b = __builtin_ctz(bits);
            out_vals[npx] = block[j + b];
            out_adr[npx] = (uint32_t) (j + i0 + b);
            npx++;
            bits &= bits - 1;
        }
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<bool S = false>
__attribute__((target("sse2")))
inline int bslz4_collect_sse2_u32(const uint32_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t block_elems, uint32_t cut,
                                   uint32_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m128i sign_bias = _mm_set1_epi32((int) (S ? 0 : 0x80000000u));
    const __m128i vcut_b = _mm_xor_si128(_mm_set1_epi32((int) cut), sign_bias);
    for (; j + 4 <= block_elems; j += 4) {
        uint32_t mnz4 = 0xFu;
        if (mask) {
            uint32_t m4;
            memcpy(&m4, &mask[j + i0], 4);
            mnz4 = (~(uint32_t) _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_cvtsi32_si128((int) m4), _mm_setzero_si128()))) & 0xFu;
        }

        __m128i v = _mm_loadu_si128((const __m128i *) &block[j]);
        __m128i vb = _mm_xor_si128(v, sign_bias);
        __m128i gt = _mm_cmpgt_epi32(vb, vcut_b);
        uint32_t mgt4 = (uint32_t) _mm_movemask_ps(_mm_castsi128_ps(gt));

        uint32_t bits = mnz4 & mgt4;
        while (bits) {
            int b = __builtin_ctz(bits);
            out_vals[npx] = block[j + b];
            out_adr[npx] = (uint32_t) (j + i0 + b);
            npx++;
            bits &= bits - 1;
        }
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/* u8 version of collect sse2: 16 pixels per compare, then the bit loop. */
template<bool S = false>
__attribute__((target("sse2")))
inline int bslz4_collect_sse2_u8(const uint8_t *BSLZ4_RESTRICT block,
                                  const uint8_t *BSLZ4_RESTRICT mask,
                                  size_t i0, size_t block_elems, uint8_t cut,
                                  uint8_t *BSLZ4_RESTRICT out_vals,
                                  uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m128i bias = _mm_set1_epi8((char) (S ? 0 : 0x80));
    const __m128i vcut_b = _mm_xor_si128(_mm_set1_epi8((char) cut), bias);
    for (; j + 16 <= block_elems; j += 16) {
        uint32_t mnz = mask
            ? (~(uint32_t) _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *) &mask[j + i0]),
                                                           _mm_setzero_si128()))) & 0xFFFFu
            : 0xFFFFu;
        const __m128i v = _mm_loadu_si128((const __m128i *) &block[j]);
        uint32_t bits = mnz & (uint32_t) _mm_movemask_epi8(_mm_cmpgt_epi8(_mm_xor_si128(v, bias), vcut_b));
        while (bits) {
            const int b = __builtin_ctz(bits);
            out_vals[npx] = block[j + b];
            out_adr[npx] = (uint32_t) (j + i0 + b);
            npx++;
            bits &= bits - 1;
        }
    }
    for (; j < block_elems; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

#endif /* BSLZ4_HAVE_SSE2_COLLECT */

#if BSLZ4_HAVE_VSX_COLLECT

/*
 * VSX has native unsigned compare-greater (vec_cmpgt on unsigned vector
 * types) -- no sign-bias trick needed at all, unlike x86 SSE2/AVX2.
 * Unaligned loads use the same memcpy-into-a-vector-typed-local idiom
 * as the x86 kernels.
 *
 * There is no movemask on POWER. vec_any_gt(v, vcut) gates the whole
 * chunk instead: if no lane exceeds cut, the mask cannot change that
 * (mask>0 & value>cut needs value>cut), so the per-lane extraction is
 * skipped entirely. Real detector data is mostly exact zeros, so the
 * gate fires for nearly every chunk -- much faster than scalar on
 * POWER9, where an ungated version is slower than scalar.
 *
 * When the gate does not fire, per-lane extraction is vec_extract in a
 * small fixed-trip loop (8 iterations for u16, 4 for u32).
 *
 * Dense data defeats the gate: with pixels above the cut placed at random
 * the vector loop is slower than the scalar one once a few percent of
 * pixels are kept.  So the block is
 * walked in windows of BSLZ4_VSX_WINDOW pixels, each running the vector or
 * the scalar loop by the pixels the previous window kept, with hysteresis
 * (bslz4_vsx_next_dense: to scalar above 5%, back to vector below 2%):
 * sparse stretches keep the gate, dense ones (rings, a low cut) go scalar,
 * at one compare per window.  The mode restarts as vector every block, so
 * the window is kept short (a longer window left dense data slower than
 * scalar).
 *
 * __vector/__bool, not bare vector/bool: <altivec.h> defines the bare
 * keyword-macros in C mode only, leaving them undefined in C++ so they
 * cannot collide with std::vector and bool. This header is included
 * from src/steps/kernels.cpp, so it needs the prefixed spelling.
 */

#define BSLZ4_VSX_WINDOW 128   /* pixels; a multiple of both vector widths */
#define BSLZ4_VSX_TO_DENSE 20  /* sparse -> dense when a window keeps > 1/20 (5%) */
#define BSLZ4_VSX_TO_SPARSE 50 /* dense -> sparse when a window keeps < 1/50 (2%) */

/* Hysteresis: the mode only changes on a clear signal, so a density near
 * the crossover does not flip the loop every window. */
static inline int bslz4_vsx_next_dense(int dense, int kept) {
    return dense ? (size_t) kept * BSLZ4_VSX_TO_SPARSE >= BSLZ4_VSX_WINDOW
                 : (size_t) kept * BSLZ4_VSX_TO_DENSE > BSLZ4_VSX_WINDOW;
}

template<bool S = false>
inline int bslz4_collect_vsx_u16(const uint16_t *BSLZ4_RESTRICT block,
                                  const uint8_t *BSLZ4_RESTRICT mask,
                                  size_t i0, size_t n, uint16_t cut,
                                  uint16_t *BSLZ4_RESTRICT out_vals,
                                  uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const size_t nv = n & ~(size_t) 7;
    int dense = 0;
    __vector unsigned short vcut = vec_splats(cut);
    while (j < nv) {
        const size_t end = j + BSLZ4_VSX_WINDOW < nv ? j + BSLZ4_VSX_WINDOW : nv;
        const int start = npx;
        if (dense) {
            for (; j < end; j++) {
                if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
                    out_vals[npx] = block[j];
                    out_adr[npx] = (uint32_t) (j + i0);
                    npx++;
                }
            }
        } else {
            for (; j < end; j += 8) {
                __vector unsigned short v;
                memcpy(&v, &block[j], sizeof(v));
                if (!bslz4_vsx_any_gt<S>(v, vcut)) continue;
                __vector __bool short gt = bslz4_vsx_gt<S>(v, vcut);
                for (int lane = 0; lane < 8; lane++) {
                    if (bslz4_mask_ok(mask, j + i0 + lane) && vec_extract((__vector unsigned short) gt, lane)) {
                        out_vals[npx] = vec_extract(v, lane);
                        out_adr[npx] = (uint32_t) (j + i0 + lane);
                        npx++;
                    }
                }
            }
        }
        dense = bslz4_vsx_next_dense(dense, npx - start);
    }
    for (; j < n; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<bool S = false>
inline int bslz4_collect_vsx_u32(const uint32_t *BSLZ4_RESTRICT block,
                                  const uint8_t *BSLZ4_RESTRICT mask,
                                  size_t i0, size_t n, uint32_t cut,
                                  uint32_t *BSLZ4_RESTRICT out_vals,
                                  uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const size_t nv = n & ~(size_t) 3;
    int dense = 0;
    __vector unsigned int vcut = vec_splats(cut);
    while (j < nv) {
        const size_t end = j + BSLZ4_VSX_WINDOW < nv ? j + BSLZ4_VSX_WINDOW : nv;
        const int start = npx;
        if (dense) {
            for (; j < end; j++) {
                if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
                    out_vals[npx] = block[j];
                    out_adr[npx] = (uint32_t) (j + i0);
                    npx++;
                }
            }
        } else {
            for (; j < end; j += 4) {
                __vector unsigned int v;
                memcpy(&v, &block[j], sizeof(v));
                if (!bslz4_vsx_any_gt<S>(v, vcut)) continue;
                __vector __bool int gt = bslz4_vsx_gt<S>(v, vcut);
                for (int lane = 0; lane < 4; lane++) {
                    if (bslz4_mask_ok(mask, j + i0 + lane) && vec_extract((__vector unsigned int) gt, lane)) {
                        out_vals[npx] = vec_extract(v, lane);
                        out_adr[npx] = (uint32_t) (j + i0 + lane);
                        npx++;
                    }
                }
            }
        }
        dense = bslz4_vsx_next_dense(dense, npx - start);
    }
    for (; j < n; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/* u8 version of collect vsx: 16 lanes of bytes. */
template<bool S = false>
inline int bslz4_collect_vsx_u8(const uint8_t *BSLZ4_RESTRICT block,
                                  const uint8_t *BSLZ4_RESTRICT mask,
                                  size_t i0, size_t n, uint8_t cut,
                                  uint8_t *BSLZ4_RESTRICT out_vals,
                                  uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const size_t nv = n & ~(size_t) 15;
    int dense = 0;
    __vector unsigned char vcut = vec_splats(cut);
    while (j < nv) {
        const size_t end = j + BSLZ4_VSX_WINDOW < nv ? j + BSLZ4_VSX_WINDOW : nv;
        const int start = npx;
        if (dense) {
            for (; j < end; j++) {
                if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
                    out_vals[npx] = block[j];
                    out_adr[npx] = (uint32_t) (j + i0);
                    npx++;
                }
            }
        } else {
            for (; j < end; j += 16) {
                __vector unsigned char v;
                memcpy(&v, &block[j], sizeof(v));
                if (!bslz4_vsx_any_gt<S>(v, vcut)) continue;
                __vector __bool char gt = bslz4_vsx_gt<S>(v, vcut);
                for (int lane = 0; lane < 16; lane++) {
                    if (bslz4_mask_ok(mask, j + i0 + lane) && vec_extract((__vector unsigned char) gt, lane)) {
                        out_vals[npx] = vec_extract(v, lane);
                        out_adr[npx] = (uint32_t) (j + i0 + lane);
                        npx++;
                    }
                }
            }
        }
        dense = bslz4_vsx_next_dense(dense, npx - start);
    }
    for (; j < n; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

#endif /* BSLZ4_HAVE_VSX_COLLECT */

#if BSLZ4_HAVE_NEON_COLLECT

/*
 * NEON has native unsigned compare-greater (vcgtq_u16/u32) -- no
 * sign-bias trick, like VSX, unlike x86 SSE2/AVX2.
 *
 * vmaxvq_u16/u32 (AArch64-only) reduces a compare result to its max
 * lane in one instruction: if that max is 0, no lane exceeded cut, so
 * the whole per-lane extraction is safe to skip -- the same gate VSX
 * uses via vec_any_gt, faster than scalar on a Cortex-A72, see
 * tools/bslz4_neon_collect_probe.c.
 *
 * vgetq_lane_u16/u32 need a compile-time-constant lane index (a real
 * ACLE restriction VSX's vec_extract does not have), so when the gate
 * does not fire, the vector and compare result are spilled to small
 * stack arrays via vst1q_u16/u32 and indexed as plain scalars instead
 * of fighting that restriction with an unrolled constant-index ladder.
 */

template<bool S = false>
inline int bslz4_collect_neon_u16(const uint16_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t n, uint16_t cut,
                                   uint16_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    uint16x8_t vcut = vdupq_n_u16(cut);
    for (; j + 8 <= n; j += 8) {
        uint16x8_t v = vld1q_u16(&block[j]);
        uint16x8_t gt = (S ? vcgtq_s16(vreinterpretq_s16_u16(v), vreinterpretq_s16_u16(vcut)) : vcgtq_u16(v, vcut));
        if (vmaxvq_u16(gt) == 0) continue;
        uint16_t vbuf[8], gbuf[8];
        vst1q_u16(vbuf, v);
        vst1q_u16(gbuf, gt);
        for (int lane = 0; lane < 8; lane++) {
            if (bslz4_mask_ok(mask, j + i0 + lane) && gbuf[lane]) {
                out_vals[npx] = vbuf[lane];
                out_adr[npx] = (uint32_t) (j + i0 + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<bool S = false>
inline int bslz4_collect_neon_u32(const uint32_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t n, uint32_t cut,
                                   uint32_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    uint32x4_t vcut = vdupq_n_u32(cut);
    for (; j + 4 <= n; j += 4) {
        uint32x4_t v = vld1q_u32(&block[j]);
        uint32x4_t gt = (S ? vcgtq_s32(vreinterpretq_s32_u32(v), vreinterpretq_s32_u32(vcut)) : vcgtq_u32(v, vcut));
        if (vmaxvq_u32(gt) == 0) continue;
        uint32_t vbuf[4], gbuf[4];
        vst1q_u32(vbuf, v);
        vst1q_u32(gbuf, gt);
        for (int lane = 0; lane < 4; lane++) {
            if (bslz4_mask_ok(mask, j + i0 + lane) && gbuf[lane]) {
                out_vals[npx] = vbuf[lane];
                out_adr[npx] = (uint32_t) (j + i0 + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/* u8 version of collect neon: 16 lanes of bytes. */
template<bool S = false>
inline int bslz4_collect_neon_u8(const uint8_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t n, uint8_t cut,
                                   uint8_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    uint8x16_t vcut = vdupq_n_u8(cut);
    for (; j + 16 <= n; j += 16) {
        uint8x16_t v = vld1q_u8(&block[j]);
        uint8x16_t gt = (S ? vcgtq_s8(vreinterpretq_s8_u8(v), vreinterpretq_s8_u8(vcut)) : vcgtq_u8(v, vcut));
        if (vmaxvq_u8(gt) == 0) continue;
        uint8_t vbuf[16], gbuf[16];
        vst1q_u8(vbuf, v);
        vst1q_u8(gbuf, gt);
        for (int lane = 0; lane < 16; lane++) {
            if (bslz4_mask_ok(mask, j + i0 + lane) && gbuf[lane]) {
                out_vals[npx] = vbuf[lane];
                out_adr[npx] = (uint32_t) (j + i0 + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if (bslz4_mask_ok(mask, j + i0) & bslz4_gt<S>(block[j], cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

#endif /* BSLZ4_HAVE_NEON_COLLECT */

/*
 * bslz4_collect_gt<T>: mask[j+i0]>0 & block[j]>cut, compacted into
 * (out_vals, out_adr). Generic (scalar) for every T; explicitly
 * specialized for uint8_t/uint16_t/uint32_t to dispatch on the pipeline's
 * collect value (avx512cs, avx2cs, sse2, vsx, neon -- see the top of this
 * file), any other value meaning the scalar loop. Used by the plain sparse
 * route and by every dense dot (dot_*.hpp).
 */
template<typename T>
inline int bslz4_collect_gt(const T *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                             size_t i0, size_t n, T cut, int collect,
                             T *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    (void) collect;  /* only the scalar tier is valid for non-SIMD dtypes */
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/*
 * bslz4_collect_nz<T>: mask[j+i0]>0 & block[j]!=0 (T's own domain, not
 * a cast through an unsigned type). Used by the CSC
 * sparse-route compaction pass. Deliberately NOT implemented in terms of
 * bslz4_collect_gt(..., cut=0): that would be equivalent for unsigned T
 * (this template's own SIMD specializations rely on exactly that
 * equivalence) but wrong for signed T, where !=0 must keep including
 * negative values.
 */
template<typename T>
inline int bslz4_collect_nz(const T *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                             size_t i0, size_t n, int collect,
                             T *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    (void) collect;  /* only the scalar tier is valid for non-SIMD dtypes */
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        T px = block[j];
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (px != 0))) {
            out_vals[npx] = px;
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

#if BSLZ4_HAVE_AVX512_COLLECT || BSLZ4_HAVE_AVX2_COLLECT || BSLZ4_HAVE_SSE2_COLLECT || BSLZ4_HAVE_VSX_COLLECT || BSLZ4_HAVE_NEON_COLLECT

template<>
inline int bslz4_collect_gt<uint16_t>(const uint16_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n, uint16_t cut, int collect,
                                       uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_gt<uint32_t>(const uint32_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n, uint32_t cut, int collect,
                                       uint32_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_nz<uint16_t>(const uint16_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n, int collect,
                                       uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* unsigned T: !=0 is exactly >0, so cut==0 reuses the same kernels. */
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        uint16_t px = block[j];
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (px != 0))) {
            out_vals[npx] = px;
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_nz<uint32_t>(const uint32_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n, int collect,
                                       uint32_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        uint32_t px = block[j];
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (px != 0))) {
            out_vals[npx] = px;
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_gt<uint8_t>(const uint8_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n, uint8_t cut, int collect,
                                       uint8_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u8(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u8(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u8(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u8(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u8(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u8(block, mask, i0, n, cut, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_nz<uint8_t>(const uint8_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n, int collect,
                                       uint8_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* unsigned T: !=0 is exactly >0, so cut==0 reuses the same kernels. */
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u8(block, mask, i0, n, (uint8_t) 0, out_vals, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u8(block, mask, i0, n, (uint8_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u8(block, mask, i0, n, (uint8_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u8(block, mask, i0, n, (uint8_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u8(block, mask, i0, n, (uint8_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u8(block, mask, i0, n, (uint8_t) 0, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        uint8_t px = block[j];
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (px != 0))) {
            out_vals[npx] = px;
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/* signed pixels: the SIMD collects with a signed compare */

template<>
inline int bslz4_collect_gt<int8_t>(const int8_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                    size_t i0, size_t n, int8_t cut, int collect,
                                    int8_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* the unsigned kernels on the same bits, comparing as signed */
    const uint8_t *ub = (const uint8_t *) (const void *) block;
    uint8_t *uo = (uint8_t *) (void *) out_vals;
    const uint8_t uc = (uint8_t) cut;
    (void) ub; (void) uo; (void) uc;
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u8<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u8<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u8<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u8<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u8<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u8<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_nz<int8_t>(const int8_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t n, int collect,
                                   int8_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* != 0 is the same test on the unsigned bits (> 0 there): the unsigned kernels, cut 0 */
    return bslz4_collect_nz<uint8_t>((const uint8_t *) (const void *) block, mask, i0, n, collect,
                                    (uint8_t *) (void *) out_vals, out_adr);
}

template<>
inline int bslz4_collect_gt<int16_t>(const int16_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                    size_t i0, size_t n, int16_t cut, int collect,
                                    int16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* the unsigned kernels on the same bits, comparing as signed */
    const uint16_t *ub = (const uint16_t *) (const void *) block;
    uint16_t *uo = (uint16_t *) (void *) out_vals;
    const uint16_t uc = (uint16_t) cut;
    (void) ub; (void) uo; (void) uc;
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u16<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u16<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u16<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u16<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u16<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u16<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_nz<int16_t>(const int16_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t n, int collect,
                                   int16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* != 0 is the same test on the unsigned bits (> 0 there): the unsigned kernels, cut 0 */
    return bslz4_collect_nz<uint16_t>((const uint16_t *) (const void *) block, mask, i0, n, collect,
                                    (uint16_t *) (void *) out_vals, out_adr);
}

template<>
inline int bslz4_collect_gt<int32_t>(const int32_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                    size_t i0, size_t n, int32_t cut, int collect,
                                    int32_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* the unsigned kernels on the same bits, comparing as signed */
    const uint32_t *ub = (const uint32_t *) (const void *) block;
    uint32_t *uo = (uint32_t *) (void *) out_vals;
    const uint32_t uc = (uint32_t) cut;
    (void) ub; (void) uo; (void) uc;
#if BSLZ4_HAVE_AVX512_COLLECT
#if BSLZ4_HAVE_VBMI_GFNI
    if (collect == BSLZ4_COLLECT_AVX512CS) return bslz4_collect_avx512cs_u32<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#endif
#if BSLZ4_HAVE_AVX512_COLLECT
    if (collect == BSLZ4_COLLECT_AVX512F) return bslz4_collect_avx512f_u32<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (collect == BSLZ4_COLLECT_AVX2CS) return bslz4_collect_avx2cs_u32<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (collect == BSLZ4_COLLECT_SSE2) return bslz4_collect_sse2_u32<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (collect == BSLZ4_COLLECT_VSX) return bslz4_collect_vsx_u32<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (collect == BSLZ4_COLLECT_NEON) return bslz4_collect_neon_u32<true>(ub, mask, i0, n, uc, uo, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY(bslz4_mask_ok(mask, j + i0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_nz<int32_t>(const int32_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t n, int collect,
                                   int32_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* != 0 is the same test on the unsigned bits (> 0 there): the unsigned kernels, cut 0 */
    return bslz4_collect_nz<uint32_t>((const uint32_t *) (const void *) block, mask, i0, n, collect,
                                    (uint32_t *) (void *) out_vals, out_adr);
}

#endif /* any tier available */

} /* namespace bslz4 */
