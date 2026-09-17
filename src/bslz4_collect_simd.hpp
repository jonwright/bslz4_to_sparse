#pragma once
/*
 * SIMD mask+threshold collection for the u16/u32 decode hot path:
 * bslz4_decode_multi's threshold-collect loop (mask>0 & value>cut) and
 * bslz4_csc_decode_multi's sparse-route compaction (mask>0 & value!=0,
 * called here with cut==0 -- identical to !=0 for unsigned T).
 *
 * Five tiers, u16/u32 only: AVX-512 (F+BW+VL), AVX2, SSE2 (x86-64 ABI
 * baseline, always present), VSX (POWER8+, probe in
 * tools/bslz4_power9_collect_probe.c) and NEON (AArch64, ASIMD is
 * ARMv8-A baseline, probe in tools/bslz4_neon_collect_probe.c). The x86
 * tiers carry __attribute__((target(...))), so this file compiles under
 * the project's ordinary -O2; VSX needs setup.py to pass -maltivec -mvsx
 * globally on ppc64le, NEON needs nothing extra on aarch64 (ASIMD is
 * baseline, like SSE2 on x86-64). MSVC has none of these, and the
 * BSLZ4_HAVE_*_COLLECT guards below make this file inert there.
 *
 * Each tier has a runtime flag (bslz4_avx512_collect_enabled() /
 * _avx2_ / _sse2_ / _vsx_ / _neon_), enabled by default iff it is the
 * highest-priority tier compiled in and bslz4_<tier>_collect_capable()
 * says this CPU supports it -- checked once, at first use, via c2py23's
 * cpuid-equivalent globals (c2py_amd64_avx512f/bw/vl, c2py_amd64_avx2,
 * c2py_ppc64_vsx, c2py_arm64_asimd; SSE2 needs no check). capable() is
 * the single source of truth, shared with the Python-visible
 * <tier>_collect_available() query in bslz4_to_sparse.cpp.
 *
 * set_<tier>_collect(True/False) overrides at runtime. AVX-512 can
 * trigger frequency throttling on some chips that outweighs the wider
 * vector for this workload; set_avx512_collect(False) drops to the next
 * tier, which does not auto-promote, so call again to reach avx2/sse2.
 *
 * bslz4_collect_gt<T>/bslz4_collect_nz<T> (bottom of this file) try the
 * tiers in that order and fall back to the plain scalar loop.
 */

#include "bslz4_common.hpp"

#include <cstring>

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

/* Single source of truth for "is this tier actually usable" -- shared
 * by the default-enabled logic below and the Python-visible
 * <tier>_collect_available() query (bslz4_to_sparse.cpp delegates to
 * these rather than duplicating the cpuid-check logic). */

inline bool bslz4_avx512_collect_capable() {
#if BSLZ4_HAVE_AVX512_COLLECT
    return (c2py_amd64_avx512f && c2py_amd64_avx512bw && c2py_amd64_avx512vl) != 0;
#else
    return false;
#endif
}

inline bool bslz4_avx2_collect_capable() {
#if BSLZ4_HAVE_AVX2_COLLECT
    return c2py_amd64_avx2 != 0;
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

/* Each defaults to true iff it's the highest-priority capable tier --
 * so exactly one of these (or none, on a build/CPU with nothing
 * available) starts enabled, matching the priority order
 * bslz4_collect_gt<T>/bslz4_collect_nz<T> dispatch in below. */

inline bool &bslz4_avx512_collect_enabled() {
    static bool x = bslz4_avx512_collect_capable();
    return x;
}

inline bool &bslz4_avx2_collect_enabled() {
    static bool x = !bslz4_avx512_collect_capable() && bslz4_avx2_collect_capable();
    return x;
}

inline bool &bslz4_vsx_collect_enabled() {
    /* Mutually exclusive with the x86 tiers by compilation (BSLZ4_HAVE_
     * VSX_COLLECT and BSLZ4_HAVE_{AVX512,AVX2,SSE2}_COLLECT can't both
     * be 1 in the same build), so no priority check against them needed. */
    static bool x = bslz4_vsx_collect_capable();
    return x;
}

inline bool &bslz4_neon_collect_enabled() {
    /* Mutually exclusive with every other tier by compilation, same as VSX. */
    static bool x = bslz4_neon_collect_capable();
    return x;
}

inline bool &bslz4_sse2_collect_enabled() {
    static bool x = !bslz4_avx512_collect_capable() && !bslz4_avx2_collect_capable()
                     && bslz4_sse2_collect_capable();
    return x;
}

#if BSLZ4_HAVE_AVX512_COLLECT

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline int bslz4_collect_avx512_u16(const uint16_t *BSLZ4_RESTRICT block,
                                     const uint8_t *BSLZ4_RESTRICT mask,
                                     size_t i0, size_t block_elems, uint16_t cut,
                                     uint16_t *BSLZ4_RESTRICT out_vals,
                                     uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m512i vcut = _mm512_set1_epi16((short) cut);
    for (; j + 32 <= block_elems; j += 32) {
        __mmask32 mnz = _mm256_cmpneq_epu8_mask(
            _mm256_loadu_si256((const __m256i *) &mask[j + i0]), _mm256_setzero_si256());
        __mmask32 mgt = _mm512_cmpgt_epu16_mask(
            _mm512_loadu_si512((const void *) &block[j]), vcut);
        uint32_t bits = (uint32_t) (mnz & mgt);
        while (bits) {
            int b = __builtin_ctz(bits);
            out_vals[npx] = block[j + b];
            out_adr[npx] = (uint32_t) (j + i0 + b);
            npx++;
            bits &= bits - 1;
        }
    }
    for (; j < block_elems; j++) {
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline int bslz4_collect_avx512_u32(const uint32_t *BSLZ4_RESTRICT block,
                                     const uint8_t *BSLZ4_RESTRICT mask,
                                     size_t i0, size_t block_elems, uint32_t cut,
                                     uint32_t *BSLZ4_RESTRICT out_vals,
                                     uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m512i vcut = _mm512_set1_epi32((int) cut);
    for (; j + 16 <= block_elems; j += 16) {
        __mmask16 mnz = _mm_cmpneq_epu8_mask(
            _mm_loadu_si128((const __m128i *) &mask[j + i0]), _mm_setzero_si128());
        __mmask16 mgt = _mm512_cmpgt_epu32_mask(
            _mm512_loadu_si512((const void *) &block[j]), vcut);
        uint32_t bits = (uint32_t) (mnz & mgt);
        while (bits) {
            int b = __builtin_ctz(bits);
            out_vals[npx] = block[j + b];
            out_adr[npx] = (uint32_t) (j + i0 + b);
            npx++;
            bits &= bits - 1;
        }
    }
    for (; j < block_elems; j++) {
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
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
 * Once there's a scalar bitmask, extraction (ctz/bits&=bits-1) is
 * identical to the AVX-512 kernels above -- that part isn't ISA-specific
 * at all.
 */

__attribute__((target("avx2")))
inline int bslz4_collect_avx2_u16(const uint16_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t block_elems, uint16_t cut,
                                   uint16_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m256i sign_bias = _mm256_set1_epi16((short) 0x8000);
    const __m256i vcut_b = _mm256_xor_si256(_mm256_set1_epi16((short) cut), sign_bias);
    for (; j + 16 <= block_elems; j += 16) {
        __m128i m = _mm_loadu_si128((const __m128i *) &mask[j + i0]);
        __m128i mz = _mm_cmpeq_epi8(m, _mm_setzero_si128());
        uint32_t mnz16 = (~(uint32_t) _mm_movemask_epi8(mz)) & 0xFFFFu;

        __m256i v = _mm256_loadu_si256((const __m256i *) &block[j]);
        __m256i vb = _mm256_xor_si256(v, sign_bias);
        __m256i gt = _mm256_cmpgt_epi16(vb, vcut_b);
        uint32_t packed = (uint32_t) _mm256_movemask_epi8(_mm256_packs_epi16(gt, gt));
        uint32_t mgt16 = (packed & 0xFFu) | ((packed >> 8) & 0xFF00u);

        uint32_t bits = mnz16 & mgt16;
        while (bits) {
            int b = __builtin_ctz(bits);
            out_vals[npx] = block[j + b];
            out_adr[npx] = (uint32_t) (j + i0 + b);
            npx++;
            bits &= bits - 1;
        }
    }
    for (; j < block_elems; j++) {
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

__attribute__((target("avx2")))
inline int bslz4_collect_avx2_u32(const uint32_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t block_elems, uint32_t cut,
                                   uint32_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m256i sign_bias = _mm256_set1_epi32((int) 0x80000000u);
    const __m256i vcut_b = _mm256_xor_si256(_mm256_set1_epi32((int) cut), sign_bias);
    for (; j + 8 <= block_elems; j += 8) {
        __m128i m = _mm_loadl_epi64((const __m128i *) &mask[j + i0]);
        __m128i mz = _mm_cmpeq_epi8(m, _mm_setzero_si128());
        uint32_t mnz8 = (~(uint32_t) _mm_movemask_epi8(mz)) & 0xFFu;

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
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
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

__attribute__((target("sse2")))
inline int bslz4_collect_sse2_u16(const uint16_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t block_elems, uint16_t cut,
                                   uint16_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m128i sign_bias = _mm_set1_epi16((short) 0x8000);
    const __m128i vcut_b = _mm_xor_si128(_mm_set1_epi16((short) cut), sign_bias);
    for (; j + 8 <= block_elems; j += 8) {
        __m128i m = _mm_loadl_epi64((const __m128i *) &mask[j + i0]);
        __m128i mz = _mm_cmpeq_epi8(m, _mm_setzero_si128());
        uint32_t mnz8 = (~(uint32_t) _mm_movemask_epi8(mz)) & 0xFFu;

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
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

__attribute__((target("sse2")))
inline int bslz4_collect_sse2_u32(const uint32_t *BSLZ4_RESTRICT block,
                                   const uint8_t *BSLZ4_RESTRICT mask,
                                   size_t i0, size_t block_elems, uint32_t cut,
                                   uint32_t *BSLZ4_RESTRICT out_vals,
                                   uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    const __m128i sign_bias = _mm_set1_epi32((int) 0x80000000u);
    const __m128i vcut_b = _mm_xor_si128(_mm_set1_epi32((int) cut), sign_bias);
    for (; j + 4 <= block_elems; j += 4) {
        uint32_t m4;
        memcpy(&m4, &mask[j + i0], 4);
        __m128i m = _mm_cvtsi32_si128((int) m4);
        __m128i mz = _mm_cmpeq_epi8(m, _mm_setzero_si128());
        uint32_t mnz4 = (~(uint32_t) _mm_movemask_epi8(mz)) & 0xFu;

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
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
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
 * gate fires for nearly every chunk -- 3.75-8.2x faster than scalar on
 * real POWER9 hardware, where an ungated version is ~2x slower.
 *
 * When the gate does not fire, per-lane extraction is vec_extract in a
 * small fixed-trip loop (8 iterations for u16, 4 for u32).
 *
 * __vector/__bool, not bare vector/bool: <altivec.h> defines the bare
 * keyword-macros in C mode only, leaving them undefined in C++ so they
 * cannot collide with std::vector and bool. This header is included
 * from bslz4_to_sparse.cpp, so it needs the prefixed spelling.
 */

inline int bslz4_collect_vsx_u16(const uint16_t *BSLZ4_RESTRICT block,
                                  const uint8_t *BSLZ4_RESTRICT mask,
                                  size_t i0, size_t n, uint16_t cut,
                                  uint16_t *BSLZ4_RESTRICT out_vals,
                                  uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    __vector unsigned short vcut = vec_splats(cut);
    for (; j + 8 <= n; j += 8) {
        __vector unsigned short v;
        memcpy(&v, &block[j], sizeof(v));
        if (!vec_any_gt(v, vcut)) continue;
        __vector __bool short gt = vec_cmpgt(v, vcut);
        for (int lane = 0; lane < 8; lane++) {
            if (mask[j + i0 + lane] > 0 && vec_extract((__vector unsigned short) gt, lane)) {
                out_vals[npx] = vec_extract(v, lane);
                out_adr[npx] = (uint32_t) (j + i0 + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

inline int bslz4_collect_vsx_u32(const uint32_t *BSLZ4_RESTRICT block,
                                  const uint8_t *BSLZ4_RESTRICT mask,
                                  size_t i0, size_t n, uint32_t cut,
                                  uint32_t *BSLZ4_RESTRICT out_vals,
                                  uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    size_t j = 0;
    __vector unsigned int vcut = vec_splats(cut);
    for (; j + 4 <= n; j += 4) {
        __vector unsigned int v;
        memcpy(&v, &block[j], sizeof(v));
        if (!vec_any_gt(v, vcut)) continue;
        __vector __bool int gt = vec_cmpgt(v, vcut);
        for (int lane = 0; lane < 4; lane++) {
            if (mask[j + i0 + lane] > 0 && vec_extract((__vector unsigned int) gt, lane)) {
                out_vals[npx] = vec_extract(v, lane);
                out_adr[npx] = (uint32_t) (j + i0 + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
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
 * uses via vec_any_gt, real-hardware-measured on a Cortex-A72 at
 * 4.18x/4.38x (u16/u32) over scalar, see
 * tools/bslz4_neon_collect_probe.c.
 *
 * vgetq_lane_u16/u32 need a compile-time-constant lane index (a real
 * ACLE restriction VSX's vec_extract does not have), so when the gate
 * does not fire, the vector and compare result are spilled to small
 * stack arrays via vst1q_u16/u32 and indexed as plain scalars instead
 * of fighting that restriction with an unrolled constant-index ladder.
 */

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
        uint16x8_t gt = vcgtq_u16(v, vcut);
        if (vmaxvq_u16(gt) == 0) continue;
        uint16_t vbuf[8], gbuf[8];
        vst1q_u16(vbuf, v);
        vst1q_u16(gbuf, gt);
        for (int lane = 0; lane < 8; lane++) {
            if (mask[j + i0 + lane] > 0 && gbuf[lane]) {
                out_vals[npx] = vbuf[lane];
                out_adr[npx] = (uint32_t) (j + i0 + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

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
        uint32x4_t gt = vcgtq_u32(v, vcut);
        if (vmaxvq_u32(gt) == 0) continue;
        uint32_t vbuf[4], gbuf[4];
        vst1q_u32(vbuf, v);
        vst1q_u32(gbuf, gt);
        for (int lane = 0; lane < 4; lane++) {
            if (mask[j + i0 + lane] > 0 && gbuf[lane]) {
                out_vals[npx] = vbuf[lane];
                out_adr[npx] = (uint32_t) (j + i0 + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if ((mask[j + i0] > 0) & (block[j] > cut)) {
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
 * specialized for uint16_t/uint32_t to try SIMD tiers in priority order
 * (avx512, avx2, sse2, vsx, neon -- each checked via its own runtime
 * enabled flag, the best available one being enabled by default: see
 * the top of this file), falling back to the same scalar loop otherwise.
 * Used by bslz4_decode_multi's real threshold-collect (bslz4_core.hpp).
 */
template<typename T>
inline int bslz4_collect_gt(const T *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                             size_t i0, size_t n, T cut,
                             T *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

/*
 * bslz4_collect_nz<T>: mask[j+i0]>0 & block[j]!=0 (T's own domain, not
 * a cast through an unsigned type -- see bslz4_core.hpp's TODO about
 * what "!=0" should mean for signed sentinel values). Used by
 * bslz4_csc_decode_multi's sparse-route compaction pass. Deliberately
 * NOT implemented in terms of bslz4_collect_gt(..., cut=0): that would
 * be equivalent for unsigned T (this template's own SIMD specializations
 * rely on exactly that equivalence) but wrong for signed T, where !=0
 * must keep including negative values.
 */
template<typename T>
inline int bslz4_collect_nz(const T *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                             size_t i0, size_t n,
                             T *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        T px = block[j];
        if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (px != 0))) {
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
                                       size_t i0, size_t n, uint16_t cut,
                                       uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
#if BSLZ4_HAVE_AVX512_COLLECT
    if (bslz4_avx512_collect_enabled())
        return bslz4_collect_avx512_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (bslz4_avx2_collect_enabled())
        return bslz4_collect_avx2_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (bslz4_sse2_collect_enabled())
        return bslz4_collect_sse2_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (bslz4_vsx_collect_enabled())
        return bslz4_collect_vsx_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (bslz4_neon_collect_enabled())
        return bslz4_collect_neon_u16(block, mask, i0, n, cut, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_gt<uint32_t>(const uint32_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n, uint32_t cut,
                                       uint32_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
#if BSLZ4_HAVE_AVX512_COLLECT
    if (bslz4_avx512_collect_enabled())
        return bslz4_collect_avx512_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (bslz4_avx2_collect_enabled())
        return bslz4_collect_avx2_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (bslz4_sse2_collect_enabled())
        return bslz4_collect_sse2_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (bslz4_vsx_collect_enabled())
        return bslz4_collect_vsx_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (bslz4_neon_collect_enabled())
        return bslz4_collect_neon_u32(block, mask, i0, n, cut, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (block[j] > cut))) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_nz<uint16_t>(const uint16_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n,
                                       uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    /* unsigned T: !=0 is exactly >0, so cut==0 reuses the same kernels. */
#if BSLZ4_HAVE_AVX512_COLLECT
    if (bslz4_avx512_collect_enabled())
        return bslz4_collect_avx512_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (bslz4_avx2_collect_enabled())
        return bslz4_collect_avx2_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (bslz4_sse2_collect_enabled())
        return bslz4_collect_sse2_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (bslz4_vsx_collect_enabled())
        return bslz4_collect_vsx_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (bslz4_neon_collect_enabled())
        return bslz4_collect_neon_u16(block, mask, i0, n, (uint16_t) 0, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        uint16_t px = block[j];
        if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (px != 0))) {
            out_vals[npx] = px;
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<>
inline int bslz4_collect_nz<uint32_t>(const uint32_t *BSLZ4_RESTRICT block, const uint8_t *BSLZ4_RESTRICT mask,
                                       size_t i0, size_t n,
                                       uint32_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
#if BSLZ4_HAVE_AVX512_COLLECT
    if (bslz4_avx512_collect_enabled())
        return bslz4_collect_avx512_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (bslz4_avx2_collect_enabled())
        return bslz4_collect_avx2_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (bslz4_sse2_collect_enabled())
        return bslz4_collect_sse2_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (bslz4_vsx_collect_enabled())
        return bslz4_collect_vsx_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (bslz4_neon_collect_enabled())
        return bslz4_collect_neon_u32(block, mask, i0, n, (uint32_t) 0, out_vals, out_adr);
#endif
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        uint32_t px = block[j];
        if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (px != 0))) {
            out_vals[npx] = px;
            out_adr[npx] = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

#endif /* any tier available */

} /* namespace bslz4 */
