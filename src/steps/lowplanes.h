#ifndef BSLZ4_STEP_LOWPLANES_H
#define BSLZ4_STEP_LOWPLANES_H
/*
 * Step "untranspose", the low-planes values: u16 blocks whose high
 * byte-planes are all zero.
 *
 * A bitshuffled block of ne elements of NB bytes stores the bit-planes of
 * byte 0 first (ne bytes), then those of byte 1.  When a u16 block's high
 * byte-planes are all zero -- every value < 256, as with low counts and a
 * zero-filled mask -- its first ne bytes alone are a bitshuffled u8 block:
 * half the bit transpose and no byte transpose give the whole result.  Two
 * forms:
 *
 *   planes -> block  bslz4_lowplanes_block: for the dense dot route, which
 *                    needs the pixel block (VBMI: straight into u16; AVX2:
 *                    kcb's u8 transpose, widened).
 *   planes -> list   bslz4_lowplanes_list: the transpose fused with the
 *                    collect, for plain sparsify and the sparse route, which
 *                    need only the selected pixels; the block is never
 *                    written (VBMI, AVX2 in kernels.cpp, VSX, portable C).
 *
 * Blocks that do not qualify (a value >= 256, or a block size a kernel does
 * not take) return "not handled" and the caller untransposes them with kcb.
 * Included by src/pipeline/driver.c only.
 */

#include "../pipeline/common.h"
#include "../pipeline/pipeline.h"
#include "caps.h"            /* bslz4_lowplanes_collect_avx2 (kernels.cpp) */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if BSLZ4_X86_SIMD
#include <immintrin.h>
#include "c2py_amd64.h"
#endif

/* kcb (bitshuffle.c): untransposes `size` elements of elem_size bytes */
int bitshuf_decode_block(char *out, const char *in, char *scratch, size_t size, size_t elem_size);

#ifndef BSLZ4_LOWPLANES_SKIP_RATIO
/* The AVX-512 fused collect transposes only the 64-pixel groups holding
 * data (a bitmap from ORing the planes) in blocks compressed more than this;
 * every group otherwise.  hpc8: 0.1 % frames -25 %, WAu0012 -8 %; at 16x
 * WAu0008 blocks (~44 % of groups busy) gained nothing. */
#define BSLZ4_LOWPLANES_SKIP_RATIO 48
#endif

/* POWER: branch-free emit for blocks compressed more than this (9 % frames
 * -18 %; on dense blocks nearly every pixel passes and the store chain is
 * serial: +89 %) */
#define BSLZ4_LOWPLANES_BRANCHFREE_RATIO 8

/* planes at or past the decoder's nz_end are read from here (planes of up
 * to 8192 pixels) instead of being zero filled */
static const uint8_t bslz4_zero_plane[1024];

#if BSLZ4_X86_SIMD
/* u16 blocks whose high bytes are all zero: untranspose the 8 low bit-planes
 * straight into u16 (AVX-512 VBMI + GFNI).  The bit transpose is kcb's
 * (bitshuf_untrans_bit_avx512vbmi_gfni, Copyright (c) 2023 Kal Conley,
 * MIT / Apache-2.0): per 64 pixels, 8 bytes from each plane are permuted so
 * each qword holds one 8x8 bit matrix, which gf2p8affineqb transposes.  Here
 * the 64 result bytes are zero-extended and stored as 64 u16 at once, so the
 * low bytes are never stored and widened separately. */
#if BSLZ4_HAVE_VBMI_GFNI
__attribute__((target("avx512f,avx512bw,avx512vbmi,gfni")))
static void bslz4_untrans_lowplanes_u16(uint16_t *BSLZ4_RESTRICT out, const uint8_t *BSLZ4_RESTRICT in,
                                        size_t ne) {
    const size_t size = ne / 8;                 /* bytes per plane; a multiple of 8 */
    const __m512i C = _mm512_set_epi64(0x070f171f474f575f, 0x060e161e464e565e, 0x050d151d454d555d,
                                       0x040c141c444c545c, 0x030b131b434b535b, 0x020a121a424a525a,
                                       0x0109111941495159, 0x0008101840485058);
    const __m512i I8 = _mm512_set1_epi64(0x8040201008040201);
    for (size_t i = 0; i + 8 <= size; i += 8) {
        int64_t a1, a3, a5, a7;
        memcpy(&a1, &in[1 * size + i], 8);
        memcpy(&a3, &in[3 * size + i], 8);
        memcpy(&a5, &in[5 * size + i], 8);
        memcpy(&a7, &in[7 * size + i], 8);
        const __m128i u0 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) &in[0 * size + i]), a1, 1);
        const __m128i u1 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) &in[2 * size + i]), a3, 1);
        const __m128i u2 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) &in[4 * size + i]), a5, 1);
        const __m128i u3 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) &in[6 * size + i]), a7, 1);
        const __m256i v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(u0), u1, 1);
        const __m256i v1 = _mm256_inserti128_si256(_mm256_castsi128_si256(u2), u3, 1);
        __m512i u = _mm512_permutex2var_epi8(_mm512_castsi256_si512(v0), C, _mm512_castsi256_si512(v1));
        u = _mm512_gf2p8affine_epi64_epi8(I8, u, 0x00);
        _mm512_storeu_si512((void *) (out + 8 * i), _mm512_cvtepu8_epi16(_mm512_castsi512_si256(u)));
        _mm512_storeu_si512((void *) (out + 8 * i + 32), _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(u, 1)));
    }
}
#else
static void bslz4_untrans_lowplanes_u16(uint16_t *out, const uint8_t *in, size_t ne) {
    (void) out; (void) in; (void) ne;            /* never called: not capable */
}
#endif

/* 1 if p[0..n) are all zero; n a multiple of 64.  512-byte steps. */
__attribute__((target("avx512f")))
static int bslz4_all_zero_avx512(const uint8_t *BSLZ4_RESTRICT p, size_t n) {
    for (size_t i = 0; i < n; i += 512) {
        __m512i acc = _mm512_setzero_si512();
        for (size_t k = i; k < i + 512 && k < n; k += 64)
            acc = _mm512_or_si512(acc, _mm512_loadu_si512((const void *) (p + k)));
        if (_mm512_test_epi64_mask(acc, acc)) return 0;
    }
    return 1;
}

static int bslz4_lowplanes_u16_capable(void) {
#if BSLZ4_HAVE_VBMI_GFNI
    /* Racing first calls (free-threaded Python) all store the same value;
     * relaxed atomics make that defined and compile to plain loads/stores. */
    static int cached = -1;
    int c = __atomic_load_n(&cached, __ATOMIC_RELAXED);
    if (c < 0) {
        c = c2py_amd64_avx512f && c2py_amd64_avx512bw && __builtin_cpu_supports("avx512vbmi") &&
            __builtin_cpu_supports("gfni");
        __atomic_store_n(&cached, c, __ATOMIC_RELAXED);
    }
    return c;
#else
    return 0;
#endif
}

/* The low-planes transpose fused with the collect, for u16 blocks whose high
 * byte-planes are all zero: per 64 pixels the 8 plane bytes are transposed
 * (as bslz4_untrans_lowplanes_u16) and the pixels > cut (and unmasked) are
 * compressed straight out of the register -- values (vpcompressb, widened to
 * u16) and pixel indices (vpcompressd) -- so the u16 block is never written
 * or re-read.  (No early exit for an all-zero group: on photon-noise frames
 * that branch mispredicts; the one after the cut is predictable.)  Planes at or past nz_end are read from `zeros` (ne/8 zero
 * bytes) rather than zero-filled; only the partial last plane is.  Needs
 * cut < 255 (nothing in these blocks exceeds 255).  Full-width stores reach
 * at most 64 entries past this block's count, so they stay below
 * i0 + j + 64 <= i0 + ne (see bslz4_collect_avx512cs_u16). */
#if BSLZ4_HAVE_VBMI_GFNI
/* Emit one 64-pixel group: the k-selected bytes of u (widened to u16) and
 * their pixel indices (base + lane), full-width stores, branch-free.
 * Returns the new count. */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi2,popcnt")))
static inline int bslz4_lowplanes_emit(__m512i u, __mmask64 k, size_t base_px, int npx,
                                       uint16_t *BSLZ4_RESTRICT out_vals,
                                       uint32_t *BSLZ4_RESTRICT out_adr) {
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i c = _mm512_maskz_compress_epi8(k, u);
    _mm512_storeu_si512((void *) (out_vals + npx), _mm512_cvtepu8_epi16(_mm512_castsi512_si256(c)));
    _mm512_storeu_si512((void *) (out_vals + npx + 32),
                        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(c, 1)));
    const __m512i sixteen = _mm512_set1_epi32(16);
    __m512i ix = _mm512_add_epi32(_mm512_set1_epi32((int) base_px), iota);
    int at = npx;
    for (int q = 0; q < 4; q++) {
        const __mmask16 mq = (__mmask16) (k >> (16 * q));
        _mm512_storeu_si512((void *) (out_adr + at), _mm512_maskz_compress_epi32(mq, ix));
        at += __builtin_popcount((unsigned) mq);
        ix = _mm512_add_epi32(ix, sixteen);
    }
    return at;
}

/* One 64-pixel group (byte offset i in each plane) of the fused collect:
 * transpose its 8 low bit-planes (vpermb + gf2p8affineqb: byte j = pixel
 * 8i + j), select > cut and unmasked, and record bytes and selection for
 * the emit pass -- no data-dependent branch. */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,gfni"), always_inline))
static inline void bslz4_lowplanes_group(const uint8_t *const *BSLZ4_RESTRICT pl, size_t i,
                                         __m512i C, __m512i I8, __m512i vcut,
                                         const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                                         uint8_t *BSLZ4_RESTRICT ubuf, __mmask64 *BSLZ4_RESTRICT kv,
                                         uint64_t *BSLZ4_RESTRICT gm) {
    int64_t a1, a3, a5, a7;
    memcpy(&a1, pl[1] + i, 8);
    memcpy(&a3, pl[3] + i, 8);
    memcpy(&a5, pl[5] + i, 8);
    memcpy(&a7, pl[7] + i, 8);
    const __m128i u0 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) (pl[0] + i)), a1, 1);
    const __m128i u1 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) (pl[2] + i)), a3, 1);
    const __m128i u2 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) (pl[4] + i)), a5, 1);
    const __m128i u3 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) (pl[6] + i)), a7, 1);
    const __m256i v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(u0), u1, 1);
    const __m256i v1 = _mm256_inserti128_si256(_mm256_castsi128_si256(u2), u3, 1);
    __m512i u = _mm512_permutex2var_epi8(_mm512_castsi256_si512(v0), C, _mm512_castsi256_si512(v1));
    u = _mm512_gf2p8affine_epi64_epi8(I8, u, 0x00);
    __mmask64 k = _mm512_cmpgt_epu8_mask(u, vcut);
    if (mask) {
        const __m512i m = _mm512_loadu_si512((const void *) (mask + i0 + 8 * i));
        k &= _mm512_test_epi8_mask(m, m);
    }
    _mm512_store_si512((void *) (ubuf + 8 * i), u);
    kv[i / 8] = k;
    gm[i / 512] |= (uint64_t) (k != 0) << ((i / 8) & 63);
}

/* The fused low-planes collect of a u16 block whose high byte-planes are
 * zero (checked by the caller).  Two passes: at cut 0 on photon noise the
 * groups holding a pixel are many and scattered, so a branch per group
 * mispredicts.  Pass 1 records every group's bytes and selection, pass 2
 * emits only the groups with pixels, found by bit scan (one mispredict per
 * block).  One pass, skipping empty groups, was ~4 % faster at cut 2 but
 * ~30 % slower at cut 0; choosing per block from the previous block's hit
 * rate kept neither (WAu, 2026-10-04). */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,gfni,popcnt")))
static int bslz4_lowplanes_collect_u16(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end,
                                       const uint8_t *BSLZ4_RESTRICT zeros,
                                       const uint8_t *BSLZ4_RESTRICT mask, size_t i0, unsigned cut,
                                       uint16_t *BSLZ4_RESTRICT out_vals,
                                       uint32_t *BSLZ4_RESTRICT out_adr, int skip) {
    const size_t size = ne / 8;                 /* bytes per plane; a multiple of 8 */
    if (nz_end > ne) nz_end = ne;               /* planes 8..15 are zero (checked by the caller) */
    const size_t np = (nz_end + size - 1) / size;
    if (nz_end < np * size) memset(raw + nz_end, 0, np * size - nz_end);
    const uint8_t *pl[8];
    for (size_t p = 0; p < 8; p++) pl[p] = p < np ? raw + p * size : zeros;
    const __m512i C = _mm512_set_epi64(0x070f171f474f575f, 0x060e161e464e565e, 0x050d151d454d555d,
                                       0x040c141c444c545c, 0x030b131b434b535b, 0x020a121a424a525a,
                                       0x0109111941495159, 0x0008101840485058);
    const __m512i I8 = _mm512_set1_epi64(0x8040201008040201);
    const __m512i vcut = _mm512_set1_epi8((char) cut);
    int npx = 0;
    _Alignas(64) uint8_t ubuf[8192];
    __mmask64 kv[128];
    uint64_t gm[2] = {0, 0};
    /* the 64-pixel groups holding any data: the planes ORed 64 bytes (8
     * groups) at a time, one test per 8 groups; with skip, only those are
     * transposed (on sparse frames most are empty).  Without skip, a plain
     * counted loop (the bit-scan loop over every group cost 2-5 % there). */
    if (skip && size % 64 == 0) {
        uint64_t gnz[2] = {0, 0};
        for (size_t i = 0; i < size; i += 64) {
            __m512i acc = _mm512_loadu_si512((const void *) (pl[0] + i));
            for (size_t p = 1; p < 8; p++)
                acc = _mm512_or_si512(acc, _mm512_loadu_si512((const void *) (pl[p] + i)));
            const size_t g = i / 8;
            gnz[g >> 6] |= (uint64_t) _mm512_test_epi64_mask(acc, acc) << (g & 63);
        }
        for (int gw = 0; gw < 2; gw++)
            for (uint64_t gbits = gnz[gw]; gbits; gbits &= gbits - 1)
                bslz4_lowplanes_group(pl, 8 * (size_t) (64 * gw + bslz4_ctz64(gbits)), C, I8, vcut,
                                      mask, i0, ubuf, kv, gm);
    } else {
        for (size_t i = 0; i < size; i += 8)
            bslz4_lowplanes_group(pl, i, C, I8, vcut, mask, i0, ubuf, kv, gm);
    }
    for (int w = 0; w < 2; w++) {
        uint64_t bits = gm[w];
        while (bits) {
            const size_t g = (size_t) (64 * w + bslz4_ctz64(bits));
            bits &= bits - 1;
            npx = bslz4_lowplanes_emit(_mm512_load_si512((const void *) (ubuf + 64 * g)), kv[g],
                                       i0 + 64 * g, npx, out_vals, out_adr);
        }
    }
    return npx;
}
#else
static int bslz4_lowplanes_collect_u16(uint8_t *raw, size_t ne, size_t nz_end, const uint8_t *zeros,
                                       const uint8_t *mask, size_t i0, unsigned cut,
                                       uint16_t *out_vals, uint32_t *out_adr, int skip) {
    (void) raw; (void) ne; (void) nz_end; (void) zeros; (void) mask; (void) i0; (void) cut;
    (void) out_vals; (void) out_adr; (void) skip;
    return 0;                                     /* never called: not capable */
}
#endif

static int bslz4_lowplanes_collect_capable(void) {
#if BSLZ4_HAVE_VBMI_GFNI
    static int cached = -1;                       /* as bslz4_lowplanes_u16_capable */
    int c = __atomic_load_n(&cached, __ATOMIC_RELAXED);
    if (c < 0) {
        c = bslz4_lowplanes_u16_capable() && c2py_amd64_avx512vl &&
            __builtin_cpu_supports("avx512vbmi2") && __builtin_cpu_supports("popcnt");
        __atomic_store_n(&cached, c, __ATOMIC_RELAXED);
    }
    return c;
#else
    return 0;
#endif
}

/* 1 if the u16 block's high byte-planes (raw[ne..2ne)) are all zero; zero
 * fills raw[nz_end..2ne) when it has to look. */
static int bslz4_high_planes_zero(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end) {
    if (nz_end <= ne) return 1;
    if (nz_end < 2 * ne) memset(raw + nz_end, 0, 2 * ne - nz_end);
    return bslz4_all_zero_avx512(raw + ne, ne);
}
#endif

#if BSLZ4_X86_SIMD
/* The byte skip without VBMI + GFNI (AVX2, or AVX-512BW without VBMI): the 8
 * low bit-planes are untransposed as one-byte elements -- kcb picks its
 * AVX-512BW / AVX2 / SSE2 kernel -- into scratch, then widened to u16.  Half
 * the bit transpose of the full path and no byte transpose (EPYC 7543: the
 * bit transpose was 20-48 % of sparsify). */
__attribute__((target("avx2")))
static void bslz4_widen_u8_u16_avx2(uint16_t *BSLZ4_RESTRICT out, const uint8_t *BSLZ4_RESTRICT in,
                                    size_t n) {
    for (size_t i = 0; i < n; i += 32) {
        const __m256i v = _mm256_loadu_si256((const __m256i *) (const void *) (in + i));
        _mm256_storeu_si256((__m256i *) (void *) (out + i), _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v)));
        _mm256_storeu_si256((__m256i *) (void *) (out + i + 16),
                            _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1)));
    }
}


#endif

/* 1 if p[0..n) are all zero; n a multiple of 8.  Plain C, 512-byte steps. */
static inline int bslz4_all_zero_c(const uint8_t *BSLZ4_RESTRICT p, size_t n) {
    for (size_t i = 0; i < n; i += 512) {
        uint64_t acc = 0;
        for (size_t k = i; k < i + 512 && k < n; k += 8) {
            uint64_t x;
            memcpy(&x, p + k, 8);
            acc |= x;
        }
        if (acc) return 0;
    }
    return 1;
}

/* The byte-skip collect for any architecture: the pixels > cut (and
 * unmasked) of a block whose values are all < 256, from its untransposed low
 * bytes v8[0..n), n a multiple of 8; an all-zero 8-byte word (most of a
 * sparse frame) is skipped with one test. */
static inline int bslz4_collect_u8_swar(const uint8_t *BSLZ4_RESTRICT v8, const uint8_t *BSLZ4_RESTRICT mask,
                                 size_t i0, size_t n, unsigned cut,
                                 uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    int npx = 0;
    for (size_t j = 0; j < n; j += 8) {
        uint64_t w;
        memcpy(&w, v8 + j, 8);
        if (!w) continue;
        for (size_t b = j; b < j + 8; b++) {
            const unsigned v = v8[b];
            if (v > cut && (!mask || mask[i0 + b])) {
                out_vals[npx] = (uint16_t) v;
                out_adr[npx] = (uint32_t) (i0 + b);
                npx++;
            }
        }
    }
    return npx;
}

#if defined(__POWER8_VECTOR__) && defined(__ALTIVEC__)
#include <altivec.h>
#undef vector
#undef pixel
#undef bool
/* The byte skip and collect fused on POWER8+ (VSX): per 64-pixel group the
 * 8 planes' 8-byte chunks are regrouped by two vec_perm rounds so that each
 * doubleword holds one byte of every plane, then vgbbd (vec_gb: an 8x8 bit
 * transpose per doubleword) gives the 64 pixel bytes; planes at or past
 * nz_end are read from `zeros`; a group whose plane bytes are all zero is
 * skipped; the pixels > cut (and unmasked) are collected from the bytes with
 * all-zero 8-byte words skipped.  POWER9: kcb's scalar transpose was ~60 %
 * of WAu sparsify after the byte skip. */
static int bslz4_lowplanes_collect_vsx(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end,
                                       const uint8_t *BSLZ4_RESTRICT zeros,
                                       const uint8_t *BSLZ4_RESTRICT mask, size_t i0, unsigned cut,
                                       uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr,
                                       int branchfree) {
    typedef __vector unsigned char vu8;
    typedef __vector unsigned long long vu64;
    const size_t size = ne / 8;
    if (nz_end > ne) nz_end = ne;
    const size_t np = (nz_end + size - 1) / size;
    if (nz_end < np * size) memset(raw + nz_end, 0, np * size - nz_end);
    const uint8_t *pl[8];
    for (size_t p = 0; p < 8; p++) pl[p] = p < np ? raw + p * size : zeros;
    /* stage 1: bytes b = 0..3 (lo) or 4..7 (hi) of 4 planes, 4 per b */
    const vu8 P1l = {0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27};
    const vu8 P1h = {4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31};
    /* stage 2: two b's, planes 0..3 then 4..7 */
    const vu8 P2a = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    const vu8 P2b = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    int npx = 0;
    uint8_t ub[64] __attribute__((aligned(16)));
    for (size_t i = 0; i < size; i += 8) {
        uint64_t a[8];
        for (int p = 0; p < 8; p++) memcpy(&a[p], pl[p] + i, 8);
        if (!(a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7])) continue;   /* 64 zero pixels */
        const vu8 L0 = (vu8) (vu64) {a[0], a[1]}, L1 = (vu8) (vu64) {a[2], a[3]};
        const vu8 L2 = (vu8) (vu64) {a[4], a[5]}, L3 = (vu8) (vu64) {a[6], a[7]};
        const vu8 Xl = vec_perm(L0, L1, P1l), Xh = vec_perm(L0, L1, P1h);
        const vu8 Yl = vec_perm(L2, L3, P1l), Yh = vec_perm(L2, L3, P1h);
        vec_xst(vec_gb(vec_perm(Xl, Yl, P2a)), 0, ub);        /* pixels 8i + 0..15 */
        vec_xst(vec_gb(vec_perm(Xl, Yl, P2b)), 16, ub);
        vec_xst(vec_gb(vec_perm(Xh, Yh, P2a)), 32, ub);
        vec_xst(vec_gb(vec_perm(Xh, Yh, P2b)), 48, ub);
        const size_t base = i0 + 8 * i;
        for (size_t w = 0; w < 64; w += 8) {
            uint64_t x;
            memcpy(&x, ub + w, 8);
            if (!x) continue;
            if (branchfree) {
                /* store every pixel, advance by its selection (the stores past
                 * the count stay below base + b <= the block's end): no
                 * mispredicts where the selection is scattered (9 % frames
                 * -18 %), but a serial chain where nearly every pixel passes
                 * (dense +89 %), so only for well-compressed blocks */
                for (size_t b = w; b < w + 8; b++) {
                    const unsigned v = ub[b];
                    out_vals[npx] = (uint16_t) v;
                    out_adr[npx] = (uint32_t) (base + b);
                    npx += (v > cut) & (!mask || mask[base + b] != 0);
                }
            } else {
                for (size_t b = w; b < w + 8; b++) {
                    const unsigned v = ub[b];
                    if (v > cut && (!mask || mask[base + b])) {
                        out_vals[npx] = (uint16_t) v;
                        out_adr[npx] = (uint32_t) (base + b);
                        npx++;
                    }
                }
            }
        }
    }
    return npx;
}
#endif

/* As bslz4_high_planes_zero, without AVX-512 (any architecture); also zero
 * fills raw[nz_end..ne) for kcb, which reads the whole low half. */
static inline int bslz4_high_planes_zero_c(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end) {
    if (nz_end <= ne) {
        memset(raw + nz_end, 0, ne - nz_end);
        return 1;
    }
    if (nz_end < 2 * ne) memset(raw + nz_end, 0, 2 * ne - nz_end);
    return bslz4_all_zero_c(raw + ne, ne);
}

/* As bslz4_high_planes_zero_c, leaving raw[nz_end..ne) alone (the fused
 * collects handle the low half themselves). */
static inline int bslz4_high_planes_zero_c_nofill(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end) {
    if (nz_end <= ne) return 1;
    if (nz_end < 2 * ne) memset(raw + nz_end, 0, 2 * ne - nz_end);
    return bslz4_all_zero_c(raw + ne, ne);
}


/* ---- what the driver calls ---------------------------------------------- */

/* 1 if this CPU/build can run the value (registry.c, through the driver) */
static inline int bslz4_lowplanes_capable(int value) {
    switch (value) {
    case BSLZ4_UNTRANSPOSE_LOWPLANES_VBMI:
#if BSLZ4_X86_SIMD
        return bslz4_lowplanes_u16_capable() && bslz4_lowplanes_collect_capable();
#else
        return 0;
#endif
    case BSLZ4_UNTRANSPOSE_LOWPLANES_AVX2:
#if BSLZ4_X86_SIMD
        return c2py_amd64_avx2 && __builtin_cpu_supports("popcnt");
#else
        return 0;
#endif
    case BSLZ4_UNTRANSPOSE_LOWPLANES_VSX:
#if defined(__POWER8_VECTOR__) && defined(__ALTIVEC__)
        return 1;
#else
        return 0;
#endif
    case BSLZ4_UNTRANSPOSE_LOWPLANES_C:
        return 1;
    default:
        return 0;
    }
}

/* planes -> list.  raw: a decoded u16 block of ne pixels (bytes past nz_end
 * zero but maybe unwritten), n its compressed size of blocksize; scratch: ne
 * bytes.  Writes the pixels > cut (unmasked) and returns their count, or -1
 * if the block does not qualify (raw is then still valid for kcb up to
 * nz_end). */
static inline int bslz4_lowplanes_list(int value, uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end,
                                       size_t n, size_t blocksize, uint8_t *BSLZ4_RESTRICT scratch,
                                       const uint8_t *BSLZ4_RESTRICT mask, size_t i0, unsigned cut,
                                       uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
    (void) n; (void) blocksize; (void) scratch;
    switch (value) {
#if BSLZ4_X86_SIMD
    case BSLZ4_UNTRANSPOSE_LOWPLANES_VBMI:
        if (ne % 64 || ne / 8 > sizeof(bslz4_zero_plane) || !bslz4_high_planes_zero(raw, ne, nz_end))
            return -1;
        if (cut >= 255) return 0;
        return bslz4_lowplanes_collect_u16(raw, ne, nz_end, bslz4_zero_plane, mask, i0, cut,
                                           out_vals, out_adr, n * BSLZ4_LOWPLANES_SKIP_RATIO < blocksize);
    case BSLZ4_UNTRANSPOSE_LOWPLANES_AVX2:
        if (ne % 256 || ne > 8192 || !bslz4_high_planes_zero_c_nofill(raw, ne, nz_end)) return -1;
        if (cut >= 255) return 0;
        return bslz4_lowplanes_collect_avx2(raw, ne, nz_end, mask, i0, cut, out_vals, out_adr);
#endif
#if defined(__POWER8_VECTOR__) && defined(__ALTIVEC__)
    case BSLZ4_UNTRANSPOSE_LOWPLANES_VSX:
        if (ne % 8 || ne / 8 > sizeof(bslz4_zero_plane) || !bslz4_high_planes_zero_c_nofill(raw, ne, nz_end))
            return -1;
        if (cut >= 255) return 0;
        return bslz4_lowplanes_collect_vsx(raw, ne, nz_end, bslz4_zero_plane, mask, i0, cut, out_vals,
                                           out_adr, n * BSLZ4_LOWPLANES_BRANCHFREE_RATIO < blocksize);
#endif
    case BSLZ4_UNTRANSPOSE_LOWPLANES_C:
        if (ne % 8 || !bslz4_high_planes_zero_c(raw, ne, nz_end)) return -1;
        if (cut >= 255) return 0;
        if (bitshuf_decode_block((char *) scratch, (const char *) raw, NULL, ne, 1) < 0) return -1;
        return bslz4_collect_u8_swar(scratch, mask, i0, ne, cut, out_vals, out_adr);
    default:
        return -1;
    }
}

/* planes -> block, for the dense route.  Returns 1 if the u16 block was
 * written from its low planes, 0 if not; either way raw is then zero filled
 * to the block's end (ready for kcb). */
static inline int bslz4_lowplanes_block(int value, uint16_t *BSLZ4_RESTRICT block, uint8_t *BSLZ4_RESTRICT raw,
                                        uint8_t *BSLZ4_RESTRICT scratch, size_t ne, size_t nz_end) {
    (void) block; (void) scratch;
    if (nz_end <= ne) {                         /* the decoder says: planes 8..15 are empty */
        memset(raw + nz_end, 0, ne - nz_end);
        if (ne % 64 == 0) {
            switch (value) {
#if BSLZ4_X86_SIMD
            case BSLZ4_UNTRANSPOSE_LOWPLANES_VBMI:
                bslz4_untrans_lowplanes_u16(block, raw, ne);
                return 1;
            case BSLZ4_UNTRANSPOSE_LOWPLANES_AVX2:
                if (bitshuf_decode_block((char *) scratch, (const char *) raw, NULL, ne, 1) < 0) break;
                bslz4_widen_u8_u16_avx2(block, scratch, ne);
                return 1;
#endif
            default:                            /* VSX, C: the dense route takes kcb */
                break;
            }
        }
        memset(raw + ne, 0, ne);
        return 0;
    }
    if (nz_end < 2 * ne) memset(raw + nz_end, 0, 2 * ne - nz_end);
    if (ne % 64) return 0;
    switch (value) {
#if BSLZ4_X86_SIMD
    case BSLZ4_UNTRANSPOSE_LOWPLANES_VBMI:
        if (!bslz4_all_zero_avx512(raw + ne, ne)) return 0;
        bslz4_untrans_lowplanes_u16(block, raw, ne);
        return 1;
    case BSLZ4_UNTRANSPOSE_LOWPLANES_AVX2:
        if (!bslz4_all_zero_c(raw + ne, ne)) return 0;
        if (bitshuf_decode_block((char *) scratch, (const char *) raw, NULL, ne, 1) < 0) return 0;
        bslz4_widen_u8_u16_avx2(block, scratch, ne);
        return 1;
#endif
    default:
        return 0;
    }
}

#endif /* BSLZ4_STEP_LOWPLANES_H */
