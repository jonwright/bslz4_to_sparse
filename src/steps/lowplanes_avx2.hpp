#pragma once
/*
 * Step "untranspose", value lowplanes-avx2 (see lowplanes.h): the fused
 * low-planes transpose + collect on AVX2.  C++ only for the avx2cs lane
 * table it shares with the collect step (collect.hpp).
 */

#include "collect.hpp"

/* The byte-skip transpose fused with the collect on AVX2 (no VBMI/GFNI):
 * pass A untransposes the 8 low bit-planes 64 pixels at a time -- kcb's
 * bitshuf_untrans_bit_avx2 (Copyright (c) 2023 Kal Conley, MIT /
 * Apache-2.0), reading planes at or past nz_end from `zeros` instead of
 * zero-filling them.  First the planes are ORed 32 bytes at a time into a
 * bitmap of the 64-pixel groups holding any data (no branch); pass A
 * then transposes and compares (> cut, unmasked) only those groups, recording
 * bytes and selection with no branch; pass B packs only the groups with
 * selected pixels, with the avx2cs lane table (a group with one pixel: the
 * bit loop).  On sparse frames most groups are never touched; on 9 %-
 * occupied ones there is still no branch per group.  Needs ne <= 8192 (128 groups)
 * and ne % 256 == 0.  Stores reach at most 8 entries past the count, within
 * i0 + 64 g + 64 <= i0 + ne. */
#if BSLZ4_HAVE_AVX2_COLLECT
static const uint8_t bslz4_zero_plane8[1024] = {0};

__attribute__((target("avx2,popcnt")))
static int bslz4_lowplanes_collect_avx2_impl(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end,
                                             const uint8_t *BSLZ4_RESTRICT mask, size_t i0, unsigned cut,
                                             uint16_t *BSLZ4_RESTRICT out_vals,
                                             uint32_t *BSLZ4_RESTRICT out_adr) {
    const size_t size = ne / 8;                       /* bytes per plane; a multiple of 32 */
    if (nz_end > ne) nz_end = ne;
    const size_t np = (nz_end + size - 1) / size;     /* planes holding data */
    if (nz_end < np * size) memset(raw + nz_end, 0, np * size - nz_end);
    const uint8_t *pl[8];
    for (size_t p = 0; p < 8; p++) pl[p] = p < np ? raw + p * size : bslz4_zero_plane8;
    /* groups of 64 pixels with any bit set in any plane: OR the planes, 4
     * groups (32 bytes) per step, no branch */
    uint64_t gm[2] = {0, 0};
    for (size_t i = 0; i < size; i += 32) {
        __m256i acc = _mm256_loadu_si256((const __m256i *) (const void *) (pl[0] + i));
        for (size_t p = 1; p < 8; p++)
            acc = _mm256_or_si256(acc, _mm256_loadu_si256((const __m256i *) (const void *) (pl[p] + i)));
        const uint32_t zero4 = (uint32_t) _mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpeq_epi64(acc, _mm256_setzero_si256())));
        const size_t g = i / 8;
        gm[g >> 6] |= (uint64_t) (~zero4 & 0xFu) << (g & 63);
    }
    const __m256i PERM = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    const __m256i MASK0 = _mm256_set1_epi64x(0x00aa00aa00aa00aa);
    const __m256i MASK1 = _mm256_set1_epi64x(0x0000cccc0000cccc);
    const __m256i MASK2 = _mm256_set1_epi64x(0x00000000f0f0f0f0);
    const __m256i bias = _mm256_set1_epi8((char) 0x80);
    const __m256i vcut = _mm256_xor_si256(_mm256_set1_epi8((char) cut), bias);
    const __m256i iota8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    alignas(32) uint8_t ubuf[8192];
    uint64_t kv[128];
    uint64_t km[2] = {0, 0};
    for (int w = 0; w < 2; w++) {
        uint64_t gbits = gm[w];
        while (gbits) {                               /* A: only the groups holding data */
            const size_t g = (size_t) (64 * w + __builtin_ctzll(gbits));
            gbits &= gbits - 1;
            const size_t i = 8 * g;
            const __m128i a0 = _mm_loadl_epi64((const __m128i *) (const void *) (pl[0] + i));
            const __m128i a1 = _mm_loadl_epi64((const __m128i *) (const void *) (pl[1] + i));
            const __m128i a2 = _mm_loadl_epi64((const __m128i *) (const void *) (pl[2] + i));
            const __m128i a3 = _mm_loadl_epi64((const __m128i *) (const void *) (pl[3] + i));
            const __m128i a4 = _mm_loadl_epi64((const __m128i *) (const void *) (pl[4] + i));
            const __m128i a5 = _mm_loadl_epi64((const __m128i *) (const void *) (pl[5] + i));
            const __m128i a6 = _mm_loadl_epi64((const __m128i *) (const void *) (pl[6] + i));
            const __m128i a7 = _mm_loadl_epi64((const __m128i *) (const void *) (pl[7] + i));
            __m256i u0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(a0, a1)), _mm_unpacklo_epi8(a4, a5), 1);
            __m256i u1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(a2, a3)), _mm_unpacklo_epi8(a6, a7), 1);
            __m256i v0 = _mm256_unpacklo_epi16(u0, u1);
            __m256i v1 = _mm256_unpackhi_epi16(u0, u1);
            u0 = _mm256_permutevar8x32_epi32(v0, PERM);
            u1 = _mm256_permutevar8x32_epi32(v1, PERM);
            v0 = _mm256_and_si256(_mm256_xor_si256(u0, _mm256_srli_epi64(u0, 7)), MASK0);
            v1 = _mm256_and_si256(_mm256_xor_si256(u1, _mm256_srli_epi64(u1, 7)), MASK0);
            u0 = _mm256_xor_si256(_mm256_xor_si256(u0, _mm256_slli_epi64(v0, 7)), v0);
            u1 = _mm256_xor_si256(_mm256_xor_si256(u1, _mm256_slli_epi64(v1, 7)), v1);
            v0 = _mm256_and_si256(_mm256_xor_si256(u0, _mm256_srli_epi64(u0, 14)), MASK1);
            v1 = _mm256_and_si256(_mm256_xor_si256(u1, _mm256_srli_epi64(u1, 14)), MASK1);
            u0 = _mm256_xor_si256(_mm256_xor_si256(u0, _mm256_slli_epi64(v0, 14)), v0);
            u1 = _mm256_xor_si256(_mm256_xor_si256(u1, _mm256_slli_epi64(v1, 14)), v1);
            v0 = _mm256_and_si256(_mm256_xor_si256(u0, _mm256_srli_epi64(u0, 28)), MASK2);
            v1 = _mm256_and_si256(_mm256_xor_si256(u1, _mm256_srli_epi64(u1, 28)), MASK2);
            u0 = _mm256_xor_si256(_mm256_xor_si256(u0, _mm256_slli_epi64(v0, 28)), v0);
            u1 = _mm256_xor_si256(_mm256_xor_si256(u1, _mm256_slli_epi64(v1, 28)), v1);
            /* u0, u1: the bytes of pixels 64g .. 64g+31, 64g+32 .. 64g+63 */
            uint64_t k = (uint64_t) (uint32_t) _mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_xor_si256(u0, bias), vcut)) |
                         (uint64_t) (uint32_t) _mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_xor_si256(u1, bias), vcut)) << 32;
            if (mask) {
                const uint8_t *mm = mask + i0 + 64 * g;
                const __m256i z = _mm256_setzero_si256();
                k &= ~((uint64_t) (uint32_t) _mm256_movemask_epi8(_mm256_cmpeq_epi8(
                            _mm256_loadu_si256((const __m256i *) (const void *) mm), z)) |
                       (uint64_t) (uint32_t) _mm256_movemask_epi8(_mm256_cmpeq_epi8(
                            _mm256_loadu_si256((const __m256i *) (const void *) (mm + 32)), z)) << 32);
            }
            /* record; no branch on k (mispredicts on 9 %-occupied frames) */
            _mm256_store_si256((__m256i *) (void *) (ubuf + 64 * g), u0);
            _mm256_store_si256((__m256i *) (void *) (ubuf + 64 * g + 32), u1);
            kv[g] = k;
            km[g >> 6] |= (uint64_t) (k != 0) << (g & 63);
        }
    }
    const uint64_t *lut = bslz4::bslz4_avx2cs_table();
    int npx = 0;
    for (int w = 0; w < 2; w++) {
        uint64_t kbits = km[w];
        while (kbits) {                               /* B: only the groups with pixels */
            const size_t g = (size_t) (64 * w + __builtin_ctzll(kbits));
            kbits &= kbits - 1;
            const uint64_t k = kv[g];
            const uint8_t *ub = ubuf + 64 * g;
            const size_t base = i0 + 64 * g;
            if (__builtin_popcountll(k) <= BSLZ4_AVX2CS_SCALAR_MAX) {
                const int b = __builtin_ctzll(k);
                out_vals[npx] = ub[b];
                out_adr[npx] = (uint32_t) (base + b);
                npx++;
                continue;
            }
            for (int q = 0; q < 8; q++) {
                const uint32_t m = (uint32_t) (k >> (8 * q)) & 0xFFu;
                const __m128i sel = _mm_loadl_epi64((const __m128i *) (const void *) &lut[m]);
                _mm256_storeu_si256((__m256i *) (void *) &out_adr[npx],
                                    _mm256_permutevar8x32_epi32(
                                        _mm256_add_epi32(_mm256_set1_epi32((int) (base + 8 * q)), iota8),
                                        _mm256_cvtepu8_epi32(sel)));
                const __m128i b8 = _mm_loadl_epi64((const __m128i *) (const void *) (ub + 8 * q));
                _mm_storeu_si128((__m128i *) (void *) &out_vals[npx], _mm_cvtepu8_epi16(_mm_shuffle_epi8(b8, sel)));
                npx += __builtin_popcount(m);
            }
        }
    }
    return npx;
}
#endif

