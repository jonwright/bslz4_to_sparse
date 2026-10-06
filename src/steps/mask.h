#ifndef BSLZ4_STEP_MASK_H
#define BSLZ4_STEP_MASK_H
/*
 * Step "mask": how the caller's fixed pixel mask (bytes, != 0 = use) is
 * applied.
 *
 *   BSLZ4_MASK_PIXEL   tested per pixel by the collect (the default: free on
 *                      data whose masked pixels hold 0, as our Eigers write).
 *   BSLZ4_MASK_PLANES  packed once per block per batch into the bit-plane
 *                      layout (bit e % 8 of byte e / 8) and ANDed into every
 *                      bit-plane straight after decode, so masked pixels are
 *                      0 before any untranspose and the block needs no mask
 *                      tests.  Blocks with nothing masked skip it.  Wins
 *                      (-1..-5 % at cut 0) only where masked pixels hold the
 *                      dtype maximum: then they no longer stop the u16 byte
 *                      skip; pure overhead (+2..+17 %) on zero-filled data.
 *   BSLZ4_MASK_NONE    every pixel valid: no tests at all.
 *
 * The matrix of a dot object is mask-folded (masked pixels have no entries),
 * so the mask only selects the sparse output and the sparse-route list.
 */

#include "../pipeline/common.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if BSLZ4_X86_SIMD
#include <immintrin.h>
#include "c2py_amd64.h"
#endif

#define BSLZ4_MASKPLANE_MAX_ELEMS 8192   /* largest block packed (8 kB of u8) */

/* AND a pixel bitmap with the mask (bytes mask[e] != 0), packed 64 pixels
 * per word. */
#if BSLZ4_X86_SIMD
__attribute__((target("avx512f,avx512bw")))
static inline void bslz4_and_mask_avx512(uint64_t *BSLZ4_RESTRICT bitmap, const uint8_t *BSLZ4_RESTRICT m,
                                  size_t nw) {
    for (size_t w = 0; w < nw; w++) {
        const __m512i v = _mm512_loadu_si512((const void *) (m + 64 * w));
        bitmap[w] &= (uint64_t) _mm512_test_epi8_mask(v, v);
    }
}
#endif

/* Pack mask[0..ne) (bytes, != 0 = use) into ne / 64 words in the bit-plane
 * layout; returns 1 if every pixel is unmasked (nothing to AND). */
static inline int bslz4_pack_mask(uint64_t *BSLZ4_RESTRICT bits, const uint8_t *BSLZ4_RESTRICT m, size_t ne) {
    const size_t nw = ne / 64;
    uint64_t all = ~(uint64_t) 0;
#if BSLZ4_X86_SIMD
    if (c2py_amd64_avx512f && c2py_amd64_avx512bw) {
        for (size_t w = 0; w < nw; w++) bits[w] = ~(uint64_t) 0;
        bslz4_and_mask_avx512(bits, m, nw);
    } else
#endif
    {
        for (size_t w = 0; w < nw; w++) {
            uint64_t keep = 0;
            for (int k = 0; k < 64; k++) keep |= (uint64_t) (m[64 * w + k] != 0) << k;
            bits[w] = keep;
        }
    }
    for (size_t w = 0; w < nw; w++) all &= bits[w];
    return all == ~(uint64_t) 0;
}

/* AND the packed mask into raw[0..nz_end) plane by plane (planes of nb = ne/8
 * bytes); bytes past nz_end are zero already (or are zero-filled later). */
#if BSLZ4_X86_SIMD
__attribute__((target("avx512f")))
static inline size_t bslz4_mask_planes_avx512(uint8_t *BSLZ4_RESTRICT raw, const uint64_t *BSLZ4_RESTRICT bits,
                                       size_t nb, size_t nz_end) {
    /* whole 64-byte chunks below nz_end, plane by plane (no division in the
     * loop); returns where it stopped */
    size_t p0 = 0, j = 0;
    for (; p0 < nz_end; p0 += nb) {
        for (j = 0; j < nb && p0 + j + 64 <= nz_end; j += 64) {
            __m512i v = _mm512_loadu_si512((const void *) (raw + p0 + j));
            v = _mm512_and_si512(v, _mm512_loadu_si512((const void *) ((const uint8_t *) bits + j)));
            _mm512_storeu_si512((void *) (raw + p0 + j), v);
        }
        if (j < nb) break;
    }
    return p0 + j < nz_end ? p0 + j : nz_end;
}
#endif

static inline void bslz4_mask_planes(uint8_t *BSLZ4_RESTRICT raw, const uint64_t *BSLZ4_RESTRICT bits,
                              size_t nb, size_t nz_end) {
    const uint8_t *mb = (const uint8_t *) bits;
    size_t i = 0;
#if BSLZ4_X86_SIMD
    if (c2py_amd64_avx512f && nb % 64 == 0) i = bslz4_mask_planes_avx512(raw, bits, nb, nz_end);
#endif
    for (size_t p0 = i - i % nb; p0 < nz_end; p0 += nb) {      /* the rest, plane by plane */
        size_t j = p0 < i ? i - p0 : 0;
        const size_t jend = nz_end - p0 < nb ? nz_end - p0 : nb;
        for (; j < jend; j++) raw[p0 + j] &= mb[j];
    }
}

#endif /* BSLZ4_STEP_MASK_H */
