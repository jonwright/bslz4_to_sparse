/*
 * bslz4_driver.c -- the single decode driver for bslz4_to_sparse.
 *
 * This is the dtype-agnostic C port of the block/tail loop.  The only
 * dtype-generic work (the per-block sparse / sparse_dot call) is a direct
 * call into the C++ dtype switch (kernels_generic.cpp) with the resolved
 * collect/dot ids in the bslz4_stage; decompress and untranspose are plain
 * non-generic function pointers.  The per-block instrumentation counters
 * (decompress/untranspose/collect/dot) are bumped here.
 *
 * The block/tail loop is written once (bslz4_driver_run).  The two public
 * entry points only differ in which inner dispatch they request -- the plain
 * sparse versus the CSC-sparse (powder) one -- and are thin wrappers.
 */

#include "bslz4_common.h"
#include "bslz4_registry.h"
#include "bslz4_codec.h"
#include "bslz4_lz4zero.h"
#include "bslz4_untranspose.h"      /* bitshuf_decode_block (kcb) */
#include "bslz4_collect_caps.h"     /* bslz4_collect_u8_avx2cs */

#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define BSLZ4_HAVE_AVX2_OR 1
#include <immintrin.h>
#include "c2py_amd64.h"
#else
#define BSLZ4_HAVE_AVX2_OR 0
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline int bslz4_ctz64(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (int) i; }
#else
static inline int bslz4_ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif


static int bslz4_driver_check_frames(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                      const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                      int nframes,
                                      uint64_t *BSLZ4_RESTRICT out_total,
                                      size_t *BSLZ4_RESTRICT out_blocksize) {
    if (nframes <= 0) return BSLZ4_ERR_BAD_NFRAMES;
    for (int f = 0; f < nframes; f++) {
        if (BSLZ4_UNLIKELY(compressed_lengths[f] < 12)) return BSLZ4_ERR_CORRUPT_CHUNK;
    }

    const char *BSLZ4_RESTRICT compressed0 = (const char *) (intptr_t) compressed_ptrs[0];
    const uint64_t total_output_length = bslz4_read_be64((const uint8_t *) compressed0);

    size_t blocksize = (size_t) bslz4_read_be32((const uint8_t *) compressed0 + 8);
    if (blocksize == 0) blocksize = BSLZ4_DEFAULT_BLOCK_BYTES;

    for (int f = 1; f < nframes; f++) {
        const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
        if (BSLZ4_UNLIKELY(bslz4_read_be64((const uint8_t *) cf) != total_output_length))
            return BSLZ4_ERR_FRAME_MISMATCH;
        size_t bsf = (size_t) bslz4_read_be32((const uint8_t *) cf + 8);
        if (bsf == 0) bsf = BSLZ4_DEFAULT_BLOCK_BYTES;
        if (BSLZ4_UNLIKELY(bsf != blocksize)) return BSLZ4_ERR_FRAME_MISMATCH;
    }

    *out_total = total_output_length;
    *out_blocksize = blocksize;
    return 0;
}

/* ---- byte skip --------------------------------------------------------
 *
 * A bitshuffled block of ne elements of NB bytes stores the bit-planes of
 * byte 0 first (ne bytes), then those of byte 1.  When a u16 block's high
 * byte-planes are all zero -- every value < 256, as with low counts and a
 * zero-filled mask -- its first ne bytes alone are a bitshuffled u8 block,
 * and untransposing just those straight into u16 gives exactly the full
 * result for about half the work (bslz4_untrans_lowplanes_u16 below; AVX-512
 * VBMI + GFNI).  A generic version (any element size: untranspose the low
 * bytes, then widen) measured slower than the plain untranspose and was
 * dropped (2026-10-03). */

/* ---- plane extraction -------------------------------------------------
 *
 * In a bitshuffled block of ne elements, bit p of element e is bit (e % 8)
 * of byte e / 8 of plane p, and plane p starts at p * ne / 8.  So the OR of
 * all planes is a bitmap of the non-zero elements.  For a very sparse block
 * (a few set bits) each non-zero value is read straight from its plane bits:
 * no untranspose and no scan of ne values.  Unsigned u8/u16/u32 only. */

#ifndef BSLZ4_EXTRACT_MAX
#define BSLZ4_EXTRACT_MAX 16   /* above this many non-zero pixels: untranspose (48 lost on real frames) */
#endif
#ifndef BSLZ4_LZ4ZERO_RATIO
/* The zero-aware lz4 decoder takes blocks compressed more than RATIO, and
 * those compressed less than DENSE; stock LZ4_decompress_safe the band in
 * between.  Per block (2026-10-05): it wins on sparse blocks (skipped zero
 * runs) and on dense ones (literal heavy; it never writes the empty high
 * byte-planes), but blocks of ~8-24x are many short sequences (~90-130 per
 * 8 kB) where stock's loop is faster on Zen 5 (8-16x: 2408 vs 3142 cycles)
 * and level on Cascade Lake (6170 vs 6463). */
#define BSLZ4_LZ4ZERO_RATIO 24
#endif
#ifndef BSLZ4_LZ4ZERO_DENSE
#define BSLZ4_LZ4ZERO_DENSE 8
#endif
#ifndef BSLZ4_EXTRACT_RATIO
/* only blocks compressed more than this are tried: at 8x, medium-density
 * frames (~9 % non-zero, blocks ~13x) built the bitmap and fell back, ~4 %
 * slower; at 24x they are untouched and very sparse frames gain as much
 * (2026-10-03, EPYC 9454) */
#define BSLZ4_EXTRACT_RATIO 24
#endif

#ifndef BSLZ4_FUSED_GENERIC
/* 1: on non-x86 builds, u16 blocks with empty high planes take kcb's low
 * planes as bytes and the portable u8 collect (POWER9: kcb's scalar 16-plane
 * transpose + byte transpose were ~80 % of sparsify) */
#define BSLZ4_FUSED_GENERIC 1
#endif

#ifndef BSLZ4_FUSED_VSX
/* 1: on POWER8+ the byte-skip collect transposes with VSX vgbbd, fused */
#define BSLZ4_FUSED_VSX 1
#endif

#ifndef BSLZ4_FUSED_U8
/* With the generic byte skip, plain sparsify and the sparse dot route
 * collect without a u16 block: 1 = kcb's low planes as bytes into scratch,
 * then bslz4_collect_u8_avx2cs; 2 = bslz4_lowplanes_collect_avx2 (transpose
 * and collect fused, two passes, empty planes not zero filled); 0 = off */
#define BSLZ4_FUSED_U8 2
#endif

#ifndef BSLZ4_BYTESKIP_GENERIC
/* 1: the byte skip also on CPUs without AVX-512 VBMI + GFNI (low planes
 * through kcb as one-byte elements, then widened) */
#define BSLZ4_BYTESKIP_GENERIC 1
#endif

#ifndef BSLZ4_LOWPLANES_SKIP
/* 1: the AVX-512 fused collect transposes only the 64-pixel groups holding
 * data (a bitmap from ORing the planes), for blocks compressed more than
 * BSLZ4_LOWPLANES_SKIP_RATIO; 0: every group.  hpc8: 0.1 % frames -25 %,
 * WAu0012 -8 %; at 16x WAu0008 blocks (~44 % of groups busy) gained nothing */
#define BSLZ4_LOWPLANES_SKIP 1
#endif
#ifndef BSLZ4_LOWPLANES_SKIP_RATIO
#define BSLZ4_LOWPLANES_SKIP_RATIO 48
#endif

#ifndef BSLZ4_FUSED_TWOPASS
/* 1: the fused collect takes two passes (record all groups, then emit those
 * with pixels by bit scan); 0: one pass, skipping groups with no pixel */
#define BSLZ4_FUSED_TWOPASS 1
#endif

#ifndef BSLZ4_LOWPLANES_FUSED
/* 1: u16 blocks taking the byte skip are transposed and collected in one
 * pass (bslz4_lowplanes_collect_u16) for plain sparsify and the sparse dot
 * route; 0: transpose to the block, then collect */
#define BSLZ4_LOWPLANES_FUSED 1
#endif

/* Set bits in n words, with the hardware popcount where there is one (the
 * driver is not built with -mpopcnt, so __builtin_popcountll alone would be
 * a libgcc call). */
#if BSLZ4_HAVE_AVX2_OR
__attribute__((target("popcnt")))
static int bslz4_popcount_words_hw(const uint64_t *BSLZ4_RESTRICT w, size_t n) {
    int c = 0;
    for (size_t i = 0; i < n; i++) c += __builtin_popcountll(w[i]);
    return c;
}
#endif
static int bslz4_popcount_words(const uint64_t *BSLZ4_RESTRICT w, size_t n) {
#if BSLZ4_HAVE_AVX2_OR
    if (c2py_amd64_popcnt) return bslz4_popcount_words_hw(w, n);
#endif
    int c = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t x = w[i];
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        c += (int) ((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
    }
    return c;
}

static void bslz4_or_planes_scalar(const uint8_t *BSLZ4_RESTRICT raw, size_t nb, int np,
                                   uint64_t *BSLZ4_RESTRICT bitmap) {
    for (size_t w = 0; w < nb / 8; w++) {
        uint64_t acc = 0, v;
        for (int p = 0; p < np; p++) {
            memcpy(&v, raw + (size_t) p * nb + 8 * w, 8);
            acc |= v;
        }
        bitmap[w] = acc;
    }
}

#if BSLZ4_HAVE_AVX2_OR
__attribute__((target("avx512f")))
static void bslz4_or_planes_avx512(const uint8_t *BSLZ4_RESTRICT raw, size_t nb, int np,
                                   uint64_t *BSLZ4_RESTRICT bitmap) {
    for (size_t w = 0; w + 64 <= nb; w += 64) {
        __m512i acc = _mm512_loadu_si512((const void *) (raw + w));
        for (int p = 1; p < np; p++)
            acc = _mm512_or_si512(acc, _mm512_loadu_si512((const void *) (raw + (size_t) p * nb + w)));
        _mm512_storeu_si512((void *) (bitmap + w / 8), acc);
    }
}

__attribute__((target("avx2")))
static void bslz4_or_planes_avx2(const uint8_t *BSLZ4_RESTRICT raw, size_t nb, int np,
                                 uint64_t *BSLZ4_RESTRICT bitmap) {
    for (size_t w = 0; w + 32 <= nb; w += 32) {
        __m256i acc = _mm256_loadu_si256((const __m256i *) (const void *) (raw + w));
        for (int p = 1; p < np; p++)
            acc = _mm256_or_si256(acc, _mm256_loadu_si256((const __m256i *) (const void *) (raw + (size_t) p * nb + w)));
        _mm256_storeu_si256((__m256i *) (void *) (bitmap + w / 8), acc);
    }
}
#endif

/* AND the pixel bitmap with the mask (bytes mask[i0 + e] != 0), packed 64
 * pixels per word: masked pixels -- whatever value they hold, 0 or the
 * dtype maximum -- then neither count towards the limit nor get gathered. */
#if BSLZ4_HAVE_AVX2_OR
__attribute__((target("avx512f,avx512bw")))
static void bslz4_and_mask_avx512(uint64_t *BSLZ4_RESTRICT bitmap, const uint8_t *BSLZ4_RESTRICT m,
                                  size_t nw) {
    for (size_t w = 0; w < nw; w++) {
        const __m512i v = _mm512_loadu_si512((const void *) (m + 64 * w));
        bitmap[w] &= (uint64_t) _mm512_test_epi8_mask(v, v);
    }
}
#endif

static void bslz4_and_mask(uint64_t *BSLZ4_RESTRICT bitmap, const uint8_t *BSLZ4_RESTRICT m, size_t nw) {
#if BSLZ4_HAVE_AVX2_OR
    if (c2py_amd64_avx512f && c2py_amd64_avx512bw) {
        bslz4_and_mask_avx512(bitmap, m, nw);
        return;
    }
#endif
    for (size_t w = 0; w < nw; w++) {
        uint64_t keep = 0;
        for (int k = 0; k < 64; k++) keep |= (uint64_t) (m[64 * w + k] != 0) << k;
        bitmap[w] &= keep;
    }
}

/* ---- mask in the bitshuffled domain ------------------------------------ */

#define BSLZ4_MASKPLANE_MAX_ELEMS 8192   /* largest block packed (8 kB of u8) */

/* Pack mask[0..ne) (bytes, != 0 = use) into ne / 64 words in the bit-plane
 * layout; returns 1 if every pixel is unmasked (nothing to AND). */
static int bslz4_pack_mask(uint64_t *BSLZ4_RESTRICT bits, const uint8_t *BSLZ4_RESTRICT m, size_t ne) {
    const size_t nw = ne / 64;
    uint64_t all = ~(uint64_t) 0;
#if BSLZ4_HAVE_AVX2_OR
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
#if BSLZ4_HAVE_AVX2_OR
__attribute__((target("avx512f")))
static size_t bslz4_mask_planes_avx512(uint8_t *BSLZ4_RESTRICT raw, const uint64_t *BSLZ4_RESTRICT bits,
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

static void bslz4_mask_planes(uint8_t *BSLZ4_RESTRICT raw, const uint64_t *BSLZ4_RESTRICT bits,
                              size_t nb, size_t nz_end) {
    const uint8_t *mb = (const uint8_t *) bits;
    size_t i = 0;
#if BSLZ4_HAVE_AVX2_OR
    if (c2py_amd64_avx512f && nb % 64 == 0) i = bslz4_mask_planes_avx512(raw, bits, nb, nz_end);
#endif
    for (size_t p0 = i - i % nb; p0 < nz_end; p0 += nb) {      /* the rest, plane by plane */
        size_t j = p0 < i ? i - p0 : 0;
        const size_t jend = nz_end - p0 < nb ? nz_end - p0 : nb;
        for (; j < jend; j++) raw[p0 + j] &= mb[j];
    }
}

/* Returns the number of pixels > cut written to out_vals/out_adr, or -1 if
 * the block has more than BSLZ4_EXTRACT_MAX non-zero pixels (the caller then
 * untransposes it; what was written here is overwritten, as the caller's
 * count did not advance).  bitmap: ne / 64 words of scratch. */
static int bslz4_plane_extract(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t NB, size_t nz_end,
                               const uint8_t *BSLZ4_RESTRICT mask, size_t i0, uint32_t cut,
                               void *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr,
                               uint64_t *BSLZ4_RESTRICT bitmap) {
    const size_t nb = ne / 8, nw = ne / 64;
    const uint32_t top = NB == 1 ? 0xFFu : (NB == 2 ? 0xFFFFu : 0xFFFFFFFFu);
    if (cut >= top) return 0;                       /* nothing can be > cut */
    /* only the planes holding data up to nz_end (the rest are zero) */
    const int np = (int) ((nz_end + nb - 1) / nb);
    if (np == 0) return 0;
    if (nz_end < (size_t) np * nb) memset(raw + nz_end, 0, (size_t) np * nb - nz_end);
#if BSLZ4_HAVE_AVX2_OR
    if (c2py_amd64_avx512f && nb % 64 == 0) bslz4_or_planes_avx512(raw, nb, np, bitmap);
    else if (c2py_amd64_avx2 && nb % 32 == 0) bslz4_or_planes_avx2(raw, nb, np, bitmap);
    else
#endif
    bslz4_or_planes_scalar(raw, nb, np, bitmap);
    if (mask) bslz4_and_mask(bitmap, mask + i0, nw);
    if (bslz4_popcount_words(bitmap, nw) > BSLZ4_EXTRACT_MAX) return -1;
    int npx = 0;
    for (size_t w = 0; w < nw; w++) {
        uint64_t bits = bitmap[w];
        while (bits) {
            const size_t e = 64 * w + (size_t) bslz4_ctz64(bits);
            bits &= bits - 1;
            const uint8_t *q = raw + (e >> 3);
            const unsigned sh = (unsigned) (e & 7);
            uint32_t v = 0;
            for (int p = 0; p < np; p++) v |= (uint32_t) ((q[(size_t) p * nb] >> sh) & 1u) << p;
            if (v <= cut) continue;
            if (NB == 1) ((uint8_t *) out_vals)[npx] = (uint8_t) v;
            else if (NB == 2) ((uint16_t *) out_vals)[npx] = (uint16_t) v;
            else ((uint32_t *) out_vals)[npx] = v;
            out_adr[npx] = (uint32_t) (i0 + e);
            npx++;
        }
    }
    return npx;
}

#if BSLZ4_HAVE_AVX2_OR
/* u16 blocks whose high bytes are all zero: untranspose the 8 low bit-planes
 * straight into u16 (AVX-512 VBMI + GFNI).  The bit transpose is kcb's
 * (bitshuf_untrans_bit_avx512vbmi_gfni, Copyright (c) 2023 Kal Conley,
 * MIT / Apache-2.0): per 64 pixels, 8 bytes from each plane are permuted so
 * each qword holds one 8x8 bit matrix, which gf2p8affineqb transposes.  Here
 * the 64 result bytes are zero-extended and stored as 64 u16 at once, so the
 * low bytes are never stored and widened separately. */
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
    static int cached = -1;
    if (cached < 0)
        cached = c2py_amd64_avx512f && c2py_amd64_avx512bw && __builtin_cpu_supports("avx512vbmi") &&
                 __builtin_cpu_supports("gfni");
    return cached;
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

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,gfni,popcnt"), always_inline))
static inline int bslz4_lowplanes_collect_body(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end,
                                       const uint8_t *BSLZ4_RESTRICT zeros,
                                       const uint8_t *BSLZ4_RESTRICT mask, size_t i0, unsigned cut,
                                       uint16_t *BSLZ4_RESTRICT out_vals,
                                       uint32_t *BSLZ4_RESTRICT out_adr,
                                       const int twopass, const int skip) {
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
    /* Two passes: at cut 0 on photon noise the groups holding a pixel are
     * many and scattered, so a branch per group mispredicts.  Pass 1 records
     * every group's bytes and selection, pass 2 emits only the groups with
     * pixels, found by bit scan (one mispredict per block).  One pass,
     * skipping empty groups, is ~4 % faster at cut 2 but ~30 % slower at
     * cut 0; choosing per block from the previous block's hit rate kept
     * neither (WAu, 2026-10-04). */
    _Alignas(64) uint8_t ubuf[8192];
    __mmask64 kv[128];
    uint64_t gm[2] = {0, 0};
    /* the 64-pixel groups holding any data: the planes ORed 64 bytes (8
     * groups) at a time, one test per 8 groups; with skip, only those are
     * transposed (on sparse frames most are empty).  Without skip, a plain
     * counted loop (the bit-scan loop over every group cost 2-5 % there). */
    uint64_t gnz[2] = {0, 0};
    const int use_gnz = skip && size % 64 == 0;
    if (use_gnz) {
        for (size_t i = 0; i < size; i += 64) {
            __m512i acc = _mm512_loadu_si512((const void *) (pl[0] + i));
            for (size_t p = 1; p < 8; p++)
                acc = _mm512_or_si512(acc, _mm512_loadu_si512((const void *) (pl[p] + i)));
            const size_t g = i / 8;
            gnz[g >> 6] |= (uint64_t) _mm512_test_epi64_mask(acc, acc) << (g & 63);
        }
    }
#define BSLZ4_LOWPLANES_GROUP(i) do { \
        int64_t a1, a3, a5, a7; \
        memcpy(&a1, pl[1] + i, 8); \
        memcpy(&a3, pl[3] + i, 8); \
        memcpy(&a5, pl[5] + i, 8); \
        memcpy(&a7, pl[7] + i, 8); \
        const __m128i u0 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) (pl[0] + i)), a1, 1); \
        const __m128i u1 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) (pl[2] + i)), a3, 1); \
        const __m128i u2 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) (pl[4] + i)), a5, 1); \
        const __m128i u3 = _mm_insert_epi64(_mm_loadl_epi64((const __m128i *) (const void *) (pl[6] + i)), a7, 1); \
        const __m256i v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(u0), u1, 1); \
        const __m256i v1 = _mm256_inserti128_si256(_mm256_castsi128_si256(u2), u3, 1); \
        __m512i u = _mm512_permutex2var_epi8(_mm512_castsi256_si512(v0), C, _mm512_castsi256_si512(v1)); \
        u = _mm512_gf2p8affine_epi64_epi8(I8, u, 0x00);   /* byte j = pixel 8i + j */ \
        __mmask64 k = _mm512_cmpgt_epu8_mask(u, vcut); \
        if (mask) { \
            const __m512i m = _mm512_loadu_si512((const void *) (mask + i0 + 8 * i)); \
            k &= _mm512_test_epi8_mask(m, m); \
        } \
        if (twopass) {                  /* record only: no data-dependent branch */ \
            _mm512_store_si512((void *) (ubuf + 8 * i), u); \
            kv[i / 8] = k; \
            gm[i / 512] |= (uint64_t) (k != 0) << ((i / 8) & 63); \
            break; \
        } \
        if (!k) break; \
        npx = bslz4_lowplanes_emit(u, k, i0 + 8 * i, npx, out_vals, out_adr); \
    } while (0)
    if (use_gnz) {
        for (int gw = 0; gw < 2; gw++)
            for (uint64_t gbits = gnz[gw]; gbits; gbits &= gbits - 1) {
                const size_t ig = 8 * (size_t) (64 * gw + bslz4_ctz64(gbits));
                BSLZ4_LOWPLANES_GROUP(ig);
            }
    } else {
        for (size_t ig = 0; ig < size; ig += 8)
            BSLZ4_LOWPLANES_GROUP(ig);
    }
#undef BSLZ4_LOWPLANES_GROUP
    for (int w = 0; w < 2 && twopass; w++) {
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

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,gfni,popcnt")))
static int bslz4_lowplanes_collect_u16(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end,
                                       const uint8_t *BSLZ4_RESTRICT zeros,
                                       const uint8_t *BSLZ4_RESTRICT mask, size_t i0, unsigned cut,
                                       uint16_t *BSLZ4_RESTRICT out_vals,
                                       uint32_t *BSLZ4_RESTRICT out_adr, int skip) {
    return bslz4_lowplanes_collect_body(raw, ne, nz_end, zeros, mask, i0, cut, out_vals, out_adr,
                                        BSLZ4_FUSED_TWOPASS, skip);
}

static const uint8_t bslz4_zero_plane[1024];      /* planes of up to 8192 pixels */

static int bslz4_lowplanes_collect_capable(void) {
    static int cached = -1;
    if (cached < 0)
        cached = bslz4_lowplanes_u16_capable() && c2py_amd64_avx512vl &&
                 __builtin_cpu_supports("avx512vbmi2") && __builtin_cpu_supports("popcnt");
    return cached;
}

/* 1 if the u16 block's high byte-planes (raw[ne..2ne)) are all zero; zero
 * fills raw[nz_end..2ne) when it has to look. */
static int bslz4_high_planes_zero(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end) {
    if (nz_end <= ne) return 1;
    if (nz_end < 2 * ne) memset(raw + nz_end, 0, 2 * ne - nz_end);
    return bslz4_all_zero_avx512(raw + ne, ne);
}
#endif

#if BSLZ4_HAVE_AVX2_OR
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

static void bslz4_widen_u8_u16_sse2(uint16_t *BSLZ4_RESTRICT out, const uint8_t *BSLZ4_RESTRICT in,
                                    size_t n) {
    const __m128i z = _mm_setzero_si128();
    for (size_t i = 0; i < n; i += 16) {
        const __m128i v = _mm_loadu_si128((const __m128i *) (const void *) (in + i));
        _mm_storeu_si128((__m128i *) (void *) (out + i), _mm_unpacklo_epi8(v, z));
        _mm_storeu_si128((__m128i *) (void *) (out + i + 8), _mm_unpackhi_epi8(v, z));
    }
}

#endif

/* 1 if p[0..n) are all zero; n a multiple of 8.  Plain C, 512-byte steps. */
static int bslz4_all_zero_c(const uint8_t *BSLZ4_RESTRICT p, size_t n) {
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

#if !BSLZ4_HAVE_AVX2_OR && defined(__POWER8_VECTOR__) && defined(__ALTIVEC__)
#include <altivec.h>
#undef vector
#undef pixel
#undef bool
#define BSLZ4_HAVE_LOWPLANES_VSX 1
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
                                       uint16_t *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr) {
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
    return npx;
}
static const uint8_t bslz4_zero_plane_vsx[1024];
#endif

/* As bslz4_high_planes_zero, without AVX-512 (any architecture); also zero
 * fills raw[nz_end..ne) for kcb, which reads the whole low half. */
static int bslz4_high_planes_zero_c(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end) {
    if (nz_end <= ne) {
        memset(raw + nz_end, 0, ne - nz_end);
        return 1;
    }
    if (nz_end < 2 * ne) memset(raw + nz_end, 0, 2 * ne - nz_end);
    return bslz4_all_zero_c(raw + ne, ne);
}

/* As bslz4_high_planes_zero_c, leaving raw[nz_end..ne) alone (the fused
 * collects handle the low half themselves). */
static int bslz4_high_planes_zero_c_nofill(uint8_t *BSLZ4_RESTRICT raw, size_t ne, size_t nz_end) {
    if (nz_end <= ne) return 1;
    if (nz_end < 2 * ne) memset(raw + nz_end, 0, 2 * ne - nz_end);
    return bslz4_all_zero_c(raw + ne, ne);
}

/* Untranspose one full block; with the byte skip allowed, a u16 block whose
 * high byte-planes are all zero takes the low-planes path. */
static int64_t bslz4_untranspose_block(const bslz4_stage *BSLZ4_RESTRICT st,
                                       uint8_t *BSLZ4_RESTRICT block, uint8_t *BSLZ4_RESTRICT raw,
                                       uint8_t *BSLZ4_RESTRICT scratch, size_t ne, size_t NB, int allow,
                                       size_t nz_end) {
    (void) allow;
#if BSLZ4_HAVE_AVX2_OR
    if (allow && NB == 2 && ne % 64 == 0 && bslz4_lowplanes_u16_capable()) {
        if (nz_end <= ne) {                                 /* the decoder says: planes 8..15 empty */
            memset(raw + nz_end, 0, ne - nz_end);
            bslz4_untrans_lowplanes_u16((uint16_t *) (void *) block, raw, ne);
            return (int64_t) (ne * NB);
        }
        if (nz_end < ne * NB) memset(raw + nz_end, 0, ne * NB - nz_end);
        if (bslz4_all_zero_avx512(raw + ne, ne)) {          /* planes 8..15 empty */
            bslz4_untrans_lowplanes_u16((uint16_t *) (void *) block, raw, ne);
            return (int64_t) (ne * NB);
        }
        return st->untranspose(block, raw, scratch, ne, NB);
    }
#if BSLZ4_BYTESKIP_GENERIC
    if (allow && NB == 2 && ne % 64 == 0) {
        int high_zero;
        if (nz_end <= ne) {                                 /* the decoder says: planes 8..15 empty */
            memset(raw + nz_end, 0, ne - nz_end);
            high_zero = 1;
        } else {
            if (nz_end < ne * NB) memset(raw + nz_end, 0, ne * NB - nz_end);
            high_zero = bslz4_all_zero_c(raw + ne, ne);
        }
        if (!high_zero) return st->untranspose(block, raw, scratch, ne, NB);
        if (bitshuf_decode_block((char *) scratch, (const char *) raw, NULL, ne, 1) < 0) return -1;
        if (c2py_amd64_avx2) bslz4_widen_u8_u16_avx2((uint16_t *) (void *) block, scratch, ne);
        else bslz4_widen_u8_u16_sse2((uint16_t *) (void *) block, scratch, ne);
        return (int64_t) (ne * NB);
    }
#endif
#endif
    if (nz_end < ne * NB) memset(raw + nz_end, 0, ne * NB - nz_end);
    return st->untranspose(block, raw, scratch, ne, NB);
}

/* Shared driver core.  kind: 0 = plain sparse, 1 = dot (powder).  For
 * kind 1, `mat` is the layout descriptor (interpreted by the kernel per
 * dot_id), `powder` the output buffer (double* for the float layouts;
 * element size from the dot table), `nout` the number of output bins and
 * `dense_sparse_x` the dense/sparse route threshold.  For kind 0 these are
 * ignored. */
static int bslz4_driver_run(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                            const int32_t *BSLZ4_RESTRICT compressed_lengths,
                            int nframes, int codec,
                            const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                            void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                            int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                            void *BSLZ4_RESTRICT powder, int nout,
                            const void *BSLZ4_RESTRICT mat,
                            double dense_sparse_x,
                            uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                            int64_t *BSLZ4_RESTRICT cursors,
                            const bslz4_stage *BSLZ4_RESTRICT st, int kind) {
    const size_t NB = st->elem_size;
    const int is_dot = (kind != 0);
    int out_size = is_dot ? bslz4_dot_out_size(st->dot_id) : 0;
    if (is_dot && out_size == 0) out_size = (int) NB;     /* packed output: the pixel dtype */

    if (threshold < 0) return BSLZ4_ERR_BAD_THRESHOLD;

    uint64_t total_output_length;
    size_t blocksize;
    int rc = bslz4_driver_check_frames(compressed_ptrs, compressed_lengths, nframes,
                                       &total_output_length, &blocksize);
    if (rc) return rc;

    /* A frame must hold exactly NIJ pixels (frame shape == mask shape): the
     * derived layouts (padded rows, bsb-csr entries) cover every mask pixel
     * and would read past a shorter decoded tail. */
    if (total_output_length / NB > (uint64_t) NIJ) return BSLZ4_ERR_TOO_MANY_PIXELS;
    if (total_output_length != (uint64_t) NIJ * NB) return BSLZ4_ERR_TOO_FEW_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return BSLZ4_ERR_TOO_LARGE;

    const size_t block_elems = blocksize / NB;
    const int byteskip = (st->options & BSLZ4_OPT_BYTESKIP) != 0;
    /* plane extraction: plain sparsify of u8/u16/u32 (dtype indices 0-2) */
    const int extract = !is_dot && (st->options & BSLZ4_OPT_PLANE_EXTRACT) != 0 && st->dtype <= 2;
    const uint8_t *BSLZ4_RESTRICT emask = (st->options & BSLZ4_OPT_NO_MASK) ? NULL : mask;
    const int lz4zero = (st->options & BSLZ4_OPT_LZ4_ZERO) != 0 && codec == BSLZ4_CODEC_LZ4;
    /* mask in the bitshuffled domain: needs a mask, blocks of <= 8192 pixels */
    const int maskplanes = (st->options & BSLZ4_OPT_MASK_PLANES) != 0 && emask != NULL &&
                           block_elems <= BSLZ4_MASKPLANE_MAX_ELEMS && block_elems % 64 == 0;
    uint64_t maskbits[BSLZ4_MASKPLANE_MAX_ELEMS / 64];
    /* the byte skip's fused transpose + collect (u16; not csc-permute-runs,
     * whose sparse route reads the block) */
    /* 1: AVX-512 VBMI + GFNI low-planes collect; 2: without them, kcb's low
     * planes as bytes into scratch, then the AVX2 u8 collect */
    int fused = 0;
#if !BSLZ4_HAVE_AVX2_OR && BSLZ4_FUSED_GENERIC
    const int fused_generic = byteskip && NB == 2 && st->dtype == 1 && block_elems % 8 == 0 &&
                              (!is_dot || st->dot_id != 24);
#if defined(BSLZ4_HAVE_LOWPLANES_VSX) && BSLZ4_FUSED_VSX
    const int use_vsx = block_elems / 8 <= sizeof(bslz4_zero_plane_vsx);
#else
    const int use_vsx = 0;
#endif
    /* the low bytes' collect: fused VSX transpose, or kcb's bytes in scratch */
#if defined(BSLZ4_HAVE_LOWPLANES_VSX) && BSLZ4_FUSED_VSX
#define BSLZ4_COLLECT_LOW(cut_, vals_, adr_) (use_vsx \
        ? bslz4_lowplanes_collect_vsx(raw, block_elems, nz_end, bslz4_zero_plane_vsx, bmask, (size_t) i0, (cut_), (vals_), (adr_)) \
        : bslz4_collect_u8_swar(scratch, bmask, (size_t) i0, block_elems, (cut_), (vals_), (adr_)))
#else
#define BSLZ4_COLLECT_LOW(cut_, vals_, adr_) \
        bslz4_collect_u8_swar(scratch, bmask, (size_t) i0, block_elems, (cut_), (vals_), (adr_))
#endif
#endif
#if BSLZ4_HAVE_AVX2_OR && BSLZ4_LOWPLANES_FUSED
    if (byteskip && NB == 2 && st->dtype == 1 && block_elems % 64 == 0 &&
        (!is_dot || st->dot_id != 24)) {
        if (block_elems / 8 <= sizeof(bslz4_zero_plane) && bslz4_lowplanes_collect_capable())
            fused = 1;
#if BSLZ4_BYTESKIP_GENERIC && BSLZ4_FUSED_U8
        else if (c2py_amd64_avx2 && bslz4_available_avx2cs_collect() && block_elems <= 8192 &&
                 (BSLZ4_FUSED_U8 != 2 || block_elems % 256 == 0))
            fused = 2;
#endif
    }
#endif
    if (is_dot) {
        /* The derived layouts index a per-block pointer array by
         * i0 / block_elems, so they must be built at this block size and
         * cover every block of the frame (nblocks + 1 entries). */
        const int layout = bslz4_dot_layout(st->dot_id);
        const size_t nblocks = block_elems ? ((size_t) NIJ + block_elems - 1) / block_elems : 0;
        if (layout == BSLZ4_LAYOUT_PADDED) {
            const bslz4_mat_padded *m = (const bslz4_mat_padded *) mat;
            if (block_elems == 0 || m->block_elems != block_elems || m->row_ptr_n < nblocks + 1)
                return BSLZ4_ERR_BAD_LAYOUT;
        } else if (layout == BSLZ4_LAYOUT_BSBCSR) {
            const bslz4_mat_bsbcsr *m = (const bslz4_mat_bsbcsr *) mat;
            if (block_elems == 0 || m->block_elems != block_elems || m->blk_ptr_n < nblocks + 1)
                return BSLZ4_ERR_BAD_LAYOUT;
        } else if (bslz4_dot_stream(st->dot_id)) {
            /* a headed index stream (bslz4_csc_variants.hpp): built for this
             * block size, with a table entry for every block */
            const uint32_t *h = ((const bslz4_mat_csc *) mat)->indices;
            if (h[0] != 0x58495342u ||
                (h[2] != 0 && (h[2] != block_elems || h[3] < nblocks)))
                return BSLZ4_ERR_BAD_LAYOUT;
        }
    }
    if (workspace_len < 3 * blocksize + (is_dot ? block_elems * (sizeof(uint32_t) + NB) : 0))
        return BSLZ4_ERR_WORKSPACE_TOO_SMALL;

    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + blocksize;
    uint8_t *BSLZ4_RESTRICT block = workspace + 2 * blocksize;
    uint32_t *BSLZ4_RESTRICT tidx = NULL;
    void *BSLZ4_RESTRICT tval = NULL;
    if (is_dot) {
        tidx = (uint32_t *) (workspace + 3 * blocksize);
        tval = (void *) ((uint8_t *) tidx + block_elems * sizeof(uint32_t));
    }

    for (int f = 0; f < nframes; f++) {
        if (is_dot) {
            uint8_t *outf = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
            memset(outf, 0, (size_t) nout * (size_t) out_size);
        }
        npx_out[f] = 0;
        cursors[f] = 12;
    }

    /* Filled once, then n/i0/out_vals/out_adr/route/powder are updated per
     * block or tail.  A new scheme adds a dot row + kernel; the template is
     * never widened. */
    bslz4_work wbase = {0};
    wbase.dtype = st->dtype;
    wbase.collect_id = st->collect_id;
    wbase.dot_id = st->dot_id;
    wbase.no_mask = (st->options & BSLZ4_OPT_NO_MASK) != 0;
    wbase.threshold = (int64_t) threshold;
    wbase.block = block;
    wbase.mask = mask;
    wbase.mat = mat;
    wbase.tidx = tidx;
    wbase.tval = tval;
    wbase.nout = nout;

    int i0 = 0;
    int64_t remaining = (int64_t) total_output_length;

    for (; remaining >= (int64_t) blocksize; remaining -= (int64_t) blocksize) {
        /* the block's mask, packed once for all frames; 1 = nothing masked */
        const int block_unmasked = maskplanes ? bslz4_pack_mask(maskbits, emask + i0, block_elems) : 0;
        for (int f = 0; f < nframes; f++) {
            const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
            const int64_t clen = compressed_lengths[f];
            int64_t p = cursors[f];
            if (BSLZ4_UNLIKELY(clen - p < 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            uint32_t nbytes = bslz4_read_be32((const uint8_t *) cf + p);
            if (BSLZ4_UNLIKELY((int64_t) nbytes > clen - p - 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            size_t nz_end = blocksize;      /* raw[nz_end..blocksize) is zero, maybe unwritten */
            int ret;
            if (lz4zero && ((size_t) nbytes * BSLZ4_LZ4ZERO_RATIO < blocksize ||
                            (size_t) nbytes * BSLZ4_LZ4ZERO_DENSE > blocksize))
                /* slack: the scratch buffer (blocksize bytes) follows raw */
                ret = bslz4_lz4_decode_zero((const uint8_t *) cf + p + 4, nbytes, raw, blocksize,
                                            blocksize, &nz_end);
            else
                ret = st->decompress(codec, cf + p + 4, (int) nbytes, (char *) raw, (int) blocksize);
            cursors[f] = p + (int64_t) nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) blocksize)) return BSLZ4_ERR_DECOMPRESS;
            if (maskplanes && !block_unmasked)
                bslz4_mask_planes(raw, maskbits, block_elems / 8, nz_end);
            /* masked pixels are 0 now: no per-pixel mask tests for this block */
            const uint8_t *BSLZ4_RESTRICT bmask = maskplanes ? NULL : emask;
            if (extract && (size_t) nbytes * BSLZ4_EXTRACT_RATIO < blocksize) {
                const int32_t npx0 = npx_out[f];
                int r = bslz4_plane_extract(raw, block_elems, NB, nz_end, bmask, (size_t) i0,
                                            (uint32_t) threshold,
                                            (uint8_t *) outpx + ((size_t) f * NIJ + (size_t) npx0) * NB,
                                            output_adr + (size_t) f * NIJ + npx0,
                                            (uint64_t *) (void *) scratch);
                if (r >= 0) {
                    bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, codec);
                    npx_out[f] = npx0 + r;
                    continue;
                }
            }
#if BSLZ4_HAVE_AVX2_OR
            if (fused && (!is_dot || (double) blocksize > dense_sparse_x * (double) nbytes) &&
                (fused == 1 ? bslz4_high_planes_zero(raw, block_elems, nz_end)
                 : BSLZ4_FUSED_U8 == 2 ? bslz4_high_planes_zero_c_nofill(raw, block_elems, nz_end)
                                       : bslz4_high_planes_zero_c(raw, block_elems, nz_end))) {
                if (fused == 2 && BSLZ4_FUSED_U8 == 1 &&   /* the low bytes into scratch */
                    bitshuf_decode_block((char *) scratch, (const char *) raw, NULL, block_elems, 1) < 0)
                    return BSLZ4_ERR_UNTRANSPOSE;
                bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, codec);
                bslz4_counters_bump(BSLZ4_STAGE_UNTRANSPOSE, st->untranspose_id);
                bslz4_counters_bump(BSLZ4_STAGE_COLLECT, st->collect_id);
                const int32_t npx0 = npx_out[f];
                /* skip empty 64-px groups only in well-compressed blocks (on
                 * 9 % frames nearly every group holds data) */
                const int skip_groups = BSLZ4_LOWPLANES_SKIP &&
                                        (size_t) nbytes * BSLZ4_LOWPLANES_SKIP_RATIO < blocksize;
                (void) skip_groups;
                if (!is_dot) {                  /* the >cut pixels straight to the output */
                    if (threshold < 255)
                        npx_out[f] = npx0 + (fused == 1
                            ? bslz4_lowplanes_collect_u16(
                                  raw, block_elems, nz_end, bslz4_zero_plane, bmask, (size_t) i0,
                                  (unsigned) threshold,
                                  (uint16_t *) (void *) outpx + (size_t) f * NIJ + (size_t) npx0,
                                  output_adr + (size_t) f * NIJ + npx0, skip_groups)
                            : BSLZ4_FUSED_U8 == 2
                            ? bslz4_lowplanes_collect_avx2(
                                  raw, block_elems, nz_end, bmask, (size_t) i0, (unsigned) threshold,
                                  (uint16_t *) (void *) outpx + (size_t) f * NIJ + (size_t) npx0,
                                  output_adr + (size_t) f * NIJ + npx0)
                            : bslz4_collect_u8_avx2cs(
                                  scratch, bmask, (size_t) i0, block_elems, (unsigned) threshold,
                                  (uint16_t *) (void *) outpx + (size_t) f * NIJ + (size_t) npx0,
                                  output_adr + (size_t) f * NIJ + npx0));
                    continue;
                }
                /* dot: the non-zero pixels into the sparse route's list */
                bslz4_work w = wbase;
                if (maskplanes) w.no_mask = 1;
                w.n = block_elems;
                w.i0 = (size_t) i0;
                w.out_vals = (uint8_t *) outpx + ((size_t) f * NIJ + (size_t) npx0) * NB;
                w.out_adr = output_adr + (size_t) f * NIJ + npx0;
                w.route = 1;
                w.powder = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
                w.precompacted = 1;
                w.pre_nz = fused == 1
                    ? bslz4_lowplanes_collect_u16(raw, block_elems, nz_end, bslz4_zero_plane,
                                                  bmask, (size_t) i0, 0u, (uint16_t *) tval, tidx,
                                                  skip_groups)
                    : BSLZ4_FUSED_U8 == 2
                    ? bslz4_lowplanes_collect_avx2(raw, block_elems, nz_end, bmask, (size_t) i0, 0u,
                                                   (uint16_t *) tval, tidx)
                    : bslz4_collect_u8_avx2cs(scratch, bmask, (size_t) i0, block_elems, 0u,
                                              (uint16_t *) tval, tidx);
                bslz4_counters_bump(BSLZ4_STAGE_DOT, st->dot_id);
                npx_out[f] = npx0 + bslz4_sparse_dot_dispatch(&w);
                continue;
            }
#endif
#if !BSLZ4_HAVE_AVX2_OR && BSLZ4_FUSED_GENERIC
            /* other architectures (ppc64le, aarch64): the byte skip and the
             * collect from the low bytes, in portable C over kcb */
            if (fused_generic && (!is_dot || (double) blocksize > dense_sparse_x * (double) nbytes) &&
                (use_vsx ? bslz4_high_planes_zero_c_nofill(raw, block_elems, nz_end)
                         : bslz4_high_planes_zero_c(raw, block_elems, nz_end))) {
                if (!use_vsx &&
                    bitshuf_decode_block((char *) scratch, (const char *) raw, NULL, block_elems, 1) < 0)
                    return BSLZ4_ERR_UNTRANSPOSE;
                bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, codec);
                bslz4_counters_bump(BSLZ4_STAGE_UNTRANSPOSE, st->untranspose_id);
                bslz4_counters_bump(BSLZ4_STAGE_COLLECT, st->collect_id);
                const int32_t npx0 = npx_out[f];
                if (!is_dot) {
                    if (threshold < 255)
                        npx_out[f] = npx0 + BSLZ4_COLLECT_LOW(
                            (unsigned) threshold,
                            (uint16_t *) (void *) outpx + (size_t) f * NIJ + (size_t) npx0,
                            output_adr + (size_t) f * NIJ + npx0);
                    continue;
                }
                bslz4_work w = wbase;
                if (maskplanes) w.no_mask = 1;
                w.n = block_elems;
                w.i0 = (size_t) i0;
                w.out_vals = (uint8_t *) outpx + ((size_t) f * NIJ + (size_t) npx0) * NB;
                w.out_adr = output_adr + (size_t) f * NIJ + npx0;
                w.route = 1;
                w.powder = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
                w.precompacted = 1;
                w.pre_nz = BSLZ4_COLLECT_LOW(0u, (uint16_t *) tval, tidx);
                bslz4_counters_bump(BSLZ4_STAGE_DOT, st->dot_id);
                npx_out[f] = npx0 + bslz4_sparse_dot_dispatch(&w);
                continue;
            }
#endif
            if (BSLZ4_UNLIKELY(bslz4_untranspose_block(st, block, raw, scratch, block_elems, NB,
                                                       byteskip, nz_end) < 0))
                return BSLZ4_ERR_UNTRANSPOSE;
            bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, codec);
            bslz4_counters_bump(BSLZ4_STAGE_UNTRANSPOSE, st->untranspose_id);

            uint8_t *BSLZ4_RESTRICT outf = (uint8_t *) outpx + (size_t) f * NIJ * NB;
            uint32_t *BSLZ4_RESTRICT outadrf = output_adr + (size_t) f * NIJ;
            int32_t npx = npx_out[f];
            bslz4_counters_bump(BSLZ4_STAGE_COLLECT, st->collect_id);
            bslz4_work w = wbase;
            if (maskplanes) w.no_mask = 1;
            w.n = block_elems;
            w.i0 = (size_t) i0;
            w.out_vals = outf + (size_t) npx * NB;
            w.out_adr = outadrf + npx;
            if (is_dot) {
                w.route = (double) blocksize > dense_sparse_x * (double) nbytes;
                w.powder = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
                bslz4_counters_bump(BSLZ4_STAGE_DOT, st->dot_id);
                npx += bslz4_sparse_dot_dispatch(&w);
            } else {
                npx += bslz4_sparse_dispatch(&w);
            }
            npx_out[f] = npx;
        }
        i0 += (int) block_elems;
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    for (int f = 0; f < nframes; f++) {
        const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
        uint32_t tail_nbytes = 0;
        if (tail_block > 0) {
            const int64_t clen = compressed_lengths[f];
            int64_t p = cursors[f];
            if (BSLZ4_UNLIKELY(clen - p < 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            tail_nbytes = bslz4_read_be32((const uint8_t *) cf + p);
            if (BSLZ4_UNLIKELY((int64_t) tail_nbytes > clen - p - 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            int ret = st->decompress(codec, cf + p + 4, (int) tail_nbytes, (char *) raw, (int) tail_block);
            cursors[f] = p + (int64_t) tail_nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return BSLZ4_ERR_DECOMPRESS;
            if (BSLZ4_UNLIKELY(st->untranspose(block, raw, scratch, tail_block / NB, NB) < 0))
                return BSLZ4_ERR_UNTRANSPOSE;
            bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, codec);
            bslz4_counters_bump(BSLZ4_STAGE_UNTRANSPOSE, st->untranspose_id);
        }
        int64_t rem_f = remaining - (int64_t) tail_block;
        if (rem_f > 0) {
            if (BSLZ4_UNLIKELY(compressed_lengths[f] < rem_f ||
                               cursors[f] > (int64_t) compressed_lengths[f] - rem_f))
                return BSLZ4_ERR_CORRUPT_CHUNK;
            memcpy(block + tail_block, cf + compressed_lengths[f] - rem_f, (size_t) rem_f);
        }
        uint8_t *BSLZ4_RESTRICT outf = (uint8_t *) outpx + (size_t) f * NIJ * NB;
        uint32_t *BSLZ4_RESTRICT outadrf = output_adr + (size_t) f * NIJ;
        int32_t npx = npx_out[f];
        size_t ntail = ((size_t) rem_f + tail_block) / NB;
        bslz4_counters_bump(BSLZ4_STAGE_COLLECT, st->collect_id);
        bslz4_work w = wbase;
        w.n = ntail;
        w.i0 = (size_t) i0;
        w.out_vals = outf + (size_t) npx * NB;
        w.out_adr = outadrf + npx;
        if (is_dot) {
            w.route = tail_block > 0
                ? (double) tail_block > dense_sparse_x * (double) tail_nbytes
                : 1; /* pure literal remainder: trivially cheap either way */
            w.powder = (uint8_t *) powder + (size_t) f * nout * (size_t) out_size;
            bslz4_counters_bump(BSLZ4_STAGE_DOT, st->dot_id);
            npx += bslz4_sparse_dot_dispatch(&w);
        } else {
            npx += bslz4_sparse_dispatch(&w);
        }
        npx_out[f] = npx;
    }
    return 0;
}

int bslz4_driver_sparsify(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                          const int32_t *BSLZ4_RESTRICT compressed_lengths,
                          int nframes, int codec,
                          const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                          void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                          int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                          uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                          int64_t *BSLZ4_RESTRICT cursors,
                          const bslz4_stage *BSLZ4_RESTRICT st) {
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, codec,
                            mask, NIJ, outpx, output_adr, npx_out, threshold,
                            NULL, 0, NULL, 0.0,
                            workspace, workspace_len, cursors, st, 0);
}

int bslz4_driver_sparsify_and_dot(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                  const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                  int nframes, int codec,
                                  const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                  void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                  int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                  void *BSLZ4_RESTRICT powder, int nout,
                                  const void *BSLZ4_RESTRICT data,
                                  const void *BSLZ4_RESTRICT indices,
                                  const uint32_t *BSLZ4_RESTRICT indptr,
                                  double dense_sparse_x,
                                  uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                  int64_t *BSLZ4_RESTRICT cursors,
                                  const bslz4_stage *BSLZ4_RESTRICT st) {
    bslz4_mat_csc csc;
    csc.data = data;
    csc.indices = (const uint32_t *) indices;
    csc.indptr = indptr;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, codec,
                            mask, NIJ, outpx, output_adr, npx_out, threshold,
                            powder, nout, &csc, dense_sparse_x,
                            workspace, workspace_len, cursors, st, 1);
}

int bslz4_driver_sparsify_and_dot_padded(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                         const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                         int nframes, int codec,
                                         const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                         void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                         int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                         double *BSLZ4_RESTRICT powder, int nout,
                                         const int32_t *BSLZ4_RESTRICT base,
                                         const float *BSLZ4_RESTRICT weights,
                                         const int32_t *BSLZ4_RESTRICT pixels,
                                         const int32_t *BSLZ4_RESTRICT rowmap,
                                         const int32_t *BSLZ4_RESTRICT row_ptr, int nrow_ptr,
                                         int width, int listed, size_t block_elems,
                                         double dense_sparse_x,
                                         uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                         int64_t *BSLZ4_RESTRICT cursors,
                                         const bslz4_stage *BSLZ4_RESTRICT st) {
    bslz4_mat_padded m;
    m.base = base;
    m.weights = weights;
    m.pixels = pixels;
    m.rowmap = rowmap;
    m.row_ptr = row_ptr;
    m.row_ptr_n = nrow_ptr > 0 ? (size_t) nrow_ptr : 0;
    m.width = width;
    m.listed = listed;
    m.block_elems = block_elems;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, codec,
                            mask, NIJ, outpx, output_adr, npx_out, threshold,
                            powder, nout, &m, dense_sparse_x,
                            workspace, workspace_len, cursors, st, 1);
}

int bslz4_driver_sparsify_and_dot_bsbcsr(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                         const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                         int nframes, int codec,
                                         const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                         void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                         int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                         double *BSLZ4_RESTRICT powder, int nout,
                                         const uint32_t *BSLZ4_RESTRICT blk_ptr, int nblk_ptr,
                                         const uint32_t *BSLZ4_RESTRICT bins,
                                         const uint32_t *BSLZ4_RESTRICT bin_ptr,
                                         const uint16_t *BSLZ4_RESTRICT idx,
                                         const float *BSLZ4_RESTRICT data,
                                         const void *BSLZ4_RESTRICT csc_data,
                                         const uint32_t *BSLZ4_RESTRICT csc_indices,
                                         const uint32_t *BSLZ4_RESTRICT csc_indptr,
                                         size_t block_elems,
                                         double dense_sparse_x,
                                         uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                         int64_t *BSLZ4_RESTRICT cursors,
                                         const bslz4_stage *BSLZ4_RESTRICT st) {
    bslz4_mat_bsbcsr m;
    m.blk_ptr = blk_ptr;
    m.blk_ptr_n = nblk_ptr > 0 ? (size_t) nblk_ptr : 0;
    m.bins = bins;
    m.bin_ptr = bin_ptr;
    m.idx = idx;
    m.data = data;
    m.csc.data = csc_data;
    m.csc.indices = csc_indices;
    m.csc.indptr = csc_indptr;
    m.block_elems = block_elems;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, codec,
                            mask, NIJ, outpx, output_adr, npx_out, threshold,
                            powder, nout, &m, dense_sparse_x,
                            workspace, workspace_len, cursors, st, 1);
}
