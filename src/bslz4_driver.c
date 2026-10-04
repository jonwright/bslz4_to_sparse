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
#define BSLZ4_EXTRACT_MAX 48   /* above this many non-zero pixels: untranspose */
#endif
#ifndef BSLZ4_LZ4ZERO_RATIO
/* the zero-aware lz4 decoder is used for blocks compressed more than this:
 * at 8x, medium-density frames (blocks ~13x, many sequences) were 50-60 %
 * slower than with stock lz4; at 24x they are 1-3 % faster and very sparse
 * frames 40-60 % faster (2026-10-03, EPYC 9454) */
#define BSLZ4_LZ4ZERO_RATIO 24
#endif
#ifndef BSLZ4_EXTRACT_RATIO
/* only blocks compressed more than this are tried: at 8x, medium-density
 * frames (~9 % non-zero, blocks ~13x) built the bitmap and fell back, ~4 %
 * slower; at 24x they are untouched and very sparse frames gain as much
 * (2026-10-03, EPYC 9454) */
#define BSLZ4_EXTRACT_RATIO 24
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
    if (bslz4_popcount_words(bitmap, nw) > BSLZ4_EXTRACT_MAX) return -1;
    int npx = 0;
    for (size_t w = 0; w < nw; w++) {
        uint64_t bits = bitmap[w];
        while (bits) {
            const size_t e = 64 * w + (size_t) bslz4_ctz64(bits);
            bits &= bits - 1;
            if (mask && !mask[i0 + e]) continue;
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
#endif

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
        for (int f = 0; f < nframes; f++) {
            const char *BSLZ4_RESTRICT cf = (const char *) (intptr_t) compressed_ptrs[f];
            const int64_t clen = compressed_lengths[f];
            int64_t p = cursors[f];
            if (BSLZ4_UNLIKELY(clen - p < 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            uint32_t nbytes = bslz4_read_be32((const uint8_t *) cf + p);
            if (BSLZ4_UNLIKELY((int64_t) nbytes > clen - p - 4)) return BSLZ4_ERR_CORRUPT_CHUNK;
            size_t nz_end = blocksize;      /* raw[nz_end..blocksize) is zero, maybe unwritten */
            int ret;
            if (lz4zero && (size_t) nbytes * BSLZ4_LZ4ZERO_RATIO < blocksize)
                ret = bslz4_lz4_decode_zero((const uint8_t *) cf + p + 4, nbytes, raw, blocksize, &nz_end);
            else
                ret = st->decompress(codec, cf + p + 4, (int) nbytes, (char *) raw, (int) blocksize);
            cursors[f] = p + (int64_t) nbytes + 4;
            if (BSLZ4_UNLIKELY(ret != (int) blocksize)) return BSLZ4_ERR_DECOMPRESS;
            if (extract && (size_t) nbytes * BSLZ4_EXTRACT_RATIO < blocksize) {
                const int32_t npx0 = npx_out[f];
                int r = bslz4_plane_extract(raw, block_elems, NB, nz_end, emask, (size_t) i0,
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
