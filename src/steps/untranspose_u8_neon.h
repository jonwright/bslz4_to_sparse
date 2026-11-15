#ifndef BSLZ4_STEP_UNTRANSPOSE_U8_NEON_H
#define BSLZ4_STEP_UNTRANSPOSE_U8_NEON_H
/*
 * Bit untranspose of one-byte elements on AArch64 NEON: eight bit-planes of
 * ne/8 bytes (plane b = bit b, pixel 8j+i at bit i of byte j) -> ne bytes.
 *
 * kcb has no NEON code, and bitshuffle's NEON untranspose does only the byte
 * transpose in NEON for one-byte elements (bshuf_shuffle_bit_eightelem_NEON
 * calls the scalar code when elem_size is odd).  Here 16 bytes of each plane
 * (128 pixels) are loaded, the 8x8 bit transpose is done lane-wise in three
 * masked-swap rounds, and three zip rounds interleave the eight result
 * vectors into pixel order.  No scratch.  ne must be a multiple of 8.
 */
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#define BSLZ4_U8_NEON_SWAP(a, b, d, m)                                          \
    do {                                                                          \
        uint8x16_t na_ = vbslq_u8(m, a, vshlq_n_u8(b, d));                        \
        uint8x16_t nb_ = vbslq_u8(m, vshrq_n_u8(a, d), b);                        \
        a = na_; b = nb_;                                                         \
    } while (0)

static inline void bslz4_untrans_u8_neon_128(uint8_t *out, const uint8_t *in, size_t n8, size_t j) {
    uint8x16_t p0 = vld1q_u8(in + 0 * n8 + j), p1 = vld1q_u8(in + 1 * n8 + j);
    uint8x16_t p2 = vld1q_u8(in + 2 * n8 + j), p3 = vld1q_u8(in + 3 * n8 + j);
    uint8x16_t p4 = vld1q_u8(in + 4 * n8 + j), p5 = vld1q_u8(in + 5 * n8 + j);
    uint8x16_t p6 = vld1q_u8(in + 6 * n8 + j), p7 = vld1q_u8(in + 7 * n8 + j);
    const uint8x16_t m1 = vdupq_n_u8(0x55), m2 = vdupq_n_u8(0x33), m4 = vdupq_n_u8(0x0f);
    /* rows = planes, columns = pixels: transpose so row i = pixel i */
    BSLZ4_U8_NEON_SWAP(p0, p1, 1, m1); BSLZ4_U8_NEON_SWAP(p2, p3, 1, m1);
    BSLZ4_U8_NEON_SWAP(p4, p5, 1, m1); BSLZ4_U8_NEON_SWAP(p6, p7, 1, m1);
    BSLZ4_U8_NEON_SWAP(p0, p2, 2, m2); BSLZ4_U8_NEON_SWAP(p1, p3, 2, m2);
    BSLZ4_U8_NEON_SWAP(p4, p6, 2, m2); BSLZ4_U8_NEON_SWAP(p5, p7, 2, m2);
    BSLZ4_U8_NEON_SWAP(p0, p4, 4, m4); BSLZ4_U8_NEON_SWAP(p1, p5, 4, m4);
    BSLZ4_U8_NEON_SWAP(p2, p6, 4, m4); BSLZ4_U8_NEON_SWAP(p3, p7, 4, m4);
    /* p_i lane l = pixel 8(j+l)+i: interleave into pixel order */
    uint8x16x2_t a01 = vzipq_u8(p0, p1), a23 = vzipq_u8(p2, p3);
    uint8x16x2_t a45 = vzipq_u8(p4, p5), a67 = vzipq_u8(p6, p7);
    uint16x8x2_t b0 = vzipq_u16(vreinterpretq_u16_u8(a01.val[0]), vreinterpretq_u16_u8(a23.val[0]));
    uint16x8x2_t b1 = vzipq_u16(vreinterpretq_u16_u8(a01.val[1]), vreinterpretq_u16_u8(a23.val[1]));
    uint16x8x2_t c0 = vzipq_u16(vreinterpretq_u16_u8(a45.val[0]), vreinterpretq_u16_u8(a67.val[0]));
    uint16x8x2_t c1 = vzipq_u16(vreinterpretq_u16_u8(a45.val[1]), vreinterpretq_u16_u8(a67.val[1]));
    uint32x4x2_t d = vzipq_u32(vreinterpretq_u32_u16(b0.val[0]), vreinterpretq_u32_u16(c0.val[0]));
    uint32x4x2_t e = vzipq_u32(vreinterpretq_u32_u16(b0.val[1]), vreinterpretq_u32_u16(c0.val[1]));
    uint32x4x2_t f = vzipq_u32(vreinterpretq_u32_u16(b1.val[0]), vreinterpretq_u32_u16(c1.val[0]));
    uint32x4x2_t g = vzipq_u32(vreinterpretq_u32_u16(b1.val[1]), vreinterpretq_u32_u16(c1.val[1]));
    uint8_t *o = out + 8 * j;
    vst1q_u8(o + 0, vreinterpretq_u8_u32(d.val[0]));   vst1q_u8(o + 16, vreinterpretq_u8_u32(d.val[1]));
    vst1q_u8(o + 32, vreinterpretq_u8_u32(e.val[0]));  vst1q_u8(o + 48, vreinterpretq_u8_u32(e.val[1]));
    vst1q_u8(o + 64, vreinterpretq_u8_u32(f.val[0]));  vst1q_u8(o + 80, vreinterpretq_u8_u32(f.val[1]));
    vst1q_u8(o + 96, vreinterpretq_u8_u32(g.val[0]));  vst1q_u8(o + 112, vreinterpretq_u8_u32(g.val[1]));
}

/* out must not overlap in.  Returns ne. */
static inline int64_t bslz4_untrans_u8_neon(uint8_t *out, const uint8_t *in, size_t ne) {
    const size_t n8 = ne / 8;
    size_t j = 0;
    if (n8 >= 16) {
        for (; j + 16 <= n8; j += 16) bslz4_untrans_u8_neon_128(out, in, n8, j);
        /* the last part group: one overlapping step (rewrites the same bytes) */
        if (j < n8) bslz4_untrans_u8_neon_128(out, in, n8, n8 - 16);
        return (int64_t) ne;
    }
    for (; j < n8; j++)
        for (int i = 0; i < 8; i++) {
            unsigned v = 0;
            for (int b = 0; b < 8; b++) v |= (unsigned) ((in[b * n8 + j] >> i) & 1) << b;
            out[8 * j + i] = (uint8_t) v;
        }
    return (int64_t) ne;
}
#undef BSLZ4_U8_NEON_SWAP
#endif
#endif
