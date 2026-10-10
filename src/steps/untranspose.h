#ifndef BSLZ4_STEP_UNTRANSPOSE_H
#define BSLZ4_STEP_UNTRANSPOSE_H
/*
 * Step "untranspose": bit-planes -> pixel block, with kcb or bitshuffle's
 * kernels (the low-planes values, for u16 blocks < 256, are lowplanes.h).
 *
 * Each backend exposes a function of the uniform signature
 *   int64_t (*)(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch, size_t size, size_t elem_size)
 * where "size" is a number of elements (a multiple of 8) and "elem_size"
 * is the byte width of one element.  "scratch" must point at size*elem_size
 * writable bytes; it is a slice of the caller-owned workspace buffer.
 *
 * kcb does its own CPU dispatch internally (SSE2/AVX2/AVX-512 on x86; it
 * has no NEON code).  The upstream bitshuffle kernels (scalar, NEON) are the
 * two lower-level primitives called here (we supply the workspace slice as
 * the temporary buffer, so nothing is allocated).
 *
 * The values (the low-planes ones: src/steps/lowplanes.h, lowplanes_avx2.hpp).
 * Each low-planes value handles only u16 blocks whose high byte-planes are
 * all zero (every value < 256), with half the bit transpose and no byte
 * transpose; any other block, or a block size the kernel does not take, is
 * untransposed with kcb.  They write the pixel list directly, fused with the
 * collect (the block is never written), on the sparse route and in plain
 * sparsify; the dense dot route needs the pixel block, which only the VBMI
 * and AVX2 values build from the low planes (VSX, C: kcb).
 *
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_VBMI  needs AVX-512 F/BW/VL/VBMI/VBMI2, GFNI
 *                      and popcnt.  Bit transpose by gf2p8affineqb, 64 pixels
 *                      at a time, stored straight into u16 for the dense
 *                      route; the fused collect transposes only the 64-pixel
 *                      groups holding data in well-compressed blocks.
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_AVX2  needs AVX2 and popcnt.  Fused collect:
 *                      planes ORed into a bitmap of non-empty 64-pixel
 *                      groups, only those transposed (kcb's AVX2 kernel) and
 *                      compacted; block sizes ne <= 8192, ne % 256 == 0.
 *                      Dense route: kcb's u8 transpose, widened to u16.
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_VSX  needs POWER8+ (vgbbd).  Fused collect:
 *                      per 64-pixel group two vec_perm rounds and vec_gb give
 *                      the pixel bytes; all-zero groups and words are
 *                      skipped.  The dense route takes kcb.
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_C  portable C, always available: kcb's u8
 *                      transpose of the low planes, then a SWAR collect of
 *                      the pixels > cut.  The dense route takes kcb.
 *   BSLZ4_UNTRANSPOSE_KCB  kcb, any dtype and block: dispatches SSE2, AVX2
 *                      or AVX-512 itself (no NEON code); the full bit and
 *                      byte transpose into the pixel block, collect separate.
 *   BSLZ4_UNTRANSPOSE_NEON  bitshuffle's NEON kernels (bshuf_trans_byte_
 *                      bitrow_NEON, then bshuf_shuffle_bit_eightelem_NEON);
 *                      AArch64 only, any dtype, into the pixel block.
 *   BSLZ4_UNTRANSPOSE_SCALAR  bitshuffle's scalar kernels (bshuf_trans_byte_
 *                      bitrow_scal, then bshuf_shuffle_bit_eightelem_scal);
 *                      any CPU, any dtype, into the pixel block.
 */

#include "../pipeline/common.h"
#include "../pipeline/pipeline.h"

#include <stdint.h>
#include <stddef.h>

/* kcb: single entry point, does its own CPU dispatch. */
int bitshuf_decode_block(char *out, const char *in, char *scratch,
                          size_t size, size_t elem_size);

/* Upstream bitshuffle low-level primitives (bitshuffle_core.c). */
int64_t bshuf_trans_byte_bitrow_scal(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_scal(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_trans_byte_bitrow_NEON(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_NEON(const void *in, void *out, size_t size, size_t elem_size);

static inline int64_t bslz4_untranspose_kcb(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                             size_t size, size_t elem_size) {
    return bitshuf_decode_block((char *) out, (const char *) in, (char *) scratch, size, elem_size);
}

static inline int64_t bslz4_untranspose_bshuf_scal(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                                    size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_scal(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_scal(scratch, out, size, elem_size);
}

static inline int64_t bslz4_untranspose_bshuf_neon(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                                    size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_NEON(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_NEON(scratch, out, size, elem_size);
}

#if defined(__ARM_NEON) && defined(__aarch64__)
#define BSLZ4_HAVE_BACKEND_NEON  1
#else
#define BSLZ4_HAVE_BACKEND_NEON  0
#endif

/* The backend for an untranspose value: kcb also for the low-planes values
 * (their blocks >= 256, and every block of another dtype). */
static inline int64_t bslz4_untranspose_backend(int value, void *BSLZ4_RESTRICT out,
                                                const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                                size_t size, size_t elem_size) {
    switch (value) {
#if BSLZ4_HAVE_BACKEND_NEON
    case BSLZ4_UNTRANSPOSE_NEON:   return bslz4_untranspose_bshuf_neon(out, in, scratch, size, elem_size);
#endif
    case BSLZ4_UNTRANSPOSE_SCALAR: return bslz4_untranspose_bshuf_scal(out, in, scratch, size, elem_size);
    default:                       return bslz4_untranspose_kcb(out, in, scratch, size, elem_size);
    }
}

#endif /* BSLZ4_STEP_UNTRANSPOSE_H */
