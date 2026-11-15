#ifndef BSLZ4_STEP_UNTRANSPOSE_H
#define BSLZ4_STEP_UNTRANSPOSE_H
/*
 * Step "untranspose": bit-planes -> pixel block, with kcb or bitshuffle's
 * kernels (the low-planes values, for u8 blocks and u16 blocks < 256, are lowplanes.h).
 *
 * Each backend exposes a function of the uniform signature
 *   int64_t (*)(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch, size_t size, size_t elem_size)
 * where "size" is a number of elements (a multiple of 8) and "elem_size"
 * is the byte width of one element.  "scratch" must point at size*elem_size
 * writable bytes; it is a slice of the caller-owned workspace buffer.
 *
 * kcb does its own CPU dispatch internally (SSE2/AVX2/AVX-512 on x86; it
 * has no NEON or VSX code, so on AArch64 the kcb value calls bitshuffle's NEON
 * kernels, and on POWER bitshuffle's emulated SSE2 ones for 2- and 4-byte pixels).  The upstream bitshuffle kernels (scalar, NEON) are the
 * two lower-level primitives called here (we supply the workspace slice as
 * the temporary buffer, so nothing is allocated).
 *
 * The values (the low-planes ones: src/steps/lowplanes.h, lowplanes_avx2.hpp).
 * Each low-planes value handles u8 blocks (eight bit-planes are all there is)
 * and u16 blocks whose high byte-planes are all zero (every value < 256),
 * with half the bit transpose and no byte transpose; any other block, or a
 * block size the kernel does not take, is untransposed with kcb.  They write
 * the pixel list directly, fused with the collect (the block is never
 * written), on the sparse route and in plain sparsify; the dense dot route
 * needs the pixel block, which only the VBMI and AVX2 values build from the
 * low planes of a u16 block (VSX, C, and every u8 block: kcb).
 *
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_VBMI  needs AVX-512 F/BW/VL/VBMI/VBMI2, GFNI
 *                      and popcnt.  Bit transpose by gf2p8affineqb, 64 pixels
 *                      at a time, stored straight into u16 for the dense
 *                      route; the fused collect transposes only the 64-pixel
 *                      groups holding data in well-compressed blocks.
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_AVX512BW  needs AVX-512 F/BW and popcnt only
 *                      (Skylake-SP, Cascade Lake: no VBMI or GFNI).  Bit
 *                      transpose by mask loads and masked adds (kcb's
 *                      avx512bw kernel), 64 pixels at a time; the fused
 *                      collect skips empty groups as the VBMI value does and
 *                      compresses as dwords; the u16 dense route block too.
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_AVX2  needs AVX2 and popcnt.  Fused collect:
 *                      planes ORed into a bitmap of non-empty 64-pixel
 *                      groups, only those transposed (kcb's AVX2 kernel) and
 *                      compacted; ne % 256 == 0, any block size (worked in
 *                      tiles of BSLZ4_LOWPLANES_TILE pixels).
 *                      Dense route: kcb's u8 transpose, widened to u16.
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_VSX  needs POWER8+ (vgbbd).  Fused collect:
 *                      per 64-pixel group two vec_perm rounds and vec_gb give
 *                      the pixel bytes; all-zero groups and words are
 *                      skipped.  The dense route takes kcb.
 *   BSLZ4_UNTRANSPOSE_LOWPLANES_C  portable C, always available: kcb's u8
 *                      transpose of the low planes (on AArch64 the NEON one
 *                      of untranspose_u8_neon.h), then a SWAR collect of
 *                      the pixels > cut.  The dense route takes kcb.
 *   BSLZ4_UNTRANSPOSE_KCB  kcb, any dtype and block: dispatches SSE2, AVX2
 *                      or AVX-512 itself; the full bit and byte transpose
 *                      into the pixel block, collect separate.  kcb has no
 *                      NEON or VSX code: on AArch64 this value runs our
 *                      own NEON bit transpose for 1-, 2- and 4-byte pixels
 *                      (untranspose_u8_neon.h: bitshuffle's does the 1-byte
 *                      bit step in scalar code, and is slower than kcb's
 *                      portable C for u16 and u32),
 *                      on POWER bitshuffle's SSE2 ones (GCC emulates SSE2 on
 *                      VSX) for 2- and 4-byte pixels.
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

#include "untranspose_u8_neon.h"

/* kcb: single entry point, does its own CPU dispatch. */
int bitshuf_decode_block(char *out, const char *in, char *scratch,
                          size_t size, size_t elem_size);

/* Upstream bitshuffle low-level primitives (bitshuffle_core.c). */
int64_t bshuf_trans_byte_bitrow_scal(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_scal(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_trans_byte_bitrow_SSE(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_SSE(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_trans_byte_bitrow_NEON(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_NEON(const void *in, void *out, size_t size, size_t elem_size);

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

/* POWER built with GCC's x86 intrinsics emulation (tools/build_extension.py passes
 * -DNO_WARN_X86_INTRINSICS): bitshuffle compiles its SSE2 kernels there, kcb does not (it takes
 * its SIMD only on x86). */
#if defined(__powerpc64__) && defined(NO_WARN_X86_INTRINSICS)
#define BSLZ4_HAVE_BACKEND_SSE_EMU  1
static inline int64_t bslz4_untranspose_bshuf_sse(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                                   size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_SSE(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_SSE(scratch, out, size, elem_size);
}
#else
#define BSLZ4_HAVE_BACKEND_SSE_EMU  0
#endif

/* kcb, and the fallback of the low-planes values.  kcb has SIMD only on x86; elsewhere it runs its
 * portable C.  AArch64: our own NEON bit transpose (untranspose_u8_neon.h) for 1-, 2- and
 * 4-byte pixels (Pinebook, one core, 8 kB blocks, GB/s new vs kcb: A53 u8 1.71 vs 0.30, u16
 * 1.27 vs 0.29, u32 1.33 vs 0.29; A72 u8 3.25 vs 0.75, u16 2.15 vs 0.67, u32 2.04 vs 0.65;
 * bitshuffle's NEON is slower than kcb's portable C), kcb for other widths.  POWER: bitshuffle's SSE2 kernels
 * (emulated on VSX) for 2- and 4-byte pixels, 1.4 to 1.45x kcb on POWER9 (one core: u32 ceo2
 * 60.5 vs 41.6 fps, u16 silicon full transpose 114 vs 82); kcb for 1-byte pixels (160 vs 134). */
static inline int64_t bslz4_untranspose_kcb(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                             size_t size, size_t elem_size) {
#if BSLZ4_HAVE_BACKEND_NEON
    if (elem_size == 1) return bslz4_untrans_u8_neon((uint8_t *) out, (const uint8_t *) in, size);
    if (elem_size == 2 || elem_size == 4)
        return bslz4_untrans_wide_neon((uint8_t *) out, (const uint8_t *) in, (uint8_t *) scratch, size, elem_size);
    return bitshuf_decode_block((char *) out, (const char *) in, (char *) scratch, size, elem_size);
#elif BSLZ4_HAVE_BACKEND_SSE_EMU
    if (elem_size > 1) return bslz4_untranspose_bshuf_sse(out, in, scratch, size, elem_size);
    return bitshuf_decode_block((char *) out, (const char *) in, (char *) scratch, size, elem_size);
#else
    return bitshuf_decode_block((char *) out, (const char *) in, (char *) scratch, size, elem_size);
#endif
}

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
