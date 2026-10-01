#ifndef BSLZ4_UNTRANSPOSE_H
#define BSLZ4_UNTRANSPOSE_H
/*
 * C untranspose backend adapters (the C side of bslz4_backends.hpp).
 *
 * Each backend exposes a function of the uniform signature
 *   int64_t (*)(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch, size_t size, size_t elem_size)
 * where "size" is a number of elements (a multiple of 8) and "elem_size"
 * is the byte width of one element.  "scratch" must point at size*elem_size
 * writable bytes; it is a slice of the caller-owned workspace buffer.
 *
 * kcb does its own CPU dispatch internally; the upstream bitshuffle
 * kernels are the two lower-level primitives called here (we supply the
 * workspace slice as the temporary buffer, so nothing is allocated).
 */

#include "bslz4_common.h"

#include <stdint.h>
#include <stddef.h>

/* kcb: single entry point, does its own CPU dispatch. */
int bitshuf_decode_block(char *out, const char *in, char *scratch,
                          size_t size, size_t elem_size);

/* Upstream bitshuffle low-level primitives (bitshuffle_core.c). */
int64_t bshuf_trans_byte_bitrow_scal(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_scal(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_trans_byte_bitrow_SSE(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_SSE(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_trans_byte_bitrow_AVX(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_AVX(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_AVX512(const void *in, void *out, size_t size, size_t elem_size);
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

static inline int64_t bslz4_untranspose_bshuf_sse(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                                   size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_SSE(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_SSE(scratch, out, size, elem_size);
}

static inline int64_t bslz4_untranspose_bshuf_avx2(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                                    size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_AVX(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_AVX(scratch, out, size, elem_size);
}

static inline int64_t bslz4_untranspose_bshuf_avx512(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                                      size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_AVX(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_AVX512(scratch, out, size, elem_size);
}

static inline int64_t bslz4_untranspose_bshuf_neon(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in, void *BSLZ4_RESTRICT scratch,
                                                    size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_NEON(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_NEON(scratch, out, size, elem_size);
}

/* Availability (build + CPU) for the untranspose backends. */
#define BSLZ4_HAVE_BACKEND_KCB   1
#define BSLZ4_HAVE_BACKEND_SCAL  1
#if defined(__SSE2__) || defined(NO_WARN_X86_INTRINSICS)
#define BSLZ4_HAVE_BACKEND_SSE   1
#else
#define BSLZ4_HAVE_BACKEND_SSE   0
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
#define BSLZ4_HAVE_BACKEND_NEON  1
#else
#define BSLZ4_HAVE_BACKEND_NEON  0
#endif

#endif /* BSLZ4_UNTRANSPOSE_H */
