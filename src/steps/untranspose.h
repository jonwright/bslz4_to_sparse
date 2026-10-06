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
