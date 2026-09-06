#pragma once
/*
 * Backend adapters for the bit/byte-untranspose step of bitshuffle-lz4
 * decoding. Each backend exposes a function of the uniform signature
 *
 *   int64_t (*)(void *out, const void *in, void *scratch, size_t size, size_t elem_size)
 *
 * where "size" is a number of elements (a multiple of 8) and "elem_size"
 * is the byte width of one element. "scratch" must point at "size*elem_size"
 * writable bytes; it is never allocated here -- it is a slice of the
 * caller-owned workspace buffer.
 *
 * kcb (https://github.com/kalcutter/bitshuffle) already takes an explicit
 * scratch buffer and does its own CPU dispatch internally, so it needs only
 * one adapter.
 *
 * Upstream bitshuffle (https://github.com/kiyo-masui/bitshuffle) instead
 * exposes distinct per-ISA kernels (scalar/SSE2/AVX2/AVX512/NEON), each of
 * which -- in the upstream library -- mallocs its own temporary buffer
 * internally. We call the two lower-level primitives that make up each of
 * those kernels directly, with our own workspace slice as the temporary
 * buffer, so nothing is allocated here either.
 */

#include "bslz4_common.hpp"

extern "C" {

/* kcb: single entry point, does its own CPU dispatch. */
int bitshuf_decode_block(char *out, const char *in, char *scratch,
                          size_t size, size_t elem_size);

/* Upstream bitshuffle low-level primitives (bitshuffle_core.c).
 * Not part of bitshuffle's public header, but exported (non-static)
 * symbols -- see bshuf_untrans_bit_elem_scal/_SSE/_AVX/_AVX512/_NEON
 * in bitshuffle_core.c for the reference (malloc'ing) composition
 * of these same two calls. */
int64_t bshuf_trans_byte_bitrow_scal(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_scal(const void *in, void *out, size_t size, size_t elem_size);

int64_t bshuf_trans_byte_bitrow_SSE(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_SSE(const void *in, void *out, size_t size, size_t elem_size);

int64_t bshuf_trans_byte_bitrow_AVX(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_AVX(const void *in, void *out, size_t size, size_t elem_size);
/* AVX512 reuses the AVX bitrow transpose; only the eight-element shuffle
 * has its own AVX512 kernel (see bshuf_untrans_bit_elem_AVX512). */
int64_t bshuf_shuffle_bit_eightelem_AVX512(const void *in, void *out, size_t size, size_t elem_size);

int64_t bshuf_trans_byte_bitrow_NEON(const void *in, void *out, size_t size, size_t elem_size);
int64_t bshuf_shuffle_bit_eightelem_NEON(const void *in, void *out, size_t size, size_t elem_size);

} /* extern "C" */

namespace bslz4 {

inline int64_t untranspose_kcb(void *out, const void *in, void *scratch,
                                size_t size, size_t elem_size) {
    return bitshuf_decode_block((char *) out, (const char *) in, (char *) scratch,
                                 size, elem_size);
}

inline int64_t untranspose_bshuf_scal(void *out, const void *in, void *scratch,
                                       size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_scal(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_scal(scratch, out, size, elem_size);
}

inline int64_t untranspose_bshuf_sse(void *out, const void *in, void *scratch,
                                      size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_SSE(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_SSE(scratch, out, size, elem_size);
}

inline int64_t untranspose_bshuf_avx2(void *out, const void *in, void *scratch,
                                       size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_AVX(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_AVX(scratch, out, size, elem_size);
}

inline int64_t untranspose_bshuf_avx512(void *out, const void *in, void *scratch,
                                         size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_AVX(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_AVX512(scratch, out, size, elem_size);
}

inline int64_t untranspose_bshuf_neon(void *out, const void *in, void *scratch,
                                       size_t size, size_t elem_size) {
    int64_t c = bshuf_trans_byte_bitrow_NEON(in, scratch, size, elem_size);
    if (c < 0) return c;
    return bshuf_shuffle_bit_eightelem_NEON(scratch, out, size, elem_size);
}

} /* namespace bslz4 */
