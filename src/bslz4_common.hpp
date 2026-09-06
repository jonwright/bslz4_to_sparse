#pragma once
/*
 * Shared macros and small helpers for the bslz4_to_sparse C++ core.
 *
 * Style: C99 intersected with C++ -- templates instead of macro-based
 * type variants, no STL, no exceptions, no mallocs. All memory is owned
 * by the Python caller and passed in as flat pointers.
 */

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
#define BSLZ4_RESTRICT __restrict
#else
#define BSLZ4_RESTRICT __restrict__
#endif

#ifndef BSLZ4_UNLIKELY
#if defined(_MSC_VER)
#define BSLZ4_UNLIKELY(expr) (expr)
#else
#define BSLZ4_UNLIKELY(expr) (__builtin_expect(!!(expr), 0))
#endif
#endif

namespace bslz4 {

/* bitshuffle-lz4 stream headers are big endian, see https://justine.lol/endian.html */
inline uint32_t read_be32(const uint8_t *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

inline uint64_t read_be64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v = (v << 8) | uint64_t(p[i]);
    }
    return v;
}

/* Default bitshuffle block size (bytes) when the stream header encodes zero. */
constexpr size_t DEFAULT_BLOCK_BYTES = 8192;

/* Error codes returned by bslz4_decode / bslz4_csc_decode.
 * Kept numerically compatible with the previous C implementation
 * where a code already existed for the same condition. */
enum : int {
    ERR_TOO_MANY_PIXELS = -99,     /* decompressed size needs more room than NIJ */
    ERR_TOO_LARGE = -98,           /* decompressed size does not fit an int */
    ERR_LZ4 = -2,                  /* LZ4_decompress_safe returned an unexpected size */
    ERR_BAD_THRESHOLD = -100,      /* threshold < 0 */
    ERR_WORKSPACE_TOO_SMALL = -103,/* caller-supplied workspace smaller than 3*blocksize */
    ERR_UNTRANSPOSE = -104,        /* backend untranspose kernel reported failure */
};

} /* namespace bslz4 */
