#ifndef BSLZ4_COMMON_H
#define BSLZ4_COMMON_H
/*
 * Shared macros and small helpers for the C and C++ sources: the error
 * codes, big-endian readers, the default block size and the compiler
 * portability macros.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER)
#define BSLZ4_RESTRICT __restrict
#define BSLZ4_NOINLINE __declspec(noinline)
#else
#define BSLZ4_RESTRICT __restrict__
#define BSLZ4_NOINLINE __attribute__((noinline))
#endif

/* AVX-512 VBMI/VBMI2 and GFNI need gcc >= 8 or clang >= 7 (target
 * attributes, intrinsics, __builtin_cpu_supports names); older compilers
 * build without those paths and pick the AVX2 ones at run time.
 * BSLZ4_NO_VBMI_GFNI forces them off (to test the fallback). */
#ifndef BSLZ4_HAVE_VBMI_GFNI
#if !defined(BSLZ4_NO_VBMI_GFNI) && \
    ((defined(__clang__) && __clang_major__ >= 7) || (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 8))
#define BSLZ4_HAVE_VBMI_GFNI 1
#else
#define BSLZ4_HAVE_VBMI_GFNI 0
#endif
#endif

/* x86-64 with gcc/clang: the SIMD paths use target attributes and
 * __builtin_cpu_supports, so they are built only there (MSVC builds the
 * portable C ones). */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define BSLZ4_X86_SIMD 1
#else
#define BSLZ4_X86_SIMD 0
#endif

#ifndef BSLZ4_UNLIKELY
#if defined(_MSC_VER)
#define BSLZ4_UNLIKELY(expr) (expr)
#else
#define BSLZ4_UNLIKELY(expr) (__builtin_expect(!!(expr), 0))
#endif
#endif

/* index of the lowest set bit of a non-zero word */
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline int bslz4_ctz64(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (int) i; }
#else
static inline int bslz4_ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

/* bitshuffle-lz4 stream headers are big endian, see https://justine.lol/endian.html */
static inline uint32_t bslz4_read_be32(const uint8_t *BSLZ4_RESTRICT p) {
    return ((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16) |
           ((uint32_t) p[2] << 8) | (uint32_t) p[3];
}

static inline uint64_t bslz4_read_be64(const uint8_t *BSLZ4_RESTRICT p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v = (v << 8) | (uint64_t) p[i];
    }
    return v;
}

/* Default bitshuffle block size (bytes) when the stream header encodes zero. */
#define BSLZ4_DEFAULT_BLOCK_BYTES 8192

/* Error codes.  The values are part of the C API: keep them stable. */
enum {
    BSLZ4_ERR_TOO_MANY_PIXELS = -99,     /* decompressed size needs more room than NIJ */
    BSLZ4_ERR_TOO_LARGE = -98,           /* decompressed size does not fit an int */
    BSLZ4_ERR_DECOMPRESS = -2,           /* LZ4/zstd decompress returned an unexpected size */
    BSLZ4_ERR_BAD_THRESHOLD = -100,      /* threshold < 0 */
    BSLZ4_ERR_WORKSPACE_TOO_SMALL = -103,/* caller-supplied workspace smaller than required */
    BSLZ4_ERR_UNTRANSPOSE = -104,        /* backend untranspose kernel reported failure */
    BSLZ4_ERR_BAD_NFRAMES = -105,        /* nframes <= 0 in a multi-frame call */
    BSLZ4_ERR_FRAME_MISMATCH = -106,     /* frames don't share total size/block size */
    BSLZ4_ERR_CORRUPT_CHUNK = -107,      /* chunk too short, or a block/raw tail lies outside it */
    BSLZ4_ERR_BAD_CHUNK_BOUNDS = -108,   /* a chunk offset/size lies outside the buffer */
    BSLZ4_ERR_BAD_LAYOUT = -109,         /* the dot's layout does not match the entry point, or the decoded block_elems mismatches the descriptor */
    BSLZ4_ERR_TOO_FEW_PIXELS = -110,     /* decompressed size is smaller than NIJ (frame and mask shapes differ) */
    BSLZ4_ERR_BAD_PIPELINE = -111,       /* a pipeline step value is unknown, 0 (auto: resolve it first) or invalid here */
    BSLZ4_ERR_UNAVAILABLE = -112,        /* a known step value needs an instruction set this CPU/build lacks */
    BSLZ4_ERR_DTYPE = -113,              /* dtype index out of range or unsupported */
};

#endif /* BSLZ4_COMMON_H */
