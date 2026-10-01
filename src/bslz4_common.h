#ifndef BSLZ4_COMMON_H
#define BSLZ4_COMMON_H
/*
 * Shared macros and small helpers for the bslz4_to_sparse C99 core.
 *
 * This is the C side of what was bslz4_common.hpp.  It holds the error
 * codes, big-endian readers and the default block size -- all things the
 * driver (bslz4_driver.c) needs that do not depend on C++ templates.
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

#ifndef BSLZ4_UNLIKELY
#if defined(_MSC_VER)
#define BSLZ4_UNLIKELY(expr) (expr)
#else
#define BSLZ4_UNLIKELY(expr) (__builtin_expect(!!(expr), 0))
#endif
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

/* Decode "options" are passed as a small array of uint16, one entry per
 * stage/option, instead of a packed bit word.  c2py23 accepts only signed
 * int scalars (plus float/buffer), so a u64 pipeline word could never reach
 * the C code as more than 31 bits; a buffer of one value per option avoids
 * any packing/unpacking entirely.  The array is stages[BSLZ4_STAGES_N],
 * indexed by BSLZ4_STAGE_* (0..3) for the four stage ids, with the last
 * entry holding the options bitmask (see BSLZ4_OPT_*).  All options zero
 * reproduces today's behaviour.  `stages` is const uint16_t*. */
#define BSLZ4_STAGES_N 5
#define BSLZ4_STAGES_OPTIONS (BSLZ4_STAGES_N - 1)

/* First reserved option bit: opt in to dropping negative pixel values from
 * the powder/sparse output (issue #9).  Currently unused (default keeps
 * negatives, matching today's behaviour). */
#define BSLZ4_OPT_DROP_NEGATIVES ((uint16_t) 1 << 0)

/* First reserved option bit: opt in to dropping negative pixel values from
 * the powder/sparse output (issue #9).  Currently unused (default keeps
 * negatives, matching today's behaviour). */
#define BSLZ4_OPT_DROP_NEGATIVES ((uint64_t) 1 << 0)

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
    BSLZ4_ERR_BAD_LAYOUT = -109,         /* a padded-CSC layout argument is inconsistent */
    BSLZ4_ERR_BAD_MATRIX = -110,         /* CSC matrix arrays inconsistent (sizes/itemsize) */
    BSLZ4_ERR_BAD_PIPELINE = -111,       /* unknown stage id or unknown option bit */
    BSLZ4_ERR_UNAVAILABLE = -112,        /* a known implementation is unavailable here */
    BSLZ4_ERR_DTYPE = -113,              /* dtype index out of range or unsupported */
};

#endif /* BSLZ4_COMMON_H */
