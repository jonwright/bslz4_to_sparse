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

/* bitshuffle-lz4 stream headers are big endian, see https://justine.lol/endian.html */
static inline uint32_t bslz4_read_be32(const uint8_t *p) {
    return ((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16) |
           ((uint32_t) p[2] << 8) | (uint32_t) p[3];
}

static inline uint64_t bslz4_read_be64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v = (v << 8) | (uint64_t) p[i];
    }
    return v;
}

/* Default bitshuffle block size (bytes) when the stream header encodes zero. */
#define BSLZ4_DEFAULT_BLOCK_BYTES 8192

/* Pipeline word (uint64_t, pass by value).  12 bits per stage:
 *  63            48 47            36 35            24 23             12 11              0
 * +----------------+----------------+----------------+-----------------+----------------+
 * |  options (16)  |    dot (12)    |  collect (12)  | untranspose(12) | decompress (12)|
 * +----------------+----------------+----------------+-----------------+----------------+
 * "options" bits are named below.  All options zero reproduces today's
 * behaviour exactly. */
#define BSLZ4_PIPE_SHIFT_DECOMPRESS  0
#define BSLZ4_PIPE_SHIFT_UNTRANSPOSE 12
#define BSLZ4_PIPE_SHIFT_COLLECT     24
#define BSLZ4_PIPE_SHIFT_DOT         36
#define BSLZ4_PIPE_SHIFT_OPTIONS     48
#define BSLZ4_PIPE_MASK             0xFFFu

#define BSLZ4_PIPE_DECOMPRESS(p)   ((int) (((p) >> BSLZ4_PIPE_SHIFT_DECOMPRESS) & BSLZ4_PIPE_MASK))
#define BSLZ4_PIPE_UNTRANSPOSE(p)  ((int) (((p) >> BSLZ4_PIPE_SHIFT_UNTRANSPOSE) & BSLZ4_PIPE_MASK))
#define BSLZ4_PIPE_COLLECT(p)      ((int) (((p) >> BSLZ4_PIPE_SHIFT_COLLECT) & BSLZ4_PIPE_MASK))
#define BSLZ4_PIPE_DOT(p)          ((int) (((p) >> BSLZ4_PIPE_SHIFT_DOT) & BSLZ4_PIPE_MASK))
#define BSLZ4_PIPE_OPTIONS(p)      ((int) (((p) >> BSLZ4_PIPE_SHIFT_OPTIONS) & 0xFFFFu))
#define BSLZ4_PIPE_OPT_BIT(p, bit) ((((p) >> BSLZ4_PIPE_SHIFT_OPTIONS) & ((uint64_t) 1 << (bit))) != 0)

/* Build a pipeline word from the four stage ids + options. */
#define BSLZ4_PIPE_MAKE(decomp, untranspose, collect, dot, opt) \
    ((((uint64_t) (opt) & 0xFFFFu) << BSLZ4_PIPE_SHIFT_OPTIONS) | \
     (((uint64_t) (dot) & BSLZ4_PIPE_MASK) << BSLZ4_PIPE_SHIFT_DOT) | \
     (((uint64_t) (collect) & BSLZ4_PIPE_MASK) << BSLZ4_PIPE_SHIFT_COLLECT) | \
     (((uint64_t) (untranspose) & BSLZ4_PIPE_MASK) << BSLZ4_PIPE_SHIFT_UNTRANSPOSE) | \
     ((uint64_t) (decomp) & BSLZ4_PIPE_MASK))

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
