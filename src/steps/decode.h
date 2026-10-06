#ifndef BSLZ4_STEP_DECODE_H
#define BSLZ4_STEP_DECODE_H
/*
 * Step "decode": one compressed block -> its bit-planes (raw bytes).
 *
 * The bitshuffle HDF5 filter frames lz4 and zstd identically -- each block
 * is [4 byte BE compressed size][compressed bytes] -- so the codec only
 * changes the function called.  Which codec a dataset used is filter
 * metadata (cd_values[4]: 2 = lz4, 3 = zstd); for lz4 the decoder is a
 * pipeline choice:
 *
 *   BSLZ4_DECODE_LZ4_BAND   the zero-aware decoder (lz4zero.h) for blocks
 *                           compressed more than BSLZ4_LZ4ZERO_RATIO or less
 *                           than BSLZ4_LZ4ZERO_DENSE, stock lz4 in between.
 *                           Zero-aware wins on sparse blocks (skipped zero
 *                           runs) and on dense ones (literal heavy; it never
 *                           writes the empty high byte-planes), but blocks of
 *                           ~8-24x are many short sequences (~90-130 per
 *                           8 kB) where stock's loop is faster: Zen 5 8-16x
 *                           2408 vs 3142 cycles; Zen 4 9 % frames +16-21 %
 *                           with the zero decoder everywhere (2026-10-06).
 *   BSLZ4_DECODE_LZ4_ZERO   the zero-aware decoder for every block: POWER9,
 *                           where stock lz4 is slower in every band (8-16x
 *                           3781 vs 4812 ns/block), and Cascade Lake.
 *   BSLZ4_DECODE_LZ4_STOCK  LZ4_decompress_safe.
 *   BSLZ4_DECODE_ZSTD       ZSTD_decompress.
 *
 * *nz_end returns how many leading bytes of raw are valid: the zero-aware
 * decoder leaves the zero tail of the block unwritten (raw[nz_end..) is zero
 * but maybe not stored); the others set it to the block size.
 */

#include "../pipeline/common.h"
#include "../pipeline/pipeline.h"
#include "lz4zero.h"

#include <stddef.h>
#include <stdint.h>

#include "lz4.h"
#include "zstd.h"

#define BSLZ4_CODEC_LZ4  2  /* matches BSHUF_H5_COMPRESS_LZ4 */
#define BSLZ4_CODEC_ZSTD 3  /* matches BSHUF_H5_COMPRESS_ZSTD */

#ifndef BSLZ4_LZ4ZERO_RATIO
#define BSLZ4_LZ4ZERO_RATIO 24
#endif
#ifndef BSLZ4_LZ4ZERO_DENSE
#define BSLZ4_LZ4ZERO_DENSE 8
#endif

/* The plain codec call: decoded size (== cap) or a negative value. */
static inline int bslz4_decode_plain(int decode, const char *BSLZ4_RESTRICT src, int n,
                                     char *BSLZ4_RESTRICT dst, int cap) {
    if (decode == BSLZ4_DECODE_ZSTD) {
        int64_t ret = (int64_t) ZSTD_decompress(dst, (size_t) cap, src, (size_t) n);
        return (ret == (int64_t) cap) ? cap : -1;
    }
    return LZ4_decompress_safe(src, dst, n, cap);
}

/* One full block of blocksize bytes into raw (which must be followed by
 * blocksize bytes of slack: the zero-aware decoder's chunked copies may run
 * into it).  Returns blocksize or a negative value. */
static inline int bslz4_decode_block(int decode, const uint8_t *BSLZ4_RESTRICT src, uint32_t n,
                                     uint8_t *BSLZ4_RESTRICT raw, size_t blocksize,
                                     size_t *BSLZ4_RESTRICT nz_end) {
    *nz_end = blocksize;
    if (decode == BSLZ4_DECODE_LZ4_ZERO ||
        (decode == BSLZ4_DECODE_LZ4_BAND && ((size_t) n * BSLZ4_LZ4ZERO_RATIO < blocksize ||
                                             (size_t) n * BSLZ4_LZ4ZERO_DENSE > blocksize)))
        return bslz4_lz4_decode_zero(src, n, raw, blocksize, blocksize, nz_end);
    return bslz4_decode_plain(decode, (const char *) src, (int) n, (char *) raw, (int) blocksize);
}

#endif /* BSLZ4_STEP_DECODE_H */
