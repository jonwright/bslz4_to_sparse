#ifndef BSLZ4_CODEC_H
#define BSLZ4_CODEC_H
/*
 * Per-block decompression codec (the C side of bslz4_codec.hpp).
 *
 * The bitshuffle HDF5 filter frames lz4 and zstd identically -- each block
 * is [4 byte BE compressed size][compressed bytes] -- so decoding either
 * only differs in which decompression function is called.  Which codec a
 * dataset used is HDF5 filter metadata (cd_values[4]: 2 = lz4, 3 = zstd)
 * known up front, so codec is an ordinary runtime parameter.
 */

#include "bslz4_common.h"

#include <stddef.h>
#include <stdint.h>

#include "lz4.h"
#include "zstd.h"

#define BSLZ4_CODEC_LZ4  2  /* matches BSHUF_H5_COMPRESS_LZ4 */
#define BSLZ4_CODEC_ZSTD 3  /* matches BSHUF_H5_COMPRESS_ZSTD */

/* Decompress one block.  Returns the number of decompressed bytes on
 * success (== dst_capacity), or -1 on error -- same contract as the old
 * C++ bslz4_decompress. */
static inline int bslz4_decompress(int codec, const char *BSLZ4_RESTRICT src, int compressed_size,
                                   char *BSLZ4_RESTRICT dst, int dst_capacity) {
    if (codec == BSLZ4_CODEC_ZSTD) {
        int64_t ret = (int64_t) ZSTD_decompress(dst, (size_t) dst_capacity,
                                                src, (size_t) compressed_size);
        return (ret == (int64_t) dst_capacity) ? dst_capacity : -1;
    }
    return LZ4_decompress_safe(src, dst, compressed_size, dst_capacity);
}

#endif /* BSLZ4_CODEC_H */
