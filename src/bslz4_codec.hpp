#pragma once
/*
 * Per-block decompression codec used inside a bitshuffle-lz4/zstd stream.
 *
 * The bitshuffle HDF5 filter frames lz4 and zstd identically -- each block
 * is [4 byte BE compressed size][compressed bytes] -- so decoding either
 * only differs in which decompression function is called. Which codec a
 * given dataset used is HDF5 filter metadata (cd_values[4]: 2 = lz4,
 * 3 = zstd, see bitshuffle/src/bshuf_h5filter.h) known to the Python
 * caller up front, not something encoded in the per-chunk stream itself.
 * So codec, unlike the untranspose backend, is an ordinary runtime
 * parameter rather than a compile-time template axis or a c2py23 variant
 * -- the same compiled function must serve whichever codec the caller's
 * dataset happens to use, one call at a time.
 *
 * No malloc here either: ZSTD_decompress(), like LZ4_decompress_safe(),
 * only ever reads compressed bytes and writes into caller-supplied
 * buffers.
 */

#include "bslz4_common.hpp"
#include "../lz4/lib/lz4.h"
#include "../zstd/lib/zstd.h"

namespace bslz4 {

enum : int {
    CODEC_LZ4 = 2,  /* matches BSHUF_H5_COMPRESS_LZ4 */
    CODEC_ZSTD = 3, /* matches BSHUF_H5_COMPRESS_ZSTD */
};

/*
 * Decompress one block. Returns the number of decompressed bytes on
 * success (== dst_capacity), or a negative value on error -- same
 * contract as LZ4_decompress_safe, which this mirrors for zstd too.
 *
 * ZSTD_decompress returns a size_t, with errors encoded as
 * (size_t)-<code> (see ZSTD_isError); reinterpreting that as an int64_t
 * turns those into ordinary small negative numbers on the 64-bit
 * platforms this project targets -- the same trick bitshuffle's own
 * bshuf_decompress_zstd_block uses (CHECK_ERR_FREE_LZ on an int64_t).
 */
inline int bslz4_decompress(int codec, const char *BSLZ4_RESTRICT src, int compressed_size,
                             char *BSLZ4_RESTRICT dst, int dst_capacity) {
    if (codec == CODEC_ZSTD) {
        int64_t ret = (int64_t) ZSTD_decompress(dst, (size_t) dst_capacity,
                                                 src, (size_t) compressed_size);
        return (ret == (int64_t) dst_capacity) ? dst_capacity : -1;
    }
    return LZ4_decompress_safe(src, dst, compressed_size, dst_capacity);
}

} /* namespace bslz4 */
