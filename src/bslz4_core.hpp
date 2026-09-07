#pragma once
/*
 * Bitshuffle-LZ4/zstd decode, generic over pixel type T and the
 * untranspose backend. Templates replace the previous #define DATATYPE /
 * #include macro-repetition; each (T, Untranspose) combination compiles
 * to its own ordinary function, instantiated explicitly (see
 * bslz4_to_sparse.cpp) behind a plain extern "C" name.
 *
 * codec (lz4 vs zstd, see bslz4_codec.hpp) is an ordinary runtime int
 * parameter rather than a template axis: which codec a dataset used is
 * HDF5 filter metadata the Python caller already has, and can differ
 * between two datasets read with the very same compiled function in the
 * same process, unlike the untranspose backend (a session-wide choice
 * for benchmarking kernels against each other).
 *
 * No malloc, no VLA: the only scratch space used is a caller-owned
 * "workspace" buffer, carved by pointer arithmetic into three
 * same-sized regions (one block each): the decompressed-but-still
 * -bitshuffled data, the untranspose backend's own scratch, and the
 * final untransposed block. Its required size (3 * blocksize, where
 * blocksize is read from the compressed stream header) is not known
 * until runtime, since HDF5 datasets may use any block size -- so it
 * cannot be a template constant or a fixed stack array.
 */

#include "bslz4_codec.hpp"
#include "bslz4_common.hpp"

#include <cstring>
#include <cstdint>

namespace bslz4 {

template<typename T,
         int64_t (*Untranspose)(void *out, const void *in, void *scratch,
                                 size_t size, size_t elem_size)>
int bslz4_decode(const char *BSLZ4_RESTRICT compressed, int compressed_length, int codec,
                  const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                  T *BSLZ4_RESTRICT output, uint32_t *BSLZ4_RESTRICT output_adr,
                  int threshold,
                  uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len) {
    constexpr size_t NB = sizeof(T);

    if (threshold < 0) return ERR_BAD_THRESHOLD;
    const T cut = (T) threshold;

    const uint64_t total_output_length = read_be64((const uint8_t *) compressed);
    if (total_output_length / NB > (uint64_t) NIJ) return ERR_TOO_MANY_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return ERR_TOO_LARGE;

    size_t blocksize = read_be32((const uint8_t *) compressed + 8);
    if (blocksize == 0) blocksize = DEFAULT_BLOCK_BYTES;
    if (workspace_len < 3 * blocksize) return ERR_WORKSPACE_TOO_SMALL;

    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + blocksize;
    T *BSLZ4_RESTRICT block = (T *) (workspace + 2 * blocksize);

    int npx = 0;
    int i0 = 0;
    int p = 12;
    int64_t remaining = (int64_t) total_output_length;

    for (; remaining >= (int64_t) blocksize; remaining -= (int64_t) blocksize) {
        uint32_t nbytes = read_be32((const uint8_t *) compressed + p);
        int ret = bslz4_decompress(codec, compressed + p + 4, (int) nbytes,
                                    (char *) raw, (int) blocksize);
        p += (int) nbytes + 4;
        if (BSLZ4_UNLIKELY(ret != (int) blocksize)) return ERR_DECOMPRESS;

        if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, blocksize / NB, NB) < 0))
            return ERR_UNTRANSPOSE;

        for (size_t j = 0; j < blocksize / NB; j++) {
            /* Plain "&" (not "&&"): both sides are always evaluated, so this
             * stays branchless -- unlike the old mask*value>cut trick, it
             * compares in T's own signed/float domain instead of forcing
             * everything through uint32_t (wrong for negative or
             * fractional values). */
            if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (block[j] > cut))) {
                *(output++) = block[j];
                *(output_adr++) = (uint32_t) (j + i0);
                npx++;
            }
        }
        i0 += (int) (blocksize / NB);
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    if (tail_block > 0) {
        uint32_t nbytes = read_be32((const uint8_t *) compressed + p);
        int ret = bslz4_decompress(codec, compressed + p + 4, (int) nbytes,
                                    (char *) raw, (int) tail_block);
        p += (int) nbytes + 4;
        if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return ERR_DECOMPRESS;
        if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, tail_block / NB, NB) < 0))
            return ERR_UNTRANSPOSE;
    }
    remaining -= (int64_t) tail_block;
    if (remaining > 0) {
        memcpy(&block[tail_block / NB], compressed + compressed_length - remaining, (size_t) remaining);
    }
    for (size_t j = 0; j < (size_t(remaining) + tail_block) / NB; j++) {
        if (BSLZ4_UNLIKELY((mask[j + i0] > 0) & (block[j] > cut))) {
            *(output++) = block[j];
            *(output_adr++) = (uint32_t) (j + i0);
            npx++;
        }
    }
    return npx;
}

template<typename T,
         int64_t (*Untranspose)(void *out, const void *in, void *scratch,
                                 size_t size, size_t elem_size)>
int bslz4_csc_decode(const char *BSLZ4_RESTRICT compressed, int compressed_length, int codec,
                      const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                      T *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                      int threshold,
                      double *BSLZ4_RESTRICT output, int NOUT,
                      const float *BSLZ4_RESTRICT data,
                      const uint32_t *BSLZ4_RESTRICT indices,
                      const uint32_t *BSLZ4_RESTRICT indptr,
                      uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len) {
    constexpr size_t NB = sizeof(T);

    if (threshold < 0) return ERR_BAD_THRESHOLD;
    const T cut = (T) threshold;

    const uint64_t total_output_length = read_be64((const uint8_t *) compressed);
    if (total_output_length / NB > (uint64_t) NIJ) return ERR_TOO_MANY_PIXELS;
    if (total_output_length > (uint64_t) INT32_MAX) return ERR_TOO_LARGE;

    size_t blocksize = read_be32((const uint8_t *) compressed + 8);
    if (blocksize == 0) blocksize = DEFAULT_BLOCK_BYTES;
    if (workspace_len < 3 * blocksize) return ERR_WORKSPACE_TOO_SMALL;

    uint8_t *BSLZ4_RESTRICT raw = workspace;
    uint8_t *BSLZ4_RESTRICT scratch = workspace + blocksize;
    T *BSLZ4_RESTRICT block = (T *) (workspace + 2 * blocksize);

    for (int j = 0; j < NOUT; j++) output[j] = 0.0;

    int npx = 0;
    int i0 = 0;
    int p = 12;
    int64_t remaining = (int64_t) total_output_length;

    for (; remaining >= (int64_t) blocksize; remaining -= (int64_t) blocksize) {
        uint32_t nbytes = read_be32((const uint8_t *) compressed + p);
        int ret = bslz4_decompress(codec, compressed + p + 4, (int) nbytes,
                                    (char *) raw, (int) blocksize);
        p += (int) nbytes + 4;
        if (BSLZ4_UNLIKELY(ret != (int) blocksize)) return ERR_DECOMPRESS;

        if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, blocksize / NB, NB) < 0))
            return ERR_UNTRANSPOSE;

        for (size_t j = 0; j < blocksize / NB; j++) {
            T px = block[j];
            if (BSLZ4_UNLIKELY(px * mask[j + i0] > 0)) {
                for (uint32_t k = indptr[j + i0]; k < indptr[j + i0 + 1]; k++) {
                    output[indices[k]] += (double) data[k] * (double) px;
                }
                if (BSLZ4_UNLIKELY(px > cut)) {
                    *(outpx++) = px;
                    *(output_adr++) = (uint32_t) (j + i0);
                    npx++;
                }
            }
        }
        i0 += (int) (blocksize / NB);
    }

    size_t tail_block = (8 * NB) * ((size_t) remaining / (8 * NB));
    if (tail_block > 0) {
        uint32_t nbytes = read_be32((const uint8_t *) compressed + p);
        int ret = bslz4_decompress(codec, compressed + p + 4, (int) nbytes,
                                    (char *) raw, (int) tail_block);
        p += (int) nbytes + 4;
        if (BSLZ4_UNLIKELY(ret != (int) tail_block)) return ERR_DECOMPRESS;
        if (BSLZ4_UNLIKELY(Untranspose(block, raw, scratch, tail_block / NB, NB) < 0))
            return ERR_UNTRANSPOSE;
    }
    remaining -= (int64_t) tail_block;
    if (remaining > 0) {
        memcpy(&block[tail_block / NB], compressed + compressed_length - remaining, (size_t) remaining);
    }
    for (size_t j = 0; j < (size_t(remaining) + tail_block) / NB; j++) {
        T px = block[j];
        if (BSLZ4_UNLIKELY(px * mask[j + i0] > 0)) {
            for (uint32_t k = indptr[j + i0]; k < indptr[j + i0 + 1]; k++) {
                output[indices[k]] += (double) data[k] * (double) px;
            }
            if (BSLZ4_UNLIKELY(px > cut)) {
                *(outpx++) = px;
                *(output_adr++) = (uint32_t) (j + i0);
                npx++;
            }
        }
    }
    return npx;
}

} /* namespace bslz4 */
