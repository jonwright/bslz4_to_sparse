#ifndef BSLZ4_REGISTRY_H
#define BSLZ4_REGISTRY_H
/*
 * Stage/inner function-pointer types shared by the C driver and the
 * generic C++ kernels (kernels_generic.cpp), plus the driver entry
 * points themselves.
 *
 * Phase 1 keeps the existing generated layer and native API unchanged:
 * the old C++ templates (bslz4_core.hpp) build a bslz4_stage and call
 * the driver, which owns the block/tail loop once.  The only genuinely
 * dtype-generic work (the per-block sparse / sparse_dot call) is reached
 * through the pointers below.
 */

#include <stddef.h>
#include <stdint.h>

typedef int (*bslz4_decompress_fn)(int codec, const char *src, int compressed_size,
                                   char *dst, int dst_capacity);

typedef int64_t (*bslz4_untranspose_fn)(void *out, const void *in, void *scratch,
                                         size_t size, size_t elem_size);

/* Per-block plain-sparse call: mask>0 & val>threshold, compacted. */
typedef int (*bslz4_sparse_fn)(const void *block, size_t n, const uint8_t *mask, size_t i0,
                                int64_t threshold, void *out_vals, uint32_t *out_adr);

/* Per-block CSC call: may route dense or sparse (see kernels_generic.cpp). */
typedef int (*bslz4_sparse_dot_fn)(const void *block, size_t n, size_t decoded_bytes,
                                    size_t nbytes, const uint8_t *mask, size_t i0,
                                    int64_t threshold, void *out_vals, uint32_t *out_adr,
                                    double *out, int nout, const float *data,
                                    const uint32_t *indices, const uint32_t *indptr,
                                    double dense_sparse_x, uint32_t *tidx, void *tval);

typedef struct bslz4_stage {
    size_t elem_size;            /* sizeof(pixel dtype) */
    bslz4_decompress_fn decompress;
    bslz4_untranspose_fn untranspose;
    bslz4_sparse_fn sparse;      /* used by the plain-sparse driver */
    bslz4_sparse_dot_fn sparse_dot; /* used by the CSC driver */
} bslz4_stage;

typedef struct bslz4_inner_entry {
    size_t elem_size;
    bslz4_sparse_fn sparse;
    bslz4_sparse_dot_fn sparse_dot;
} bslz4_inner_entry;

#ifdef __cplusplus
extern "C" {
#endif

/* Defined in kernels_generic.cpp; indexed by dtype 0..9
 * (u8,u16,u32,u64,i8,i16,i32,i64,f32,f64). */
extern const bslz4_inner_entry bslz4_inner_table[10];

int bslz4_driver_sparsify(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                          int nframes, int codec, const uint8_t *mask, int NIJ,
                          void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                          uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                          const bslz4_stage *st);

int bslz4_driver_sparsify_and_dot(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                                  int nframes, int codec, const uint8_t *mask, int NIJ,
                                  void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                                  double *powder, int nout, const float *data,
                                  const uint32_t *indices, const uint32_t *indptr,
                                  double dense_sparse_x,
                                  uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                                  const bslz4_stage *st);

#ifdef __cplusplus
}
#endif

#endif /* BSLZ4_REGISTRY_H */
