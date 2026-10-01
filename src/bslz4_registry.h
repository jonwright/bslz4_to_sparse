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

#include "bslz4_common.h"

typedef int (*bslz4_decompress_fn)(int codec, const char *BSLZ4_RESTRICT src, int compressed_size,
                                   char *BSLZ4_RESTRICT dst, int dst_capacity);

typedef int64_t (*bslz4_untranspose_fn)(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in,
                                         void *BSLZ4_RESTRICT scratch,
                                         size_t size, size_t elem_size);

/* Per-block plain-sparse call: mask>0 & val>threshold, compacted. */
typedef int (*bslz4_sparse_fn)(const void *BSLZ4_RESTRICT block, size_t n,
                                const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                                int64_t threshold, void *BSLZ4_RESTRICT out_vals,
                                uint32_t *BSLZ4_RESTRICT out_adr);

/* Per-block CSC call: may route dense or sparse (see kernels_generic.cpp). */
typedef int (*bslz4_sparse_dot_fn)(const void *BSLZ4_RESTRICT block, size_t n,
                                    size_t decoded_bytes, size_t nbytes,
                                    const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                                    int64_t threshold, void *BSLZ4_RESTRICT out_vals,
                                    uint32_t *BSLZ4_RESTRICT out_adr,
                                    double *BSLZ4_RESTRICT out, int nout,
                                    const float *BSLZ4_RESTRICT data,
                                    const uint32_t *BSLZ4_RESTRICT indices,
                                    const uint32_t *BSLZ4_RESTRICT indptr,
                                    double dense_sparse_x, uint32_t *BSLZ4_RESTRICT tidx,
                                    void *BSLZ4_RESTRICT tval);

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

int bslz4_driver_sparsify(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                          const int32_t *BSLZ4_RESTRICT compressed_lengths,
                          int nframes, int codec,
                          const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                          void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                          int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                          uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                          int64_t *BSLZ4_RESTRICT cursors,
                          const bslz4_stage *BSLZ4_RESTRICT st);

int bslz4_driver_sparsify_and_dot(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                  const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                  int nframes, int codec,
                                  const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                  void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                  int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                  double *BSLZ4_RESTRICT powder, int nout,
                                  const float *BSLZ4_RESTRICT data,
                                  const uint32_t *BSLZ4_RESTRICT indices,
                                  const uint32_t *BSLZ4_RESTRICT indptr,
                                  double dense_sparse_x,
                                  uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                  int64_t *BSLZ4_RESTRICT cursors,
                                  const bslz4_stage *BSLZ4_RESTRICT st);

/* ---- Registry / resolve (bslz4_registry.c) ---- */

/* Stage ids for impl_available / resolve.  Stable ABI (plan.md section 5). */
#define BSLZ4_STAGE_DECOMPRESS  0
#define BSLZ4_STAGE_UNTRANSPOSE 1
#define BSLZ4_STAGE_COLLECT     2
#define BSLZ4_STAGE_DOT         3

/* Resolve dtype + pipeline into a stage table.  Validates, in order:
 *   dtype in [0,9]                     -> BSLZ4_ERR_DTYPE
 *   stage ids known + available        -> BSLZ4_ERR_UNAVAILABLE / BSLZ4_ERR_BAD_PIPELINE
 *   no unknown option bits             -> BSLZ4_ERR_BAD_PIPELINE
 * Returns 0 on success. */
int bslz4_resolve(int dtype, uint64_t pipeline, bslz4_stage *BSLZ4_RESTRICT st);

/* availability: 1 available, 0 known but not usable here, -1 unknown id */
int bslz4_impl_available(int stage, int id);

/* optional test instrumentation */
void bslz4_reset_counters(void);
int  bslz4_read_counters(uint64_t *BSLZ4_RESTRICT out, int n);   /* flattened [stage][impl] */
void bslz4_counters_bump(int stage, int id);       /* internal */

/* ---- Native entry points (bslz4_to_sparse.c) ---- */

int bslz4_sparsify(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes,
                   const uint8_t *mask, int NIJ,
                   void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                   uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                   int dtype, uint64_t pipeline);

int bslz4_sparsify_and_dot(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                           int nframes, const uint8_t *mask, int NIJ,
                           void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                           double *powder, int nout,
                           const float *weights, const uint32_t *indices, const uint32_t *indptr,
                           double route_threshold,
                           uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                           int dtype, uint64_t pipeline);

int bslz4_offsets_to_pointers(const char *base, size_t base_len, int64_t *offsets,
                              const int32_t *lengths, int nframes);

void bslz4_note_chunk(const char *chunk, size_t chunk_len, int index,
                      int64_t *pointers, int32_t *lengths);

#ifdef __cplusplus
}
#endif

#endif /* BSLZ4_REGISTRY_H */
