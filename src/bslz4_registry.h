#ifndef BSLZ4_REGISTRY_H
#define BSLZ4_REGISTRY_H
/*
 * Types shared by the C driver and the generic C++ kernels
 * (kernels_generic.cpp), plus the driver entry points themselves.
 *
 * The driver is dtype-agnostic C and owns the block/tail loop once; the only
 * dtype-generic work (the per-block sparse / sparse_dot call) is a direct
 * call into a dtype-switching dispatch in the C++ TU, so the per-dtype
 * kernels can be inlined there (no function-pointer indirection).
 */

#include <stddef.h>
#include <stdint.h>

#include "bslz4_common.h"

typedef int (*bslz4_decompress_fn)(int codec, const char *BSLZ4_RESTRICT src, int compressed_size,
                                   char *BSLZ4_RESTRICT dst, int dst_capacity);

typedef int64_t (*bslz4_untranspose_fn)(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in,
                                         void *BSLZ4_RESTRICT scratch,
                                         size_t size, size_t elem_size);

/* The CSC matrix + powder destination, bundled so the per-block dispatch
 * takes one pointer instead of five flat arguments. */
typedef struct bslz4_csc {
    double *BSLZ4_RESTRICT out;        /* powder output (nout doubles) */
    int nout;
    const float *BSLZ4_RESTRICT data;  /* CSC weights */
    const uint32_t *BSLZ4_RESTRICT indices;
    const uint32_t *BSLZ4_RESTRICT indptr;
} bslz4_csc;

/* What the dtype-agnostic driver needs to decode: element width, dtype
 * index (for the C++ dispatch switch) and the per-block, non-generic
 * decompress / untranspose operations.  The per-block dtype-generic work is
 * a direct call into a dtype-dispatching switch rather than a function
 * pointer, so the compiler can inline the per-dtype kernels. */
typedef struct bslz4_stage {
    size_t elem_size;            /* sizeof(pixel dtype) */
    int dtype;                   /* pixel dtype index 0..9 */
    bslz4_decompress_fn decompress;
    bslz4_untranspose_fn untranspose;
} bslz4_stage;

#ifdef __cplusplus
extern "C" {
#endif

/* Direct-call dispatch (kernels_generic.cpp): route is 0=dense, 1=sparse.
 * The per-dtype handlers are static/inline in that TU, so the switch bodies
 * inline.  The CSC matrix comes bundled in *csc. */
int bslz4_sparse_dispatch(int dtype,
                          const void *BSLZ4_RESTRICT block, size_t n,
                          const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                          int64_t threshold, void *BSLZ4_RESTRICT out_vals,
                          uint32_t *BSLZ4_RESTRICT out_adr);

int bslz4_sparse_dot_dispatch(int dtype, int route,
                              const void *BSLZ4_RESTRICT block, size_t n,
                              const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                              int64_t threshold, void *BSLZ4_RESTRICT out_vals,
                              uint32_t *BSLZ4_RESTRICT out_adr,
                              const bslz4_csc *BSLZ4_RESTRICT csc,
                              uint32_t *BSLZ4_RESTRICT tidx,
                              void *BSLZ4_RESTRICT tval);

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
