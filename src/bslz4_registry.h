#ifndef BSLZ4_REGISTRY_H
#define BSLZ4_REGISTRY_H
/*
 * Types shared by the C driver and the generic C++ kernels
 * (kernels_generic.cpp), plus the driver entry points themselves.
 *
 * The driver is dtype-agnostic C and owns the block/tail loop once; the only
 * dtype-generic work (the per-block sparse / sparse_dot call) is reached
 * through a thin dtype-dispatching switch in the C++ TU.  That switch calls
 * a small, per-dtype noinline kernel (so the switch body stays small and
 * each kernel is independently register-allocated), and every kernel takes
 * a single pointer to a bslz4_work context rather than a dozen scalar args.
 */

#include <stddef.h>
#include <stdint.h>

#include "bslz4_common.h"

typedef int (*bslz4_decompress_fn)(int codec, const char *BSLZ4_RESTRICT src, int compressed_size,
                                   char *BSLZ4_RESTRICT dst, int dst_capacity);

typedef int64_t (*bslz4_untranspose_fn)(void *BSLZ4_RESTRICT out, const void *BSLZ4_RESTRICT in,
                                         void *BSLZ4_RESTRICT scratch,
                                         size_t size, size_t elem_size);

/* Per-block working state for the dtype-generic inner call.  The driver
 * fills one of these per block and passes a single pointer; the C++ kernel
 * unpacks it once into locals (no repeated struct-pointer indirection, no
 * ABI stack spill of a dozen scalar arguments).  The CSC-only fields are
 * ignored by the plain sparse path. */
typedef struct bslz4_work {
    int    dtype;                    /* pixel dtype index 0..9 */
    int    route;                    /* 0=dense, 1=sparse (CSC only) */
    int    collect_id;               /* resolved collect tier 0..5 */
    int    dot_id;                   /* resolved dot impl: 0=csc, 1=csc-fused */
    size_t n;                        /* pixels in this block/tail */
    size_t i0;                       /* global pixel offset of block start */
    int64_t threshold;
    const void   *BSLZ4_RESTRICT block;    /* decoded + transposed block */
    const uint8_t *BSLZ4_RESTRICT mask;
    void          *BSLZ4_RESTRICT out_vals;/* sparse output values */
    uint32_t      *BSLZ4_RESTRICT out_adr;
    double        *BSLZ4_RESTRICT powder;  /* CSC output (nout doubles) */
    int           nout;
    const float   *BSLZ4_RESTRICT data;    /* CSC weights */
    const uint32_t *BSLZ4_RESTRICT indices;
    const uint32_t *BSLZ4_RESTRICT indptr;
    uint32_t      *BSLZ4_RESTRICT tidx;    /* sparse-route compaction scratch */
    void          *BSLZ4_RESTRICT tval;
} bslz4_work;

/* What the dtype-agnostic driver needs to decode: element width, dtype
 * index and the per-block, non-generic decompress / untranspose
 * operations.  The per-block dtype-generic work is reached through a thin
 * dtype-dispatching switch (kernels_generic.cpp) that calls small,
 * per-dtype noinline kernels, each taking a bslz4_work*. */
typedef struct bslz4_stage {
    size_t elem_size;            /* sizeof(pixel dtype) */
    int dtype;                   /* pixel dtype index 0..9 */
    int collect_id;              /* resolved collect tier 0..5 */
    int dot_id;                  /* resolved dot impl 0.. */
    int untranspose_id;          /* resolved untranspose backend 0..3 */
    bslz4_decompress_fn decompress;
    bslz4_untranspose_fn untranspose;
} bslz4_stage;

#ifdef __cplusplus
extern "C" {
#endif

/* Thin dtype-dispatching switch (kernels_generic.cpp).  Each case is a
 * direct call to a small per-dtype noinline kernel, so the switch body
 * stays small and each kernel is independently register-allocated.  A
 * single bslz4_work* carries all per-block state; the kernel unpacks the
 * fields once into locals. */
int bslz4_sparse_dispatch(const bslz4_work *BSLZ4_RESTRICT w);
int bslz4_sparse_dot_dispatch(const bslz4_work *BSLZ4_RESTRICT w);

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

/* Counter layout (flat read via bslz4_read_counters). */
#define BSLZ4_NSTAGES  4
#define BSLZ4_ID_SLOTS 16

/* Resolve dtype + stages (a uint16 array, one entry per stage/option, see
 * BSLZ4_STAGES_N in bslz4_common.h) into a stage table.  Validates, in
 * order:
 *   dtype in [0,9]                     -> BSLZ4_ERR_DTYPE
 *   stage ids known + available        -> BSLZ4_ERR_UNAVAILABLE / BSLZ4_ERR_BAD_PIPELINE
 *   no unknown option bits             -> BSLZ4_ERR_BAD_PIPELINE
 * Returns 0 on success. */
int bslz4_resolve(int dtype, const uint16_t *stages, bslz4_stage *BSLZ4_RESTRICT st);

/* availability: 1 available, 0 known but not usable here, -1 unknown id */
int bslz4_impl_available(int stage, int id);

/* optional test instrumentation */
void bslz4_reset_counters(void);
int  bslz4_read_counters(uint64_t *BSLZ4_RESTRICT out, int n);   /* flattened [stage][impl] */

#ifdef __cplusplus
}
#endif

/* A tiny inlined store (no call, no bounds check -- callers are internal and
 * pass a valid (stage, id)); the array itself lives in bslz4_registry.c. */
extern uint64_t bslz4_counters[BSLZ4_NSTAGES][BSLZ4_ID_SLOTS];
static inline void bslz4_counters_bump(int stage, int id) {
    bslz4_counters[(unsigned) stage][(unsigned) id]++;
}

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Native entry points (bslz4_to_sparse.c) ---- */

int bslz4_sparsify(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes,
                   const uint8_t *mask, int NIJ,
                   void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                   uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                   int dtype, const uint16_t *stages);

int bslz4_sparsify_and_dot(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                           int nframes, const uint8_t *mask, int NIJ,
                           void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                           double *powder, int nout,
                           const float *weights, const uint32_t *indices, const uint32_t *indptr,
                           double route_threshold,
                           uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                           int dtype, const uint16_t *stages);

int bslz4_offsets_to_pointers(const char *base, size_t base_len, int64_t *offsets,
                              const int32_t *lengths, int nframes);

void bslz4_note_chunk(const char *chunk, size_t chunk_len, int index,
                      int64_t *pointers, int32_t *lengths);

#ifdef __cplusplus
}
#endif

#endif /* BSLZ4_REGISTRY_H */
