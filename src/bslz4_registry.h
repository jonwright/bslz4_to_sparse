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

/* Matrix layouts.  Each dot id maps to one layout (see the dots[] table in
 * bslz4_registry.c).  The driver and the work struct stay layout-agnostic:
 * bslz4_work carries an opaque `mat` descriptor and an opaque `powder`
 * buffer, and the kernel interprets them by dot_id. */
enum { BSLZ4_LAYOUT_CSC = 0, BSLZ4_LAYOUT_PADDED = 1, BSLZ4_LAYOUT_BSBCSR = 2 };

/* CSC matrix.  data is float* (csc / csc-fused / ...) or uint32_t* for the
 * later fixed-point dot; the element width is implied by the dot id.  The
 * dot id also says what indices holds: one bin per entry (csc, csc-fused),
 * the first bin per pixel (csc-run: start + length, length from indptr), or
 * the one bin per pixel with weight 1 (csc-nosplit: data/indptr not read),
 * or the first of a pixel's two bins b, b+1 with weights 1 and data[p]
 * (csc-nosplit-moment: data has one entry per pixel, indptr not read).
 * See the column walkers in kernels_generic.cpp. */
typedef struct {
    const void *data;
    const uint32_t *indices;
    const uint32_t *indptr;
} bslz4_mat_csc;

/* Padded CSC: one row per pixel (implicit, listed==0) or per *listed* pixel
 * (listed==1).  A row has base[r] (first output bin) and weights[r*width+k]
 * for the consecutive bins base[r]..base[r]+width-1, zero padded.  row_ptr
 * (nblocks+1) gives the rows of each block (replaces the pre-refactor rcur
 * cursor): rows of block b are row_ptr[b]..row_ptr[b+1], i.e. the rows whose
 * pixel lies in [b*block_elems, (b+1)*block_elems).  rowmap (listed only)
 * maps a pixel to its row, or -1 for a pixel with no row.  Built from the
 * mask-folded matrix, so masked pixels have zero weights or no row. */
typedef struct {
    const int32_t *base;
    const float *weights;             /* nrows x width, row-major, zero padded */
    const int32_t *pixels;            /* listed only, else NULL */
    const int32_t *rowmap;            /* listed only, else NULL */
    const int32_t *row_ptr;           /* nblocks+1 */
    size_t row_ptr_n;                 /* entries in row_ptr (checked >= nblocks+1) */
    int width;
    int listed;
    size_t block_elems;
} bslz4_mat_padded;

/* Bit-shuffle-block-sized CSR: for each decode block, the active (non-empty)
 * bins and, per bin, the (in-block pixel index, weight) entries.  bin_ptr
 * has nactive+1 entries, nactive = total active (bin,block) pairs.  The CSC
 * is kept for the sparse route.  Both are built from the mask-folded matrix
 * (masked pixels have no entries). */
typedef struct {
    const uint32_t *blk_ptr;          /* nblocks+1 -> into bins/bin_ptr */
    size_t blk_ptr_n;                 /* entries in blk_ptr (checked >= nblocks+1) */
    const uint32_t *bins;             /* active bin id per (block) entry */
    const uint32_t *bin_ptr;          /* nactive+1 -> into idx/data */
    const uint16_t *idx;              /* in-block pixel index (block_elems <= 8192) */
    const float *data;                /* weights, as in the (mask-folded) CSC */
    bslz4_mat_csc csc;                /* for the sparse route */
    size_t block_elems;
} bslz4_mat_bsbcsr;

/* Per-block working state for the dtype-generic inner call.  The driver
 * fills one of these per block and passes a single pointer; the C++ kernel
 * unpacks it once into locals (no repeated struct-pointer indirection, no
 * ABI stack spill of a dozen scalar arguments).  `mat` is the layout
 * descriptor (interpreted per dot_id) and `powder` the output buffer
 * (double* for the float layouts, int64_t* for fixed-point).  Neither is
 * used by the plain sparse path. */
typedef struct bslz4_work {
    int    dtype;                    /* pixel dtype index 0..9 */
    int    route;                    /* 0=dense, 1=sparse */
    int    collect_id;               /* resolved collect tier 0..5 */
    int    dot_id;                   /* resolved dot impl (bslz4_dots[] in bslz4_registry.c) */
    int    no_mask;                  /* BSLZ4_OPT_NO_MASK: skip all mask checks */
    size_t n;                        /* pixels in this block/tail */
    size_t i0;                       /* global pixel offset of block start */
    int64_t threshold;
    const void   *BSLZ4_RESTRICT block;    /* decoded + transposed block */
    const uint8_t *BSLZ4_RESTRICT mask;
    void          *BSLZ4_RESTRICT out_vals;/* sparse output values */
    uint32_t      *BSLZ4_RESTRICT out_adr;
    const void    *BSLZ4_RESTRICT mat;     /* layout descriptor, per dot_id */
    void          *BSLZ4_RESTRICT powder;  /* double* or int64_t*, per dot_id */
    int           nout;
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
    uint16_t options;            /* BSLZ4_OPT_* bitmask */
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
                                  void *BSLZ4_RESTRICT powder, int nout,
                                  const void *BSLZ4_RESTRICT data,
                                  const void *BSLZ4_RESTRICT indices,
                                  const uint32_t *BSLZ4_RESTRICT indptr,
                                  double dense_sparse_x,
                                  uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                  int64_t *BSLZ4_RESTRICT cursors,
                                  const bslz4_stage *BSLZ4_RESTRICT st);

int bslz4_driver_sparsify_and_dot_padded(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                         const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                         int nframes, int codec,
                                         const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                         void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                         int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                         double *BSLZ4_RESTRICT powder, int nout,
                                         const int32_t *BSLZ4_RESTRICT base,
                                         const float *BSLZ4_RESTRICT weights,
                                         const int32_t *BSLZ4_RESTRICT pixels,
                                         const int32_t *BSLZ4_RESTRICT rowmap,
                                         const int32_t *BSLZ4_RESTRICT row_ptr, int nrow_ptr,
                                         int width, int listed, size_t block_elems,
                                         double dense_sparse_x,
                                         uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                                         int64_t *BSLZ4_RESTRICT cursors,
                                         const bslz4_stage *BSLZ4_RESTRICT st);

int bslz4_driver_sparsify_and_dot_bsbcsr(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                                         const int32_t *BSLZ4_RESTRICT compressed_lengths,
                                         int nframes, int codec,
                                         const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                                         void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                                         int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                                         double *BSLZ4_RESTRICT powder, int nout,
                                         const uint32_t *BSLZ4_RESTRICT blk_ptr, int nblk_ptr,
                                         const uint32_t *BSLZ4_RESTRICT bins,
                                         const uint32_t *BSLZ4_RESTRICT bin_ptr,
                                         const uint16_t *BSLZ4_RESTRICT idx,
                                         const float *BSLZ4_RESTRICT data,
                                         const void *BSLZ4_RESTRICT csc_data,
                                         const uint32_t *BSLZ4_RESTRICT csc_indices,
                                         const uint32_t *BSLZ4_RESTRICT csc_indptr,
                                         size_t block_elems,
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
 *   if expected_layout >= 0 and the dot id's layout mismatches
 *                                      -> BSLZ4_ERR_BAD_LAYOUT
 *   no unknown option bits             -> BSLZ4_ERR_BAD_PIPELINE
 * Pass expected_layout = -1 to skip the layout cross-check (used by the
 * plain sparsify entry, which has no dot layout).
 * Returns 0 on success. */
int bslz4_resolve(int dtype, const uint16_t *stages, int expected_layout,
                  bslz4_stage *BSLZ4_RESTRICT st);

/* availability: 1 available, 0 known but not usable here, -1 unknown id */
int bslz4_impl_available(int stage, int id);

/* dot-id introspection: layout id (one of BSLZ4_LAYOUT_*), or -1 unknown;
 * and the output element size in bytes of the powder buffer (8: a double,
 * or an int64 for the fixed-point dots; 0: the pixel dtype's size, for a
 * packed output such as csc-permute). */
int bslz4_dot_layout(int id);
int bslz4_dot_stream(int id);   /* 1: indices is a headed byte stream (BSLZ4_STREAM_MAGIC) */
int bslz4_dot_out_size(int id);

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
                           void *powder, int nout,
                           const void *weights, const void *indices, const uint32_t *indptr,
                           double route_threshold,
                           uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                           int dtype, const uint16_t *stages);

int bslz4_sparsify_and_dot_padded(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                                  int nframes, const uint8_t *mask, int NIJ,
                                  void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                                  double *powder, int nout,
                                  const int32_t *base, const float *weights,
                                  const int32_t *pixels, const int32_t *rowmap,
                                  const int32_t *row_ptr, int nrow_ptr, int width, int listed,
                                  size_t block_elems, double route_threshold,
                                  uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                                  int dtype, const uint16_t *stages);

int bslz4_sparsify_and_dot_bsbcsr(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                                  int nframes, const uint8_t *mask, int NIJ,
                                  void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                                  double *powder, int nout,
                                  const uint32_t *blk_ptr, int nblk_ptr, const uint32_t *bins,
                                  const uint32_t *bin_ptr, const uint16_t *idx,
                                  const float *data,
                                  const float *csc_data, const uint32_t *csc_indices,
                                  const uint32_t *csc_indptr,
                                  size_t block_elems, double route_threshold,
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
