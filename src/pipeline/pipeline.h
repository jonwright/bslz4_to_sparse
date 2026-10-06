#ifndef BSLZ4_PIPELINE_H
#define BSLZ4_PIPELINE_H
/*
 * The pipeline: one uint16 value per step, chosen per object (never global).
 *
 *   pipeline[BSLZ4_STEP_DECODE]       compressed block -> bit-planes
 *   pipeline[BSLZ4_STEP_MASK]         how the pixel mask is applied
 *   pipeline[BSLZ4_STEP_UNTRANSPOSE]  bit-planes -> pixel block (or, for the
 *                                     low-planes values, straight to the
 *                                     pixel list for u16 blocks < 256)
 *   pipeline[BSLZ4_STEP_COLLECT]      pixel block -> list of (index, value)
 *   pipeline[BSLZ4_STEP_ROUTE]        matrix objects: per block, dot over
 *                                     the block (dense) or the list (sparse)
 *   pipeline[BSLZ4_STEP_DOT]          matrix objects: the matrix layout
 *
 * Within each step 0 means "auto" (src/_pipeline.py resolves it before C
 * sees the array; C rejects a 0) and the values are numbered best first, by
 * measurement (notes_pipeline_cases.md).  A value that needs an instruction
 * set is its own value (collect avx512cs, untranspose lowplanes-vbmi, ...)
 * and is rejected where the CPU or build lacks it -- never a silent
 * fallback.  The names and the checks are mirrored in src/_pipeline.py.
 * route and dot are 0 (unused) for a plain sparsify.
 */

#include <stddef.h>
#include <stdint.h>

#include "common.h"

enum {
    BSLZ4_STEP_DECODE = 0,
    BSLZ4_STEP_MASK = 1,
    BSLZ4_STEP_UNTRANSPOSE = 2,
    BSLZ4_STEP_COLLECT = 3,
    BSLZ4_STEP_ROUTE = 4,
    BSLZ4_STEP_DOT = 5,
    BSLZ4_NSTEPS = 6
};

/* decode: the codec is a property of the data (lz4 or zstd); for lz4 the
 * decoder is a choice */
enum {
    BSLZ4_DECODE_LZ4_BAND = 1,    /* zero-aware decoder below 8x and above 24x, stock in between */
    BSLZ4_DECODE_LZ4_ZERO = 2,    /* zero-aware decoder (src/steps/lz4zero.h) for every block */
    BSLZ4_DECODE_LZ4_STOCK = 3,   /* LZ4_decompress_safe */
    BSLZ4_DECODE_ZSTD = 4,
    BSLZ4_DECODE_N = 5
};

enum {
    BSLZ4_MASK_PIXEL = 1,         /* tested per pixel when collecting */
    BSLZ4_MASK_PLANES = 2,        /* ANDed into the bit-planes after decode */
    BSLZ4_MASK_NONE = 3,          /* every pixel valid: no mask tests */
    BSLZ4_MASK_N = 4
};

/* untranspose: the low-planes values handle u16 blocks whose high byte-planes
 * are zero (every value < 256) with half the bit transpose, and fused with
 * the collect where the block itself is not needed; other blocks take kcb */
enum {
    BSLZ4_UNTRANSPOSE_LOWPLANES_VBMI = 1,   /* AVX-512 VBMI/VBMI2 + GFNI */
    BSLZ4_UNTRANSPOSE_LOWPLANES_AVX2 = 2,
    BSLZ4_UNTRANSPOSE_LOWPLANES_VSX = 3,    /* POWER8+ */
    BSLZ4_UNTRANSPOSE_LOWPLANES_C = 4,      /* portable C over kcb */
    BSLZ4_UNTRANSPOSE_KCB = 5,              /* kcb (dispatches SSE2/AVX2/AVX-512 itself) */
    BSLZ4_UNTRANSPOSE_NEON = 6,             /* bitshuffle's NEON kernels */
    BSLZ4_UNTRANSPOSE_SCALAR = 7,           /* bitshuffle's scalar kernels */
    BSLZ4_UNTRANSPOSE_N = 8
};

/* collect: the SIMD values are u16/u32 only */
enum {
    BSLZ4_COLLECT_AVX512CS = 1,
    BSLZ4_COLLECT_AVX2CS = 2,
    BSLZ4_COLLECT_VSX = 3,
    BSLZ4_COLLECT_NEON = 4,
    BSLZ4_COLLECT_SSE2 = 5,
    BSLZ4_COLLECT_SCALAR = 6,
    BSLZ4_COLLECT_N = 7
};

enum {
    BSLZ4_ROUTE_RATIO_RULE = 1,   /* sparse for blocks compressed more than BSLZ4_ROUTE_RATIO */
    BSLZ4_ROUTE_DENSE = 2,
    BSLZ4_ROUTE_SPARSE = 3,
    BSLZ4_ROUTE_N = 4
};
#define BSLZ4_ROUTE_RATIO 8

/* dot: each value names a layout of the matrix arrays (src/_matrix.py) */
enum {
    BSLZ4_DOT_CSC = 1,
    BSLZ4_DOT_PADDED = 2,
    BSLZ4_DOT_PADDED_AVX2 = 3,
    BSLZ4_DOT_CSC_RUN = 4,
    BSLZ4_DOT_CSC_NOSPLIT = 5,
    BSLZ4_DOT_BSBCSR = 6,
    BSLZ4_DOT_BSBCSR_NOSPLIT = 7,
    BSLZ4_DOT_CSC_NOSPLIT_MOMENT = 8,
    BSLZ4_DOT_CSC_PERMUTE = 9,
    BSLZ4_DOT_CSC_INT = 10,
    BSLZ4_DOT_CSC_RUN_INT = 11,
    BSLZ4_DOT_CSC_RUN_INT16 = 12,
    BSLZ4_DOT_N = 13
};

/* the C entry each dot is reached through */
enum { BSLZ4_LAYOUT_CSC = 0, BSLZ4_LAYOUT_PADDED = 1, BSLZ4_LAYOUT_BSBCSR = 2 };

/* counters: one slot per value of each step (test instrumentation) */
#define BSLZ4_VALUE_SLOTS 16

typedef struct {
    const void *data;
    const uint32_t *indices;
    const uint32_t *indptr;
} bslz4_mat_csc;

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

/* The resolved pipeline of one call: the step values (all non-zero but route
 * and dot of a plain sparsify) and the pixel type. */
typedef struct bslz4_pipe {
    uint16_t step[BSLZ4_NSTEPS];
    int dtype;                   /* pixel dtype index 0..9: u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 */
    size_t elem_size;            /* sizeof(pixel dtype) */
} bslz4_pipe;

/* One block (or the tail) handed to the dtype-generic collect / dot kernels
 * (src/steps/kernels.cpp). */
typedef struct bslz4_work {
    int    dtype;                    /* pixel dtype index 0..9 */
    int    route;                    /* 0 = dense (dot over the block), 1 = sparse (over the list) */
    int    collect;                  /* BSLZ4_COLLECT_* */
    int    dot;                      /* BSLZ4_DOT_* */
    int    no_mask;                  /* skip all mask tests */
    size_t n;                        /* pixels in this block/tail */
    size_t i0;                       /* global pixel offset of block start */
    int64_t threshold;
    const void   *BSLZ4_RESTRICT block;    /* decoded + untransposed block */
    const uint8_t *BSLZ4_RESTRICT mask;
    void          *BSLZ4_RESTRICT out_vals;/* sparse output values */
    uint32_t      *BSLZ4_RESTRICT out_adr;
    const void    *BSLZ4_RESTRICT mat;     /* layout descriptor, per dot */
    void          *BSLZ4_RESTRICT powder;  /* double*, int64_t* or the pixel type, per dot */
    int           nout;
    uint32_t      *BSLZ4_RESTRICT tidx;    /* sparse-route list scratch */
    void          *BSLZ4_RESTRICT tval;
    /* 1: the driver already listed this block's non-zero (unmasked) pixels
     * in tval/tidx (pre_nz entries) and `block` is not filled; only set for
     * the sparse route, which then reads just the list. */
    int           precompacted;
    int           pre_nz;
} bslz4_work;

#ifdef __cplusplus
extern "C" {
#endif

/* ---- registry (src/pipeline/registry.c) ---- */

/* 1 available, 0 known but this CPU/build lacks what it needs, -1 unknown
 * (0 included: auto is resolved in Python) */
int bslz4_step_available(int step, int value);

/* Check a pipeline for the dtype and entry point (layout; -1 for the plain
 * sparsify, which needs route and dot 0) and fill p.  Returns 0 or
 * BSLZ4_ERR_BAD_PIPELINE / BSLZ4_ERR_UNAVAILABLE / BSLZ4_ERR_DTYPE /
 * BSLZ4_ERR_BAD_LAYOUT. */
int bslz4_resolve(int dtype, const uint16_t *pipeline, int layout, bslz4_pipe *BSLZ4_RESTRICT p);

/* dot introspection: layout (BSLZ4_LAYOUT_*, -1 unknown); the powder element
 * size in bytes (8: double or int64; 0: the pixel dtype's size, for
 * csc-permute) */
int bslz4_dot_layout(int dot);
int bslz4_dot_out_size(int dot);

/* test instrumentation: blocks run per (step, value) */
void bslz4_reset_counters(void);
int  bslz4_read_counters(uint64_t *BSLZ4_RESTRICT out, int n);   /* flattened [step][value] */

/* ---- dtype-generic kernels (src/steps/kernels.cpp) ---- */
int bslz4_sparse_dispatch(const bslz4_work *BSLZ4_RESTRICT w);
int bslz4_sparse_dot_dispatch(const bslz4_work *BSLZ4_RESTRICT w);

/* ---- the frame/block loop (src/pipeline/driver.c) ---- */

/* kind 0: plain sparsify (mat, powder, nout unused); otherwise mat is the
 * layout descriptor for p->step[BSLZ4_STEP_DOT] */
int bslz4_driver_run(const int64_t *BSLZ4_RESTRICT compressed_ptrs,
                     const int32_t *BSLZ4_RESTRICT compressed_lengths, int nframes,
                     const uint8_t *BSLZ4_RESTRICT mask, int NIJ,
                     void *BSLZ4_RESTRICT outpx, uint32_t *BSLZ4_RESTRICT output_adr,
                     int32_t *BSLZ4_RESTRICT npx_out, int threshold,
                     void *BSLZ4_RESTRICT powder, int nout, const void *BSLZ4_RESTRICT mat,
                     uint8_t *BSLZ4_RESTRICT workspace, size_t workspace_len,
                     int64_t *BSLZ4_RESTRICT cursors, const bslz4_pipe *BSLZ4_RESTRICT p);

#ifdef __cplusplus
}
#endif

/* A tiny inlined store (no call, no bounds check: callers pass a valid
 * (step, value)); the array lives in registry.c. */
extern uint64_t bslz4_counters[BSLZ4_NSTEPS][BSLZ4_VALUE_SLOTS];
static inline void bslz4_counters_bump(int step, int value) {
    bslz4_counters[(unsigned) step][(unsigned) value]++;
}

#endif /* BSLZ4_PIPELINE_H */
