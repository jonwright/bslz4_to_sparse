/*
 * registry.c -- what each pipeline step value needs, and the check of a
 * pipeline before a call.  The names and the same checks, with the error
 * messages, are in src/_pipeline.py; this is the C side, so a pipeline built
 * by hand still cannot run an instruction set the CPU lacks.
 */

#include "pipeline.h"
#include "../steps/caps.h"
#include "../steps/untranspose.h"

#include <string.h>

/* driver.c (from src/steps/lowplanes.h) */
extern int bslz4_lowplanes_available(int value);

uint64_t bslz4_counters[BSLZ4_NSTEPS][BSLZ4_VALUE_SLOTS];

typedef char bslz4_value_slots_cover_every_step
    [(BSLZ4_DECODE_N <= BSLZ4_VALUE_SLOTS && BSLZ4_MASK_N <= BSLZ4_VALUE_SLOTS &&
      BSLZ4_UNTRANSPOSE_N <= BSLZ4_VALUE_SLOTS && BSLZ4_COLLECT_N <= BSLZ4_VALUE_SLOTS &&
      BSLZ4_ROUTE_N <= BSLZ4_VALUE_SLOTS && BSLZ4_DOT_N <= BSLZ4_VALUE_SLOTS) ? 1 : -1];

/* dtype masks: bit d set => pixel dtype index d accepted */
#define BSLZ4_ALL_DTYPES 0x3FFu
#define BSLZ4_INT_DTYPES 0x0FFu                 /* u8 .. i64 */
#define BSLZ4_U16_U32 ((1u << 1) | (1u << 2))
#define BSLZ4_U16 (1u << 1)

/* Each dot: the C entry (layout) it is reached through, the pixel dtypes it
 * takes, the powder element size (bytes; 0 = the pixel dtype, a packed
 * output) and what it needs. */
typedef struct {
    int layout;
    uint32_t dtypes;
    int out_size;
    int (*available)(void);
} bslz4_dot_desc;

static int bslz4_always(void) { return 1; }

/*
 * Step "dot": each value is a layout of the pixel -> bin matrix (built from
 * a mask-folded CSC by src/_matrix.py; kernels in src/steps/dot_*.hpp), with
 * a dense and a sparse route.  Weights are float32 and the powder float64
 * unless stated.
 *
 *   BSLZ4_DOT_CSC  general CSC: any matrix, one (bin, weight) entry per
 *                      pixel/bin pair.
 *   BSLZ4_DOT_PADDED  every pixel one ascending run of consecutive bins, at
 *                      most 64 wide: per row a first bin and a fixed width
 *                      of weights, zero padded; no index read per entry.
 *                      Rows are per pixel, or only the pixels with entries
 *                      when under half have any (listed).
 *   BSLZ4_DOT_PADDED_AVX2  the padded layout with AVX2+FMA row kernels on
 *                      the dense route, for widths <= 8 (wider: scalar).
 *   BSLZ4_DOT_CSC_RUN  CSC where each pixel's bins are one ascending run of
 *                      consecutive bins: stores only the first bin per pixel
 *                      beside the weights; any width.
 *   BSLZ4_DOT_CSC_NOSPLIT  a histogram: at most one bin per pixel, weight
 *                      exactly 1; stores that bin per pixel, no weights.
 *   BSLZ4_DOT_BSBCSR  any matrix, for ones where most pixels have no entry
 *                      (rings): per decode block a CSR of the bins it
 *                      touches, so the dense route walks only those bins
 *                      with register accumulators; the sparse route uses
 *                      the nested CSC.
 *   BSLZ4_DOT_BSBCSR_NOSPLIT  bsb-csr for a histogram (one bin per pixel,
 *                      weight 1).
 *   BSLZ4_DOT_CSC_NOSPLIT_MOMENT  a histogram with a first moment: each
 *                      pixel reaches no bin or the pair b, b+1 with weights
 *                      exactly 1 and q; the output interleaves sum I and
 *                      sum qI.  Stores b and q per pixel.
 *   BSLZ4_DOT_CSC_PERMUTE  a permutation: each pixel its own bin, weight 1;
 *                      the pixel value is stored (not added) at its bin, so
 *                      the output is the re-ordered image, in the pixel
 *                      dtype rather than a float64 powder.
 *   BSLZ4_DOT_CSC_INT  general CSC with u32 fixed-point weights
 *                      round(w * 2**b) (unsigned, so no negative weights),
 *                      integer pixel dtypes only; exact int64 sums of
 *                      w * v, rescaled by 2**-b afterwards.
 *   BSLZ4_DOT_CSC_RUN_INT  csc-run's layout (first bin per pixel) with u32
 *                      fixed-point weights and an int64 powder, as csc-int.
 *   BSLZ4_DOT_CSC_RUN_INT16  as csc-run-int with u16 weights.
 *
 * The integer values choose b so every sum stays exact in int64 and give each
 * pixel's weights the rounding residual so they sum to round(sum w * 2**b).
 */
static const bslz4_dot_desc bslz4_dots[BSLZ4_DOT_N] = {
    /* 0 (auto)              */ {-1, 0, 0, 0},
    /* csc                   */ {BSLZ4_LAYOUT_CSC,    BSLZ4_ALL_DTYPES, 8, bslz4_always},
    /* padded                */ {BSLZ4_LAYOUT_PADDED, BSLZ4_ALL_DTYPES, 8, bslz4_always},
    /* padded-avx2           */ {BSLZ4_LAYOUT_PADDED, BSLZ4_ALL_DTYPES, 8, bslz4_available_avx2_padded},
    /* csc-run               */ {BSLZ4_LAYOUT_CSC,    BSLZ4_ALL_DTYPES, 8, bslz4_always},
    /* csc-nosplit           */ {BSLZ4_LAYOUT_CSC,    BSLZ4_ALL_DTYPES, 8, bslz4_always},
    /* bsb-csr               */ {BSLZ4_LAYOUT_BSBCSR, BSLZ4_ALL_DTYPES, 8, bslz4_always},
    /* bsb-csr-nosplit       */ {BSLZ4_LAYOUT_BSBCSR, BSLZ4_ALL_DTYPES, 8, bslz4_always},
    /* csc-nosplit-moment    */ {BSLZ4_LAYOUT_CSC,    BSLZ4_ALL_DTYPES, 8, bslz4_always},
    /* csc-permute           */ {BSLZ4_LAYOUT_CSC,    BSLZ4_ALL_DTYPES, 0, bslz4_always},
    /* csc-int               */ {BSLZ4_LAYOUT_CSC,    BSLZ4_INT_DTYPES, 8, bslz4_always},
    /* csc-run-int           */ {BSLZ4_LAYOUT_CSC,    BSLZ4_INT_DTYPES, 8, bslz4_always},
    /* csc-run-int16         */ {BSLZ4_LAYOUT_CSC,    BSLZ4_INT_DTYPES, 8, bslz4_always},
};

int bslz4_dot_layout(int dot) {
    if (dot <= 0 || dot >= BSLZ4_DOT_N) return -1;
    return bslz4_dots[dot].layout;
}

int bslz4_dot_out_size(int dot) {
    if (dot <= 0 || dot >= BSLZ4_DOT_N) return 0;
    return bslz4_dots[dot].out_size;
}

int bslz4_step_available(int step, int value) {
    switch (step) {
    case BSLZ4_STEP_DECODE:
        return (value > 0 && value < BSLZ4_DECODE_N) ? 1 : -1;
    case BSLZ4_STEP_MASK:
        return (value > 0 && value < BSLZ4_MASK_N) ? 1 : -1;
    case BSLZ4_STEP_UNTRANSPOSE:
        switch (value) {
        case BSLZ4_UNTRANSPOSE_LOWPLANES_VBMI:
        case BSLZ4_UNTRANSPOSE_LOWPLANES_AVX2:
        case BSLZ4_UNTRANSPOSE_LOWPLANES_VSX:
        case BSLZ4_UNTRANSPOSE_LOWPLANES_C:  return bslz4_lowplanes_available(value) ? 1 : 0;
        case BSLZ4_UNTRANSPOSE_KCB:          return 1;
        case BSLZ4_UNTRANSPOSE_NEON:         return BSLZ4_HAVE_BACKEND_NEON ? 1 : 0;
        case BSLZ4_UNTRANSPOSE_SCALAR:       return 1;
        default:                             return -1;
        }
    case BSLZ4_STEP_COLLECT:
        switch (value) {
        case BSLZ4_COLLECT_AVX512CS: return bslz4_available_avx512cs_collect() ? 1 : 0;
        case BSLZ4_COLLECT_AVX2CS:   return bslz4_available_avx2cs_collect() ? 1 : 0;
        case BSLZ4_COLLECT_VSX:      return bslz4_available_vsx_collect() ? 1 : 0;
        case BSLZ4_COLLECT_NEON:     return bslz4_available_neon_collect() ? 1 : 0;
        case BSLZ4_COLLECT_SSE2:     return bslz4_available_sse2_collect() ? 1 : 0;
        case BSLZ4_COLLECT_SCALAR:   return 1;
        default:                     return -1;
        }
    case BSLZ4_STEP_ROUTE:
        return (value > 0 && value < BSLZ4_ROUTE_N) ? 1 : -1;
    case BSLZ4_STEP_DOT:
        if (value <= 0 || value >= BSLZ4_DOT_N) return -1;
        return bslz4_dots[value].available() ? 1 : 0;
    default:
        return -1;
    }
}

static size_t bslz4_elem_size(int dtype) {
    static const unsigned char size[10] = {1, 2, 4, 8, 1, 2, 4, 8, 4, 8};
    return size[dtype];
}

int bslz4_resolve(int dtype, const uint16_t *pipeline, int layout, bslz4_pipe *BSLZ4_RESTRICT p) {
    if (dtype < 0 || dtype > 9) return BSLZ4_ERR_DTYPE;
    const int is_dot = layout >= 0;
    for (int s = 0; s < BSLZ4_NSTEPS; s++) {
        const int v = pipeline[s];
        if (!is_dot && (s == BSLZ4_STEP_ROUTE || s == BSLZ4_STEP_DOT)) {
            if (v != 0) return BSLZ4_ERR_BAD_PIPELINE;      /* a plain sparsify has no dot */
            continue;
        }
        const int a = bslz4_step_available(s, v);
        if (a < 0) return BSLZ4_ERR_BAD_PIPELINE;
        if (a == 0) return BSLZ4_ERR_UNAVAILABLE;
        p->step[s] = (uint16_t) v;
    }
    if (!is_dot) p->step[BSLZ4_STEP_ROUTE] = p->step[BSLZ4_STEP_DOT] = 0;
    const uint32_t bit = 1u << dtype;
    if (p->step[BSLZ4_STEP_UNTRANSPOSE] <= BSLZ4_UNTRANSPOSE_LOWPLANES_C && !(bit & BSLZ4_U16))
        return BSLZ4_ERR_DTYPE;
    if (p->step[BSLZ4_STEP_COLLECT] != BSLZ4_COLLECT_SCALAR && !(bit & BSLZ4_U16_U32))
        return BSLZ4_ERR_DTYPE;
    if (is_dot) {
        const bslz4_dot_desc *d = &bslz4_dots[p->step[BSLZ4_STEP_DOT]];
        if (d->layout != layout) return BSLZ4_ERR_BAD_LAYOUT;
        if (!(bit & d->dtypes)) return BSLZ4_ERR_DTYPE;
    }
    p->dtype = dtype;
    p->elem_size = bslz4_elem_size(dtype);
    return 0;
}

void bslz4_reset_counters(void) {
    memset(bslz4_counters, 0, sizeof(bslz4_counters));
}

int bslz4_read_counters(uint64_t *BSLZ4_RESTRICT out, int n) {
    int k = 0;
    for (int s = 0; s < BSLZ4_NSTEPS; s++)
        for (int i = 0; i < BSLZ4_VALUE_SLOTS && k < n; i++)
            out[k++] = bslz4_counters[s][i];
    return k;
}
