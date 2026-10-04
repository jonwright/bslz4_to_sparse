/*
 * bslz4_registry.c -- stage tables, resolve, availability and counters.
 *
 * The stable stage ids, names and dtype masks are mirrored in Python
 * (src/__init__.py) and in a C header; C only answers availability and
 * resolves a (dtype, pipeline) into a usable bslz4_stage.
 */

#include "bslz4_common.h"
#include "bslz4_codec.h"
#include "bslz4_untranspose.h"
#include "bslz4_collect_caps.h"
#include "bslz4_registry.h"

#include <string.h>

/* C-linkage capability probes for the padded SIMD tiers (defined in
 * kernels_generic.cpp, using bslz4_padded.hpp). */
extern int bslz4_available_avx2_padded(void);
extern int bslz4_available_sse2_padded(void);
extern int bslz4_available_avx512_padded(void);

uint64_t bslz4_counters[BSLZ4_NSTAGES][BSLZ4_ID_SLOTS];

/* Dot implementations as a table.  Each entry says which layout the dot id
 * is, the dtype mask it accepts, the output element size (bytes) for the
 * powder buffer, and an availability predicate.  A new scheme is just a new
 * row here plus its kernel in kernels_generic.cpp; the work struct, driver
 * and the other entries stay untouched. */
typedef struct {
    int layout;
    uint32_t dtype_mask;
    int out_size;
    int (*available)(void);
} bslz4_dot_desc;

static int bslz4_dot_always(void) { return 1; }

static const bslz4_dot_desc bslz4_dots[] = {
    /* 0 csc          */ {BSLZ4_LAYOUT_CSC,    0x3FFu, 8, bslz4_dot_always},
    /* 1 csc-fused    */ {BSLZ4_LAYOUT_CSC,    0x3FFu, 8, bslz4_dot_always},
    /* 2 padded       */ {BSLZ4_LAYOUT_PADDED, 0x3FFu, 8, bslz4_dot_always},       /* scalar */
    /* 3 padded-sse2  */ {BSLZ4_LAYOUT_PADDED, 0x3FFu, 8, bslz4_available_sse2_padded},
    /* 4 padded-avx2  */ {BSLZ4_LAYOUT_PADDED, 0x3FFu, 8, bslz4_available_avx2_padded},
    /* 5 padd-avx512  */ {BSLZ4_LAYOUT_PADDED, 0x3FFu, 8, bslz4_available_avx512_padded},
    /* 6 bsb-csr      */ {BSLZ4_LAYOUT_BSBCSR, 0x3FFu, 8, bslz4_dot_always},
    /* 7 csc-run      */ {BSLZ4_LAYOUT_CSC,    0x3FFu, 8, bslz4_dot_always},  /* start + length */
    /* 8 csc-nosplit  */ {BSLZ4_LAYOUT_CSC,    0x3FFu, 8, bslz4_dot_always},  /* one bin, weight 1 */
    /* 9 bsb-csr-nosplit */ {BSLZ4_LAYOUT_BSBCSR, 0x3FFu, 8, bslz4_dot_always},  /* weight 1 */
    /* 10 csc-nosplit-moment */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},  /* bins b, b+1: 1, q */
    /* Experimental CSC-entry dots (bslz4_csc_variants.hpp); 20-22 are
     * integer weights with an int64 powder, integer pixels only. */
    /* 11 csc-nosplit-dump         */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 12 csc-nosplit-moment-dump  */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 13 csc-run-u16              */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 14 csc-nosplit-u16          */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 15 csc-run-delta            */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 16 csc-nosplit-delta        */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 17 csc-nosplit-walk         */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 18 csc-tile                 */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 19 csc-run-moment           */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 8, bslz4_dot_always},
    /* 20 csc-int                  */ {BSLZ4_LAYOUT_CSC, 0x0FFu, 8, bslz4_dot_always},
    /* 21 csc-run-int              */ {BSLZ4_LAYOUT_CSC, 0x0FFu, 8, bslz4_dot_always},
    /* 22 csc-run-int16            */ {BSLZ4_LAYOUT_CSC, 0x0FFu, 8, bslz4_dot_always},
    /* out_size 0: the output element is the pixel dtype (a packed output) */
    /* 23 csc-permute              */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 0, bslz4_dot_always},
    /* 24 csc-permute-runs         */ {BSLZ4_LAYOUT_CSC, 0x3FFu, 0, bslz4_dot_always},
};

static int bslz4_dot_count(void) {
    return (int) (sizeof(bslz4_dots) / sizeof(bslz4_dots[0]));
}

int bslz4_dot_layout(int id) {
    if (id < 0 || id >= bslz4_dot_count()) return -1;
    return bslz4_dots[id].layout;
}

/* 1 if the dot's `indices` is a byte stream with a header (bslz4_csc_variants.hpp) */
int bslz4_dot_stream(int id) {
    return id == 15 || id == 16 || id == 17 || id == 18 || id == 24;
}

int bslz4_dot_out_size(int id) {
    if (id < 0 || id >= bslz4_dot_count()) return 0;
    return bslz4_dots[id].out_size;
}

int bslz4_impl_available(int stage, int id) {
    switch (stage) {
    case BSLZ4_STAGE_DECOMPRESS:
        return (id == BSLZ4_CODEC_LZ4 || id == BSLZ4_CODEC_ZSTD) ? 1 : -1;
    case BSLZ4_STAGE_UNTRANSPOSE:
        switch (id) {
        case 0: return 1;  /* kcb */
        case 1: return BSLZ4_HAVE_BACKEND_SSE ? 1 : 0;
        case 2: return BSLZ4_HAVE_BACKEND_NEON ? 1 : 0;
        case 3: return 1;  /* scal */
        default: return -1;
        }
    case BSLZ4_STAGE_COLLECT:
        switch (id) {
        case 0: return 1;  /* scalar */
        case 1: return bslz4_available_avx512_collect() ? 1 : 0;
        case 2: return bslz4_available_avx2_collect() ? 1 : 0;
        case 3: return bslz4_available_sse2_collect() ? 1 : 0;
        case 4: return bslz4_available_vsx_collect() ? 1 : 0;
        case 5: return bslz4_available_neon_collect() ? 1 : 0;
        case 6: return bslz4_available_avx512cs_collect() ? 1 : 0;   /* compress-store */
        default: return -1;
        }
    case BSLZ4_STAGE_DOT:
        if (id < 0 || id >= bslz4_dot_count()) return -1;
        return bslz4_dots[id].available() ? 1 : 0;
    default:
        return -1;
    }
}

/* dtype masks: bit d set => supports dtype index d.  Scalar includes all
 * 10; the SIMD collect tiers are u16/u32 only (indices 1, 2); csc includes
 * all 10 (it is dtype-generic). */
static uint32_t collect_dtype_mask(int id) {
    if (id == 0) return 0x3FFu;                       /* scalar: all 10 */
    return (1u << 1) | (1u << 2);                     /* u16, u32 */
}

static uint32_t dot_dtype_mask(int id) {
    if (id < 0 || id >= bslz4_dot_count()) return 0;
    return bslz4_dots[id].dtype_mask;
}

/* Known options bits.  Only the reserved DROP_NEGATIVES flag exists; it is
 * not yet implemented, so requesting it is currently rejected. */
static int options_ok(uint16_t options) {
    return (options & (uint16_t) ~(BSLZ4_OPT_DROP_NEGATIVES | BSLZ4_OPT_NO_MASK |
                                   BSLZ4_OPT_BYTESKIP | BSLZ4_OPT_PLANE_EXTRACT |
                                   BSLZ4_OPT_LZ4_ZERO | BSLZ4_OPT_MASK_PLANES)) == 0;
}

static bslz4_untranspose_fn untranspose_by_id(int id) {
    switch (id) {
    case 0:  return bslz4_untranspose_kcb;
    case 1:  return bslz4_untranspose_bshuf_sse;
    case 2:  return bslz4_untranspose_bshuf_neon;
    case 3:  return bslz4_untranspose_bshuf_scal;
    default: return 0;
    }
}

static size_t bslz4_elem_size(int dtype) {
    switch (dtype) {
    case 0: return 1;  /* u8 */
    case 1: return 2;  /* u16 */
    case 2: return 4;  /* u32 */
    case 3: return 8;  /* u64 */
    case 4: return 1;  /* i8 */
    case 5: return 2;  /* i16 */
    case 6: return 4;  /* i32 */
    case 7: return 8;  /* i64 */
    case 8: return 4;  /* f32 */
    case 9: return 8;  /* f64 */
    default: return 1;
    }
}

int bslz4_resolve(int dtype, const uint16_t *stages, int expected_layout,
                  bslz4_stage *BSLZ4_RESTRICT st) {
    if (dtype < 0 || dtype > 9) return BSLZ4_ERR_DTYPE;

    int dec = stages[BSLZ4_STAGE_DECOMPRESS];
    int unt = stages[BSLZ4_STAGE_UNTRANSPOSE];
    int col = stages[BSLZ4_STAGE_COLLECT];
    int dot = stages[BSLZ4_STAGE_DOT];
    uint16_t options = stages[BSLZ4_STAGES_OPTIONS];

    if (bslz4_impl_available(BSLZ4_STAGE_DECOMPRESS, dec) <= 0)
        return bslz4_impl_available(BSLZ4_STAGE_DECOMPRESS, dec) < 0 ? BSLZ4_ERR_BAD_PIPELINE
                                                                      : BSLZ4_ERR_UNAVAILABLE;
    if (bslz4_impl_available(BSLZ4_STAGE_UNTRANSPOSE, unt) <= 0)
        return bslz4_impl_available(BSLZ4_STAGE_UNTRANSPOSE, unt) < 0 ? BSLZ4_ERR_BAD_PIPELINE
                                                                       : BSLZ4_ERR_UNAVAILABLE;
    if (bslz4_impl_available(BSLZ4_STAGE_COLLECT, col) <= 0)
        return bslz4_impl_available(BSLZ4_STAGE_COLLECT, col) < 0 ? BSLZ4_ERR_BAD_PIPELINE
                                                                   : BSLZ4_ERR_UNAVAILABLE;
    if (bslz4_impl_available(BSLZ4_STAGE_DOT, dot) <= 0)
        return bslz4_impl_available(BSLZ4_STAGE_DOT, dot) < 0 ? BSLZ4_ERR_BAD_PIPELINE
                                                               : BSLZ4_ERR_UNAVAILABLE;
    if (expected_layout >= 0 && bslz4_dots[dot].layout != expected_layout)
        return BSLZ4_ERR_BAD_LAYOUT;
    if (!((collect_dtype_mask(col) >> dtype) & 1u)) return BSLZ4_ERR_DTYPE;
    if (!((dot_dtype_mask(dot) >> dtype) & 1u)) return BSLZ4_ERR_DTYPE;
    if (!options_ok(options)) return BSLZ4_ERR_BAD_PIPELINE;

    st->elem_size = (size_t) bslz4_elem_size(dtype);
    st->dtype = dtype;
    st->collect_id = col;
    st->dot_id = dot;
    st->untranspose_id = unt;
    st->options = options;
    st->decompress = &bslz4_decompress;
    st->untranspose = untranspose_by_id(unt);
    return 0;
}

void bslz4_reset_counters(void) {
    memset(bslz4_counters, 0, sizeof(bslz4_counters));
}

int bslz4_read_counters(uint64_t *BSLZ4_RESTRICT out, int n) {
    int k = 0;
    for (int s = 0; s < BSLZ4_NSTAGES; s++)
        for (int i = 0; i < BSLZ4_ID_SLOTS && k < n; i++)
            out[k++] = bslz4_counters[s][i];
    return k;
}
