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

uint64_t bslz4_counters[BSLZ4_NSTAGES][BSLZ4_ID_SLOTS];

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
        default: return -1;
        }
    case BSLZ4_STAGE_DOT:
        return (id == 0 || id == 1) ? 1 : -1;  /* 0=csc, 1=csc-fused (dot+threshold interleaved) */
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
    (void) id;
    return 0x3FFu;                                    /* csc: all 10 */
}

/* Known options bits.  Only the reserved DROP_NEGATIVES flag exists; it is
 * not yet implemented, so requesting it is currently rejected. */
static int options_ok(uint64_t pipeline) {
    uint64_t opt = (uint64_t) BSLZ4_PIPE_OPTIONS(pipeline);
    return (opt & ~(uint64_t) BSLZ4_OPT_DROP_NEGATIVES) == 0;
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

int bslz4_resolve(int dtype, uint64_t pipeline, bslz4_stage *BSLZ4_RESTRICT st) {
    if (dtype < 0 || dtype > 9) return BSLZ4_ERR_DTYPE;

    int dec = BSLZ4_PIPE_DECOMPRESS(pipeline);
    int unt = BSLZ4_PIPE_UNTRANSPOSE(pipeline);
    int col = BSLZ4_PIPE_COLLECT(pipeline);
    int dot = BSLZ4_PIPE_DOT(pipeline);

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
    if (!((collect_dtype_mask(col) >> dtype) & 1u)) return BSLZ4_ERR_DTYPE;
    if (!((dot_dtype_mask(dot) >> dtype) & 1u)) return BSLZ4_ERR_DTYPE;
    if (!options_ok(pipeline)) return BSLZ4_ERR_BAD_PIPELINE;

    st->elem_size = (size_t) bslz4_elem_size(dtype);
    st->dtype = dtype;
    st->collect_id = col;
    st->dot_id = dot;
    st->untranspose_id = unt;
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
