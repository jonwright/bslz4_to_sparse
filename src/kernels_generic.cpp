/*
 * kernels_generic.cpp -- the only genuinely dtype-generic code.
 *
 * The dtype-generic work is an inner per-block/tail call, never the outer
 * multi-frame loop (which lives in bslz4_driver.c).  This TU holds:
 *   * sparse<T>          -- plain route
 *   * sparse_dot<T>      -- sparse CSC route
 *   * dense_dot<T>       -- dense CSC route (dot, then a separate >cut)
 *   * dense_dot_fused<T> -- dense route with the >cut sparsify fused into
 *                           the CSC loop (one pass over the block)
 * for the 10 pixel dtypes, plus the scalar/nz/dot loops they need.
 *
 * Each dtype gets a small *noinline* kernel (bslz4_sparse_u8, ...,
 * bslz4_sparse_dot_f64) that inlines its own template, so each kernel is
 * register-allocated independently; the dispatch is a thin switch of direct
 * calls.  A new route strategy is just another leaf kernel selected by
 * dot_id.
 *
 * The SIMD collect tiers are the bslz4_collect_gt<T>/bslz4_collect_nz<T>
 * entry points (bslz4_collect_simd.hpp); the collect tier id chosen by the
 * pipeline is dispatched with a predicted if-chain.
 *
 * The per-block instrumentation counters are bumped by the driver
 * (bslz4_driver.c), one bump per block for the collect/dot ids, so a test
 * can show the selected impl ran in every route and in the tail and that
 * the scalar collect tier was not picked for a SIMD dtype.
 *
 * Compiled with -fno-exceptions -fno-rtti -fno-threadsafe-statics (C++11)
 * only for this TU.
 */

#include "bslz4_common.h"
#include "bslz4_collect_simd.hpp"
#include "bslz4_registry.h"
#include "bslz4_collect_caps.h"

using namespace bslz4;

/* C-linkage availability for the SIMD collect tiers (bslz4_collect_caps.h);
 * used by bslz4_impl_available(COLLECT, id).  The pick itself is by collect id
 * in the decode stages, so there is no set_/get_ state here anymore. */
extern "C" int bslz4_available_avx512_collect(void) { return bslz4_avx512_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_avx2_collect(void)   { return bslz4_avx2_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_sse2_collect(void)   { return bslz4_sse2_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_vsx_collect(void)    { return bslz4_vsx_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_neon_collect(void)   { return bslz4_neon_collect_capable() ? 1 : 0; }

/* ---- per-dtype generic kernels (all take a single bslz4_work*) ----
 *
 * Each is a self-contained function that inlines its own template and
 * collect/dot selection.  They are *noinline* so each kernel is
 * register-allocated independently; the dispatchers are thin switches that
 * call them.  The collect tier and dot implementation are runtime ids in the
 * work struct, so a new route strategy is a leaf function + a switch case.
 */

/* plain sparse, generic */
template<typename T>
static inline
int sparse_plain(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    return bslz4_collect_gt<T>((const T *) w->block, w->mask, w->i0, w->n, cut,
                               w->collect_id, (T *) w->out_vals, w->out_adr);
}

/* dense CSC "dot then threshold": dot.dense over every masked pixel, then a
 * separate >cut collect.  No compaction: tidx/tval are not needed. */
template<typename T>
static inline
int dense_dot(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->mask;
    const size_t i0 = w->i0, n = w->n;
    double *BSLZ4_RESTRICT out = w->powder;
    const float *BSLZ4_RESTRICT data = w->data;
    const uint32_t *BSLZ4_RESTRICT indices = w->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = w->indptr;
    for (size_t j = 0; j < n; j++) {
        if (mask[j + i0] > 0) {
            uint32_t k0 = indptr[j + i0], k1 = indptr[j + i0 + 1];
            T pv = px[j];
            for (uint32_t k = k0; k < k1; k++)
                out[indices[k]] += (double) data[k] * (double) pv;
        }
    }
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect_id,
                               (T *) w->out_vals, w->out_adr);
}

/* dense CSC "fused": dot.dense and the >cut sparse-output collect run in one
 * pass over the block, so the block/mask are scanned once instead of twice.
 * Produces the same powder and the same sparse output (ascending raster
 * order) as dense_dot, without the second SIMD collect pass. */
template<typename T>
static inline
int dense_dot_fused(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->mask;
    const size_t i0 = w->i0, n = w->n;
    double *BSLZ4_RESTRICT out = w->powder;
    const float *BSLZ4_RESTRICT data = w->data;
    const uint32_t *BSLZ4_RESTRICT indices = w->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = w->indptr;
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT oadr = w->out_adr;
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if (mask[j + i0] > 0) {
            uint32_t k0 = indptr[j + i0], k1 = indptr[j + i0 + 1];
            T pv = px[j];
            for (uint32_t k = k0; k < k1; k++)
                out[indices[k]] += (double) data[k] * (double) pv;
            if (BSLZ4_UNLIKELY(pv > cut)) {
                ov[npx] = pv;
                oadr[npx] = (uint32_t) (j + i0);
                npx++;
            }
        }
    }
    return npx;
}

/* sparse CSC route: compact non-zeros into (tval,tidx), dot.sparse over the
 * list, then the >cut collect over the same list. */
template<typename T>
static inline
int sparse_dot(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->mask;
    const size_t i0 = w->i0, n = w->n;
    T *tv = (T *) w->tval;
    uint32_t *BSLZ4_RESTRICT tidx = w->tidx;
    int npx = 0;
    int nz = bslz4_collect_nz<T>(px, mask, i0, n, w->collect_id, tv, tidx);
    double *BSLZ4_RESTRICT out = w->powder;
    const float *BSLZ4_RESTRICT data = w->data;
    const uint32_t *BSLZ4_RESTRICT indices = w->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = w->indptr;
    for (int kk = 0; kk < nz; kk++) {
        uint32_t addr = tidx[kk];
        T val = tv[kk];
        uint32_t k0 = indptr[addr], k1 = indptr[addr + 1];
        for (uint32_t k = k0; k < k1; k++)
            out[indices[k]] += (double) data[k] * (double) val;
    }
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT out_adr = w->out_adr;
    for (int kk = 0; kk < nz; kk++) {
        T val = tv[kk];
        if (BSLZ4_UNLIKELY(val > cut)) {
            ov[npx] = val;
            out_adr[npx] = tidx[kk];
            npx++;
        }
    }
    return npx;
}

/* ---- per-dtype noinline kernels ------------------------------------ */

#define BSLZ4_DTYPE_KERNELS(T, name)                                      \
    extern "C" BSLZ4_NOINLINE int bslz4_sparse_##name(const bslz4_work *w) { \
        return sparse_plain<T>(w);                                        \
    }                                                                     \
    extern "C" BSLZ4_NOINLINE int bslz4_sparse_dot_##name(const bslz4_work *w) { \
        if (w->route) return sparse_dot<T>(w);                            \
        return w->dot_id == 1 ? dense_dot_fused<T>(w) : dense_dot<T>(w);  \
    }

BSLZ4_DTYPE_KERNELS(uint8_t,  u8)
BSLZ4_DTYPE_KERNELS(uint16_t, u16)
BSLZ4_DTYPE_KERNELS(uint32_t, u32)
BSLZ4_DTYPE_KERNELS(uint64_t, u64)
BSLZ4_DTYPE_KERNELS(int8_t,   i8)
BSLZ4_DTYPE_KERNELS(int16_t,  i16)
BSLZ4_DTYPE_KERNELS(int32_t,  i32)
BSLZ4_DTYPE_KERNELS(int64_t,  i64)
BSLZ4_DTYPE_KERNELS(float,    f32)
BSLZ4_DTYPE_KERNELS(double,   f64)

#undef BSLZ4_DTYPE_KERNELS

/* ---- thin dtype-dispatching switches -------------------------------- */

extern "C" int bslz4_sparse_dispatch(const bslz4_work *BSLZ4_RESTRICT w) {
    switch (w->dtype) {
    case 0:  return bslz4_sparse_u8(w);
    case 1:  return bslz4_sparse_u16(w);
    case 2:  return bslz4_sparse_u32(w);
    case 3:  return bslz4_sparse_u64(w);
    case 4:  return bslz4_sparse_i8(w);
    case 5:  return bslz4_sparse_i16(w);
    case 6:  return bslz4_sparse_i32(w);
    case 7:  return bslz4_sparse_i64(w);
    case 8:  return bslz4_sparse_f32(w);
    case 9:  return bslz4_sparse_f64(w);
    default: return BSLZ4_ERR_DTYPE;
    }
}

extern "C" int bslz4_sparse_dot_dispatch(const bslz4_work *BSLZ4_RESTRICT w) {
    switch (w->dtype) {
    case 0:  return bslz4_sparse_dot_u8(w);
    case 1:  return bslz4_sparse_dot_u16(w);
    case 2:  return bslz4_sparse_dot_u32(w);
    case 3:  return bslz4_sparse_dot_u64(w);
    case 4:  return bslz4_sparse_dot_i8(w);
    case 5:  return bslz4_sparse_dot_i16(w);
    case 6:  return bslz4_sparse_dot_i32(w);
    case 7:  return bslz4_sparse_dot_i64(w);
    case 8:  return bslz4_sparse_dot_f32(w);
    case 9:  return bslz4_sparse_dot_f64(w);
    default: return BSLZ4_ERR_DTYPE;
    }
}
