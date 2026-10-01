/*
 * kernels_generic.cpp -- the only genuinely dtype-generic code.
 *
 * Per the plan, the dtype-generic work is an inner per-block/tail call,
 * never the outer multi-frame loop (which lives in bslz4_driver.c).  This
 * TU holds:
 *   * sparse<T>   -- mask>0 & val>threshold, compacted (plain route)
 *   * sparse_dot<T> -- dense-vs-sparse CSC route
 * for the 10 pixel dtypes, plus the scalar/nz/dot loops they need.  The
 * SIMD collect tiers are the same bslz4_collect_gt<T>/bslz4_collect_nz<T>
 * entry points as before (bslz4_collect_simd.hpp), so behaviour is
 * unchanged for Phase 1.
 *
 * Compiled with -fno-exceptions -fno-rtti -fno-threadsafe-statics (C++11)
 * only for this TU.
 */

#include "bslz4_common.hpp"
#include "bslz4_collect_simd.hpp"
#include "bslz4_registry.h"
#include "bslz4_collect_caps.h"

using namespace bslz4;

/* C-linkage access to the SIMD collect tiers (see bslz4_collect_caps.h). */

extern "C" int bslz4_available_avx512_collect(void) { return bslz4_avx512_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_avx2_collect(void)   { return bslz4_avx2_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_sse2_collect(void)   { return bslz4_sse2_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_vsx_collect(void)    { return bslz4_vsx_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_neon_collect(void)   { return bslz4_neon_collect_capable() ? 1 : 0; }

extern "C" int bslz4_get_avx512_collect(void) { return bslz4_avx512_collect_enabled() ? 1 : 0; }
extern "C" int bslz4_get_avx2_collect(void)   { return bslz4_avx2_collect_enabled() ? 1 : 0; }
extern "C" int bslz4_get_sse2_collect(void)   { return bslz4_sse2_collect_enabled() ? 1 : 0; }
extern "C" int bslz4_get_vsx_collect(void)    { return bslz4_vsx_collect_enabled() ? 1 : 0; }
extern "C" int bslz4_get_neon_collect(void)   { return bslz4_neon_collect_enabled() ? 1 : 0; }

extern "C" int bslz4_set_avx512_collect(int enabled) {
    if (enabled && !bslz4_avx512_collect_capable()) return -1;
    bslz4_avx512_collect_enabled() = (enabled != 0);
    return 0;
}
extern "C" int bslz4_set_avx2_collect(int enabled) {
    if (enabled && !bslz4_avx2_collect_capable()) return -1;
    bslz4_avx2_collect_enabled() = (enabled != 0);
    return 0;
}
extern "C" int bslz4_set_sse2_collect(int enabled) {
    if (enabled && !bslz4_sse2_collect_capable()) return -1;
    bslz4_sse2_collect_enabled() = (enabled != 0);
    return 0;
}
extern "C" int bslz4_set_vsx_collect(int enabled) {
    if (enabled && !bslz4_vsx_collect_capable()) return -1;
    bslz4_vsx_collect_enabled() = (enabled != 0);
    return 0;
}
extern "C" int bslz4_set_neon_collect(int enabled) {
    if (enabled && !bslz4_neon_collect_capable()) return -1;
    bslz4_neon_collect_enabled() = (enabled != 0);
    return 0;
}

extern "C" int bslz4_active_collect_tier(void) {
#if BSLZ4_HAVE_AVX512_COLLECT
    if (bslz4_avx512_collect_enabled()) return 1;
#endif
#if BSLZ4_HAVE_AVX2_COLLECT
    if (bslz4_avx2_collect_enabled()) return 2;
#endif
#if BSLZ4_HAVE_SSE2_COLLECT
    if (bslz4_sse2_collect_enabled()) return 3;
#endif
#if BSLZ4_HAVE_VSX_COLLECT
    if (bslz4_vsx_collect_enabled()) return 4;
#endif
#if BSLZ4_HAVE_NEON_COLLECT
    if (bslz4_neon_collect_enabled()) return 5;
#endif
    return 0;
}

/* plain sparse, generic */
template<typename T>
static inline
int sparse_plain(const void *BSLZ4_RESTRICT block, size_t n,
                 const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                 int64_t threshold, void *BSLZ4_RESTRICT out_vals,
                 uint32_t *BSLZ4_RESTRICT out_adr) {
    const T cut = (T) threshold;
    return bslz4_collect_gt<T>((const T *) block, mask, i0, n, cut, (T *) out_vals, out_adr);
}

/* dense CSC route: dot.dense over every masked pixel, then the >cut collect.
 * No compaction: the output is written once and tidx/tval are not needed. */
template<typename T>
static inline
int dense_dot(const void *BSLZ4_RESTRICT block, size_t n,
              const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
              int64_t threshold, void *BSLZ4_RESTRICT out_vals,
              uint32_t *BSLZ4_RESTRICT out_adr,
              const bslz4_csc *BSLZ4_RESTRICT csc) {
    const T cut = (T) threshold;
    const T *px = (const T *) block;
    double *BSLZ4_RESTRICT out = csc->out;
    const float *BSLZ4_RESTRICT data = csc->data;
    const uint32_t *BSLZ4_RESTRICT indices = csc->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = csc->indptr;
    for (size_t j = 0; j < n; j++) {
        if (mask[j + i0] > 0) {
            uint32_t k0 = indptr[j + i0], k1 = indptr[j + i0 + 1];
            T pv = px[j];
            for (uint32_t k = k0; k < k1; k++)
                out[indices[k]] += (double) data[k] * (double) pv;
        }
    }
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, (T *) out_vals, out_adr);
}

/* sparse CSC route: compact non-zeros into (tval,tidx), dot.sparse over the
 * list, then the >cut collect over the same list. */
template<typename T>
static inline
int sparse_dot(const void *BSLZ4_RESTRICT block, size_t n,
               const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
               int64_t threshold, void *BSLZ4_RESTRICT out_vals,
               uint32_t *BSLZ4_RESTRICT out_adr,
               const bslz4_csc *BSLZ4_RESTRICT csc, uint32_t *BSLZ4_RESTRICT tidx,
               void *BSLZ4_RESTRICT tval) {
    const T cut = (T) threshold;
    const T *px = (const T *) block;
    T *tv = (T *) tval;
    int npx = 0;
    int nz = bslz4_collect_nz<T>(px, mask, i0, n, tv, tidx);
    double *BSLZ4_RESTRICT out = csc->out;
    const float *BSLZ4_RESTRICT data = csc->data;
    const uint32_t *BSLZ4_RESTRICT indices = csc->indices;
    const uint32_t *BSLZ4_RESTRICT indptr = csc->indptr;
    for (int kk = 0; kk < nz; kk++) {
        uint32_t addr = tidx[kk];
        T val = tv[kk];
        uint32_t k0 = indptr[addr], k1 = indptr[addr + 1];
        for (uint32_t k = k0; k < k1; k++)
            out[indices[k]] += (double) data[k] * (double) val;
    }
    T *ov = (T *) out_vals;
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

/* Direct-call, dtype-switching entry points.  The per-dtype handlers are
 * static + always_inline in this TU, so the switch bodies inline: one jump
 * table, no function-pointer indirect call, no second call boundary. */

extern "C" int bslz4_sparse_dispatch(
    int dtype, const void *BSLZ4_RESTRICT block, size_t n,
    const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
    int64_t threshold, void *BSLZ4_RESTRICT out_vals,
    uint32_t *BSLZ4_RESTRICT out_adr) {
    switch (dtype) {
    case 0: return sparse_plain<uint8_t>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 1: return sparse_plain<uint16_t>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 2: return sparse_plain<uint32_t>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 3: return sparse_plain<uint64_t>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 4: return sparse_plain<int8_t>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 5: return sparse_plain<int16_t>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 6: return sparse_plain<int32_t>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 7: return sparse_plain<int64_t>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 8: return sparse_plain<float>(block, n, mask, i0, threshold, out_vals, out_adr);
    case 9: return sparse_plain<double>(block, n, mask, i0, threshold, out_vals, out_adr);
    default: return BSLZ4_ERR_DTYPE;
    }
}

extern "C" int bslz4_sparse_dot_dispatch(
    int dtype, int route, const void *BSLZ4_RESTRICT block, size_t n,
    const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
    int64_t threshold, void *BSLZ4_RESTRICT out_vals,
    uint32_t *BSLZ4_RESTRICT out_adr,
    const bslz4_csc *BSLZ4_RESTRICT csc, uint32_t *BSLZ4_RESTRICT tidx,
    void *BSLZ4_RESTRICT tval) {
    switch (dtype) {
    case 0: return route ? sparse_dot<uint8_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<uint8_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 1: return route ? sparse_dot<uint16_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<uint16_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 2: return route ? sparse_dot<uint32_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<uint32_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 3: return route ? sparse_dot<uint64_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<uint64_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 4: return route ? sparse_dot<int8_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<int8_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 5: return route ? sparse_dot<int16_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<int16_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 6: return route ? sparse_dot<int32_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<int32_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 7: return route ? sparse_dot<int64_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<int64_t>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 8: return route ? sparse_dot<float>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<float>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    case 9: return route ? sparse_dot<double>(block, n, mask, i0, threshold, out_vals, out_adr, csc, tidx, tval)
                         : dense_dot<double>(block, n, mask, i0, threshold, out_vals, out_adr, csc);
    default: return BSLZ4_ERR_DTYPE;
    }
}
