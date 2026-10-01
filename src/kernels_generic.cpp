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
static int bslz4_inner_sparse(const void *BSLZ4_RESTRICT block, size_t n,
                              const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                              int64_t threshold, void *BSLZ4_RESTRICT out_vals,
                              uint32_t *BSLZ4_RESTRICT out_adr) {
    const T cut = (T) threshold;
    return bslz4_collect_gt<T>((const T *) block, mask, i0, n, cut, (T *) out_vals, out_adr);
}

/* CSC, generic: routes dense or sparse (see bslz4_core.hpp comments). */
template<typename T>
static int bslz4_inner_sparse_dot(const void *BSLZ4_RESTRICT block, size_t n,
                                  size_t decoded_bytes, size_t nbytes,
                                  const uint8_t *BSLZ4_RESTRICT mask, size_t i0,
                                  int64_t threshold, void *BSLZ4_RESTRICT out_vals,
                                  uint32_t *BSLZ4_RESTRICT out_adr,
                                  double *BSLZ4_RESTRICT out, int nout,
                                  const float *BSLZ4_RESTRICT data,
                                  const uint32_t *BSLZ4_RESTRICT indices,
                                  const uint32_t *BSLZ4_RESTRICT indptr,
                                  double dense_sparse_x, uint32_t *BSLZ4_RESTRICT tidx,
                                  void *BSLZ4_RESTRICT tval) {
    const T cut = (T) threshold;
    const T *px = (const T *) block;
    T *tv = (T *) tval;
    const bool use_sparse = decoded_bytes > 0
        ? (double) decoded_bytes > dense_sparse_x * (double) nbytes
        : true;

    if (use_sparse) {
        int npx = 0;
        int nz = bslz4_collect_nz<T>(px, mask, i0, n, tv, tidx);
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
    } else {
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
}

/* extern "C" wrappers, one per dtype, for the bslz4_inner_table. */
#define BSLZ4_DEFINE_SPARSE_DTYPE(T, SUFFIX)                                        \
    extern "C" int bslz4_inner_sparse_##SUFFIX(                                    \
        const void *BSLZ4_RESTRICT block, size_t n, const uint8_t *BSLZ4_RESTRICT mask, \
        size_t i0, int64_t threshold, void *BSLZ4_RESTRICT out_vals,                \
        uint32_t *BSLZ4_RESTRICT out_adr) {                                        \
        return bslz4_inner_sparse<T>(block, n, mask, i0, threshold, out_vals,      \
                                     out_adr);                                      \
    }                                                                              \
    extern "C" int bslz4_inner_sparse_dot_##SUFFIX(                                \
        const void *BSLZ4_RESTRICT block, size_t n, size_t decoded_bytes, size_t nbytes, \
        const uint8_t *BSLZ4_RESTRICT mask, size_t i0, int64_t threshold,          \
        void *BSLZ4_RESTRICT out_vals, uint32_t *BSLZ4_RESTRICT out_adr,           \
        double *BSLZ4_RESTRICT out, int nout, const float *BSLZ4_RESTRICT data,    \
        const uint32_t *BSLZ4_RESTRICT indices, const uint32_t *BSLZ4_RESTRICT indptr, \
        double dense_sparse_x, uint32_t *BSLZ4_RESTRICT tidx,                      \
        void *BSLZ4_RESTRICT tval) {                                               \
        return bslz4_inner_sparse_dot<T>(block, n, decoded_bytes, nbytes, mask,    \
                                         i0, threshold, out_vals, out_adr, out,    \
                                         nout, data, indices, indptr,              \
                                         dense_sparse_x, tidx, tval);              \
    }

BSLZ4_DEFINE_SPARSE_DTYPE(uint8_t, u8)
BSLZ4_DEFINE_SPARSE_DTYPE(uint16_t, u16)
BSLZ4_DEFINE_SPARSE_DTYPE(uint32_t, u32)
BSLZ4_DEFINE_SPARSE_DTYPE(uint64_t, u64)
BSLZ4_DEFINE_SPARSE_DTYPE(int8_t, i8)
BSLZ4_DEFINE_SPARSE_DTYPE(int16_t, i16)
BSLZ4_DEFINE_SPARSE_DTYPE(int32_t, i32)
BSLZ4_DEFINE_SPARSE_DTYPE(int64_t, i64)
BSLZ4_DEFINE_SPARSE_DTYPE(float, f32)
BSLZ4_DEFINE_SPARSE_DTYPE(double, f64)

#undef BSLZ4_DEFINE_SPARSE_DTYPE

extern "C" const bslz4_inner_entry bslz4_inner_table[10] = {
    {sizeof(uint8_t),  bslz4_inner_sparse_u8,  bslz4_inner_sparse_dot_u8},
    {sizeof(uint16_t), bslz4_inner_sparse_u16, bslz4_inner_sparse_dot_u16},
    {sizeof(uint32_t), bslz4_inner_sparse_u32, bslz4_inner_sparse_dot_u32},
    {sizeof(uint64_t), bslz4_inner_sparse_u64, bslz4_inner_sparse_dot_u64},
    {sizeof(int8_t),   bslz4_inner_sparse_i8,  bslz4_inner_sparse_dot_i8},
    {sizeof(int16_t),  bslz4_inner_sparse_i16, bslz4_inner_sparse_dot_i16},
    {sizeof(int32_t),  bslz4_inner_sparse_i32, bslz4_inner_sparse_dot_i32},
    {sizeof(int64_t),  bslz4_inner_sparse_i64, bslz4_inner_sparse_dot_i64},
    {sizeof(float),    bslz4_inner_sparse_f32, bslz4_inner_sparse_dot_f32},
    {sizeof(double),   bslz4_inner_sparse_f64, bslz4_inner_sparse_dot_f64},
};
