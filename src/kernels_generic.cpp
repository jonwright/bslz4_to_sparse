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

using namespace bslz4;

/* plain sparse, generic */
template<typename T>
static int bslz4_inner_sparse(const void *block, size_t n, const uint8_t *mask, size_t i0,
                              int64_t threshold, void *out_vals, uint32_t *out_adr) {
    const T cut = (T) threshold;
    return bslz4_collect_gt<T>((const T *) block, mask, i0, n, cut, (T *) out_vals, out_adr);
}

/* CSC, generic: routes dense or sparse (see bslz4_core.hpp comments). */
template<typename T>
static int bslz4_inner_sparse_dot(const void *block, size_t n, size_t decoded_bytes,
                                  size_t nbytes, const uint8_t *mask, size_t i0,
                                  int64_t threshold, void *out_vals, uint32_t *out_adr,
                                  double *out, int nout, const float *data,
                                  const uint32_t *indices, const uint32_t *indptr,
                                  double dense_sparse_x, uint32_t *tidx, void *tval) {
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
        const void *block, size_t n, const uint8_t *mask, size_t i0,               \
        int64_t threshold, void *out_vals, uint32_t *out_adr) {                    \
        return bslz4_inner_sparse<T>(block, n, mask, i0, threshold, out_vals,      \
                                     out_adr);                                      \
    }                                                                              \
    extern "C" int bslz4_inner_sparse_dot_##SUFFIX(                                \
        const void *block, size_t n, size_t decoded_bytes, size_t nbytes,          \
        const uint8_t *mask, size_t i0, int64_t threshold, void *out_vals,         \
        uint32_t *out_adr, double *out, int nout, const float *data,               \
        const uint32_t *indices, const uint32_t *indptr, double dense_sparse_x,    \
        uint32_t *tidx, void *tval) {                                              \
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
