/*
 * kernels.cpp -- the one C++ translation unit: the dtype-generic collect and
 * dot code of the steps (templates over the pixel type), instantiated
 * explicitly for the 10 pixel dtypes and reached from the C driver through
 * bslz4_sparse_dispatch / bslz4_sparse_dot_dispatch, one call per block.
 *
 *   collect.hpp         block -> pixel list, per collect value
 *   dot_csc.hpp         csc, csc-run, csc-nosplit, csc-nosplit-moment
 *   dot_padded.hpp      padded, padded-avx2
 *   dot_bsbcsr.hpp      bsb-csr, bsb-csr-nosplit
 *   dot_fixed.hpp       csc-permute, csc-int, csc-run-int, csc-run-int16
 *   lowplanes_avx2.hpp  the AVX2 fused low-planes collect (C entry below)
 *
 * Each dtype's entry is a noinline instance, so each is register-allocated
 * on its own; the dispatchers are switches over the dtype.
 *
 * Compiled with -fno-exceptions -fno-rtti -fno-threadsafe-statics (C++11).
 */

#include "../pipeline/common.h"
#include "../pipeline/pipeline.h"
#include "caps.h"
#include "collect.hpp"
#include "dot_csc.hpp"
#include "dot_padded.hpp"
#include "dot_bsbcsr.hpp"
#include "dot_fixed.hpp"
#include "lowplanes_avx2.hpp"

/* ---- CPU checks of the C++ step values (registry.c) ---- */

extern "C" int bslz4_available_avx512cs_collect(void) { return bslz4_avx512cs_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_avx2cs_collect(void)   { return bslz4_avx2cs_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_sse2_collect(void)     { return bslz4_sse2_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_vsx_collect(void)      { return bslz4_vsx_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_neon_collect(void)     { return bslz4_neon_collect_capable() ? 1 : 0; }
extern "C" int bslz4_available_avx2_padded(void)      { return bslz4_padded_avx2_capable() ? 1 : 0; }

extern "C" int bslz4_lowplanes_collect_avx2(uint8_t *raw, size_t ne, size_t nz_end, const uint8_t *mask,
                                            size_t i0, unsigned cut, uint16_t *out_vals, uint32_t *out_adr) {
#if BSLZ4_HAVE_AVX2_COLLECT
    if (ne % 256) return -1;
    return bslz4_lowplanes_collect_avx2_impl(raw, ne, nz_end, mask, i0, cut, out_vals, out_adr);
#else
    (void) raw; (void) ne; (void) nz_end; (void) mask; (void) i0; (void) cut; (void) out_vals; (void) out_adr;
    return -1;
#endif
}

/* ---- one block, per pixel type ---- */

/* plain sparsify: the > cut (unmasked) pixels of the block */
template<typename T>
BSLZ4_NOINLINE int bslz4_sparse_block(const bslz4_work *BSLZ4_RESTRICT w) {
    if (bslz4_cut_above_max<T>(w)) return 0;
    return bslz4_collect_gt<T>((const T *) w->block, w->no_mask ? NULL : w->mask, w->i0,
                               w->n, (T) w->threshold, w->collect, (T *) w->out_vals, w->out_adr);
}

/* sparsify + dot: the dot's dense or sparse route (w->route) */
template<typename T>
BSLZ4_NOINLINE int bslz4_sparse_dot_block(const bslz4_work *BSLZ4_RESTRICT w) {
    const int sparse = w->route;
    switch (w->dot) {
    case BSLZ4_DOT_CSC:                return sparse ? sparse_dot<T>(w) : dense_dot<T>(w);
    case BSLZ4_DOT_PADDED:             return sparse ? padded_sparse<T>(w) : padded_dense<T>(w, false);
    case BSLZ4_DOT_PADDED_AVX2:        return sparse ? padded_sparse<T>(w) : padded_dense<T>(w, true);
    case BSLZ4_DOT_CSC_RUN:            return sparse ? run_sparse<T>(w) : run_dense<T>(w);
    case BSLZ4_DOT_CSC_NOSPLIT:        return sparse ? nosplit_sparse<T>(w) : nosplit_dense<T>(w);
    case BSLZ4_DOT_BSBCSR:             return sparse ? bsbcsr_sparse<T>(w) : bsbcsr_dense<T>(w);
    case BSLZ4_DOT_BSBCSR_NOSPLIT:     return sparse ? bsbcsr_nosplit_sparse<T>(w) : bsbcsr_nosplit_dense<T>(w);
    case BSLZ4_DOT_CSC_NOSPLIT_MOMENT: return sparse ? nosplit_moment_sparse<T>(w) : nosplit_moment_dense<T>(w);
    default:                           return bslz4v::dispatch<T>(w);
    }
}

template int bslz4_sparse_block<uint8_t>(const bslz4_work *);
template int bslz4_sparse_block<uint16_t>(const bslz4_work *);
template int bslz4_sparse_block<uint32_t>(const bslz4_work *);
template int bslz4_sparse_block<uint64_t>(const bslz4_work *);
template int bslz4_sparse_block<int8_t>(const bslz4_work *);
template int bslz4_sparse_block<int16_t>(const bslz4_work *);
template int bslz4_sparse_block<int32_t>(const bslz4_work *);
template int bslz4_sparse_block<int64_t>(const bslz4_work *);
template int bslz4_sparse_block<float>(const bslz4_work *);
template int bslz4_sparse_block<double>(const bslz4_work *);

template int bslz4_sparse_dot_block<uint8_t>(const bslz4_work *);
template int bslz4_sparse_dot_block<uint16_t>(const bslz4_work *);
template int bslz4_sparse_dot_block<uint32_t>(const bslz4_work *);
template int bslz4_sparse_dot_block<uint64_t>(const bslz4_work *);
template int bslz4_sparse_dot_block<int8_t>(const bslz4_work *);
template int bslz4_sparse_dot_block<int16_t>(const bslz4_work *);
template int bslz4_sparse_dot_block<int32_t>(const bslz4_work *);
template int bslz4_sparse_dot_block<int64_t>(const bslz4_work *);
template int bslz4_sparse_dot_block<float>(const bslz4_work *);
template int bslz4_sparse_dot_block<double>(const bslz4_work *);

/* ---- the dtype switches the driver calls ---- */

extern "C" int bslz4_sparse_dispatch(const bslz4_work *BSLZ4_RESTRICT w) {
    switch (w->dtype) {
    case 0:  return bslz4_sparse_block<uint8_t>(w);
    case 1:  return bslz4_sparse_block<uint16_t>(w);
    case 2:  return bslz4_sparse_block<uint32_t>(w);
    case 3:  return bslz4_sparse_block<uint64_t>(w);
    case 4:  return bslz4_sparse_block<int8_t>(w);
    case 5:  return bslz4_sparse_block<int16_t>(w);
    case 6:  return bslz4_sparse_block<int32_t>(w);
    case 7:  return bslz4_sparse_block<int64_t>(w);
    case 8:  return bslz4_sparse_block<float>(w);
    case 9:  return bslz4_sparse_block<double>(w);
    default: return BSLZ4_ERR_DTYPE;
    }
}

extern "C" int bslz4_sparse_dot_dispatch(const bslz4_work *BSLZ4_RESTRICT w) {
    switch (w->dtype) {
    case 0:  return bslz4_sparse_dot_block<uint8_t>(w);
    case 1:  return bslz4_sparse_dot_block<uint16_t>(w);
    case 2:  return bslz4_sparse_dot_block<uint32_t>(w);
    case 3:  return bslz4_sparse_dot_block<uint64_t>(w);
    case 4:  return bslz4_sparse_dot_block<int8_t>(w);
    case 5:  return bslz4_sparse_dot_block<int16_t>(w);
    case 6:  return bslz4_sparse_dot_block<int32_t>(w);
    case 7:  return bslz4_sparse_dot_block<int64_t>(w);
    case 8:  return bslz4_sparse_dot_block<float>(w);
    case 9:  return bslz4_sparse_dot_block<double>(w);
    default: return BSLZ4_ERR_DTYPE;
    }
}
