#ifndef BSLZ4_COLLECT_CAPS_H
#define BSLZ4_COLLECT_CAPS_H
/*
 * C-linkage access to the optional SIMD mask+threshold collect tiers
 * (u16/u32 only).  The functionality lives in bslz4_collect_simd.hpp;
 * these thin extern "C" wrappers (kernels_generic.cpp) let the C registry
 * (bslz4_registry.c) decide availability via bslz4_impl_available.  The
 * tier actually run is chosen by the collect id in the decode stages, so
 * no set/get flags-based state is exposed here.
 *
 * ids: 0 = scalar, 1 = avx512, 2 = avx2, 3 = sse2, 4 = vsx, 5 = neon,
 *      6 = avx512cs, 7 = avx2cs
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int bslz4_available_avx512_collect(void);
int bslz4_available_avx512cs_collect(void);
int bslz4_available_avx2cs_collect(void);
/* driver's fused byte-skip collect (kernels_generic.cpp); -1 if not built */
int bslz4_collect_u8_avx2cs(const uint8_t *v8, const uint8_t *mask, size_t i0, size_t n,
                            unsigned cut, uint16_t *out_vals, uint32_t *out_adr);
int bslz4_available_avx2_collect(void);
int bslz4_available_sse2_collect(void);
int bslz4_available_vsx_collect(void);
int bslz4_available_neon_collect(void);

#ifdef __cplusplus
}
#endif

#endif /* BSLZ4_COLLECT_CAPS_H */
