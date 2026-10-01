#ifndef BSLZ4_COLLECT_CAPS_H
#define BSLZ4_COLLECT_CAPS_H
/*
 * C-linkage access to the optional SIMD mask+threshold collect tiers
 * (u16/u32 only).  The functionality lives in bslz4_collect_simd.hpp;
 * these thin extern "C" wrappers (kernels_generic.cpp) let the C
 * registry (bslz4_registry.c) and the Python shims reach it.
 *
 * ids: 0 = scalar, 1 = avx512, 2 = avx2, 3 = sse2, 4 = vsx, 5 = neon
 */

#ifdef __cplusplus
extern "C" {
#endif

int bslz4_available_avx512_collect(void);
int bslz4_available_avx2_collect(void);
int bslz4_available_sse2_collect(void);
int bslz4_available_vsx_collect(void);
int bslz4_available_neon_collect(void);

int bslz4_get_avx512_collect(void);
int bslz4_get_avx2_collect(void);
int bslz4_get_sse2_collect(void);
int bslz4_get_vsx_collect(void);
int bslz4_get_neon_collect(void);

/* Setters return 0 on success, -1 if enabling an unavailable tier. */
int bslz4_set_avx512_collect(int enabled);
int bslz4_set_avx2_collect(int enabled);
int bslz4_set_sse2_collect(int enabled);
int bslz4_set_vsx_collect(int enabled);
int bslz4_set_neon_collect(int enabled);

#ifdef __cplusplus
}
#endif

#endif /* BSLZ4_COLLECT_CAPS_H */
