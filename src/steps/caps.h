#ifndef BSLZ4_STEP_CAPS_H
#define BSLZ4_STEP_CAPS_H
/*
 * C-linkage entries into the C++ step code (src/steps/kernels.cpp): the
 * CPU checks of the SIMD collect and padded values, and the AVX2 fused
 * low-planes collect the driver calls.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int bslz4_available_avx512cs_collect(void);
int bslz4_available_avx2cs_collect(void);
int bslz4_available_sse2_collect(void);
int bslz4_available_vsx_collect(void);
int bslz4_available_neon_collect(void);
int bslz4_available_avx2_padded(void);

/* the AVX2 low-planes collect (lowplanes.h); -1 if not built */
int bslz4_lowplanes_collect_avx2(uint8_t *raw, size_t ne, size_t nz_end, const uint8_t *mask,
                                 size_t i0, unsigned cut, uint16_t *out_vals, uint32_t *out_adr);

#ifdef __cplusplus
}
#endif

#endif /* BSLZ4_STEP_CAPS_H */
