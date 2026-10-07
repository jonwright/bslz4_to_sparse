#ifndef BSLZ4_STEP_CAPS_H
#define BSLZ4_STEP_CAPS_H
/*
 * C-linkage entries into the C++ step code (src/steps/kernels.cpp): the
 * CPU checks of the SIMD collect and padded values, and the AVX2 fused
 * low-planes collect the driver calls.
 */

#include <stddef.h>
#include <stdint.h>

/* The fused low-planes collects (lowplanes.h, lowplanes_avx2.hpp) work on a
 * block in tiles of this many pixels, so their scratch (one byte per pixel
 * plus a mask word per 64) lives on the stack at a fixed size whatever the
 * decode block is.  A multiple of 512, at most 8192 (the zero plane). */
#ifndef BSLZ4_LOWPLANES_TILE
#define BSLZ4_LOWPLANES_TILE 8192
#endif
#if BSLZ4_LOWPLANES_TILE % 512 || BSLZ4_LOWPLANES_TILE > 8192 || BSLZ4_LOWPLANES_TILE < 512
#error "BSLZ4_LOWPLANES_TILE must be a multiple of 512 in 512..8192"
#endif

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
