/* Bit-unshuffle of one 8 kB bitshuffle block, one warp, in registers; output staged
   in the decode buffer. 32x32 bit transposes (register shuffles), from bslz4decoders/gpu_batched. */
/* 32x32 bit transpose, LSB first: afterwards bit r of a[c] == (old) bit c of a[r] */
__device__ __forceinline__ void transpose32(uint32_t (&a)[32]) {
  uint32_t m = 0x0000FFFFU;
  #pragma unroll
  for (int j = 16; j != 0; j >>= 1) {
    #pragma unroll
    for (int k = 0; k < 32; k++) {
      if (k & j) continue;
      const uint32_t t = ((a[k] >> j) ^ a[k + j]) & m;
      a[k + j] ^= t; a[k] ^= (t << j);
    }
    m ^= (m << (j >> 1));
  }
}
/* output is staged in the (now free) decode buffer as 16-byte vectors with an xor swizzle (conflict free),
   then copied to global memory with fully coalesced 512-byte warp stores */
__device__ __forceinline__ void sst4(uint8_t* dec, uint32_t n, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  ((uint4*)dec)[n ^ ((n >> 3) & 7)] = make_uint4(a, b, c, d); }
__device__ __forceinline__ void flush(uint8_t* dec, uint32_t* out) {
  __syncwarp();
  #pragma unroll
  for (int i = 0; i < 16; i++) { const uint32_t n = i*32 + threadIdx.x; ((uint4*)out)[n] = ((const uint4*)dec)[n ^ ((n >> 3) & 7)]; }
}
/* a block is 8e bit-planes of N=8192/e bits; plane k word g (32 elements) = in[k*R + g], R = N/32 */
__device__ __forceinline__ void shuf32_warp(uint8_t* dec, uint32_t* out, bool fl = true) {
  const uint32_t* in = (const uint32_t*)dec; const int L = threadIdx.x; uint32_t a[2][32];
  #pragma unroll
  for (int q = 0; q < 2; q++) {
    #pragma unroll
    for (int k = 0; k < 32; k++) a[q][k] = in[k*64 + L + 32*q];
  }
  __syncwarp();
  #pragma unroll
  for (int q = 0; q < 2; q++) {
    transpose32(a[q]); const int g = L + 32*q;
    #pragma unroll
    for (int i = 0; i < 8; i++) sst4(dec, g*8 + i, a[q][4*i], a[q][4*i+1], a[q][4*i+2], a[q][4*i+3]);
  }
  if (fl) flush(dec, out); else __syncwarp();
}
__device__ __forceinline__ void shuf16_warp(uint8_t* dec, uint32_t* out, bool fl = true) {
  const uint32_t* in = (const uint32_t*)dec; const int L = threadIdx.x; uint32_t a[2][32];
  #pragma unroll
  for (int p = 0; p < 2; p++) {
    #pragma unroll
    for (int k = 0; k < 16; k++) { a[p][k] = in[k*128 + L + 64*p]; a[p][16+k] = in[k*128 + L + 64*p + 32]; }
  }
  __syncwarp();
  #pragma unroll
  for (int p = 0; p < 2; p++) {
    transpose32(a[p]); const int ga = L + 64*p, gb = ga + 32;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t wa[4], wb[4];
      #pragma unroll
      for (int m = 0; m < 4; m++) { const uint32_t x0 = a[p][2*(4*i+m)], x1 = a[p][2*(4*i+m)+1];
        wa[m] = (x0 & 0xFFFFU) | (x1 << 16); wb[m] = (x0 >> 16) | (x1 & 0xFFFF0000U); }
      sst4(dec, ga*4 + i, wa[0], wa[1], wa[2], wa[3]); sst4(dec, gb*4 + i, wb[0], wb[1], wb[2], wb[3]);
    }
  }
  if (fl) flush(dec, out); else __syncwarp();
}
__device__ __forceinline__ void shuf8_warp(uint8_t* dec, uint32_t* out, bool fl = true) {
  const uint32_t* in = (const uint32_t*)dec; const int L = threadIdx.x; uint32_t a[2][32];
  #pragma unroll
  for (int p = 0; p < 2; p++) {
    #pragma unroll
    for (int s = 0; s < 4; s++) {
      #pragma unroll
      for (int k = 0; k < 8; k++) a[p][8*s + k] = in[k*256 + L + 128*p + 32*s];
    }
  }
  __syncwarp();
  #pragma unroll
  for (int p = 0; p < 2; p++) {
    transpose32(a[p]);
    #pragma unroll
    for (int s = 0; s < 4; s++) {
      const int g = L + 128*p + 32*s; uint32_t w[8];
      #pragma unroll
      for (int m = 0; m < 8; m++)
        w[m] = ((a[p][4*m] >> (8*s)) & 0xFFU) | (((a[p][4*m+1] >> (8*s)) & 0xFFU) << 8) | (((a[p][4*m+2] >> (8*s)) & 0xFFU) << 16) | (((a[p][4*m+3] >> (8*s)) & 0xFFU) << 24);
      sst4(dec, g*2, w[0], w[1], w[2], w[3]); sst4(dec, g*2 + 1, w[4], w[5], w[6], w[7]);
    }
  }
  if (fl) flush(dec, out); else __syncwarp();
}
