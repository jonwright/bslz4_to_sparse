/* bslz4 (bitshuffle + lz4, 8 kB blocks) -> dense frames on the GPU, then a CSR matrix product
   and/or the pixels above a cut.

   Concatenated by _gpu.py after nvcomp_lz4.cuh (decompressStream) and shuffle_warp.cuh
   (shuf8/16/32_warp).

   unblock     one thread per image walks the block headers
   decompress  one warp per 8 kB block: lz4 decode into shared memory, unshuffle in registers,
   coalesced store to a dense frame (frame f at dense + f * stride bytes; stride a multiple of 8192)
   csr1d       powder = M @ frame, M in CSR (bins x pixels, mask folded in): one thread block per
   (bin, F frames), threads stride over the bin's entries, each (weight, index) read once is applied
   to F frames, warp shuffles + shared memory reduce, thread 0 writes. No atomics. (pyFAI's
   csr_integrate4 with one sum instead of four.) count_above   one warp per 8 kB block of the dense
   frame: pixels > cut (and > each top_pixels threshold), not masked scan_counts   per (threshold
   level, image): block offsets and the total choose_level  per image, as ImageD11 top_pixels: the
   cut or the first threshold leaving < howmany pixels scan_images   the lists of a batch back to
   back collect_write one warp per 8 kB block: re-reads the block, writes the pixels of the chosen
   level, the first howmany howmany <= cap = nbytes / (4 + bpp): a frame's list never takes more
   than the raw frame's bytes.  A frame with more than cap pixels above the cut, when every pixel
   was asked for (howmany None or > cap), gets level -1: no list is written and the host makes it
   from the dense frame.

   csr1d accumulators (the `acc` of _gpu.py): f32 = float weights, float sums; f64 = float
   weights, double sums (pyFAI); i64 = u32 fixed-point weights, int64 sums (exact).
*/

inline __device__ uint32_t read32be(const uint8_t *address) {
    return ((uint32_t) (255 & (address)[0]) << 24 | (uint32_t) (255 & (address)[1]) << 16 |
            (uint32_t) (255 & (address)[2]) << 8 | (uint32_t) (255 & (address)[3]));
}

/* One thread per image: walk the block headers. starts[i*nstored + b] = offset of block b's 4-byte
   length word in chunk i.  err[i] != 0 when the chunk does not end exactly where the header says.
 */
extern "C" __global__ void unblock(const uint8_t *const slab, const int64_t *const chunk_off,
                                   const int64_t *const chunk_end, uint32_t *const starts,
                                   const uint32_t nstored, const uint32_t tocopy, const uint32_t n,
                                   int32_t *const err) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t *c = slab + chunk_off[i];
    const uint64_t len = (uint64_t) (chunk_end[i] - chunk_off[i]);
    uint32_t *out = starts + (size_t) i * nstored;
    uint64_t pos = 12;
    int32_t e = 0;
    for (uint32_t b = 0; b < nstored; b++) {
        if (pos + 4 > len) {
            e = 1;
            break;
        }
        const uint32_t l = read32be(c + pos);
        if (pos + 4 + l > len) {
            e = 1;
            break;
        }
        out[b] = (uint32_t) pos;
        pos += 4 + l;
    }
    if (e == 0 && pos + tocopy != len) e = 2;
    err[i] = e;
}

/* grid (frames, (ntot + 1) / 2), block (32, 2) */
extern "C" __global__ void decompress(const uint8_t *const slab, const int64_t *const chunk_off,
                                      const int64_t *const chunk_end, const uint32_t *const starts,
                                      const int32_t *const err, const uint32_t nstored,
                                      const uint32_t ntot, const uint32_t total_bytes,
                                      const uint32_t bpp, uint8_t *const dense,
                                      const unsigned long long stride) {
    const int w = threadIdx.y, lane = threadIdx.x;
    const uint32_t im = blockIdx.x; /* images fastest */
    const uint32_t blockid = blockIdx.y * blockDim.y + w;
    __shared__ __align__(16) uint8_t inbuf[DECOMP_INPUT_BUFFER_SIZE * DECOMP_CHUNKS_PER_BLOCK];
    __shared__ __align__(16) uint8_t dec[DECOMP_CHUNKS_PER_BLOCK][8192];
    if (blockid >= ntot || err[im] != 0) return; /* a bad chunk is skipped (reported by the host) */
    const uint8_t *comp = slab + chunk_off[im];
    uint8_t *out = dense + (size_t) im * stride;
    const uint32_t nfull = total_bytes / 8192;
    const uint32_t part = total_bytes - nfull * 8192, tocopy = part % (bpp * 8),
                   bsize = part - tocopy;
    if (blockid < nstored) {
        const uint32_t *st = starts + (size_t) im * nstored;
        decompressStream(inbuf + w * DECOMP_INPUT_BUFFER_SIZE, dec[w], comp + st[blockid] + 4,
                         read32be(comp + st[blockid]));
    }
    __syncwarp();
    if (blockid < nfull) {
        uint32_t *o = (uint32_t *) (out + (size_t) blockid * 8192);
        if (bpp == 4)
            shuf32_warp(dec[w], o);
        else if (bpp == 2)
            shuf16_warp(dec[w], o);
        else
            shuf8_warp(dec[w], o);
    } else { /* the last, partial block: bsize bytes bit-shuffled (slowly unshuffled here), then
                tocopy raw bytes */
        uint8_t *ps = out + (size_t) nfull * 8192;
        const uint32_t nelb = bsize / bpp, bytestride = bsize / (8 * bpp);
        for (uint32_t dest = lane; dest < nelb; dest += 32) {
            const uint8_t *mybyte = dec[w] + dest / 8;
            const uint32_t mymask = 1U << (dest % 8);
            uint32_t val = 0;
            for (uint32_t i = 0; i < bpp * 8; i++) {
                if (*mybyte & mymask) val |= (1U << i);
                mybyte += bytestride;
            }
            for (uint32_t i = 0; i < bpp; i++)
                ps[dest * bpp + i] = (uint8_t) (val >> (8 * i));
        }
        if (lane < tocopy)
            ps[bsize + lane] = comp[(uint64_t) (chunk_end[im] - chunk_off[im]) - tocopy + lane];
    }
}

/* the sum of v over the warp, in lane 0; s: 32 entries of shared memory for this warp */
template <typename A> __device__ __forceinline__ A warp_sum(A v, A *s) {
#ifndef BSLZ4_NO_SHFL
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFU, v, o);
#else /* the same tree through shared memory */
    const int lane = threadIdx.x & 31;
    __syncwarp();
    s[lane] = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        __syncwarp();
        if (lane < o) s[lane] = v = v + s[lane + o];
    }
#endif
    return v;
}

/* T pixel, W weight (float or u32 fixed point), A accumulator; grid (nbins, ceil(nframes / F)),
 * block (WG <= 1024); on Fermi grid x may be below nbins (at most 65535) */
template <typename T, typename W, typename A, int F>
__device__ __forceinline__ void csr1d_bin(const int bin, const T *__restrict__ dense,
                                          const unsigned long long stride_el, const int nframes,
                                          const W *__restrict__ coef, const int *__restrict__ idx,
                                          const int *__restrict__ indptr, const int nbins,
                                          A *__restrict__ out) {
    __shared__ A sh[32 * F];
#ifdef BSLZ4_NO_SHFL
    __shared__ A red[32 * 32];
#else
    A *const red = 0;
#endif
    const int f0 = blockIdx.y * F;
    const int k0 = indptr[bin], k1 = indptr[bin + 1];
    const T *d = dense + (size_t) f0 * stride_el;
    const int nf = min(F, nframes - f0);
    A acc[F];
#pragma unroll
    for (int f = 0; f < F; f++)
        acc[f] = 0;
    if (nf == F) {
        for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
            const A c = (A) __ldg(coef + k);
            const size_t p = (size_t) __ldg(idx + k);
#pragma unroll
            for (int f = 0; f < F; f++)
                acc[f] += c * (A) __ldg(d + f * stride_el + p);
        }
    } else {
        for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
            const A c = (A) __ldg(coef + k);
            const size_t p = (size_t) __ldg(idx + k);
            for (int f = 0; f < nf; f++)
                acc[f] += c * (A) __ldg(d + f * stride_el + p);
        }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int f = 0; f < F; f++) {
        const A v = warp_sum(acc[f], red + wid * 32);
        if (lane == 0) sh[wid * F + f] = v;
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
        for (int f = 0; f < F; f++) {
            A v = (lane < nw) ? sh[lane * F + f] : (A) 0;
            v = warp_sum(v, red);
            if (lane == 0 && f < nf) out[(size_t) (f0 + f) * nbins + bin] = v;
        }
    }
}

template <typename T, typename W, typename A, int F>
__global__ void csr1d(const T *__restrict__ dense, const unsigned long long stride_el,
                      const int nframes, const W *__restrict__ coef, const int *__restrict__ idx,
                      const int *__restrict__ indptr, const int nbins, A *__restrict__ out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    for (int bin = blockIdx.x; bin < nbins; bin += gridDim.x) {
        csr1d_bin<T, W, A, F>(bin, dense, stride_el, nframes, coef, idx, indptr, nbins, out);
        __syncthreads(); /* sh and red are reused by the next bin */
    }
#else
    csr1d_bin<T, W, A, F>(blockIdx.x, dense, stride_el, nframes, coef, idx, indptr, nbins, out);
#endif
}

template <typename T> __device__ __forceinline__ bool above(T v, long long cut_i, float cut_f) {
    return (long long) v > cut_i;
}
template <> __device__ __forceinline__ bool above<float>(float v, long long cut_i, float cut_f) {
    return v > cut_f;
}

/* top_pixels (ImageD11.sinograms.lima_segmenter.top_pixels) on the device. Level 0 is the cut;
   level k = 1..nthr adds the sorted threshold k-1. Counts per 8 kB block and level: counts[(k * n +
   im) * ntot + block], pixels > cut, not masked, and > thresholds[k-1]. grid (frames, (ntot + 1) /
   2), block (32, 2) */
#define MAXTHR 16
template <typename T>
__global__ void count_above(const T *__restrict__ dense, const unsigned long long stride_el,
                            const uint32_t npix, const uint32_t ntot, const int32_t *const err,
                            const uint8_t *__restrict__ mask, const long long cut_i,
                            const float cut_f, const long long *__restrict__ thr_i,
                            const float *__restrict__ thr_f, const int nthr, const uint32_t n,
                            uint32_t *__restrict__ counts) {
    const int lane = threadIdx.x;
    const uint32_t im = blockIdx.x, blockid = blockIdx.y * blockDim.y + threadIdx.y;
    if (blockid >= ntot) return;
    uint32_t cnt[MAXTHR + 1];
    for (int k = 0; k <= nthr; k++)
        cnt[k] = 0;
    if (err[im] == 0) {
        const uint32_t be = 8192 / sizeof(T), p0b = blockid * be, nel = min(be, npix - p0b);
        const T *d = dense + (size_t) im * stride_el + p0b;
        for (uint32_t p0 = 0; p0 < nel; p0 += 32) {
            const uint32_t p = p0 + lane;
            T v = 0;
            if (p < nel) v = d[p];
            bool f = (p < nel) && above<T>(v, cut_i, cut_f) && mask[p0b + p] != 0;
            uint32_t b = __ballot_sync(0xFFFFFFFFU, f);
            cnt[0] += __popc(b);
            for (int k = 0; k < nthr && b != 0;
                 k++) { /* sorted thresholds: stop at the first one nobody passes */
                f = f && above<T>(v, thr_i[k], thr_f[k]);
                b = __ballot_sync(0xFFFFFFFFU, f);
                cnt[k + 1] += __popc(b);
            }
        }
    }
    if (lane == 0)
        for (int k = 0; k <= nthr; k++)
            counts[((size_t) k * n + im) * ntot + blockid] = cnt[k];
}

/* per (level, image) row: exclusive scan of the per-block counts, and the total.  grid (rows) */
extern "C" __global__ void scan_counts(const uint32_t *const counts, uint32_t *const offs,
                                       int32_t *const npx, const uint32_t nb) {
    __shared__ uint32_t part[257];
    const uint32_t im = blockIdx.x, t = threadIdx.x, per = (nb + 255) / 256;
    const uint32_t *c = counts + (size_t) im * nb;
    uint32_t *o = offs + (size_t) im * nb;
    uint32_t s = 0;
    for (uint32_t k = t * per; k < min(nb, (t + 1) * per); k++)
        s += c[k];
    part[t] = s;
    __syncthreads();
    if (t == 0) {
        uint32_t r = 0;
        for (int i = 0; i < 256; i++) {
            const uint32_t v = part[i];
            part[i] = r;
            r += v;
        }
        part[256] = r;
    }
    __syncthreads();
    uint32_t run = part[t];
    for (uint32_t k = t * per; k < min(nb, (t + 1) * per); k++) {
        o[k] = run;
        run += c[k];
    }
    if (t == 0) npx[im] = (int32_t) part[256];
}

/* per image, as top_pixels: all pixels above the cut if there are at most howmany; else the first
   threshold with fewer than howmany above it, else the last threshold; then at most howmany (the
   first ones in pixel order) */
extern "C" __global__ void choose_level(const int32_t *const npxl, const uint32_t n, const int nthr,
                                        const uint32_t howmany, const uint32_t fallback,
                                        int32_t *const level, int32_t *const npx) {
    const uint32_t im = blockIdx.x * blockDim.x + threadIdx.x;
    if (im >= n) return;
    if (fallback && (uint32_t) npxl[im] > howmany) {
        level[im] = -1;
        npx[im] = 0;
        return;
    } /* the host makes this frame's list */
    int L = 0;
    if ((uint32_t) npxl[im] > howmany && nthr > 0) {
        L = nthr;
        for (int k = 1; k <= nthr; k++)
            if ((uint32_t) npxl[(size_t) k * n + im] < howmany) {
                L = k;
                break;
            }
    }
    level[im] = L;
    npx[im] = (int32_t) min((uint32_t) npxl[(size_t) L * n + im], howmany);
}

/* exclusive prefix of the per-image totals (n <= a few hundred): the lists of a batch are stored
 * back to back */
extern "C" __global__ void scan_images(const int32_t *const npx, uint32_t *const imgoff,
                                       const uint32_t n) {
    if (threadIdx.x == 0) {
        uint32_t r = 0;
        for (uint32_t i = 0; i < n; i++) {
            imgoff[i] = r;
            r += (uint32_t) npx[i];
        }
    }
}

/* re-read each block and write the pixels of the image's level straight to their place in its list
   (pixel order), the first howmany only.  grid (frames, (ntot + 1) / 2), block (32, 2) */
template <typename T>
__global__ void
collect_write(const T *__restrict__ dense, const unsigned long long stride_el, const uint32_t npix,
              const uint32_t ntot, const int32_t *const err, const uint8_t *__restrict__ mask,
              const long long cut_i, const float cut_f, const long long *__restrict__ thr_i,
              const float *__restrict__ thr_f, const uint32_t n,
              const uint32_t *__restrict__ counts, const uint32_t *__restrict__ offs,
              const int32_t *__restrict__ level, const uint32_t *__restrict__ imgoff,
              const uint32_t howmany, T *__restrict__ out_val, uint32_t *__restrict__ out_idx) {
    const int lane = threadIdx.x;
    const uint32_t im = blockIdx.x, blockid = blockIdx.y * blockDim.y + threadIdx.y;
    if (blockid >= ntot || err[im] != 0) return;
    const int L = level[im];
    if (L < 0) return;
    const size_t row = ((size_t) L * n + im) * ntot + blockid;
    const uint32_t start = offs[row];
    if (counts[row] == 0 || start >= howmany) return;
    const long long li = L ? thr_i[L - 1] : cut_i;
    const float lf = L ? thr_f[L - 1] : cut_f;
    const uint32_t be = 8192 / sizeof(T), p0b = blockid * be, nel = min(be, npix - p0b);
    const T *d = dense + (size_t) im * stride_el + p0b;
    const size_t base = imgoff[im];
    uint32_t cnt = start;
    for (uint32_t p0 = 0; p0 < nel && cnt < howmany; p0 += 32) {
        const uint32_t p = p0 + lane;
        bool f = false;
        T v = 0;
        if (p < nel) {
            v = d[p];
            f = above<T>(v, cut_i, cut_f) && above<T>(v, li, lf) && mask[p0b + p] != 0;
        }
        const uint32_t bal = __ballot_sync(0xFFFFFFFFU, f);
        const uint32_t r = cnt + __popc(bal & ((1U << lane) - 1U));
        if (f && r < howmany) {
            out_val[base + r] = v;
            out_idx[base + r] = p0b + p;
        }
        cnt += __popc(bal);
    }
}
