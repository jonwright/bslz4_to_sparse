"""
Small benchmark for two design questions from the c2py23 rewrite:

1. zstd vs lz4 decode speed (issue #10), at two very different pixel
   statistics: Poisson mu=0.001 (near-empty, mostly-zero frames, typical
   of a fast/low-flux scan) vs mu=1 (denser, more "random-looking" data
   that compresses less well and exercises more of the untranspose/
   decompress work per frame).

2. chunk2sparseCSC (one call per frame) vs chunk2sparseCSCmulti (one
   batched call per group of frames) per-frame time (issue #12), which
   is supposed to win by doing the CSC (indptr/indices/data) lookup for
   each pixel once and applying it across every frame in the batch
   instead of re-walking it once per frame.

Run directly: python3 bench_zstd_multichunk.py
"""
import os
import time

import h5py
import hdf5plugin
import numpy as np

import bslz4_to_sparse as bslz4

SHAPE = (64, 512, 512)  # (nframes, ni, nj) -- modest size, quick to run
DTYPE = np.uint16
FNAME = "bench_zstd_multichunk.h5"


def make_dataset(f, name, mu, cname):
    rng = np.random.default_rng(12345)
    npix = SHAPE[1] * SHAPE[2]
    ary = rng.poisson(lam=mu, size=SHAPE[0] * npix).astype(DTYPE).reshape(SHAPE)
    if name in f:
        del f[name]
    f.create_dataset(
        name,
        data=ary,
        chunks=(1, SHAPE[1], SHAPE[2]),
        **hdf5plugin.Bitshuffle(nelems=0, cname=cname, clevel=3),
    )


def read_all_chunks(ds):
    return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(SHAPE[0])]


def bench_read_speed(ds, codec):
    """Per-frame time (ms) for bslz4_to_sparse() over the whole dataset."""
    npix = SHAPE[1] * SHAPE[2]
    mask = np.ones(npix, np.uint8)
    values = np.empty(npix, DTYPE)
    indices = np.empty(npix, np.uint32)
    workspace = np.empty(3 * 8192, np.uint8)
    bslz4.bslz4_to_sparse(ds, 0, 0, mask, (values, indices), workspace, codec)  # warmup

    t0 = time.perf_counter()
    npx_total = 0
    for i in range(SHAPE[0]):
        npx, _ = bslz4.bslz4_to_sparse(ds, i, 0, mask, (values, indices), workspace, codec)
        npx_total += npx
    dt_ms = (time.perf_counter() - t0) / SHAPE[0] * 1e3
    return dt_ms, npx_total / SHAPE[0]


class RadialCSC:
    """A minimal synthetic 1-bin-per-pixel radial integrator: same CSC
    shape (data, indices, indptr, shape) pyFAI/scipy.sparse would give,
    just without depending on either. One weight-1.0 entry per pixel is
    enough to exercise the same indptr/indices/data access pattern as a
    real azimuthal integrator."""

    def __init__(self, ni, nj, nbins):
        yy, xx = np.mgrid[0:ni, 0:nj]
        r = np.sqrt((yy - ni / 2.0) ** 2 + (xx - nj / 2.0) ** 2).ravel()
        bin_idx = np.clip((r / (r.max() + 1e-6) * nbins).astype(np.uint32), 0, nbins - 1)
        npix = ni * nj
        self.indptr = np.arange(npix + 1, dtype=np.uint32)
        self.indices = bin_idx
        self.data = np.ones(npix, np.float32)
        self.shape = (nbins,)


def bench_csc_single(mask, csc, buffers):
    c2s = bslz4.chunk2sparseCSC(mask, csc, dtype=DTYPE)
    c2s(buffers[0], 0)  # warmup
    t0 = time.perf_counter()
    for buf in buffers:
        c2s(buf, 0)
    return (time.perf_counter() - t0) / len(buffers) * 1e3


def bench_csc_multi(mask, csc, buffers, batch_size):
    c2sm = bslz4.chunk2sparseCSCmulti(mask, csc, dtype=DTYPE)
    c2sm(buffers[:batch_size], 0)  # warmup
    t0 = time.perf_counter()
    for start in range(0, len(buffers), batch_size):
        c2sm(buffers[start:start + batch_size], 0)
    return (time.perf_counter() - t0) / len(buffers) * 1e3


def main():
    if os.path.exists(FNAME):
        os.remove(FNAME)

    print("=== zstd vs lz4 read speed (bslz4_to_sparse) ===")
    print(f"{'mu':>8} {'codec':>6} {'ms/frame':>10} {'px/frame':>10}")
    with h5py.File(FNAME, "a") as f:
        for mu in (0.001, 1.0):
            for cname, codec in (("lz4", bslz4.CODEC_LZ4), ("zstd", bslz4.CODEC_ZSTD)):
                name = f"read_mu{mu}_{cname}"
                make_dataset(f, name, mu, cname)
                ds = f[name]
                dt_ms, npx = bench_read_speed(ds, codec)
                print(f"{mu:>8} {cname:>6} {dt_ms:>10.4f} {npx:>10.1f}")

    print()
    print("=== chunk2sparseCSC (single) vs chunk2sparseCSCmulti (batched) ===")
    npix = SHAPE[1] * SHAPE[2]
    mask = np.ones((SHAPE[1], SHAPE[2]), np.uint8)
    csc = RadialCSC(SHAPE[1], SHAPE[2], nbins=1000)
    print(f"{'mu':>8} {'batch':>6} {'single ms/f':>12} {'multi ms/f':>11} {'speedup':>8}")
    with h5py.File(FNAME, "r") as f:
        for mu in (0.001, 1.0):
            name = f"read_mu{mu}_lz4"
            ds = f[name]
            buffers = read_all_chunks(ds)
            t_single = bench_csc_single(mask, csc, buffers)
            for batch_size in (4, 16, SHAPE[0]):
                t_multi = bench_csc_multi(mask, csc, buffers, batch_size)
                print(f"{mu:>8} {batch_size:>6} {t_single:>12.4f} {t_multi:>11.4f} "
                      f"{t_single / t_multi:>8.2f}x")

    os.remove(FNAME)


if __name__ == "__main__":
    main()
