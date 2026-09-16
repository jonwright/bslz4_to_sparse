"""
Time each untranspose backend available_backends() reports, against scal.

The backends differ only in the bit/byte de-shuffle kernel, so this is the
measurement that says whether a SIMD one is worth having on this machine:
kcb is x86-only internally, sse is upstream bitshuffle's SSE2 (reached on
POWER through GCC's VSX-backed x86-intrinsic headers), neon is aarch64.

Run directly: python3 bench_backends.py
"""
import os
import sys
import time

import h5py
import hdf5plugin
import numpy as np

path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if path:
    sys.path.insert(0, path)
import bslz4_to_sparse as bslz4

print("Running from", bslz4.__file__)

SHAPE = (32, 1024, 1024)  # (nframes, ni, nj)
DTYPE = np.uint16
MU = 0.01  # sparse, like real detector frames
FNAME = "bench_backends.h5"
REPEATS = 3

if not os.path.exists(FNAME):
    rng = np.random.default_rng(12345)
    ary = rng.poisson(lam=MU, size=SHAPE).astype(DTYPE)
    with h5py.File(FNAME, "w") as f:
        f.create_dataset("data", data=ary, chunks=(1, SHAPE[1], SHAPE[2]),
                         **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))

with h5py.File(FNAME, "r") as f:
    ds = f["data"]
    chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(SHAPE[0])]

mask = np.ones(SHAPE[1:], np.uint8)


def _timed(c2sm):
    t0 = time.perf_counter()
    c2sm(chunks, 0)
    return time.perf_counter() - t0


def time_backend(name):
    """Best of REPEATS, per frame, in ms. Best rather than mean: the
    interesting number is the kernel, not the scheduler."""
    bslz4.set_backend(name)
    c2sm = bslz4.chunk2sparseMulti(mask, dtype=DTYPE)
    c2sm(chunks, 0)  # warmup
    return min(_timed(c2sm) for _ in range(REPEATS)) / SHAPE[0] * 1e3


try:
    results = {name: time_backend(name) for name in bslz4.available_backends()}
finally:
    bslz4.set_backend(None)

base = results.get("scal")
print("\n%-8s %10s %9s" % ("backend", "ms/frame", "vs scal"))
for name, ms in sorted(results.items(), key=lambda kv: kv[1]):
    speedup = "%.2fx" % (base / ms) if base else "-"
    print("%-8s %10.4f %9s" % (name, ms, speedup))
