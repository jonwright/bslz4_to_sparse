"""Threads: one processor object per thread.

The decode releases the GIL (and on free-threaded Python 3.14t/3.15t there
is no GIL), so threads decode in parallel.  Objects keep and reuse their own
work and output buffers and nothing is locked, so each thread must use its
own object, and must copy what it keeps before its next call.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import h5py
import hdf5plugin
import numpy as np

import bslz4_to_sparse

rng = np.random.default_rng(3)
frames = rng.poisson(0.05, (32, 512, 512)).astype(np.uint16)
with h5py.File("frames.h5", "w", driver="core", backing_store=False) as f:
    ds = f.create_dataset("data", data=frames, chunks=(1, 512, 512),
                          **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]
mask = np.ones((512, 512), np.uint8)

local = threading.local()


def sparsify(chunk):
    if not hasattr(local, "c2s"):                 # first call in this thread
        local.c2s = bslz4_to_sparse.chunk2sparse(mask, dtype=np.uint16)
    npx, (values, indices) = local.c2s(chunk, 0)
    return values[:npx].copy(), indices[:npx].copy()   # copies: the buffers are reused


with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(sparsify, chunks))

for frame, (values, indices) in zip(frames, results):
    ref = np.flatnonzero(frame.ravel() > 0)
    assert (indices == ref).all() and (values == frame.ravel()[ref]).all()
print("%d frames decoded on 4 threads, each matching numpy" % len(results))
print("pixels per frame:", [len(v) for v, _ in results[:8]], "...")
