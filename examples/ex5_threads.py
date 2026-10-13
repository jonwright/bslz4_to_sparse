"""Threads: read and decode many frames at once, one decoder object per thread.

The decode releases the GIL (and on free-threaded Python 3.14t/3.15t there is no GIL), so threads
decode in parallel, and so does reading a file with os.pread.  Objects keep and reuse their own work
and output buffers and nothing is locked, so each thread owns one object and copies what it keeps
before its next call.  Here every thread reads its own frames from the file by offset (the offsets
are found once, see example 2) and decodes them, with no HDF5 call in the loop.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np

import bslz4_to_sparse

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it

with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    dtype, codec = ds.dtype, bslz4_to_sparse.detect_codec(ds)
    where = bslz4_to_sparse.harvest_chunk_offsets(ds)  # {frame: (byte offset, byte size)}
    frame0 = ds[0]
mask = (frame0 != np.iinfo(dtype).max).astype(np.uint8)
nframes = min(len(where), 16)
cut = 3
nthreads = 4

# one decoder object per thread: made here, before the threads start, and used by one thread only
decoders = [bslz4_to_sparse.chunk2sparse(mask, dtype=dtype, codec=codec) for _ in range(nthreads)]
fd = os.open(h5filename, os.O_RDONLY)


def sparsify(k):
    """Thread k: frames k, k + nthreads, ...: pread each chunk, decode, keep copies."""
    out = {}
    for frame in range(k, nframes, nthreads):
        offset, size = where[frame]
        chunk = os.pread(fd, size, offset)  # releases the GIL while it waits for the disk
        npx, (values, indices) = decoders[k](chunk, cut)  # releases the GIL while it decodes
        out[frame] = (values[:npx].copy(), indices[:npx].copy())  # copies: the buffers are reused
    return out


with ThreadPoolExecutor(max_workers=nthreads) as pool:
    results = {}
    for part in pool.map(sparsify, range(nthreads)):
        results.update(part)
os.close(fd)

with h5py.File(h5filename, "r") as h5:  # only to check the result
    for frame in range(nframes):
        values, indices = results[frame]
        ref = np.flatnonzero((h5[h5dataset][frame].ravel() > cut) & (mask.ravel() > 0))
        assert (indices == ref).all()
print("%d frames read and decoded on %d threads, each matching numpy" % (nframes, nthreads))
print("pixels per frame:", [len(results[f][0]) for f in range(min(nframes, 8))], "...")
