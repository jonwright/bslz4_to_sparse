"""Quick start: compressed Eiger chunks straight to sparse pixels.

An Eiger writes each frame as one bitshuffle-LZ4 compressed HDF5 chunk.
chunk2sparse decodes such a chunk and keeps only the pixels above a cut
(and not masked), without ever building the dense frame in Python.
"""
import h5py
import hdf5plugin
import numpy as np

import bslz4_to_sparse

# A small stand-in for a detector file (kept in memory here): 4 frames of
# 512 x 512 uint16 photon counts, one frame per chunk, bitshuffle-LZ4.
rng = np.random.default_rng(42)
frames = rng.poisson(0.02, (4, 512, 512)).astype(np.uint16)
mask = np.ones((512, 512), np.uint8)       # 1 = use this pixel
mask[250:262, :] = 0                       # e.g. a gap between modules

with h5py.File("frames.h5", "w", driver="core", backing_store=False) as f:
    ds = f.create_dataset("data", data=frames, chunks=(1, 512, 512),
                          **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    codec = bslz4_to_sparse.detect_codec(ds)               # lz4 or zstd
    c2s = bslz4_to_sparse.chunk2sparse(mask, dtype=ds.dtype, codec=codec)
    chunk = ds.id.read_direct_chunk((0, 0, 0))[1]           # frame 0, still compressed
    npx, (values, indices) = c2s(chunk, 0)                  # pixels > 0, not masked

print("compressed chunk: %d bytes for %d pixels" % (len(chunk), mask.size))
print("pixels above the cut:", npx)
print("first indices:", indices[:5].tolist(), "values:", values[:5].tolist())

# values and indices are full-size buffers owned by c2s: the first npx
# entries are the answer, and the next call on c2s overwrites them.
ref = np.flatnonzero((frames[0] > 0).ravel() & (mask.ravel() > 0))
assert npx == ref.size and (indices[:npx] == ref).all()
assert (values[:npx] == frames[0].ravel()[ref]).all()
print("same pixels as numpy finds in the decompressed frame")

# coo() gives (row, column, value) copies instead
npx, row, col, vals = c2s.coo(chunk, 0)
print("as (row, col, value):", list(zip(row[:3].tolist(), col[:3].tolist(), vals[:3].tolist())))
