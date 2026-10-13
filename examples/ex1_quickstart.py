"""Quick start: one compressed Eiger chunk straight to sparse pixels.

An Eiger writes each frame as one bitshuffle-lz4 (or -zstd) compressed HDF5 chunk.  chunk2sparse
decodes such a chunk and keeps only the pixels above a cut that are not masked, without ever
building the dense frame in Python.  Here h5py is used to find the data and hand over one chunk
still compressed; example 2 shows the same without HDF5 in the loop.
"""

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np

import bslz4_to_sparse

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
mask_filename = "mask.npy"  # your detector mask, pyFAI's convention: non-zero = masked pixel

with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    print("dataset: shape %s, dtype %s, chunks %s" % (ds.shape, ds.dtype, ds.chunks))
    codec = bslz4_to_sparse.detect_codec(ds)  # lz4 or zstd, from the HDF5 filter
    frame0 = ds[0]  # decompressed by HDF5, only here to build the mask and to check
    chunk0 = ds.id.read_direct_chunk((0, 0, 0))[1]  # frame 0, still compressed

# The mask: where the detector has gaps or dead pixels.
bad = np.load(mask_filename) != 0  # the masked pixels
mask = (~bad).astype(np.uint8)  # ours: 1 = use this pixel

# The object is made once for a dataset: it knows the frame shape (from the mask) and the dtype.
c2s = bslz4_to_sparse.chunk2sparse(mask, dtype=ds.dtype, codec=codec)
cut = 3
npx, (values, indices) = c2s(chunk0, cut)  # the pixels above the cut

print("compressed chunk: %d bytes for %d pixels" % (len(chunk0), mask.size))
print("pixels above the cut of %d: %d" % (cut, npx))
print("first indices:", indices[:5].tolist(), "values:", values[:5].tolist())

# values and indices are full-size buffers owned by c2s: the first npx entries are the answer, and
# the next call on c2s overwrites them.  indices are positions in the flattened frame.
ref = np.flatnonzero((frame0.ravel() > cut) & (mask.ravel() > 0))
assert npx == ref.size and (indices[:npx] == ref).all()
assert (values[:npx] == frame0.ravel()[ref]).all()
print("same pixels as numpy finds in the frame HDF5 decompressed")

# coo() returns (row, column, value) copies instead, safe to keep
npx, row, col, vals = c2s.coo(chunk0, cut)
print("as (row, col, value):", list(zip(row[:3].tolist(), col[:3].tolist(), vals[:3].tolist())))

# A picture: the frame, and the sparse pixels that were kept from it.
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].imshow(np.ma.masked_where(mask == 0, frame0), vmin=0, vmax=10)
axes[0].set_title("frame 0, as HDF5 decompresses it (masked pixels blank)")
axes[1].plot(col, row, ".", ms=1)
axes[1].set_xlim(0, mask.shape[1])
axes[1].set_ylim(mask.shape[0], 0)
axes[1].set_aspect("equal")
axes[1].set_title("the %d pixels above the cut of %d, from the compressed chunk" % (npx, cut))
plt.show()
