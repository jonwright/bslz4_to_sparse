"""Sparse pixels straight from compressed chunks.

Examples 1 to 3 use only names that the released packages had, called as those releases took them;
everything from example 4 on is new in 0.0.22.  An Eiger writes each frame as one bitshuffle-lz4
(or -zstd) compressed HDF5 chunk.  chunk2sparse decodes such a chunk and keeps only the pixels
above a cut that are not masked, without ever building the dense frame in Python; bslz4_to_sparse
does the same for one frame of a dataset.  Here h5py finds the data and hands over one chunk still
compressed.  (lz4 or zstd is found from the chunk: detect_codec and codec= are not needed.)
"""

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np

import bslz4_to_sparse

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
mask_filename = (
    "mask.h5"  # your detector mask: a dataset "mask", pyFAI's convention (non-zero = masked)
)

with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    print("dataset: shape %s, dtype %s, chunks %s" % (ds.shape, ds.dtype, ds.chunks))
    frame0 = ds[0]  # decompressed by HDF5, only here to build the mask and to check
    chunk0 = ds.id.read_direct_chunk((0, 0, 0))[1]  # frame 0, still compressed

# The mask: where the detector has gaps or dead pixels.
with h5py.File(mask_filename, "r") as h5:
    bad = h5["mask"][()] != 0  # the masked pixels
mask = (~bad).astype(np.uint8)  # ours: 1 = use this pixel

# The object is made once for a dataset: it knows the frame shape (from the mask) and the dtype.
c2s = bslz4_to_sparse.chunk2sparse(mask, ds.dtype)
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

# bslz4_to_sparse reads the chunk of a frame from an open dataset itself (h5py's read_direct_chunk)
with h5py.File(h5filename, "r") as h5:
    npx2, (values2, indices2) = bslz4_to_sparse.bslz4_to_sparse(h5[h5dataset], 0, cut, mask.ravel())
assert npx2 == npx and (indices2[:npx] == indices[:npx]).all()
print("bslz4_to_sparse(dataset, 0, cut, mask) finds the same %d pixels" % npx2)

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
