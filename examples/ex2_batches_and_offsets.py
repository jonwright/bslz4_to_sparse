"""Without HDF5 in the loop: find the chunks once, then read the file and decode by byte offset.

HDF5 stores each frame's compressed chunk somewhere in the file.  harvest_chunk_offsets finds where
(one pass through h5py, and the only use of HDF5 here: the result could be kept beside the file).
After that the file is read with ordinary file reads, and decode_offsets decodes a batch of frames
straight from the bytes read, with one release of the GIL for the whole batch.
"""

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np

import bslz4_to_sparse

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
mask_filename = "mask.npy"  # your detector mask, pyFAI's convention: non-zero = masked pixel

# 1. once, with HDF5: where is each chunk, and what are the shape and dtype of the frames?
with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    nframes, shape, dtype = ds.shape[0], ds.shape[1:], ds.dtype
    codec = bslz4_to_sparse.detect_codec(ds)
    where = bslz4_to_sparse.harvest_chunk_offsets(ds)  # {frame: (byte offset, byte size)}
print(
    "%d frames of %s %s; frame 0 is %d bytes at offset %d"
    % ((nframes, shape, dtype) + (where[0][1], where[0][0]))
)
bad = np.load(mask_filename) != 0  # the masked pixels
mask = (~bad).astype(np.uint8)  # ours: 1 = use this pixel

# 2. no HDF5 from here on: the file is read as bytes
file_bytes = np.fromfile(h5filename, dtype=np.uint8)
nbatch = min(nframes, 8)  # the output buffers have one full-size row per frame of the batch
offsets, lengths = bslz4_to_sparse.pack_offsets_lengths(where, range(nbatch))

c2s = bslz4_to_sparse.chunk2sparse(mask, dtype=dtype, codec=codec)
cut = 3
npx, (values, indices) = c2s.decode_offsets(file_bytes, offsets, lengths, cut)
print("pixels above the cut of %d, per frame:" % cut, npx.tolist())
print("values and indices have one row per frame:", values.shape, indices.shape)
# the buffers belong to c2s and are reused by its next call: copy what you keep

with h5py.File(h5filename, "r") as h5:  # only to check the result
    frames = h5[h5dataset][:nbatch]
for k in range(nbatch):
    ref = np.flatnonzero((frames[k].ravel() > cut) & (mask.ravel() > 0))
    assert (indices[k, : npx[k]] == ref).all()
print("every frame matches numpy on the frames HDF5 decompressed")

# A list of chunks held in memory goes through multi() in the same way
chunks = [file_bytes[o : o + n].tobytes() for o, n in zip(offsets.tolist(), lengths.tolist())]
batch_npx = npx.copy()
npx, _ = c2s.multi(chunks, cut)
assert (npx == batch_npx).all()
print("multi() on the same chunks gives the same counts")
