"""Batches of frames, and reading chunks by file offset.

multi() decodes a list of chunks in one call (one GIL release for the whole
batch).  decode_offsets() does the same for chunks given as byte offsets
into one buffer, such as a memory-mapped HDF5 file: harvest_chunk_offsets
finds where each frame's chunk is, so no h5py call happens per frame.
Here the file lives in memory; with a file on disk the buffer would be
np.memmap(filename, dtype=np.uint8, mode="r").
"""
import h5py
import hdf5plugin
import numpy as np

import bslz4_to_sparse

rng = np.random.default_rng(7)
frames = rng.poisson(0.05, (10, 256, 256)).astype(np.uint16)
mask = np.ones((256, 256), np.uint8)
c2s = bslz4_to_sparse.chunk2sparse(mask, dtype=np.uint16)

with h5py.File("frames.h5", "w", driver="core", backing_store=False) as f:
    ds = f.create_dataset("data", data=frames, chunks=(1, 256, 256),
                          **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    f.flush()
    file_bytes = np.frombuffer(f.id.get_file_image(), np.uint8)   # the whole file
    # 1. a batch of chunks read with h5py
    chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(ds.shape[0])]
    offsets = bslz4_to_sparse.harvest_chunk_offsets(ds)    # {frame: (offset, size)}

npx, (values, indices) = c2s.multi(chunks, 0)
print("multi: npx per frame", npx.tolist())
print("values/indices arrays:", values.shape, indices.shape)
batch_npx = npx.copy()          # the next call on c2s reuses these buffers

# 2. the same frames by byte offset into the file's bytes
offs, lens = bslz4_to_sparse.pack_offsets_lengths(offsets, range(10))
npx, (values, indices) = c2s.decode_offsets(file_bytes, offs, lens, 0)
print("decode_offsets: npx per frame", npx.tolist())
assert (npx == batch_npx).all()

for k in range(10):
    ref = np.flatnonzero(frames[k].ravel() > 0)
    assert (indices[k, :npx[k]] == ref).all()
print("every frame matches numpy")
