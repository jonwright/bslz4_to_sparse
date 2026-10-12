"""Limit the pixels per frame, work on uncompressed frames, decompress alone.

howmany keeps at most that many pixels per frame, chosen as ImageD11's
top_pixels does: a frame with more pixels above the cut moves to the first of
the sorted thresholds that leaves fewer than howmany (the default thresholds
double from the cut), and then keeps the first howmany of them in pixel order.
The selection runs inside the decode, so a frame that is too full costs
little.  from_frames() runs the same processing on frames that are not
compressed, and decompress() gives the frames back without any sparse step.
"""

import h5py
import hdf5plugin
import numpy as np

import bslz4_to_sparse as b

rng = np.random.default_rng(3)
shape = (128, 160)
frames = np.stack([rng.poisson(lam, shape) for lam in (0.02, 1.0, 6.0)]).astype(np.uint16)
mask = np.ones(shape, np.uint8)
with h5py.File("f.h5", "w", driver="core", backing_store=False) as f:
    ds = f.create_dataset(
        "d", data=frames, chunks=(1,) + shape, **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
    )
    chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]

# 1. the default returns every pixel above the cut; howmany limits it
everything = b.chunk2sparse(mask, np.uint16)
npx_all, _ = everything.multi(chunks, 0)
npx_all = npx_all.copy()
limited = b.chunk2sparse(mask, np.uint16, howmany=500)
npx, (values, indices) = limited.multi(chunks, 0)
print("pixels above the cut:", npx_all.tolist(), " with howmany=500:", npx.tolist())
print("cut each frame ended with:", limited.last_cut.tolist())
assert (npx <= 500).all() and npx[0] == npx_all[0]  # the sparse frame is untouched

# the rule, written out in numpy for the dense frame
f2 = frames[2].ravel()
thresholds = [max(0, 1) * 2**i for i in range(16)]  # max(cut, 1) * 2**i
cut = next((t for t in thresholds if np.count_nonzero(f2 > t) < 500), thresholds[-1])
want = np.flatnonzero(f2 > cut)[:500]
assert npx[2] == len(want) and (indices[2, : npx[2]] == want).all()
assert limited.last_cut[2] == cut
print("frame 2: threshold", cut, "kept", int(npx[2]), "pixels, as the rule says")

# 2. frames that are not compressed give the same result
again = b.chunk2sparse(mask, np.uint16, howmany=500)
npx2, (values2, indices2) = again.from_frames(frames, 0)
assert (npx2 == npx).all() and (indices2[1, : npx2[1]] == indices[1, : npx[1]]).all()
print("from_frames: the same pixels as the compressed chunks")

# 3. decompress alone: the pixel values, or the blocks before the bit unshuffle
back = b.decompress(chunks, np.uint16)
assert (back.reshape(frames.shape) == frames).all()
shuffled = b.decompress(chunks, np.uint16, unshuffle=False)
print("decompressed:", back.shape, "pixels; before the unshuffle:", shuffled.shape, "bytes")
