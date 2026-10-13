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
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np

import bslz4_to_sparse as b

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
mask_filename = "mask.npy"  # your detector mask, pyFAI's convention: non-zero = masked pixel

with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    dtype, codec = ds.dtype, b.detect_codec(ds)
    frames = ds[:3]  # decompressed by HDF5: for the checks, and for from_frames below
    chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(3)]
mask = (np.load(mask_filename) == 0).astype(np.uint8)  # 1 = use this pixel
cut = 0  # every non-zero pixel: the frames are not sparse at this cut, which is what howmany is for
howmany = 2000

# 1. the default returns every pixel above the cut; howmany limits it
everything = b.chunk2sparse(mask, dtype, codec)
npx_all, _ = everything.multi(chunks, cut)
npx_all = npx_all.copy()
limited = b.chunk2sparse(mask, dtype, codec, howmany=howmany)
npx, (values, indices) = limited.multi(chunks, cut)
print("pixels above the cut:", npx_all.tolist(), "with howmany=%d:" % howmany, npx.tolist())
print("cut each frame ended with:", limited.last_cut.tolist())
assert (npx <= howmany).all()

# the rule, written out in numpy for the fullest frame
k = int(np.argmax(npx_all))
f = np.where(mask.ravel() > 0, frames[k].ravel(), 0)  # masked pixels are never kept
thresholds = [max(cut, 1) * 2**i for i in range(16)]  # max(cut, 1) * 2**i
level = next((t for t in thresholds if np.count_nonzero(f > t) < howmany), thresholds[-1])
want = np.flatnonzero(f > level)[:howmany]
assert npx[k] == len(want) and (indices[k, : npx[k]] == want).all()
assert limited.last_cut[k] == level
print("frame %d: threshold %d kept %d pixels, as the rule says" % (k, level, npx[k]))

# 2. frames that are not compressed give the same result
again = b.chunk2sparse(mask, dtype, codec, howmany=howmany)
npx2, (values2, indices2) = again.from_frames(frames, cut)
assert (npx2 == npx).all() and (indices2[1, : npx2[1]] == indices[1, : npx[1]]).all()
print("from_frames: the same pixels as the compressed chunks")

# 3. decompress alone: the pixel values, or the blocks before the bit unshuffle
back = b.decompress(chunks, dtype)
assert (back.reshape(frames.shape) == frames).all()
shuffled = b.decompress(chunks, dtype, unshuffle=False)
print("decompressed:", back.shape, "pixels; before the unshuffle:", shuffled.shape, "bytes")
