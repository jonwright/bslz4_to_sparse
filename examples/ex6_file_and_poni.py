"""Your data: an Eiger HDF5 file and a pyFAI .poni file, the whole job.

Open the file, take the geometry from the .poni, build the integrator once, then for every frame get
the pixels above the cut and the powder pattern straight from the compressed chunks, read by offset.
Nothing is written here: what to do with the results is up to you (examples 8, 10 and 11 write results, example 9 iterates over them).
"""

import os
import time

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np
import pyFAI
from pyFAI.method_registry import IntegrationMethod

import bslz4_to_sparse

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
mask_filename = (
    "mask.h5"  # your detector mask: a dataset "mask", pyFAI's convention (non-zero = masked)
)
poni_filename = "calib.poni"  # your pyFAI geometry
cut = 3  # keep the pixels above this
npt = 1000  # radial bins of the powder pattern

ai = pyFAI.load(poni_filename)
with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    dtype, codec, nframes = ds.dtype, bslz4_to_sparse.detect_codec(ds), min(ds.shape[0], 20)
    where = bslz4_to_sparse.harvest_chunk_offsets(ds)
    frame0 = ds[0]

with h5py.File(mask_filename, "r") as h5:
    bad = h5["mask"][()] != 0  # the masked pixels: gaps and dead pixels
mask = (~bad).astype(np.uint8)  # ours: 1 = use this pixel

# pyFAI builds its sparse matrix (and the normalisation of an image of ones) once
method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
ones = ai.integrate1d(
    np.ones(frame0.shape, np.float32), npt, unit="q_nm^-1", method=method, mask=bad
)
sum_normalization = np.asarray(ones.sum_normalization).ravel()
integ = bslz4_to_sparse.chunk2sparseCSC(mask, ai.engines[method].engine, dtype=dtype, codec=codec)

scale = integ.powder_scale or 1.0  # integer sums (16-bit pixels) times 2**-b, or already float
npixels = np.empty(nframes, np.int64)
powder_sums = np.empty((nframes, npt))
fd = os.open(h5filename, os.O_RDONLY)
t0 = time.perf_counter()
for i in range(nframes):
    offset, size = where[i]
    npx, (values, indices), powder_sum = integ(os.pread(fd, size, offset), cut)
    npixels[i] = npx
    powder_sums[i] = powder_sum * scale
    # values[:npx], indices[:npx] are this frame's sparse pixels: use or copy them here, the next
    # call overwrites them
elapsed = time.perf_counter() - t0
os.close(fd)

intensity = np.divide(
    powder_sums, sum_normalization, out=np.zeros_like(powder_sums), where=sum_normalization > 0
)
print("%d frames in %.3f s" % (nframes, elapsed))
print(
    "pixels above the cut of %d per frame: mean %.0f, min %d, max %d"
    % (cut, npixels.mean(), npixels.min(), npixels.max())
)
mean = intensity.mean(axis=0)
top = np.argsort(mean)[::-1][:3]
print("strongest powder bins, q (nm^-1):", np.round(ones.radial[np.sort(top)], 2).tolist())

# frame 0 against pyFAI itself and numpy
ref = ai.integrate1d(frame0, npt, unit="q_nm^-1", method=method, mask=bad)
assert np.allclose(intensity[0], ref.intensity, rtol=1.2e-7, atol=0)
assert npixels[0] == np.count_nonzero((frame0 > cut) & ~bad)
print("frame 0 matches pyFAI's integrate1d, and numpy's count above the cut")

# A picture: the mean powder pattern of the frames, and the sparse pixels each frame gave.
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(ones.radial, mean)
axes[0].set_xlabel("q (nm$^{-1}$)")
axes[0].set_ylabel("mean intensity")
axes[0].set_title("powder pattern, mean of %d frames" % nframes)
axes[1].plot(npixels, "o-")
axes[1].set_xlabel("frame")
axes[1].set_title("pixels above the cut of %d" % cut)
plt.tight_layout()
plt.show()
