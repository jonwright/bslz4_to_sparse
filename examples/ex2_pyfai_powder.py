"""A powder pattern from the same pass: pyFAI's sparse matrix in the decoder (released API).

chunk2sparseCSC(mask, csc, dtype) is as in 0.0.15 and 0.0.16: chunk2sparse plus one more output.
While a chunk is decoded, every pixel is also multiplied into a sparse matrix (bins x pixels) and
added into the bins it falls in.  pyFAI's CSC integration engine is exactly such a matrix, so the
powder pattern comes out of the decompression with no dense frame.  A call returns

    npx            the number of pixels above the cut
    values, indices  those pixels (values and their flat positions; the first npx entries)
    powder_sum     for every bin, the sum over its pixels of weight x pixel value: pyFAI's
                   sum_signal, float64

pyFAI divides powder_sum by sum_normalization, the same sum for an image of ones with the
corrections (solid angle, polarisation, flat field) folded in, to get the intensity.
sum_normalization does not depend on the data: it is computed once.  One change since the
releases: for 8 and 16-bit pixels the powder defaults to exact integer sums now, so the float64 of
the released versions is asked for with exact=False, as here; the last part of this example shows
the new default.
"""

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

with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    dtype = ds.dtype
    frame0 = ds[0]
    chunk0 = ds.id.read_direct_chunk((0, 0, 0))[1]
with h5py.File(mask_filename, "r") as h5:
    bad = h5["mask"][()] != 0  # the masked pixels
mask = (~bad).astype(np.uint8)

# pyFAI makes its sparse matrix once, when it first integrates with a sparse method (bbox pixel
# splitting, CSC layout).  Integrating an image of ones also gives sum_normalization.
ai = pyFAI.load(poni_filename)
method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
ones = ai.integrate1d(
    np.ones(frame0.shape, np.float32), 400, unit="q_nm^-1", method=method, mask=bad
)
engine = ai.engines[method].engine  # pyFAI's CSC matrix
sum_normalization = np.asarray(ones.sum_normalization).ravel()

# Our object: decode, collect the sparse pixels and multiply into the matrix, all in one pass.
integ = bslz4_to_sparse.chunk2sparseCSC(mask, engine, dtype, exact=False)
cut = 3
npx, (values, indices), powder_sum = integ(chunk0, cut)
print("pipeline dot:", bslz4_to_sparse.describe(integ.pipeline)["dot"])
print("sparse pixels above the cut of %d: %d" % (cut, npx))
print("powder_sum: %d bins of %s" % (powder_sum.size, powder_sum.dtype))
sum_signal = powder_sum

# the same frame, integrated by pyFAI from the decompressed pixels
ref = ai.integrate1d(frame0, 400, unit="q_nm^-1", method=method, mask=bad)
assert npx == np.count_nonzero((frame0 > cut) & ~bad)
signal = np.asarray(ref.sum_signal).ravel()
print(
    "largest difference from pyFAI's sum_signal: %.3g (largest bin %.3g)"
    % (abs(sum_signal - signal).max(), signal.max())
)
assert np.allclose(sum_signal, signal, rtol=1e-9, atol=0)

intensity = np.divide(
    sum_signal, sum_normalization, out=np.zeros_like(sum_signal), where=sum_normalization > 0
)
assert np.allclose(intensity, ref.intensity, rtol=1.2e-7, atol=0)
print("intensity = powder_sum / sum_normalization equals pyFAI's integrate1d")

# A picture: why powder_sum is divided by sum_normalization.  The raw sum in a bin grows with the
# number of pixels that fall in it (more pixels at large radius, none in the module gaps); the
# normalisation counts them (weighted), and the ratio is the intensity.
import matplotlib.pyplot as plt

q = ones.radial
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].plot(q, sum_signal)
axes[0].set_title("powder_sum: sum of pixel values in each bin")
axes[1].plot(q, sum_normalization)
axes[1].set_title("sum_normalization: pixels in each bin")
axes[2].plot(q, intensity, label="powder_sum / sum_normalization")
axes[2].plot(q, ref.intensity, "--", label="pyFAI integrate1d")
axes[2].set_title("intensity")
axes[2].legend()
for ax in axes:
    ax.set_xlabel("q (nm$^{-1}$)")
plt.tight_layout()
plt.show()

# Only the powder is wanted? Then no pixel needs collecting: a cut at the dtype's maximum finds none
# (and howmany=0 would keep none of any that were found).
nothing = np.iinfo(dtype).max
only_powder = bslz4_to_sparse.chunk2sparseCSC(mask, engine, dtype, exact=False)
npx, _, powder_only = only_powder(chunk0, nothing)
assert npx == 0 and np.array_equal(powder_only, powder_sum)
print("with the cut at the dtype maximum: no pixels, the same powder_sum")

# 2D (radial x azimuth): the matrix has nrad * nazim rows, in radial-major order
method2 = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=2)
ref2 = ai.integrate2d(frame0, 100, 18, unit="q_nm^-1", method=method2, mask=bad)
integ2 = bslz4_to_sparse.chunk2sparseCSC(mask, ai.engines[method2].engine, dtype, exact=False)
_, _, powder_sum2 = integ2(chunk0, nothing)
cake = powder_sum2.reshape(100, 18).T  # pyFAI's (azimuth, radial) layout
assert np.allclose(cake, ref2.sum_signal, rtol=1e-9, atol=0)
print("2D cake", cake.shape, "matches pyFAI's integrate2d sum_signal")

# New in 0.0.22, and the default for 8 and 16-bit pixels: exact sums.  The powder is then int64, the
# sums of fixed-point integer weights, and powder_scale (a power of two) turns it into float; for
# 32-bit pixels (or exact=False) the sums are float64 and powder_scale is None.
exact = bslz4_to_sparse.chunk2sparseCSC(mask, engine, dtype)
_, _, powder_int = exact(chunk0, nothing)
print("default: powder %s, powder_scale %s" % (powder_int.dtype, exact.powder_scale))
if exact.powder_scale is not None:
    assert np.allclose(powder_int * exact.powder_scale, powder_sum, rtol=1e-6, atol=0)
