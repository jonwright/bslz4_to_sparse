"""Azimuthal integration in the same pass: pyFAI's CSC matrix.

chunk2sparseCSC also multiplies every frame by a sparse matrix while it
decodes.  pyFAI's CSC integration engine is such a matrix (bins x pixels),
so the powder pattern comes out of the decompression with no dense frame.
The result is pyFAI's sum_signal: divide by pyFAI's sum_normalization (from
integrating an image of ones, done once) for the intensity.
"""
import h5py
import hdf5plugin
import numpy as np
from pyFAI.detectors import Detector
from pyFAI.method_registry import IntegrationMethod

try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
except ImportError:                                       # older pyFAI
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

import bslz4_to_sparse

shape = (512, 512)
ai = AzimuthalIntegrator(dist=0.1, poni1=0.02, poni2=0.018, wavelength=3e-11,
                         detector=Detector(75e-6, 75e-6, max_shape=shape))

# a powder-ring frame with Poisson noise, stored as the detector would
q = ai.qArray()
mean = 0.05 + sum(3.0 * np.exp(-0.5 * ((q - qk) / 0.2) ** 2) for qk in (20, 35, 48, 60))
frame = np.random.default_rng(1).poisson(mean).astype(np.uint16)
with h5py.File("rings.h5", "w", driver="core", backing_store=False) as f:
    f.create_dataset("data", data=frame[None], chunks=(1,) + shape,
                     **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    chunk = f["data"].id.read_direct_chunk((0, 0, 0))[1]

method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
ref = ai.integrate1d(frame, 400, unit="q_nm^-1", method=method)      # pyFAI itself
engine = ai.engines[method].engine                                   # its CSC matrix
mask = np.ones(shape, np.uint8)

integ = bslz4_to_sparse.chunk2sparseCSC(mask, engine, dtype=np.uint16)
npx, (values, indices), powder = integ(chunk, 5)
print("pipeline dot:", bslz4_to_sparse.describe(integ.pipeline)["dot"])
print("powder: %d bins; pixels above the cut of 5: %d" % (powder.size, npx))
assert npx == np.count_nonzero(frame > 5)

signal = np.asarray(ref.sum_signal).ravel()
print("max |powder - pyFAI sum_signal| = %.3g (largest bin %.3g)"
      % (abs(powder - signal).max(), signal.max()))
assert np.allclose(powder, signal, rtol=1e-5, atol=1e-3)

norm = np.asarray(ref.sum_normalization).ravel()
intensity = np.divide(powder, norm, out=np.zeros_like(powder), where=norm > 0)
assert np.allclose(intensity, ref.intensity, rtol=1e-4, atol=1e-4)
print("intensity matches pyFAI's integrate1d")

# 2D (radial x azimuth): the matrix has nrad * nazim rows
method2 = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=2)
ref2 = ai.integrate2d(frame, 200, 36, unit="q_nm^-1", method=method2)
integ2 = bslz4_to_sparse.chunk2sparseCSC(mask, ai.engines[method2].engine, dtype=np.uint16)
_, _, powder2 = integ2(chunk, 5)
cake = powder2.reshape(200, 36).T                     # pyFAI's (azimuth, radial) layout
assert np.allclose(cake, ref2.sum_signal, rtol=1e-5, atol=1e-3)
print("2D cake", cake.shape, "matches pyFAI's integrate2d sum_signal")
