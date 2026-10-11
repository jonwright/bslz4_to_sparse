"""Your data: an Eiger HDF5 file and a pyFAI .poni file.

The whole job in one function: open the file, take the geometry from the
.poni, build the integrator once, then for every frame get the pixels above
the cut and the powder pattern straight from the compressed chunks.  Run it
on your own data as

    python ex6_file_and_poni.py data.h5 /entry_0000/ESRF-ID11/eiger/data calib.poni [cut] [nframes]

With no arguments (as when this page was built) it first writes a small
stand-in: an Eiger 1M file of powder rings and a .poni for it.
"""
import os
import sys
import tempfile
import time

import h5py
import hdf5plugin
import numpy as np
import pyFAI
from pyFAI.method_registry import IntegrationMethod

import bslz4_to_sparse


def process(h5name, dataset, poni, cut=0, nframes=None, npt=1000):
    """Sparse pixels and a 1D powder pattern for every frame of the dataset."""
    ai = pyFAI.load(poni)
    with h5py.File(h5name, "r") as f:
        ds = f[dataset]
        n = ds.shape[0] if nframes is None else min(nframes, ds.shape[0])
        frame0 = ds[0]
        # masked: the detector's gaps from the .poni, and pixels holding the
        # detector's bad-pixel value (the dtype maximum, or 65535 for 16-bit
        # data stored as uint32) in the first frame
        bad = frame0 == np.iinfo(ds.dtype).max
        if ds.dtype.itemsize > 2:
            bad |= frame0 == 65535
        if ai.detector.mask is not None:
            bad |= ai.detector.mask.astype(bool)
        mask = (~bad).astype(np.uint8)

        # pyFAI builds its CSC matrix (and the normalisation) once
        method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
        ones = ai.integrate1d(np.ones(ds.shape[1:], np.float32), npt, unit="q_nm^-1",
                              method=method, mask=bad)
        norm = np.asarray(ones.sum_normalization).ravel()
        integ = bslz4_to_sparse.chunk2sparseCSC(mask, ai.engines[method].engine,
                                                dtype=ds.dtype,
                                                codec=bslz4_to_sparse.detect_codec(ds))

        npixels = np.empty(n, np.int64)
        powders = np.empty((n, npt))
        t0 = time.perf_counter()
        for i in range(n):
            chunk = ds.id.read_direct_chunk((i, 0, 0))[1]
            npx, (values, indices), powder = integ(chunk, cut)
            npixels[i] = npx
            powders[i] = powder
            # values[:npx], indices[:npx] are this frame's sparse pixels: copy
            # or use them here, the next call overwrites them
        elapsed = time.perf_counter() - t0
    intensity = np.divide(powders, norm, out=np.zeros_like(powders), where=norm > 0)
    return ones.radial, intensity, npixels, elapsed


def make_standin(folder):
    """An Eiger 1M stand-in: 20 frames of LaB6-like rings and its .poni."""
    from pyFAI.calibrant import get_calibrant
    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:                                   # older pyFAI
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    ai = AzimuthalIntegrator(dist=0.12, poni1=0.040, poni2=0.039, wavelength=3e-11,
                             detector=pyFAI.detector_factory("Eiger1M"))
    poni = os.path.join(folder, "calib.poni")
    ai.write(poni)
    cal = get_calibrant("LaB6")
    cal.wavelength = ai.wavelength
    q = ai.qArray()
    mean = 0.02 + sum(5.0 / (1 + 0.3 * k) * np.exp(-0.5 * ((q - qk) / 0.15) ** 2)
                      for k, qk in enumerate(cal.get_peaks("q_nm^-1")[:30]))
    rng = np.random.default_rng(0)
    frames = rng.poisson(mean, (20,) + mean.shape).astype(np.uint16)
    frames[:, ai.detector.mask.astype(bool)] = 65535          # the module gaps
    h5name = os.path.join(folder, "eiger_0000.h5")
    with h5py.File(h5name, "w") as f:
        f.create_dataset("/entry_0000/measurement/data", data=frames,
                         chunks=(1,) + mean.shape,
                         **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    return h5name, "/entry_0000/measurement/data", poni


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        h5name, dataset, poni = sys.argv[1:4]
        cut = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        nframes = int(sys.argv[5]) if len(sys.argv) > 5 else None
        q, intensity, npixels, elapsed = process(h5name, dataset, poni, cut, nframes)
    else:
        with tempfile.TemporaryDirectory() as folder:
            h5name, dataset, poni = make_standin(folder)
            cut = 3
            q, intensity, npixels, elapsed = process(h5name, dataset, poni, cut)
            # check frame 0 against pyFAI itself and numpy
            ai = pyFAI.load(poni)
            with h5py.File(h5name, "r") as f:
                f0 = f[dataset][0]
            bad = (f0 == 65535) | ai.detector.mask.astype(bool)
            method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
            ref = ai.integrate1d(f0, 1000, unit="q_nm^-1", method=method, mask=bad)
            assert np.allclose(intensity[0], ref.intensity, rtol=1e-4, atol=1e-4)
            assert npixels[0] == np.count_nonzero((f0 > cut) & ~bad)
            print("frame 0 matches pyFAI's integrate1d, and numpy's count above the cut")
    n = len(npixels)
    print("%d frames in %.3f s" % (n, elapsed))
    print("pixels above the cut of %d per frame: mean %.0f, min %d, max %d"
          % (cut, npixels.mean(), npixels.min(), npixels.max()))
    mean = intensity.mean(axis=0)
    top = np.argsort(mean)[::-1][:3]
    print("strongest powder bins, q (nm^-1):", np.round(q[np.sort(top)], 2).tolist())
