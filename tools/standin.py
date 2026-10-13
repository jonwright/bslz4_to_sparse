"""Stand-in data for the tests and the documentation build, never for the examples themselves.

The examples are written for a reader's own files:

    h5filename = "eiger_0000.h5"                     # a bitshuffle-compressed HDF5 file
    h5dataset = "/entry_0000/ESRF-ID11/eiger/data"   # its (nframes, ni, nj) dataset
    mask_filename = "mask.npy"                       # the detector mask, non-zero = masked pixel
    poni_filename = "calib.poni"                     # the pyFAI geometry, where one is used

When the examples are run by the test-suite or the documentation build there is no such file, so a
stand-in with those names is written into the working directory first: powder rings with Poisson
noise on an Eiger-like detector, module gaps that count nothing (and a mask.npy that marks them),
one frame per chunk, bitshuffle-lz4, as the detector writes it.  Only its content is synthetic.

    make_standin(folder, nfiles=3, nframes=12, dtype=np.uint16, detector="Eiger1M")
"""

import os

import numpy as np

H5DATASET = "/entry_0000/ESRF-ID11/eiger/data"
H5NAME = "eiger_%04d.h5"
PONI = "calib.poni"
MASK = "mask.npy"


def make_standin(folder, nfiles=3, nframes=12, dtype=np.uint16, detector="Eiger1M", seed=0):
    """Write eiger_0000.h5 ... and calib.poni into folder; returns the list of the h5 paths.
    dtype: the pixel type (u16 or u32);  detector: a pyFAI detector name, which gives the shape, the
    module gaps and the geometry of the .poni."""
    import h5py
    import hdf5plugin
    import pyFAI
    from pyFAI.calibrant import get_calibrant

    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:  # older pyFAI
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    dtype = np.dtype(dtype)
    ai = AzimuthalIntegrator(
        dist=0.12,
        poni1=0.040,
        poni2=0.039,
        wavelength=3e-11,
        detector=pyFAI.detector_factory(detector),
    )
    ai.write(os.path.join(folder, PONI))
    cal = get_calibrant("LaB6")
    cal.wavelength = ai.wavelength
    q = ai.qArray()
    mean = 0.02 + sum(
        5.0 / (1 + 0.3 * k) * np.exp(-0.5 * ((q - qk) / 0.15) ** 2)
        for k, qk in enumerate(cal.get_peaks("q_nm^-1")[:30])
    )
    gaps = ai.detector.mask.astype(bool)
    np.save(os.path.join(folder, MASK), gaps.astype(np.uint8))
    rng = np.random.default_rng(seed)
    names = []
    for k in range(nfiles):
        frames = rng.poisson(mean, (nframes,) + mean.shape).astype(dtype)
        frames[:, gaps] = 0  # the gaps count nothing
        name = os.path.join(folder, H5NAME % k)
        with h5py.File(name, "w") as f:
            f.create_dataset(
                H5DATASET,
                data=frames,
                chunks=(1,) + mean.shape,
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
            )
        names.append(name)
    return names
