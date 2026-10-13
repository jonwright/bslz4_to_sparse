"""Ring windows for texture pipelines: intensity[frame, window, azimuth], one file per input file.

Texture tomography needs, for every frame, the intensity of a few Bragg rings as a function of
azimuth.  The conveyor gives exactly that: for each ring a window of radial bins of pyFAI's
normalised 2D cake, summed, with a background window either side of it and no background
subtracted, so the reader decides what to do with the three.  The files follow the input files
and their frame order: sorting frames into scan order is left to the reader.
"""

import os
import tempfile

import h5py
import hdf5plugin
import numpy as np
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from bslz4_to_sparse import conveyor

DSET = "/entry/data"
shape = (96, 112)
ai = AzimuthalIntegrator(
    dist=0.1,
    poni1=0.004,
    poni2=0.0045,
    wavelength=3e-11,
    detector=Detector(75e-6, 75e-6, max_shape=shape),
)
rng = np.random.default_rng(21)

with tempfile.TemporaryDirectory() as tmp:
    paths = []
    for k in range(2):
        frames = rng.poisson(
            0.5 + 3 * np.exp(-0.5 * ((ai.twoThetaArray(shape) * 57.3 - 1.0) / 0.1) ** 2),
            (6,) + shape,
        )
        path = os.path.join(tmp, "scan%d.h5" % k)
        with h5py.File(path, "w") as h:
            h.create_dataset(
                DSET,
                data=frames.astype(np.uint16),
                chunks=(1,) + shape,
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
            )
        conveyor.write_index(conveyor.harvest(path, DSET), path)
        paths.append(path)

    # one ring at 1.0 degree 2theta, 0.12 wide, with a window of 0.1 either side
    ranges, ring, role = conveyor.peak_windows([1.0], 0.12, background_width=0.1)
    cake = conveyor.Integrator.from_pyfai(
        ai, 70, unit="2th_deg", npt_azim=12, radial_range=(0.3, 3.0), polarization_factor=0.95
    )
    rings = cake.rings(ranges)

    units, frame_shape, dtype = conveyor.make_units(paths, DSET)  # a unit is a whole file
    out = os.path.join(tmp, "rings")
    conveyor.run(
        units,
        # howmany=0: rings only, no sparse pixels
        conveyor.CpuBackend(np.ones(shape, np.uint8), dtype, csc=rings.csc, howmany=0),
        workers=2,
        sink=lambda k: conveyor.RingSink(rings, out, DSET, ring, role),
        policy=conveyor.Policy("rings"),
    )
    with h5py.File(os.path.join(out, conveyor.rings_name(paths[1])), "r") as h:
        g = h[DSET.strip("/")]
        intensity, roles = g["intensity"][:], g["role"][:]
        bins = g["window_nbins"][:]

# frames, windows (below, ring, above), azimuth bins; the ring window is the brightest
assert intensity.shape == (6, 3, 12) and list(roles) == [-1, 0, 1]
assert (intensity[:, 1] / bins[1]).mean() > (
    intensity[:, [0, 2]].mean(axis=(0, 2)) / bins[[0, 2]]
).max()
print("intensity %s: frames, windows %s, azimuth bins" % (intensity.shape, list(roles)))
