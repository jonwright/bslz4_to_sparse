"""Writing results: ImageD11 sparse files with exact powder sums, and the integrator beside them.

One output file per input file (workers never share a file; a later step gathers them), in
ImageD11's sparse layout.  The powder is kept as exact int64 sums with the power-of-two scale that
turns them into float.  What turns sums into intensities (the pyFAI geometry, the matrix and the
normalisation of an image of ones) is saved once, as an Integrator file the results point to.
"""

import os
import tempfile

import h5py
import hdf5plugin
import numpy as np
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.method_registry import IntegrationMethod

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
rng = np.random.default_rng(11)

with tempfile.TemporaryDirectory() as tmp:
    frames = rng.poisson(0.3 + 4 * np.exp(-0.5 * ((ai.qArray() - 40) / 3) ** 2), (10,) + shape)
    path = os.path.join(tmp, "scan.h5")
    with h5py.File(path, "w") as h:
        h.create_dataset(
            DSET,
            data=frames.astype(np.uint16),
            chunks=(1,) + shape,
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
        )
    conveyor.write_index(conveyor.harvest(path, DSET), path)

    # the integrator: matrix and normalisation from pyFAI, saved once
    integ = conveyor.Integrator.from_pyfai(ai, 50, unit="q_nm^-1", split="bbox")
    integ_path = os.path.join(tmp, "integrator.h5")
    integ.save(integ_path)

    # the run: sparse pixels (at most 400 per frame) and the exact powder, size-checked
    units, frame_shape, dtype = conveyor.make_units([path], DSET)
    backend = conveyor.CpuBackend(
        np.ones(shape, np.uint8), dtype, csc=integ.csc, cut=2, howmany=400, exact=True
    )
    out = os.path.join(tmp, "sparse")
    conveyor.run(
        units,
        backend,
        workers=1,
        policy=conveyor.Policy(("sparse", "powder1d")),
        sink=lambda k: conveyor.ImageD11SparseSink(out, DSET, integrator=integ_path),
    )

    # reading it back: intensities from the sums, as pyFAI would give them
    with h5py.File(os.path.join(out, conveyor.imaged11_name(path)), "r") as h:
        g = h[DSET.strip("/")]
        powder, scale = g["powder"][:], g["powder"].attrs["scale"]
        where = g["powder"].attrs["integrator"]
        nnz = g["nnz"][:]
    back = conveyor.Integrator.load(where)
    intensity = back.intensity(powder, scale)
    method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
    ref = ai.integrate1d(frames[3].astype(np.uint16), 50, unit="q_nm^-1", method=method)
    assert np.allclose(intensity[3], ref.intensity, rtol=1e-4, atol=1e-4)
    assert powder.dtype == np.int64

print(
    "%d frames: %d sparse pixels, powder %s int64 sums, scale 2**%d"
    % (len(nnz), nnz.sum(), powder.shape, round(np.log2(scale)))
)
print("intensity from the stored sums equals pyFAI's integrate1d")
