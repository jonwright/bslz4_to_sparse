"""Writing results: ImageD11 sparse files with exact powder sums, and the integrator beside them.

One output file per input file (workers never share a file; a later step gathers them), in ImageD11's
sparse layout.  The powder is kept as exact int64 sums with the power-of-two scale that turns them
into float, so nothing is lost to rounding.  What turns sums into intensities (the pyFAI geometry,
the sparse matrix and the normalisation of an image of ones) is saved once, as an Integrator file
that the results point to.  The conveyor's Policy checks that nothing written is larger than what
was read.
"""

import glob
import os

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np
import pyFAI
from pyFAI.method_registry import IntegrationMethod

from bslz4_to_sparse import conveyor

h5pattern = "eiger_*.h5"  # your files: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside each
mask_filename = "mask.npy"  # your detector mask, pyFAI's convention: non-zero = masked pixel
poni_filename = "calib.poni"  # your pyFAI geometry
outdir = "results"
h5filenames = sorted(glob.glob(h5pattern))
for name in h5filenames:
    conveyor.write_index(conveyor.harvest(name, h5dataset), name)

units, shape, dtype = conveyor.make_work_units(h5filenames, h5dataset)
with h5py.File(h5filenames[0], "r") as h5:
    frame0 = h5[h5dataset][0]
bad = np.load(mask_filename) != 0  # the masked pixels
ai = pyFAI.load(poni_filename)
mask = (~bad).astype(np.uint8)  # ours: 1 = use this pixel

# the integrator: matrix and normalisation from pyFAI, saved once
os.makedirs(outdir, exist_ok=True)
integ = conveyor.Integrator.from_pyfai(ai, 500, unit="q_nm^-1", split="bbox", mask=bad)
integ_path = os.path.join(outdir, "integrator.h5")
integ.save(integ_path)

# the run: sparse pixels (at most 5000 per frame) and the exact powder sums
backend = conveyor.CpuBackend(mask, dtype, csc=integ.csc, cut=3, howmany=5000, exact=True)
result = conveyor.run(
    units,
    backend,
    workers=2,
    policy=conveyor.Policy(("sparse", "powder1d")),
    sink=lambda k: conveyor.ImageD11SparseSink(outdir, h5dataset, integrator=integ_path),
)
print(conveyor.report(result))

# reading one file back: intensities from the stored sums, as pyFAI gives them
with h5py.File(os.path.join(outdir, conveyor.imaged11_name(h5filenames[0])), "r") as h5:
    g = h5[h5dataset.strip("/")]
    powder_sums, scale = g["powder"][:], g["powder"].attrs["scale"]
    where = g["powder"].attrs["integrator"]
    nnz = g["nnz"][:]
back = conveyor.Integrator.load(where)
intensity = back.intensity(powder_sums, scale)  # integer sums times 2**-b, over the normalisation
method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
frame3 = None
with h5py.File(h5filenames[0], "r") as h5:
    frame3 = h5[h5dataset][3]
ref = ai.integrate1d(frame3, 500, unit="q_nm^-1", method=method, mask=bad)
worst = (abs(intensity[3] - ref.intensity) / np.where(ref.intensity > 0, ref.intensity, 1)).max()
print("largest relative difference from pyFAI's intensity: %.2e" % worst)
# 16-bit pixels: only pyFAI's own float32 rounding (6e-8).  32-bit pixels: the integer weights have
# to leave room for pixel values up to 2**32, so they are coarser (2**-17 or so): about 1e-6.
assert worst < (1.2e-7 if dtype.itemsize <= 2 else 1e-5)
print(
    "%d frames: %d sparse pixels, powder_sums %s int64 with scale 2**%d"
    % (len(nnz), nnz.sum(), powder_sums.shape, round(np.log2(scale)))
)
print("the intensity from the stored sums equals pyFAI's integrate1d")
