"""Ring windows for texture pipelines: intensity[frame, window, azimuth], one file per input file.

Texture tomography needs, for every frame, the intensity of a few Bragg rings as a function of
azimuth.  The conveyor gives exactly that: for each ring a window of radial bins of pyFAI's
normalised 2D cake, summed, with a background window either side of it and no background subtracted,
so the reader decides what to do with the three.  The sparse pixels of the same frames come from the
same pass.  The files follow the input files and their frame order: sorting frames into scan order is
left to the reader.  Where the rings are is user input: usually decided by eye from a powder pattern.
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
poni_filename = "calib.poni"  # your pyFAI geometry
outdir = "rings_out"
h5filenames = sorted(glob.glob(h5pattern))
for name in h5filenames:
    conveyor.write_index(conveyor.harvest(name, h5dataset), name)
units, shape, dtype = conveyor.make_work_units(h5filenames, h5dataset)
with h5py.File(h5filenames[0], "r") as h5:
    frame0 = h5[h5dataset][0]
bad = frame0 == np.iinfo(dtype).max
ai = pyFAI.load(poni_filename)
mask = (~(bad | ai.detector.mask.astype(bool))).astype(np.uint8)

# Where are the rings?  Here: the two strongest peaks of the 1D pattern of frame 0, decided by eye in
# practice.  Windows of 0.4 nm^-1 on each ring, and 0.4 either side as background.
method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
pattern = ai.integrate1d(frame0, 400, unit="q_nm^-1", method=method, mask=bad)
smooth = np.convolve(pattern.intensity, np.ones(3) / 3, mode="same")  # noise makes spikes
peaks = np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
centres = np.sort(pattern.radial[peaks[np.argsort(smooth[peaks])[::-1][:2]]])
ranges, ring, role = conveyor.peak_windows(centres, 0.4, background_width=0.4)
print("ring centres (q, nm^-1):", np.round(centres, 2).tolist())

# One matrix: a 2D (radial x azimuth) cake of 400 x 36 bins, then the sums over the windows of
# the normalised cake: dividing by the per-bin number of pixels is folded into the weights.
cake = conveyor.Integrator.from_pyfai(ai, 400, unit="q_nm^-1", npt_azim=36, mask=bad)
rings = cake.rings(ranges)

backend = conveyor.CpuBackend(mask, dtype, csc=rings.csc, cut=3, howmany=5000)
result = conveyor.run(
    units,
    backend,
    workers=2,
    sink=lambda k: conveyor.TeeSink(
        conveyor.RingSink(rings, outdir, h5dataset, ring, role),
        conveyor.ImageD11SparseSink(outdir, h5dataset),  # the sparse pixels, from the same pass
    ),
    policy=conveyor.Policy(("sparse", "rings")),
)
print(conveyor.report(result))

with h5py.File(
    os.path.join(outdir, conveyor.rings_name(h5filenames[1 % len(h5filenames)])), "r"
) as h5:
    g = h5[h5dataset.strip("/")]
    intensity, roles, nbins = g["intensity"][:], g["role"][:], g["window_nbins"][:]
# frames, windows (below, ring, above, for each ring), azimuth bins
assert intensity.shape[1:] == (6, 36) and list(roles) == [-1, 0, 1] * 2
per_bin = intensity.mean(axis=(0, 2)) / nbins  # mean intensity per radial bin, each window
print("mean intensity per radial bin, for each window:", np.round(per_bin, 3).tolist())
assert (per_bin[1::3] > per_bin[0::3]).all() and (
    per_bin[1::3] > per_bin[2::3]
).all()  # ring over background
print("intensity %s: frames, windows %s, azimuth bins" % (intensity.shape, list(roles)))
