"""Drinking from the firehose: the results of many files as a plain Python iterator.

Writing results is a sink's job, and file formats are the user's choice.  Often the wish is simpler:
take every frame's sparse pixels (and powder sums) into your own code as they are decoded.
conveyor.iterate() runs the conveyor behind a for loop.  Each item is one batch of frames, with the
arguments of Sink.write: copies you may keep.  Batches arrive in the order the workers finish them,
not in file order: `file` (the index into the list of files) and `frame0` say where each belongs.
"""

import glob

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np

from bslz4_to_sparse import conveyor

h5pattern = "eiger_*.h5"  # your files: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside each
h5filenames = sorted(glob.glob(h5pattern))

for name in h5filenames:  # once per file, the only use of HDF5 here (see example 8)
    conveyor.write_index(conveyor.harvest(name, h5dataset), name)

# Cut the work into units of at most 50 MB of compressed input each, so that a small data set still
# gives the workers several units to share (the default, 256 MB, is a whole file in most cases).
units, shape, dtype = conveyor.make_work_units(h5filenames, h5dataset, max_unit_bytes=50_000_000)
with h5py.File(h5filenames[0], "r") as h5:
    mask = (h5[h5dataset][0] != np.iinfo(dtype).max).astype(np.uint8)
backend = conveyor.CpuBackend(mask, dtype, cut=3, howmany=5000)

# The loop gets the pixels as they are decoded.
total = 0
frames_seen = set()
for file, frame0, npx, values, indices, powder in conveyor.iterate(
    units, backend, workers=2, batch=4
):
    for j, n in enumerate(npx):  # frame frame0 + j of file number `file`
        total += int(values[j, :n].sum(dtype=np.int64))
        frames_seen.add((file, frame0 + j))
nframes = sum(u.n for u in units)
assert len(frames_seen) == nframes
print(
    "%d frames of %d files came through a for loop; sum of the sparse pixel values: %d"
    % (nframes, len(h5filenames), total)
)

# Leaving the loop early (break, or an exception) stops the workers: nothing is left running.
for batch in conveyor.iterate(units, backend, workers=2):
    print(
        "first batch: file %d, frame %d, %d frames; leaving the loop now"
        % (batch[0], batch[1], len(batch[2]))
    )
    break
