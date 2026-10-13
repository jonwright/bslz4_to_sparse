"""Drinking from the firehose: the results of many files as a plain Python iterator.

Writing results is a sink's job, and file formats are the user's choice.  Often the wish is
simpler: take every frame's sparse pixels (or powder) into your own code as they are decoded.
iterate() runs the conveyor behind a for loop; each item is one batch of frames, a copy you may keep.
Batches arrive in the order the workers finish them: file and frame0 say where each belongs.
A Policy guards what is written; here nothing is, so it only checks the kind of output.
"""

import os
import tempfile

import h5py
import hdf5plugin
import numpy as np

from bslz4_to_sparse import conveyor

DSET = "/entry/data"
shape = (96, 112)
rng = np.random.default_rng(9)
mask = np.ones(shape, np.uint8)

with tempfile.TemporaryDirectory() as tmp:
    paths, truth = [], {}
    for k in range(3):
        frames = np.where(rng.random((20,) + shape) < 0.03, rng.integers(1, 30, (20,) + shape), 0)
        for i, fr in enumerate(frames):
            truth[(k, i)] = int(fr.sum())
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

    units, frame_shape, dtype = conveyor.make_units(paths, DSET, max_bytes=3000)
    backend = conveyor.CpuBackend(mask, dtype, howmany=500)

    # the loop gets the pixels as they are decoded (a Policy guards what is written, not what is iterated)
    total = {}
    for file, frame0, npx, values, indices, powder in conveyor.iterate(
        units, backend, workers=2, batch=8, policy=conveyor.Policy("sparse")
    ):
        for j, n in enumerate(npx):
            total[(file, frame0 + j)] = int(values[j, :n].sum(dtype=np.int64))
    assert total == truth

    # stopping early stops the run
    first = next(iter(conveyor.iterate(units, backend, workers=2)))
    assert first[0] in (0, 1, 2)

print("%d frames through a for loop, equal to the sums of the input" % len(total))
