"""The conveyor: many files, one reader and one decoder per worker.

The conveyor moves work to the decoders.  Each file's chunk offsets are kept
beside it in a small index file (written once, here with h5py), so reading
never needs HDF5.  Every worker owns two buffers and a reader thread: the
reader takes the next file from a shared queue and reads it with large calls
while the worker decodes the previous one, so a faster worker takes more
files.  The sink receives each batch of results; this one only checks them.
"""

import os
import tempfile

import h5py
import hdf5plugin
import numpy as np

import bslz4_to_sparse as b
from bslz4_to_sparse import conveyor

DSET = "/entry/data"
shape = (96, 112)
rng = np.random.default_rng(5)
mask = np.ones(shape, np.uint8)

with tempfile.TemporaryDirectory() as tmp:
    paths = []
    for k in range(4):
        frames = np.where(rng.random((25,) + shape) < 0.04, rng.integers(1, 40, (25,) + shape), 0)
        path = os.path.join(tmp, "scan%d.h5" % k)
        with h5py.File(path, "w") as h:
            h.create_dataset(
                DSET,
                data=frames.astype(np.uint16),
                chunks=(1,) + shape,
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
            )
        conveyor.write_index(conveyor.harvest(path, DSET), path)  # the only HDF5 reading
        paths.append(path)

    # what a direct decode of every frame gives, reading the bytes by offset
    direct = {}
    dec = b.chunk2sparse(mask, np.uint16, b.CODEC_LZ4)
    for fi, path in enumerate(paths):
        ix = conveyor.read_index(path)
        with open(path, "rb") as fh:
            for i in range(len(ix.offsets)):
                fh.seek(int(ix.offsets[i]))
                n, (v, idx) = dec(fh.read(int(ix.lengths[i])), 0)
                direct[(fi, i)] = (int(n), int(v[:n].sum(dtype=np.int64)))

    got = {}

    class Check(conveyor.Sink):
        def write(self, file, frame0, npx, values, indices, powder):
            for j, n in enumerate(npx):
                got[(file, frame0 + j)] = (int(n), int(values[j, :n].sum(dtype=np.int64)))

    units, frame_shape, dtype = conveyor.make_units(paths, DSET)  # whole files, from the indexes
    result = conveyor.run(
        units, conveyor.CpuBackend(mask, dtype), workers=3, sink=lambda worker: Check()
    )

assert got == direct
print(
    "%d frames from %d files through %d workers, identical to a direct decode"
    % (result["frames"], len(paths), result["workers"])
)
