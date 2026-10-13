"""Many files: the conveyor, a worked example of the decoders at scale.

The decoders need nothing around them.  This is what it takes to point them at a whole data set: read
the files in large pieces, keep every core busy, and write the results.  The steps are

1. find where every frame's compressed chunk lies in each file.  `harvest` reads that from HDF5 (it
   returns an Index: the byte offset and length of every chunk, and the shape and dtype of the
   frames) and `write_index` keeps it in a small file beside the data (12 bytes per frame), so that
   nothing afterwards needs HDF5.  This is the only step that does, and it needs doing once, even
   at acquisition time;
2. cut the work into work units (nothing to do with pyFAI's radial units): consecutive frames of
   one file, whole files by default, each read with large reads and decoded by one worker;
3. run: each worker reads the next unit while it decodes the previous one, a worker that is faster
   takes more units, and the results go to a sink.  The sink here writes one sparse-pixel file per
   data file, in ImageD11's layout.  On a cluster each node would run this script on its share of
   the files.
"""

import glob
import os

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np

import bslz4_to_sparse
from bslz4_to_sparse import conveyor

h5pattern = "eiger_*.h5"  # your files: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside each
mask_filename = (
    "mask.h5"  # your detector mask: a dataset "mask", pyFAI's convention (non-zero = masked)
)
outdir = "sparse_out"  # where the sparse files go
h5filenames = sorted(glob.glob(h5pattern))

# 1. the chunk index of every file
for name in h5filenames:
    index = conveyor.harvest(name, h5dataset)  # an Index
    conveyor.write_index(index, name)  # name + ".bslz4idx"
print(
    "%d files; frames of %s %s, %d in the first"
    % (len(h5filenames), index.shape[1:], index.dtype, index.shape[0])
)

# 2. the work units, and the frame shape and dtype the index holds
units, shape, dtype = conveyor.make_work_units(h5filenames, h5dataset)
print("%d work units" % len(units))

with h5py.File(mask_filename, "r") as h5:
    mask = (h5["mask"][()] == 0).astype(np.uint8)  # 1 = use this pixel

# 3. run: 3 workers (threads), the pixels above the cut of 3, at most 5000 per frame
backend = conveyor.CpuBackend(mask, dtype, cut=3, howmany=5000)
result = conveyor.run(
    units,
    backend,
    workers=3,
    # min_pixels=2 would also drop the single pixels (connected objects smaller than that, mostly
    # noise), using ImageD11's labelling
    sink=lambda worker: conveyor.ImageD11SparseSink(outdir, h5dataset),
    policy=conveyor.Policy("sparse"),
)
print(conveyor.report(result))

# check one frame of the first file: the sparse file against a direct decode of the same chunk
index = conveyor.read_index(h5filenames[0])
with open(h5filenames[0], "rb") as fh:
    fh.seek(int(index.offsets[0]))
    chunk = fh.read(int(index.lengths[0]))
c2s = bslz4_to_sparse.chunk2sparse(mask, dtype, howmany=5000)
npx, (values, indices) = c2s(chunk, 3)
with h5py.File(os.path.join(outdir, conveyor.imaged11_name(h5filenames[0])), "r") as h5:
    g = h5[h5dataset.strip("/")]
    n0 = int(g["nnz"][0])
    row, col = g["row"][:n0], g["col"][:n0]
assert n0 == npx and (row.astype(np.int64) * shape[1] + col == indices[:npx]).all()
print(
    "frame 0 of %s: %d pixels in the sparse file, equal to a direct decode" % (h5filenames[0], n0)
)
