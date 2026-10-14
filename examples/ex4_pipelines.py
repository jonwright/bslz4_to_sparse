"""Pipelines: which kernels run, and trying every one.

Every object has a pipeline: one value per processing step (decode, mask, untranspose, collect, and
for matrix objects route and dot).  None means the best guess for this CPU and data; it is resolved
when the object is made and kept in .pipeline.  describe() names the values, and with no argument
lists every value and whether this machine can run it.  This example runs every usable value of
every step, alone with the rest chosen automatically, and checks that each gives exactly the pixels
the default gives.  (The combinations are tested by the test-suite, test_every_pipeline.py.)
"""

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np

import bslz4_to_sparse

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
mask_filename = (
    "mask.h5"  # your detector mask: a dataset "mask", pyFAI's convention (non-zero = masked)
)

with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    dtype, codec = ds.dtype, bslz4_to_sparse.detect_codec(ds)
    chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(4)]
with h5py.File(mask_filename, "r") as h5:
    bad = h5["mask"][()] != 0  # the masked pixels
mask = (~bad).astype(np.uint8)  # ours: 1 = use this pixel
cut = 3

c2s = bslz4_to_sparse.chunk2sparse(mask, dtype=dtype, codec=codec)
print("automatic pipeline:", bslz4_to_sparse.describe(c2s.pipeline))
npx, (values, indices) = c2s.multi(chunks, cut)
want = [(values[i, : npx[i]].copy(), indices[i, : npx[i]].copy()) for i in range(len(chunks))]

print(
    "\nevery value of every step this machine can run, alone (step, value, and whether its pixels equal the default's):"
)
tried = same = 0
for step, values_of_step in bslz4_to_sparse.describe().items():
    if step in (
        "route",
        "dot",
    ):  # these steps belong to the matrix product: see test_every_pipeline.py
        print("  %-12s (only for chunk2sparseCSC objects, which need a matrix)" % step)
        continue
    for _, name, usable, needs in values_of_step:
        if not usable:
            print("  %-12s %-18s not on this CPU or build (%s)" % (step, name, needs))
            continue
        try:
            c = bslz4_to_sparse.chunk2sparse(mask, dtype=dtype, codec=codec, pipeline={step: name})
        except ValueError as e:  # not for this data (the dtype, say), or for matrix objects only
            print("  %-12s %-18s refused: %s" % (step, name, str(e)[:60]))
            continue
        n, (v, ix) = c.multi(chunks, cut)
        ok = all(
            (v[i, : n[i]] == want[i][0]).all() and (ix[i, : n[i]] == want[i][1]).all()
            for i in range(len(chunks))
        )
        tried += 1
        same += ok
        print("  %-12s %-18s %s" % (step, name, "same" if ok else "DIFFERENT"))
print("\n%d pipelines run, %d identical to the default" % (tried, same))
assert tried == same

# a value the CPU or the data cannot use is refused when the object is made
try:
    bslz4_to_sparse.chunk2sparse(mask, dtype=np.uint32, pipeline={"untranspose": "lowplanes-c"})
except ValueError as e:
    print("\nrefused:", e)
