"""Pipelines: which kernels run, and choosing them.

Every object has a pipeline: one value per processing step (decode, mask,
untranspose, collect, and for matrix objects route and dot).  None means
the best guess for this CPU, data and matrix; it is resolved when the
object is made and kept in .pipeline.  describe() names the values, and
with no argument lists every value and whether this machine can run it.
The output below is from the machine that built this documentation.
"""
import numpy as np

import bslz4_to_sparse

mask = np.ones((256, 256), np.uint8)
c2s = bslz4_to_sparse.chunk2sparse(mask, dtype=np.uint16)
print("automatic pipeline:", bslz4_to_sparse.describe(c2s.pipeline))

print("\nvalues this machine can run:")
for step, values in bslz4_to_sparse.describe().items():
    usable = [name for _, name, ok, _ in values if ok]
    print("  %-12s %s" % (step, ", ".join(usable)))

# any step can be fixed by name; the rest are still chosen automatically
slow = bslz4_to_sparse.chunk2sparse(mask, dtype=np.uint16,
                                    pipeline={"untranspose": "scalar", "collect": "scalar"})
print("\nforced:", bslz4_to_sparse.describe(slow.pipeline))

# a value the CPU or the data cannot use is refused when the object is made
try:
    bslz4_to_sparse.chunk2sparse(mask, dtype=np.uint32, pipeline={"untranspose": "lowplanes-c"})
except ValueError as e:
    print("\nrefused:", e)
