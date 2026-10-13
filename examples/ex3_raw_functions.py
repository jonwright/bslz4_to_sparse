"""The raw functions and helpers of the released versions: npbuf and bslz4_uint16_t and friends.

The first releases (0.0.15, 0.0.16) also exposed the compiled functions, one for each pixel type,
and chunk2sparse and chunk2sparseCSC were thin classes around them.  They still work, with the same
arguments: the compressed chunk as a uint8 array, the mask, the output arrays that are filled in
place, and the cut.  They return the number of pixels above the cut (negative for an error).  The
csc forms also take pyFAI's matrix as float32 data, uint32 indices and uint32 indptr and fill a
float64 powder.  npbuf makes the uint8 array from the bytes h5py gives, without a copy.  A new
program should use chunk2sparse and chunk2sparseCSC (examples 1 and 2), which allocate for you,
decode batches and run on every pixel type.
"""

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np
import pyFAI
from pyFAI.method_registry import IntegrationMethod

import bslz4_to_sparse

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
mask_filename = (
    "mask.h5"  # your detector mask: a dataset "mask", pyFAI's convention (non-zero = masked)
)
poni_filename = "calib.poni"  # your pyFAI geometry

with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    dtype = ds.dtype
    chunk0 = ds.id.read_direct_chunk((0, 0, 0))[1]  # bytes: read only
with h5py.File(mask_filename, "r") as h5:
    bad = h5["mask"][()] != 0
mask = (~bad).astype(np.uint8).ravel()

cmp = bslz4_to_sparse.npbuf(chunk0)  # a uint8 array on the same memory
print("npbuf: %s of %d bytes" % (cmp.dtype, cmp.size))

# the function for the pixel type: bslz4_uint8_t, bslz4_uint16_t or bslz4_uint32_t
decode = getattr(bslz4_to_sparse, "bslz4_%s_t" % dtype.name)
values = np.empty(mask.size, dtype)
indices = np.empty(mask.size, np.uint32)
cut = 3
npx = decode(cmp, mask, values, indices, cut)
print("%s: %d pixels above the cut of %d" % (decode.__name__, npx, cut))
assert npx == bslz4_to_sparse.chunk2sparse(mask.reshape(bad.shape), dtype)(chunk0, cut)[0]

# with the matrix product: bslz4_csc_uint8_t, ..._uint16_t, ..._uint32_t
ai = pyFAI.load(poni_filename)
method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
ai.integrate1d(np.ones(bad.shape, np.float32), 400, unit="q_nm^-1", method=method, mask=bad)
engine = ai.engines[method].engine
decode_csc = getattr(bslz4_to_sparse, "bslz4_csc_%s_t" % dtype.name)
powder = np.empty(engine.bins, np.float64)
npx = decode_csc(
    cmp,
    mask,
    values,
    indices,
    cut,
    powder,
    engine.data.astype(np.float32),
    engine.indices.astype(np.uint32),
    engine.indptr.astype(np.uint32),
)
print(
    "%s: %d pixels, powder of %d bins, sum %.6g"
    % (decode_csc.__name__, npx, powder.size, powder.sum())
)
reference = bslz4_to_sparse.chunk2sparseCSC(mask.reshape(bad.shape), engine, dtype, exact=False)(
    chunk0, cut
)[2]
assert np.array_equal(powder, reference)
