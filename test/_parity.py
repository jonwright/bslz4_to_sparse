"""Emit a bit-exact digest of decode outputs for the bslz4_to_sparse build
pointed at by BSLZ4_TO_SPARSE_PATH.  Run once against the git-head
reference (build-lib) and once against the refactor (build-phase1); the
digests must be identical.

Covers: all 10 pixel dtypes, all available untranspose backends, the
plain-sparse and the CSC (powder) route, with the dense/sparse route
forced both ways plus the default.  Outputs are hashed as raw bytes so a
match means bit-identical.
"""
import os, sys, hashlib
import numpy as np
import h5py, hdf5plugin

path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if path:
    sys.path.insert(0, path)
import bslz4_to_sparse as bslz4

print("from", bslz4.__file__, file=sys.stderr)

NF, H, W = 2, 128, 128
rng = np.random.default_rng(7)

# A CSC matrix over the active (masked) pixels.
NBIN = 16
mask2d = np.ones((H, W), np.uint8)
mask = mask2d.ravel()
# random CSC: indptr over NIJ+1, some bins
NIJ = H * W
indices = np.zeros(NIJ * 3, np.uint32)
indptr = np.zeros(NIJ + 1, np.uint32)
data = np.ones(NIJ * 3, np.float32)
k = 0
for p in range(NIJ):
    indptr[p + 1] = indptr[p] + (p % 3)
    for c in range(indptr[p + 1] - indptr[p]):
        indices[k] = (p * 7) % NBIN
        data[k] = 0.5 + (p % 5) * 0.1
        k += 1
indices = indices[:k]
data = data[:k]
indptr = indptr[:NIJ + 1]

class CSC:
    def __init__(self, d, ind, ptr, shape):
        self.data = d; self.indices = ind; self.indptr = ptr; self.shape = shape

csc = CSC(data, indices, indptr, (NBIN,))

DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64,
          np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]

def make_chunks(dt):
    """Synthetic compressed chunks with a few full blocks plus a tail."""
    ary = rng.poisson(0.05, size=(NF, H, W)).astype(dt)
    # a small-hot corner and one large value above the cut, in the tail region
    ary[:, -5:, -5:] = np.array([3], dt)
    ary[:, -1, -1] = np.array([100], dt)
    name = f"d{dt().itemsize}"
    with h5py.File("/tmp/_parity.h5", "w") as f:
        f.create_dataset(name, data=ary, chunks=(1, H, W),
                         **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    with h5py.File("/tmp/_parity.h5", "r") as f:
        ds = f[name]
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(NF)]
    return chunks, ary.dtype

h = hashlib.sha256()
backends = list(bslz4.available_backends())

for dt in DTYPES:
    chunks, dtyp = make_chunks(dt)
    for be in backends:
        bslz4.set_backend(be)
        # plain sparse
        c2sm = bslz4.chunk2sparseMulti(mask2d, dtype=dtyp)
        for cut in (0, 16):
            npx, (vals, adr) = c2sm(chunks, cut)
            h.update(np.asarray(npx, np.int32).tobytes())
            for f in range(NF):
                n = int(npx[f])
                h.update(np.asarray(vals[f][:n], dtyp).tobytes())
                h.update(np.asarray(adr[f][:n], np.uint32).tobytes())
        # CSC / powder
        c2sc = bslz4.chunk2sparseCSCmulti(mask2d, csc, dtyp)
        for thr in (0.0, 8.0, 1e9):
            bslz4.set_dense_sparse_threshold(thr)
            for cut in (0, 16):
                npx, (vals, adr), powder = c2sc(chunks, cut)
                h.update(np.asarray(npx, np.int32).tobytes())
                for f in range(NF):
                    n = int(npx[f])
                    h.update(np.asarray(vals[f][:n], dtyp).tobytes())
                    h.update(np.asarray(adr[f][:n], np.uint32).tobytes())
                h.update(np.array(powder, np.float64).tobytes())
        bslz4.set_dense_sparse_threshold(8.0)

print("DIGEST", h.hexdigest())
