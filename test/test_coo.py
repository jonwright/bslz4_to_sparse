"""coo(): (row, col, value) of the sparse pixels.  The row/col split uses a
reciprocal multiply instead of a division per pixel (_unravel); it must equal
divmod exactly -- at every row boundary of the Eiger 4M/16M shapes, past the
2**26 limit (fallback), and through both coo() methods.
"""

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")
sp = pytest.importorskip("scipy.sparse")

import bslz4_to_sparse as b


@pytest.mark.parametrize("shape", [(2162, 2068), (4362, 4148), (4371, 4150), (513, 1031)])
def test_unravel_row_boundaries(shape):
    nslow, nfast = shape
    k = np.arange(1, nslow, dtype=np.int64) * nfast
    idx = np.unique(np.concatenate(([0, 1], k - 1, k, k + 1, [nslow * nfast - 1])))
    idx = idx[(idx >= 0) & (idx < nslow * nfast)].astype(np.uint32)
    row = np.empty(idx.size, np.uint16)
    col = np.empty(idx.size, np.uint16)
    b._unravel(idx, nfast, row, col)
    assert (row == idx // nfast).all()
    assert (col == idx % nfast).all()


def test_unravel_fallback_past_limit():
    nfast = 9000
    idx = np.array([0, 8999, 9000, (1 << 26) + 12345, 3_000_000_000], np.uint64)
    row = np.empty(idx.size, np.uint64)
    col = np.empty(idx.size, np.uint64)
    b._unravel(idx, nfast, row, col)
    assert (row == idx // nfast).all() and (col == idx % nfast).all()


def test_coo_methods(tmp_path):
    shape = (97, 211)
    rng = np.random.default_rng(51)
    frame = rng.poisson(0.05, shape).astype(np.uint16)
    with h5py.File(str(tmp_path / "c.h5"), "w") as f:
        ds = f.create_dataset("d", data=frame[None], chunks=(1,) + shape,
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        chunk = ds.id.read_direct_chunk((0, 0, 0))[1]
        codec = b.detect_codec(ds)
    mask = np.ones(shape, np.uint8)
    want_r, want_c = np.nonzero(frame)
    n, r, c, v = b.chunk2sparse(mask, codec=codec).coo(chunk, 0)
    assert n == want_r.size and (r == want_r).all() and (c == want_c).all()
    assert (v == frame[want_r, want_c]).all()
    M = sp.random(20, frame.size, density=0.05, format="csc", random_state=1, dtype=np.float32)
    n, r, c, v, p = b.chunk2sparseCSC(mask, M, codec=codec).coo(chunk, 0)
    assert n == want_r.size and (r == want_r).all() and (c == want_c).all()
    assert np.allclose(p, M @ frame.ravel().astype(np.float64))
