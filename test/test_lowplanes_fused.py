"""The byte skip's fused transpose + collect (u16 blocks with the high
byte-planes empty, bslz4_lowplanes_collect_u16): the plain sparse output and
the sparse-route CSC powder must equal numpy at every cut -- including cuts at
and above 255, where nothing in such a block passes -- with and without the
fixed mask, for sparse and dense-ish blocks, with masked pixels holding values.
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

SHAPE = (101, 203)     # 20503 px: full blocks and a tail


def make(kind, rng):
    n = (4,) + SHAPE
    if kind == "very sparse":
        a = np.zeros(n, np.int64)
        flat = a.reshape(4, -1)
        for f in range(4):
            idx = rng.choice(flat.shape[1], 25, replace=False)
            flat[f, idx] = rng.integers(1, 256, idx.size)
    elif kind == "clustered":            # whole 64-pixel groups full, others empty
        a = np.zeros(n, np.int64)
        flat = a.reshape(4, -1)
        for f in range(4):
            for s in rng.choice(flat.shape[1] // 64, 20, replace=False):
                flat[f, s * 64:(s + 1) * 64] = rng.integers(0, 256, 64)
    elif kind == "dense <256":
        a = rng.integers(0, 256, n)
    elif kind == "max 255":
        a = np.where(rng.random(n) < 0.02, 255, 0)
    return a.astype(np.uint16)


def chunks_of(frames, path):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))], b.detect_codec(ds)


def masks():
    rng = np.random.default_rng(3)
    m = (rng.random(SHAPE) > 0.1).astype(np.uint8)
    m[:, 60:64] = 0
    return {"ones": np.ones(SHAPE, np.uint8), "holes": m}


@pytest.fixture(autouse=True)
def restore():
    saved = (b.get_byteskip(), b.get_lz4_zero(), b.get_mask_planes())
    yield
    b.set_byteskip(saved[0])
    b.set_lz4_zero(saved[1])
    b.set_mask_planes(saved[2])


@pytest.mark.parametrize("kind", ["very sparse", "clustered", "dense <256", "max 255"])
@pytest.mark.parametrize("mname", ["ones", "holes"])
def test_sparse(tmp_path, kind, mname):
    frames = make(kind, np.random.default_rng(17))
    chunks, codec = chunks_of(frames, tmp_path / "s.h5")
    mask = masks()[mname]
    for mp in (False, True):
        b.set_byteskip(True)
        b.set_mask_planes(mp)
        integ = b.chunk2sparseMulti(mask, dtype=np.uint16, codec=codec)
        for cut in (0, 1, 100, 254, 255, 256, 1000):
            npx, (vals, adr) = integ(chunks, cut)
            for f in range(len(frames)):
                flat = frames[f].ravel()
                want = np.flatnonzero((flat > cut) & (mask.ravel() > 0))
                n = int(npx[f])
                tag = (kind, mname, mp, cut, f)
                assert n == want.size, tag
                assert (adr[f, :n] == want).all(), tag
                assert (vals[f, :n] == flat[want]).all(), tag


@pytest.mark.parametrize("kind", ["very sparse", "clustered", "max 255"])
@pytest.mark.parametrize("mname", ["ones", "holes"])
def test_csc_sparse_route(tmp_path, kind, mname):
    frames = make(kind, np.random.default_rng(18))
    chunks, codec = chunks_of(frames, tmp_path / "c.h5")
    mask = masks()[mname]
    npix = SHAPE[0] * SHAPE[1]
    M = sp.random(120, npix, density=0.02, format="csc", random_state=9, dtype=np.float32)
    ref = np.array([M @ (f.ravel().astype(np.float64) * mask.ravel()) for f in frames])
    saved = b.get_dense_sparse_threshold()
    try:
        b.set_dense_sparse_threshold(0.0)          # every block takes the sparse route
        for skip in (True, False):
            b.set_byteskip(skip)
            integ = b.chunk2sparseCSCmulti(mask, M, dtype=np.uint16, codec=codec)
            for cut in (0, 7, 255):
                npx, (vals, adr), p = integ(chunks, cut)
                assert np.allclose(p, ref, rtol=1e-9, atol=1e-9), (kind, mname, skip, cut)
                for f in range(len(frames)):
                    flat = frames[f].ravel()
                    want = np.flatnonzero((flat > cut) & (mask.ravel() > 0))
                    n = int(npx[f])
                    assert n == want.size and (adr[f, :n] == want).all(), (kind, mname, skip, cut, f)
                    assert (vals[f, :n] == flat[want]).all()
    finally:
        b.set_dense_sparse_threshold(saved)
