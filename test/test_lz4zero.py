"""The zero-aware lz4 block decoder (set_lz4_zero), plane extraction, the
u16 byte skip and the mask applied in the bitshuffled domain (set_mask_planes)
must change nothing but speed: sparse output and CSC powder identical to numpy
for every dtype and every on/off combination -- including masked pixels that
hold the dtype maximum (they must never appear) and unmasked saturated pixels
at the maximum (they must always be collected).
"""

import itertools
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

SHAPE = (99, 209)     # 20691 px: full blocks plus a tail
# the fixed mask: two module-gap-like columns and a row, plus scattered pixels
_m = np.ones(SHAPE, bool)
_m[:, 100:104] = False
_m[50, :] = False
_m.ravel()[np.random.default_rng(7).choice(_m.size, 300, replace=False)] = False
MASK = _m.astype(np.uint8)
MASKED = ~_m


def make(kind, dt, rng):
    n = (3,) + SHAPE
    info = np.iinfo(dt) if np.dtype(dt).kind in "ui" else None
    top = int(info.max) if info else 1000
    a = np.zeros(n, np.float64)
    flat = a.reshape(3, -1)
    if kind == "very sparse":
        for f in range(3):
            idx = rng.choice(flat.shape[1], 12, replace=False)
            flat[f, idx] = rng.integers(1, 30, idx.size)
    elif kind == "sparse spots":
        for f in range(3):
            idx = rng.choice(flat.shape[1], 80, replace=False)
            flat[f, idx] = rng.integers(1, min(top, 5000) + 1, idx.size)
    elif kind == "high bits":
        for f in range(3):
            idx = rng.choice(flat.shape[1], 30, replace=False)
            flat[f, idx] = top
    elif kind == "medium":
        a[:] = rng.poisson(0.1, n)
    elif kind == "dense":
        a[:] = rng.poisson(30, n)
    elif kind == "saturated":
        # sparse counts, masked pixels at the maximum (as an Eiger writes
        # them), and a few unmasked saturated pixels at the maximum too
        for f in range(3):
            idx = rng.choice(flat.shape[1], 40, replace=False)
            flat[f, idx] = rng.integers(1, 20, idx.size)
            flat[f, rng.choice(flat.shape[1], 5, replace=False)] = top
        a[:, MASKED] = top
    if info is not None:
        a = np.minimum(a, top)
    return a.astype(dt)


def chunks_of(frames, path):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))], b.detect_codec(ds)


@pytest.fixture(autouse=True)
def restore():
    saved = (b.get_lz4_zero(), b.get_plane_extract(), b.get_byteskip(), b.get_mask_planes())
    yield
    b.set_lz4_zero(saved[0])
    b.set_plane_extract(saved[1])
    b.set_byteskip(saved[2])
    b.set_mask_planes(saved[3])


@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.int32, np.float32])
@pytest.mark.parametrize("kind", ["very sparse", "sparse spots", "high bits", "medium", "dense", "saturated"])
def test_sparse_output_unchanged(tmp_path, dt, kind):
    rng = np.random.default_rng(41)
    frames = make(kind, dt, rng)
    chunks, codec = chunks_of(frames, tmp_path / "z.h5")
    mask = MASK
    for cut in (0, 5):
        for zero, ext, skip, mplanes in itertools.product((True, False), repeat=4):
            b.set_lz4_zero(zero)
            b.set_plane_extract(ext)
            b.set_byteskip(skip)
            b.set_mask_planes(mplanes)
            integ = b.chunk2sparseMulti(mask, dtype=dt, codec=codec)
            npx, (vals, adr) = integ(chunks, cut)
            for f in range(len(frames)):
                flat = frames[f].ravel()
                want = np.flatnonzero((flat > cut) & (mask.ravel() > 0))
                n = int(npx[f])
                tag = (kind, cut, zero, ext, skip, mplanes, f)
                assert n == want.size, tag
                assert (adr[f, :n] == want).all(), tag
                assert (vals[f, :n] == flat[want]).all(), tag


@pytest.mark.parametrize("dt", [np.uint16, np.uint32])
@pytest.mark.parametrize("kind", ["very sparse", "medium", "dense", "saturated"])
@pytest.mark.parametrize("route", ["dense", "sparse"])
def test_powder_unchanged(tmp_path, dt, kind, route):
    rng = np.random.default_rng(42)
    frames = make(kind, dt, rng)
    chunks, codec = chunks_of(frames, tmp_path / "p.h5")
    npix = SHAPE[0] * SHAPE[1]
    mask = MASK
    M = sp.random(150, npix, density=0.02, format="csc", random_state=6, dtype=np.float32)
    ref = np.array([M @ (f.ravel().astype(np.float64) * mask.ravel()) for f in frames])
    saved = b.get_dense_sparse_threshold()
    try:
        b.set_dense_sparse_threshold(1e9 if route == "dense" else 0.0)
        for zero, skip, mplanes in itertools.product((True, False), repeat=3):
            b.set_lz4_zero(zero)
            b.set_byteskip(skip)
            b.set_mask_planes(mplanes)
            integ = b.chunk2sparseCSCmulti(mask, M, dtype=dt, codec=codec)
            npx, (vals, adr), p = integ(chunks, 0)
            assert np.allclose(p, ref, rtol=1e-9, atol=1e-9), (kind, route, zero, skip, mplanes)
            for f in range(len(frames)):          # the >cut output: masked never, saturated always
                flat = frames[f].ravel()
                want = np.flatnonzero((flat > 0) & (mask.ravel() > 0))
                n = int(npx[f])
                assert (adr[f, :n] == want).all(), (kind, route, zero, skip, mplanes, f)
    finally:
        b.set_dense_sparse_threshold(saved)
