"""Every available collect tier (scalar, avx512, avx512cs, avx2, sse2, vsx,
neon) must give exactly numpy's selection: the values > cut of the unmasked
pixels, with their flat indices, in order.  Run for u16 and u32 over sparse,
medium, dense and all-maximum data, several cuts, with a mask and with an
all-ones mask (the no-mask path), on a shape that leaves a tail; and the CSC
powder must agree across tiers on both routes.
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

SHAPE = (93, 217)   # 20181 px: full blocks plus a tail for u16 and u32
TIERS = [i for i in sorted(b._COLLECT_ID_TO_NAME) if b.impl_available(b._STAGE_COLLECT, i) == 1]


def make(kind, dt, rng):
    n = (3,) + SHAPE
    top = np.iinfo(dt).max
    if kind == "sparse":
        a = np.zeros(n, np.int64)
        a[rng.random(n) < 1e-3] = rng.integers(1, 1000, int((rng.random(n) < 1e-3).sum()) or 1)[0]
        a[:, 10:14, 50:54] = 900
    elif kind == "medium":
        a = rng.poisson(0.1, n)
    elif kind == "dense":
        a = rng.poisson(30, n)
    else:  # "max"
        a = np.full(n, top, np.int64)
    return np.minimum(a, top).astype(dt)


def chunks_of(frames, path):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))], b.detect_codec(ds)


@pytest.mark.parametrize("dt", [np.uint16, np.uint32])
@pytest.mark.parametrize("kind", ["sparse", "medium", "dense", "max"])
@pytest.mark.parametrize("masked", [True, False])
def test_every_tier_matches_numpy(tmp_path, dt, kind, masked):
    rng = np.random.default_rng(21)
    frames = make(kind, dt, rng)
    chunks, codec = chunks_of(frames, tmp_path / "c.h5")
    mask = ((rng.random(SHAPE) > 0.15) if masked else np.ones(SHAPE, bool)).astype(np.uint8)
    # the threshold is a C int: the largest cut tried is min(max - 1, 2**31 - 1)
    for cut in (0, 3, 600, min(int(np.iinfo(dt).max) - 1, 2 ** 31 - 1)):
        for tier in TIERS:
            p = b.pack_pipeline(collect=tier, options=0 if masked else b._OPT_NO_MASK)
            integ = b.chunk2sparseMulti(mask, dtype=dt, codec=codec, pipeline=p)
            npx, (vals, adr) = integ(chunks, cut)
            for f in range(len(frames)):
                flat = frames[f].ravel()
                want = np.flatnonzero((flat > cut) & (mask.ravel() > 0))
                n = int(npx[f])
                assert n == want.size, (b._COLLECT_ID_TO_NAME[tier], kind, cut, f)
                assert (adr[f, :n] == want).all(), (b._COLLECT_ID_TO_NAME[tier], kind, cut, f)
                assert (vals[f, :n] == flat[want]).all(), (b._COLLECT_ID_TO_NAME[tier], kind, cut, f)


@pytest.mark.parametrize("dt", [np.uint16, np.uint32])
@pytest.mark.parametrize("route", ["dense", "sparse"])
def test_powder_same_for_every_tier(tmp_path, dt, route):
    rng = np.random.default_rng(22)
    frames = make("medium", dt, rng)
    chunks, codec = chunks_of(frames, tmp_path / "p.h5")
    npix = SHAPE[0] * SHAPE[1]
    mask = (rng.random(SHAPE) > 0.1).astype(np.uint8)
    M = sp.random(200, npix, density=0.02, format="csc", random_state=5, dtype=np.float32)
    saved = b.get_dense_sparse_threshold()
    try:
        b.set_dense_sparse_threshold(1e9 if route == "dense" else 0.0)
        res = {}
        for tier in TIERS:
            integ = b.chunk2sparseCSCmulti(mask, M, dtype=dt, codec=codec,
                                           pipeline=b.pack_pipeline(collect=tier))
            npx, (vals, adr), p = integ(chunks, 0)
            res[tier] = (p.copy(), npx.copy())
    finally:
        b.set_dense_sparse_threshold(saved)
    ref = np.array([M @ (f.ravel().astype(np.float64) * mask.ravel()) for f in frames])
    for tier, (p, npx) in res.items():
        assert np.allclose(p, ref, rtol=1e-9, atol=1e-9), b._COLLECT_ID_TO_NAME[tier]
        assert (npx == res[TIERS[0]][1]).all(), b._COLLECT_ID_TO_NAME[tier]
