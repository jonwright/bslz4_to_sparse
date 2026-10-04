"""Plane extraction (set_plane_extract): for well compressed blocks with few
non-zero pixels, the plain sparsify reads values straight from the bit-planes.
It must give exactly numpy's selection (values > cut of unmasked pixels, flat
indices, in order) and exactly the same as with extraction off, for u8, u16
and u32, around the 48-pixel limit, with high bit-planes set, with a mask and
without (the no-mask path), on a shape with a tail.
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

import bslz4_to_sparse as b

SHAPE = (101, 203)    # 20503 px: full blocks and a tail for every dtype


def make(kind, dt, rng):
    n = (3,) + SHAPE
    top = int(np.iinfo(dt).max)
    a = np.zeros(n, np.int64)
    flat = a.reshape(3, -1)
    if kind == "very sparse":
        for f in range(3):
            idx = rng.choice(flat.shape[1], 60, replace=False)
            flat[f, idx] = rng.integers(1, min(top, 5000) + 1, idx.size)
    elif kind == "near limit":
        # per decode block, 47..50 non-zero pixels: both sides of the limit
        be = 8192 // np.dtype(dt).itemsize
        for f in range(3):
            for s in range(0, flat.shape[1] - be, be):
                k = 47 + (s // be) % 4
                idx = s + rng.choice(be, k, replace=False)
                flat[f, idx] = rng.integers(1, min(top, 300) + 1, k)
    elif kind == "high planes":
        for f in range(3):
            idx = rng.choice(flat.shape[1], 40, replace=False)
            flat[f, idx] = top - rng.integers(0, 3, idx.size)
    elif kind == "medium":
        a[:] = rng.poisson(0.1, n)
    elif kind == "max":
        a[:] = top
    return np.minimum(a, top).astype(dt)


def chunks_of(frames, path):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))], b.detect_codec(ds)


@pytest.fixture(autouse=True)
def restore():
    saved = b.get_plane_extract()
    yield
    b.set_plane_extract(saved)


@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("kind", ["very sparse", "near limit", "high planes", "medium", "max"])
@pytest.mark.parametrize("masked", [True, False])
def test_plane_extract_matches_numpy(tmp_path, dt, kind, masked):
    rng = np.random.default_rng(31)
    frames = make(kind, dt, rng)
    chunks, codec = chunks_of(frames, tmp_path / "e.h5")
    mask = ((rng.random(SHAPE) > 0.2) if masked else np.ones(SHAPE, bool)).astype(np.uint8)
    top = int(np.iinfo(dt).max)
    for cut in sorted({0, 3, 200, min(top - 1, 2 ** 31 - 1), min(top, 2 ** 31 - 1)}):
        got = {}
        for on in (True, False):
            b.set_plane_extract(on)
            integ = b.chunk2sparseMulti(mask, dtype=dt, codec=codec)
            npx, (vals, adr) = integ(chunks, cut)
            got[on] = [(adr[f, :int(npx[f])].copy(), vals[f, :int(npx[f])].copy())
                       for f in range(len(frames))]
        for f in range(len(frames)):
            flatv = frames[f].ravel()
            want = np.flatnonzero((flatv > cut) & (mask.ravel() > 0))
            for on in (True, False):
                a, v = got[on][f]
                assert a.size == want.size and (a == want).all(), (kind, cut, on, f)
                assert (v == flatv[want]).all(), (kind, cut, on, f)


def test_extraction_path_is_taken(tmp_path):
    """The counters show the untranspose is skipped for very sparse blocks
    with extraction on, and runs for every block with it off."""
    rng = np.random.default_rng(32)
    # very sparse as in the benchmarks (~0.05 % non-zero, small counts), so the
    # blocks compress well past the extraction's compression filter
    frames = np.zeros((3,) + SHAPE, np.uint16)
    for f in range(3):
        idx = rng.choice(SHAPE[0] * SHAPE[1], 10, replace=False)
        frames[f].ravel()[idx] = rng.integers(1, 30, idx.size)
    chunks, codec = chunks_of(frames, tmp_path / "c.h5")
    mask = np.ones(SHAPE, np.uint8)
    U, D = b._STAGE_UNTRANSPOSE * b._COUNTER_SLOTS, b._STAGE_DECOMPRESS * b._COUNTER_SLOTS
    seen = {}
    for on in (True, False):
        b.set_plane_extract(on)
        integ = b.chunk2sparseMulti(mask, dtype=np.uint16, codec=codec)
        b.reset_counters()
        integ(chunks, 0)
        out = np.empty(4 * b._COUNTER_SLOTS, np.uint64)
        b.read_counters(out)
        seen[on] = (int(out[D:D + b._COUNTER_SLOTS].sum()), int(out[U:U + b._COUNTER_SLOTS].sum()))
    blocks_on, untr_on = seen[True]
    blocks_off, untr_off = seen[False]
    assert blocks_on == blocks_off
    assert untr_off == blocks_off                 # every block untransposed
    assert untr_on < blocks_on // 2, seen          # most blocks extracted instead
