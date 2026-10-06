"""Padded-CSC layout (pipeline dot "padded" / "padded-avx2").

Self-contained (no pyFAI): chunks are generated here with h5py + hdf5plugin,
and every result is checked against the conventional CSC decoder
(chunk2sparseCSC with pipeline={"dot": "csc"}) fed the same matrix, across dtypes,
both routes (dense/sparse) and every padded tier the machine can run.
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
from bslz4_to_sparse import _pipeline


def make_runs(npix, nbins, width_max, seed, empty=0.15):
    """A random CSC (nbins, npix) where every pixel reaches one run of
    consecutive bins (0..width_max of them), so it can be padded."""
    rng = np.random.default_rng(seed)
    n = rng.integers(0, width_max + 1, npix)
    n[rng.random(npix) < empty] = 0
    start = rng.integers(0, nbins - n + 1)
    indptr = np.concatenate(([0], np.cumsum(n)))
    indices = np.concatenate([start[i] + np.arange(n[i]) for i in range(npix)]) if n.sum() else np.zeros(0, int)
    data = rng.random(int(n.sum())).astype(np.float32) + 0.25
    return sp.csc_matrix((data, indices.astype(np.int32), indptr.astype(np.int32)), shape=(nbins, npix))


def write_chunks(tmp_path, frames, cname="lz4"):
    with h5py.File(str(tmp_path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname=cname, clevel=0))
        codec = b.detect_codec(ds)
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]
    return chunks, codec


def sparse_set(npx, vals, adr, f):
    n = int(npx[f])
    return set(zip(adr[f, :n].tolist(), vals[f, :n].tolist()))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """{(dtype, cname): (frames, chunks, codec)}"""
    out = {}
    d = tmp_path_factory.mktemp("padded")
    SHAPE = (60, 80)
    for dt in (np.uint8, np.uint16, np.uint32, np.int16, np.float32):
        frames = np.random.default_rng(7).poisson(0.6, (3,) + SHAPE).astype(dt)
        if np.dtype(dt).kind == "f":
            frames = frames * np.float32(1.5)
        if np.dtype(dt).kind == "i":
            frames = frames - 1
        chunks, codec = write_chunks(d / ("%s.h5" % np.dtype(dt).name), frames)
        out[(np.dtype(dt),)] = (frames, chunks, codec)
    return out


def tiers():
    out = []
    for name in ("padded-avx2",):
        if _pipeline.available("dot", _pipeline.NAMES["dot"].index(name)) == 1:
            out.append(name)
    out.append("padded")  # scalar always
    return out


def integ(mask, csc, dt, dot, route=None):
    return b.chunk2sparseCSC(mask, csc, dtype=dt, pipeline={"dot": dot, "route": route})


def _compare(csc, dt, chunks, cut=1, listed=None):
    """Run every available padded tier and bsb-csr against csc; return True if all match."""
    mask = np.ascontiguousarray(np.random.default_rng(9).random(60 * 80) > 0.1, np.uint8).reshape(60, 80)
    ok = True
    for route in ("dense", "sparse"):
        rc = integ(mask, csc, dt, "csc", route).multi(chunks, cut)
        for dot in tiers():
            rr = integ(mask, csc, dt, dot, route).multi(chunks, cut)
            for f in range(len(chunks)):
                n0, v0, a0, p0 = rc[0][f], rc[1][0][f], rc[1][1][f], rc[2][f]
                n1, v1, a1, p1 = rr[0][f], rr[1][0][f], rr[1][1][f], rr[2][f]
                if n0 != n1 or not np.array_equal(v0[:n0], v1[:n1]) or not np.array_equal(a0[:n0], a1[:n1]):
                    ok = False
                if not np.allclose(p0, p1, rtol=1e-10, atol=1e-9):
                    ok = False
    return ok


def test_padded_matches_csc_every_dtype(data):
    for (dt,), (frames, chunks, codec) in data.items():
        csc = make_runs(60 * 80, 200, 4, seed=10, empty=0.15)
        assert _compare(csc, dt, chunks), dt
    # also a listed-dense matrix (few pixels have entries)
    csc = make_runs(60 * 80, 100, 3, seed=11, empty=0.9)
    assert _compare(csc, np.uint16, data[(np.dtype(np.uint16),)][1])


def test_padded_rejects_non_consecutive():
    csc = sp.csc_matrix((np.ones(3, np.float32), np.array([0, 2, 5], np.int32), np.array([0, 1, 3], np.int32)),
                        shape=(10, 2))
    mask = np.ones((1, 2), np.uint8)
    for dot in tiers():
        with pytest.raises(ValueError, match="consecutive"):
            integ(mask, csc, np.uint16, dot)


def test_padded_sparse_output_order(data):
    _, chunks, _ = data[(np.dtype(np.uint16),)]
    csc = make_runs(60 * 80, 200, 4, seed=20)
    mask = np.ones((60, 80), np.uint8)
    for dot in tiers():
        npx, (outpx, adr), _ = integ(mask, csc, np.uint16, dot, "dense").multi(chunks, 1)
        for f in range(len(chunks)):
            n = int(npx[f])
            if n > 1:
                assert np.all(np.diff(adr[f, :n].astype(np.int64)) > 0), (dot, f, "order/repeat")


def test_padded_layout_attributes():
    csc = make_runs(60 * 80, 100, 3, seed=30, empty=0.9)
    mask = np.ones((60, 80), np.uint8)
    c = integ(mask, csc, np.uint16, "padded")
    lay = c._layout
    assert c.dot == "padded" and lay.entry == "padded"
    base, _w, _pix, _rowmap, row_ptr, _width, listed, block_elems = lay.args()
    assert listed == 1                       # sparse matrix -> listed
    assert block_elems == 4096 and lay.block_elems == 4096
    assert row_ptr[-1] == base.shape[0]


def test_padded_block_elems_mismatch():
    """A dataset whose encoded block size differs from the default (8192
    bytes, so 4096 u16 elems) still decodes correctly: the integrator rebuilds
    the layout from the first chunk header (C also rejects a mismatch)."""
    SHAPE = (60, 80)
    frames = np.random.default_rng(1).poisson(0.6, (3,) + SHAPE).astype(np.uint16)
    with h5py.File("_bshuf_block.h5", "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + SHAPE,
                              **hdf5plugin.Bitshuffle(nelems=512, cname="lz4", clevel=0))
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(3)]
    try:
        csc = make_runs(60 * 80, 100, 3, seed=40)
        mask = np.ones((60, 80), np.uint8)
        a = integ(mask, csc, np.uint16, "csc").multi(chunks, 1)
        pinteg = integ(mask, csc, np.uint16, "padded")
        p = pinteg.multi(chunks, 1)
        assert pinteg._layout.block_elems == 512
        assert pinteg._layout.args()[7] == 512
        for f in range(3):
            assert np.allclose(a[2][f], p[2][f])
    finally:
        import os
        if os.path.exists("_bshuf_block.h5"):
            os.remove("_bshuf_block.h5")
