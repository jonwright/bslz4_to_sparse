"""Bit-shuffle-block CSR layout (dot="bsb-csr").

Self-contained (no pyFAI): chunks are generated here with h5py + hdf5plugin,
and every result is checked against the conventional CSC decoder
(chunk2sparseCSCmulti with dot="csc") fed the same matrix.  bsb-csr is
general: it also handles a matrix whose pixels reach non-consecutive bins
(i.e. one padded must refuse), which is what makes the fallback layout work.
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


def make_any(npix, nbins, seed):
    """A random CSC (nbins, npix); pixels may reach any bins (no consecutive
    constraint), so padded cannot represent it."""
    rng = np.random.default_rng(seed)
    n = rng.integers(0, 4, npix)
    n[rng.random(npix) < 0.2] = 0
    indptr = np.concatenate(([0], np.cumsum(n)))
    indices = np.concatenate([rng.integers(0, nbins, n[i]) for i in range(npix)]) if n.sum() else np.zeros(0, int)
    data = rng.random(int(n.sum())).astype(np.float32) + 0.25
    return sp.csc_matrix((data, indices.astype(np.int32), indptr.astype(np.int32)), shape=(nbins, npix))


def make_frames(dt, seed, shape=(60, 80), nf=3):
    rng = np.random.default_rng(seed)
    frames = rng.poisson(0.6, (nf,) + shape).astype(dt)
    if np.dtype(dt).kind == "f":
        frames = frames * np.float32(1.5)
    if np.dtype(dt).kind == "i":
        frames = frames - 1
    return frames


def write_chunks(path, frames):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4", clevel=0))
        codec = b.detect_codec(ds)
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]
    return chunks, codec


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = {}
    d = tmp_path_factory.mktemp("bsb")
    for dt in (np.uint8, np.uint16, np.uint32, np.int16, np.float32):
        frames = make_frames(dt, 7)
        chunks, codec = write_chunks(d / ("%s.h5" % np.dtype(dt).name), frames)
        out[(np.dtype(dt),)] = (frames, chunks, codec)
    return out


@pytest.fixture
def restore_state():
    thr = b.get_dense_sparse_threshold()
    yield
    b.set_dense_sparse_threshold(thr)


def test_bsbcsr_matches_csc_every_dtype(data, restore_state):
    for (dt,), (frames, chunks, codec) in data.items():
        csc = make_any(60 * 80, 300, seed=10)
        mask = (np.random.default_rng(9).random(60 * 80) > 0.1).astype(np.uint8).reshape(60, 80)
        for route in (1e9, 0.0):
            b.set_dense_sparse_threshold(route)
            rc = b.chunk2sparseCSCmulti(mask, csc, dtype=dt, dot="csc")(chunks, 1)
            rr = b.chunk2sparseCSCmulti(mask, csc, dtype=dt, dot="bsb-csr")(chunks, 1)
            for f in range(len(chunks)):
                n0, v0, a0, p0 = rc[0][f], rc[1][0][f], rc[1][1][f], rc[2][f]
                n1, v1, a1, p1 = rr[0][f], rr[1][0][f], rr[1][1][f], rr[2][f]
                assert n0 == n1, (dt, route, f)
                assert np.array_equal(v0[:n0], v1[:n1]), (dt, route, f)
                assert np.array_equal(a0[:n0], a1[:n1]), (dt, route, f)
                assert np.allclose(p0, p1, rtol=1e-10, atol=1e-9), (dt, route, f)


def test_bsbcsr_handles_2d_padded_cannot():
    """A matrix with non-consecutive bins (2D / FAZIT-like) decodes correctly
    with bsb-csr, and is refused by padded."""
    csc = sp.csc_matrix((np.ones(3, np.float32), np.array([0, 2, 5], np.int32), np.array([0, 1, 3], np.int32)),
                        shape=(10, 2))
    shape = (1, 2)
    mask = np.ones(shape, np.uint8)
    frames = np.ones((1,) + shape, np.uint16)
    with h5py.File("_b.h5", "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + shape,
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4", clevel=0))
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(1)]
    try:
        # padded must refuse (non-consecutive)
        with pytest.raises(ValueError, match="consecutive"):
            b.chunk2sparseCSCmulti(mask, csc, dtype=np.uint16, dot="padded")
        # bsb-csr must produce the same powder as csc
        a = b.chunk2sparseCSCmulti(mask, csc, dtype=np.uint16, dot="csc")(chunks, 1)
        r = b.chunk2sparseCSCmulti(mask, csc, dtype=np.uint16, dot="bsb-csr")(chunks, 1)
        assert np.allclose(a[2], r[2])
    finally:
        import os
        if os.path.exists("_b.h5"):
            os.remove("_b.h5")


def test_bsbcsr_sparse_output_order(data, restore_state):
    _, chunks, _ = data[(np.dtype(np.uint16),)]
    csc = make_any(60 * 80, 300, seed=20)
    mask = np.ones((60, 80), np.uint8)
    b.set_dense_sparse_threshold(1e9)
    npx, (outpx, adr), _ = b.chunk2sparseCSCmulti(mask, csc, dtype=np.uint16, dot="bsb-csr")(chunks, 1)
    for f in range(len(chunks)):
        n = int(npx[f])
        if n > 1:
            assert np.all(np.diff(adr[f, :n].astype(np.int64)) > 0), (f, "order/repeat")


def test_bsbcsr_layout_attributes():
    csc = make_any(60 * 80, 100, seed=30)
    mask = np.ones((60, 80), np.uint8)
    integ = b.chunk2sparseCSCmulti(mask, csc, dtype=np.uint16, dot="bsb-csr")
    assert integ.layout == "bsb-csr" and integ.bsb_csr is not None
    q = integ.bsb_csr
    assert q.block_elems == 4096
    assert len(q.blk_ptr) == 3            # ceil(4800/4096)+1 blocks
    assert q.bins.size + 1 == q.bin_ptr.size
