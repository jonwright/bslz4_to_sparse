"""Decode blocks far from the default 8 kB: 256 kB and 1 MB, lz4 and zstd.

The SIMD steps are written for blocks of up to a few thousand pixels
(lowplanes: <= 8192 pixels; mask planes: <= 8192 pixels; bsb-csr:
<= 65536 pixels).  Larger blocks must fall back to a generic step or be
refused with an error, never give wrong numbers.

  * sparsify: every untranspose x collect x mask the CPU offers, against numpy;
  * dots: every dot against numpy `M @ (img * mask)` on both routes.  The
    bsb-csr dots only handle blocks of <= 65536 pixels: they must raise a
    ValueError, not run wrongly.  (A dot that cannot represent the matrix
    refuses it at construction; that is not a block-size matter.)

Self-contained (no pyFAI).
"""

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")
sp = pytest.importorskip("scipy.sparse")

import bslz4_to_sparse as b
from bslz4_to_sparse import _pipeline
from test_layouts import (available_dots, expected, make_nosplit,
                          make_nosplit_moment, reference)

SHAPE = (1100, 1200)      # 1.32 Mpx: 2 blocks + a tail at 1 MB (u16), more for smaller
NPIX = SHAPE[0] * SHAPE[1]
BLOCK_BYTES = {"256kB": 256 * 1024, "1MB": 1024 * 1024}
CODECS = ("lz4", "zstd")


def available(step):
    return [n for i, n in enumerate(_pipeline.NAMES[step])
            if i and _pipeline.available(step, i) == 1]


def write_chunks(frames, cname, block_bytes):
    nelems = block_bytes // frames.dtype.itemsize
    with h5py.File("blocks.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=nelems, cname=cname))
        codec = b.detect_codec(ds)
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]
    return chunks, codec


def make_split_1d(npix, nbins, seed):
    """Each pixel splits over 1-3 consecutive bins, weights summing to 1."""
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 4, npix)
    start = rng.integers(0, nbins - 3, npix)
    indptr = np.concatenate(([0], np.cumsum(n)))
    within = np.arange(int(n.sum())) - indptr[:-1].repeat(n)
    indices = start.repeat(n) + within
    w = rng.random(int(n.sum())) + 0.1
    w /= np.add.reduceat(w, indptr[:-1]).repeat(n)
    return sp.csc_matrix((w.astype(np.float32), indices.astype(np.int32),
                          indptr.astype(np.int32)), shape=(nbins, npix))


@pytest.mark.parametrize("block", sorted(BLOCK_BYTES))
@pytest.mark.parametrize("cname", CODECS)
@pytest.mark.parametrize("dt", [np.uint16, np.uint32])
def test_sparsify_every_step_matches_numpy(cname, block, dt):
    rng = np.random.default_rng(11)
    frames = np.minimum(rng.poisson(0.3, (2,) + SHAPE), np.iinfo(dt).max).astype(dt)
    frames[:, 10:14, 50:54] = 900
    chunks, codec = write_chunks(frames, cname, BLOCK_BYTES[block])
    assert int.from_bytes(bytes(chunks[0][8:12]), "big") == BLOCK_BYTES[block]
    mask = (rng.random(SHAPE) > 0.1).astype(np.uint8)
    ran = 0
    for un in available("untranspose"):
        for co in available("collect"):
            for mk in ("pixel", "none"):
                m = mask if mk == "pixel" else np.ones(SHAPE, np.uint8)
                try:
                    integ = b.chunk2sparse(m, dtype=dt, codec=codec,
                                           pipeline={"untranspose": un, "collect": co, "mask": mk})
                except ValueError:                  # step not for this dtype / codec
                    continue
                npx, (vals, adr) = integ.multi(chunks, 1)
                for f in range(len(frames)):
                    flat = frames[f].ravel()
                    want = np.flatnonzero((flat > 1) & (m.ravel() > 0))
                    n = int(npx[f])
                    assert n == want.size, (un, co, mk, f)
                    assert (adr[f, :n] == want).all(), (un, co, mk, f)
                    assert (vals[f, :n] == flat[want]).all(), (un, co, mk, f)
                ran += 1
    assert ran >= 2


@pytest.mark.parametrize("block", sorted(BLOCK_BYTES))
@pytest.mark.parametrize("cname", CODECS)
@pytest.mark.parametrize("kind", ["split1d", "nosplit", "moment"])
def test_every_dot_matches_numpy_or_is_refused(cname, block, kind):
    rng = np.random.default_rng(12)
    frames = rng.poisson(0.3, (2,) + SHAPE).astype(np.uint16)
    chunks, codec = write_chunks(frames, cname, BLOCK_BYTES[block])
    mask = (rng.random(SHAPE) > 0.1).astype(np.uint8)
    M = {"split1d": lambda: make_split_1d(NPIX, 200, 4),
         "nosplit": lambda: make_nosplit(NPIX, 250, 8),
         "moment": lambda: make_nosplit_moment(NPIX, 240, 9)}[kind]()
    ref = reference(M, frames, mask)
    keep = mask > 0
    ok, refused_at_run = set(), set()
    for dot in available_dots():
        for route in ("dense", "sparse"):
            try:
                integ = b.chunk2sparseCSC(mask, M, dtype=np.uint16, codec=codec,
                                          pipeline={"dot": dot, "route": route})
            except ValueError:                      # cannot represent this matrix
                continue
            try:
                _npx, _vp, p = integ.multi(chunks, 0)
            except ValueError as e:                 # a limit of the layout: must say so
                assert "65536" in str(e) and "block_elems" in str(e), (dot, route, str(e))
                refused_at_run.add(dot)
                continue
            r, tol = expected(integ, frames * keep, ref)
            err = np.abs(p - r).max() / max(np.abs(r).max(), 1.0)
            assert err < tol, (dot, route, err)
            ok.add(dot)
    assert "csc" in ok
    assert not (ok & refused_at_run)
    # bsb-csr is built for blocks of <= 65536 pixels; both of these are larger
    assert refused_at_run <= {"bsb-csr", "bsb-csr-nosplit"}
    assert "bsb-csr" in refused_at_run
    if kind == "split1d":
        assert "padded" in ok


@pytest.mark.parametrize("nelems", [2048, 4096, 8192, 8256, 8448, 16384, 20480, 65536])
@pytest.mark.parametrize("masked", [True, False])
def test_lowplanes_tiles(nelems, masked):
    """The fused low-planes collects work in tiles: blocks below, at, just
    above and a ragged multiple of the tile (8256 = 64 over, 8448 = 256 over)
    must give numpy's answer.  Frame 0 has every value < 256 (the fused path),
    frame 1 a few larger ones (the block falls back), frame 2 is sparse."""
    rng = np.random.default_rng(13)
    frames = np.minimum(rng.poisson(0.5, (3,) + SHAPE), 255).astype(np.uint16)
    frames[1, 500:510, 600:610] = 900
    frames[2] = np.where(rng.random(SHAPE) < 1e-3, 7, 0)
    chunks, codec = write_chunks(frames, "lz4", nelems * 2)
    mask = ((rng.random(SHAPE) > 0.1) if masked else np.ones(SHAPE, bool)).astype(np.uint8)
    ran = 0
    for un in [n for n in available("untranspose") if n.startswith("lowplanes")]:
        for co in available("collect"):
            integ = b.chunk2sparse(mask, dtype=np.uint16, codec=codec,
                                   pipeline={"untranspose": un, "collect": co,
                                             "mask": "pixel" if masked else "none"})
            for cut in (0, 3):
                npx, (vals, adr) = integ.multi(chunks, cut)
                for f in range(3):
                    flat = frames[f].ravel()
                    want = np.flatnonzero((flat > cut) & (mask.ravel() > 0))
                    n = int(npx[f])
                    assert n == want.size, (un, co, cut, f)
                    assert (adr[f, :n] == want).all(), (un, co, cut, f)
                    assert (vals[f, :n] == flat[want]).all(), (un, co, cut, f)
            ran += 1
    assert ran >= 1


def test_auto_dot_falls_back_on_large_blocks():
    """An auto dot that is bsb-csr on the default block becomes the csc form
    on a 1 MB block, and is right; an explicit bsb-csr still refuses."""
    rng = np.random.default_rng(14)
    frames = rng.poisson(0.3, (2,) + SHAPE).astype(np.uint16)
    mask = (rng.random(SHAPE) > 0.1).astype(np.uint8)
    M = make_nosplit(NPIX, 250, 8)
    ref = reference(M, frames, mask)
    for nelems_bytes, want_fallback in ((8192, False), (1024 * 1024, True)):
        chunks, codec = write_chunks(frames, "lz4", nelems_bytes)
        integ = b.chunk2sparseCSC(mask, M, dtype=np.uint16, codec=codec)
        assert integ.dot == "bsb-csr-nosplit"
        _npx, _vp, p = integ.multi(chunks, 0)
        assert (integ.dot == "csc-nosplit") == want_fallback
        assert _pipeline.NAMES["dot"][integ.pipeline[_pipeline.DOT]] == integ.dot
        r, tol = expected(integ, frames * (mask > 0), ref)
        assert np.abs(p - r).max() / max(np.abs(r).max(), 1.0) < tol
