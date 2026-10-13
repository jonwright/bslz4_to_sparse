"""Untranspose values: the low-planes untransposes (u16 blocks whose high
byte-planes are all zero untranspose only the low byte's planes), kcb and
scalar must all be byte-exact, so every frame is reconstructed from the
sparse output with every untranspose value usable for its dtype, and a CSC
powder must be identical for each.  Shapes leave a partial tail block.
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

SHAPE = (97, 211)   # 20467 px: several full blocks and a tail for every dtype


def frames_for(kind, rng):
    n = (3,) + SHAPE
    if kind == "u16 <256":
        return rng.poisson(30, n).astype(np.uint16)
    if kind == "u16 mixed":
        a = rng.poisson(30, n)
        a[:, ::9, ::17] = rng.integers(256, 65535, a[:, ::9, ::17].shape)
        return a.astype(np.uint16)
    if kind == "u32 <256":
        return rng.poisson(30, n).astype(np.uint32)
    if kind == "u32 <65536":
        return rng.poisson(3000, n).astype(np.uint32)
    if kind == "u32 mixed":
        a = rng.poisson(30, n)
        a[:, ::7, ::13] = rng.integers(70000, 10 ** 9, a[:, ::7, ::13].shape)
        return a.astype(np.uint32)
    if kind == "i16 negatives":
        return (rng.poisson(30, n) - 15).astype(np.int16)
    if kind == "i32 small":
        return rng.poisson(20, n).astype(np.int32)
    if kind == "f32":
        return rng.poisson(5, n).astype(np.float32)
    if kind == "u8":
        return rng.poisson(5, n).astype(np.uint8)
    if kind == "u64 <256":
        return rng.poisson(30, n).astype(np.uint64)
    if kind == "u64 <2^32":
        return rng.integers(0, 2 ** 32, n).astype(np.uint64)
    if kind == "u64 5-8 bytes":
        a = rng.poisson(30, n).astype(np.uint64)
        a[:, ::5, ::7] = rng.integers(2 ** 33, 2 ** 62, a[:, ::5, ::7].shape)
        return a
    if kind == "i64 big":
        return rng.integers(0, 10 ** 12, n).astype(np.int64)
    if kind == "f64":
        return rng.poisson(5, n).astype(np.float64)
    raise ValueError(kind)


KINDS = ("u16 <256", "u16 mixed", "u32 <256", "u32 <65536", "u32 mixed",
         "i16 negatives", "i32 small", "f32", "u8",
         "u64 <256", "u64 <2^32", "u64 5-8 bytes", "i64 big", "f64")


def chunks_of(frames, path):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        codec = b.detect_codec(ds)
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))], codec


def untransposes(dt):
    """The untranspose values usable here for pixels of dtype dt (the
    low-planes ones are for u16 only)."""
    out = []
    for i, n in enumerate(_pipeline.NAMES["untranspose"]):
        if not i or _pipeline.available("untranspose", i) != 1:
            continue
        if n.startswith("lowplanes") and np.dtype(dt) != np.uint16:
            continue
        out.append(n)
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_frames_identical_for_every_untranspose(tmp_path, kind):
    rng = np.random.default_rng(12)
    frames = frames_for(kind, rng)
    chunks, codec = chunks_of(frames, tmp_path / "f.h5")
    mask = np.ones(SHAPE, np.uint8)
    dt = frames.dtype
    cut = 0 if dt.kind in "uf" else np.iinfo(dt).min   # every pixel above the cut
    got = {}
    names = untransposes(dt)
    assert "kcb" in names
    for skip in names:
        integ = b.chunk2sparse(mask, dtype=dt, codec=codec, pipeline={"untranspose": skip})
        npx, (vals, adr) = integ.multi(chunks, max(int(cut), 0))
        rec = np.zeros((len(frames), SHAPE[0] * SHAPE[1]), dt)
        for f in range(len(frames)):
            n = int(npx[f])
            rec[f, adr[f, :n]] = vals[f, :n]
        got[skip] = rec
    keep = frames.reshape(len(frames), -1).copy()
    if dt.kind in "ui":
        keep[keep <= 0] = 0     # the sparse output holds values > cut (0)
    for skip in names:
        assert (got[skip] == got["kcb"]).all(), (kind, skip)
        assert (got[skip] == keep).all(), (kind, skip)


@pytest.mark.parametrize("kind", ("u16 <256", "u16 mixed", "u32 <256", "u32 <65536"))
@pytest.mark.parametrize("route", ("dense", "sparse"))
def test_powder_identical_for_every_untranspose(tmp_path, kind, route):
    rng = np.random.default_rng(13)
    frames = frames_for(kind, rng)
    chunks, codec = chunks_of(frames, tmp_path / "p.h5")
    npix = SHAPE[0] * SHAPE[1]
    mask = (rng.random(npix) > 0.1).astype(np.uint8).reshape(SHAPE)
    M = sp.random(300, npix, density=0.01, format="csc", random_state=4, dtype=np.float32)
    out = {}
    names = untransposes(frames.dtype)
    for skip in names:
        integ = b.chunk2sparseCSC(mask, M, dtype=frames.dtype, codec=codec,
                                  pipeline={"untranspose": skip, "route": route})
        out[skip] = integ.multi(chunks, 0)[2].copy()
    ref = np.array([M @ (f.ravel().astype(np.float64) * mask.ravel()) for f in frames])
    for skip in names:
        assert (out[skip] == out["kcb"]).all(), skip
        assert np.allclose(out[skip], ref, rtol=1e-9, atol=1e-9), skip
