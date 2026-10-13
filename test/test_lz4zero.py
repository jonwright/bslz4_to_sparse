"""The zero-aware lz4 block decoder (pipeline decode lz4-band / lz4-zero vs
lz4-stock), the u16 low-planes untranspose (lowplanes-* vs kcb / scalar) and
the mask applied in the bitshuffled domain (mask planes vs pixel) must change
nothing but speed: sparse output and CSC powder identical to numpy for every
dtype and every combination -- including masked pixels that
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
from bslz4_to_sparse import _pipeline

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


DECODES = ("lz4-band", "lz4-zero", "lz4-stock")
MASKS = ("pixel", "planes")


def untransposes(dt):
    """The untranspose values usable here for this dtype (lowplanes-* are u16 only)."""
    out = []
    for i, n in enumerate(_pipeline.NAMES["untranspose"]):
        if i == 0 or _pipeline.available("untranspose", i) != 1:
            continue
        if n.startswith("lowplanes") and np.dtype(dt) != np.uint16:
            continue
        out.append(n)
    return out


@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.int32, np.float32])
@pytest.mark.parametrize("kind", ["very sparse", "sparse spots", "high bits", "medium", "dense", "saturated"])
def test_sparse_output_unchanged(tmp_path, dt, kind):
    rng = np.random.default_rng(41)
    frames = make(kind, dt, rng)
    chunks, codec = chunks_of(frames, tmp_path / "z.h5")
    mask = MASK
    for cut in (0, 5):
        for dec, unt, msk in itertools.product(DECODES, untransposes(dt), MASKS):
            integ = b.chunk2sparse(mask, dtype=dt, codec=codec,
                                   pipeline={"decode": dec, "untranspose": unt, "mask": msk})
            npx, (vals, adr) = integ.multi(chunks, cut)
            for f in range(len(frames)):
                flat = frames[f].ravel()
                want = np.flatnonzero((flat > cut) & (mask.ravel() > 0))
                n = int(npx[f])
                tag = (kind, cut, dec, unt, msk, f)
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
    for dec, unt, msk in itertools.product(DECODES, untransposes(dt), MASKS):
        integ = b.chunk2sparseCSC(mask, M, dtype=dt, codec=codec,
                                  pipeline={"decode": dec, "untranspose": unt, "mask": msk,
                                            "route": route})
        npx, (vals, adr), p = integ.multi(chunks, 0)
        tag = (kind, route, dec, unt, msk)
        assert np.allclose(p, ref, rtol=1e-9, atol=1e-9), tag
        for f in range(len(frames)):          # the >cut output: masked never, saturated always
            flat = frames[f].ravel()
            want = np.flatnonzero((flat > 0) & (mask.ravel() > 0))
            n = int(npx[f])
            assert n == want.size, tag + (f,)
            assert (adr[f, :n] == want).all(), tag + (f,)


def test_bad_values_refused():
    """lowplanes-* is u16 only; 'none' needs a mask with no zeros; the old
    option names are not pipeline values."""
    with pytest.raises(ValueError, match="is for u16 pixels"):
        b.chunk2sparse(MASK, dtype=np.uint32, pipeline={"untranspose": "lowplanes-c"})
    with pytest.raises(ValueError, match="masked pixels"):
        b.chunk2sparse(MASK, dtype=np.uint16, pipeline={"mask": "none"})
    with pytest.raises(ValueError, match="unknown"):
        b.chunk2sparse(MASK, dtype=np.uint16, pipeline={"decode": "lz4_zero"})
    with pytest.raises(ValueError, match="unknown steps"):
        b.chunk2sparse(MASK, dtype=np.uint16, pipeline={"byteskip": 1})
    with pytest.raises(ValueError, match="zstd"):
        b.chunk2sparse(MASK, dtype=np.uint16, codec=b.CODEC_LZ4, pipeline={"decode": "zstd"})
