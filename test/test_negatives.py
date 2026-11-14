"""negatives="list": for signed integer pixels the pixel list also holds every negative pixel,
whatever the cut (markers such as an older Pilatus's -1/-2 that change from frame to frame),
while the powder is unchanged (every unmasked pixel as it is).  Checked against numpy for i8,
i16, i32 and i64, every usable collect value, the low-planes untranspose (i16), chunks and
uncompressed frames, both dot routes and several dots; the correction the docs describe
(docs/index.md, "Pixel types and negative values") recovers the powder of the unmarked pixels;
the option is refused where it does not apply.
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

SHAPE = (93, 217)      # full blocks plus a tail
DTYPES = [np.int8, np.int16, np.int32, np.int64]
COLLECTS = [n for i, n in enumerate(_pipeline.NAMES["collect"])
            if i and _pipeline.available("collect", i) == 1]
LOWPLANES = [n for i, n in enumerate(_pipeline.NAMES["untranspose"])
             if n.startswith("lowplanes") and _pipeline.available("untranspose", i) == 1]


def frames_with_markers(dt, rng, n=3):
    """Poisson counts with -1/-2 markers at positions that change per frame."""
    f = np.minimum(rng.poisson(0.3, (n,) + SHAPE), 100).astype(np.int64)
    for i in range(n):
        d = rng.random(SHAPE) < 0.01
        f[i][d] = np.where(rng.random(int(d.sum())) < 0.5, -1, -2)
    f[:, 5, 10:80] = 50                  # a run above any small cut
    return f.astype(dt)


def chunks_of(frames, path):
    with h5py.File(str(path), "w") as h:
        ds = h.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]


def mask_of(rng):
    m = (rng.random(SHAPE) > 0.05).astype(np.uint8)
    m[:, 100:103] = 0
    return m


def check_list(frames, mask, cut, npx, vals, adr, tag):
    for f in range(len(frames)):
        flat = frames[f].ravel()
        want = np.flatnonzero(((flat > cut) | (flat < 0)) & (mask.ravel() > 0))
        n = int(npx[f])
        assert n == want.size, tag + (f, n, want.size)
        assert (adr[f, :n] == want).all(), tag + (f,)
        assert (vals[f, :n] == flat[want]).all(), tag + (f,)


@pytest.mark.parametrize("dt", DTYPES)
def test_list_holds_negatives(tmp_path, dt):
    rng = np.random.default_rng(31)
    frames = frames_with_markers(dt, rng)
    chunks = chunks_of(frames, tmp_path / "n.h5")
    mask = mask_of(rng)
    big = int(np.iinfo(dt).max) + 10
    unt = ["kcb"] + (LOWPLANES if dt == np.int16 else [])
    for collect in (COLLECTS if np.dtype(dt).itemsize < 8 else ["scalar"]):    # i64: scalar only
        for u in unt:
            o = b.chunk2sparse(mask, dtype=dt, codec=b.CODEC_LZ4, negatives="list",
                               pipeline={"collect": collect, "untranspose": u})
            for cut in (0, 3, big):          # above the maximum: the negatives only
                npx, (v, a) = o.multi(chunks, cut)
                check_list(frames, mask, min(cut, int(np.iinfo(dt).max)), npx, v, a,
                           ("chunks", np.dtype(dt).name, collect, u, cut))
                npx, (v, a) = o.from_frames(frames, cut)
                check_list(frames, mask, min(cut, int(np.iinfo(dt).max)), npx, v, a,
                           ("frames", np.dtype(dt).name, collect, u, cut))
            # the default leaves them out, as before
            d = b.chunk2sparse(mask, dtype=dt, codec=b.CODEC_LZ4,
                               pipeline={"collect": collect, "untranspose": u})
            npx, (v, a) = d.multi(chunks, 0)
            assert all((v[f, :npx[f]] > 0).all() for f in range(len(frames)))


@pytest.mark.parametrize("dt", [np.int16, np.int32])
@pytest.mark.parametrize("route", ["dense", "sparse"])
def test_powder_and_correction(tmp_path, dt, route):
    rng = np.random.default_rng(32)
    frames = frames_with_markers(dt, rng)
    chunks = chunks_of(frames, tmp_path / "p.h5")
    mask = mask_of(rng)
    npix = mask.size
    M = sp.random(150, npix, density=0.02, format="csc", random_state=4, dtype=np.float32)
    b0 = rng.integers(0, 149, npix)                         # each pixel in two consecutive bins
    runs = sp.csc_matrix((np.tile([0.6, 0.4], npix).astype(np.float32),
                          np.stack([b0, b0 + 1], 1).ravel(), np.arange(0, 2 * npix + 1, 2)),
                         shape=(150, npix))
    nosplit = sp.csc_matrix((np.ones(npix, np.float32), rng.integers(0, 40, npix),
                             np.arange(npix + 1)), shape=(40, npix))
    norm = (0.5 + rng.random(npix)).astype(np.float32)      # solid angle, polarisation, ...
    cases = [(M, "csc")] + [(runs, d) for d in ("csc-run", "padded", "bsb-csr")] + \
            [(nosplit, d) for d in ("csc-nosplit", "bsb-csr-nosplit", "csc-nosplit-int")]
    for mat, dot in cases:
        for collect in COLLECTS:
            o = b.chunk2sparseCSC(mask, mat, dtype=dt, codec=b.CODEC_LZ4, negatives="list",
                                  pipeline={"route": route, "dot": dot, "collect": collect})
            npx, (v, a), p = o.multi(chunks, 2)
            check_list(frames, mask, 2, npx, v, a, (np.dtype(dt).name, route, dot, collect))
            Md = mat.tocsc().astype(np.float64)
            for f in range(len(frames)):
                flat = frames[f].ravel().astype(np.float64)
                m = mask.ravel() > 0
                assert np.allclose(p[f], Md @ (flat * m), rtol=1e-9, atol=1e-9), (dot, route, f)
                # the caller's correction from the list: the marked pixels out of the powder and
                # out of the normalisation
                n = int(npx[f])
                vals, idx = v[f, :n], a[f, :n].astype(np.int64)
                marked = vals < 0
                fixed = p[f] - Md[:, idx[marked]] @ vals[marked].astype(np.float64)
                keep = m & (flat >= 0)
                assert np.allclose(fixed, Md @ (flat * keep), rtol=1e-9, atol=1e-9), (dot, route, f)
                nfix = Md @ (norm * m) - Md[:, idx[marked]] @ norm[idx[marked]]
                assert np.allclose(nfix, Md @ (norm * keep), rtol=1e-6, atol=1e-6), (dot, route, f)


def test_refused():
    mask = np.ones(SHAPE, np.uint8)
    with pytest.raises(ValueError, match="signed integer"):
        b.chunk2sparse(mask, dtype=np.uint16, negatives="list")
    with pytest.raises(ValueError, match="signed integer"):
        b.chunk2sparse(mask, dtype=np.float32, negatives="list")
    with pytest.raises(ValueError, match="howmany"):
        b.chunk2sparse(mask, dtype=np.int32, negatives="list", howmany=100)
    with pytest.raises(ValueError, match='"drop" or "list"'):
        b.chunk2sparse(mask, dtype=np.int32, negatives="keep")
    assert b.chunk2sparse(mask, dtype=np.int32).negatives == "drop"
