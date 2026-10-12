"""howmany / thresholds: limit the pixels returned per frame (ImageD11's
top_pixels rules).

  * howmany=None (default): every pixel above the cut, as before;
  * howmany=N: the result is top_pixels(N, thresholds) applied to what the
    default call returns: the pixels above the first sorted threshold that
    leaves fewer than N (else the last), then the first N in pixel order;
  * default thresholds max(cut, 1) * 2**i, i = 0..15;
  * the powder never depends on howmany.

The oracle is written here from the rules (a plain loop), and checked against
ImageD11's own numba top_pixels when ImageD11 is installed.
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
from bslz4_to_sparse import _toppix

SHAPE = (96, 130)
NPIX = SHAPE[0] * SHAPE[1]


def oracle(values, indices, nnz, howmany, thresholds):
    """Reference: (values, indices) kept, by the rules, one frame."""
    values, indices = values[:nnz], indices[:nnz]
    if howmany is None or nnz <= howmany:
        return values.copy(), indices.copy()
    tcut = thresholds[-1]
    for t in thresholds:
        if sum(1 for v in values if v > t) < howmany:
            tcut = t
            break
    keep = [k for k in range(nnz) if values[k] > tcut][:howmany]
    return values[keep], indices[keep]


def frames_of(kind, dtype, n=3, seed=0):
    """sparse: ~1 % of pixels hit, small counts; dense: Poisson(6) nearly everywhere."""
    rng = np.random.default_rng(seed)
    top = 100 if np.dtype(dtype).itemsize == 1 else 5000
    out = np.zeros((n,) + SHAPE, dtype)
    for i in range(n):
        if kind == "sparse":
            hit = rng.random(SHAPE) < 0.01
            out[i][hit] = rng.integers(1, top, hit.sum())
        else:
            out[i] = np.minimum(rng.poisson(6, SHAPE), top)
    return out


def chunks_of(frames):
    with h5py.File("tp.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))], b.detect_codec(ds)


MASK = np.ones(SHAPE, np.uint8)
MASK[3, 5:9] = 0
DTYPES = (np.uint8, np.uint16, np.uint32, np.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["sparse", "dense"])
@pytest.mark.parametrize("cut", [0, 1, 3])
@pytest.mark.parametrize("howmany", [0, 1, 50, 400, 10 ** 9])
def test_matches_oracle(dtype, kind, cut, howmany):
    chunks, codec = chunks_of(frames_of(kind, dtype))
    full = b.chunk2sparse(MASK, dtype, codec)
    lim = b.chunk2sparse(MASK, dtype, codec, howmany=howmany)
    npx0, (v0, i0) = full.multi(chunks, cut)
    npx0 = npx0.copy()
    v0, i0 = v0.copy(), i0.copy()
    npx, (v, i) = lim.multi(chunks, cut)
    thr = _toppix.default_thresholds(cut)
    for f in range(len(chunks)):
        wv, wi = oracle(v0[f], i0[f], int(npx0[f]), howmany, thr)
        assert npx[f] == len(wv), (f, npx[f], len(wv))
        assert (v[f, :npx[f]] == wv).all() and (i[f, :npx[f]] == wi).all()
        assert npx[f] <= howmany
        assert (np.diff(i[f, :npx[f]].astype(np.int64)) > 0).all()
    assert (lim.last_nabove == npx0).all()
    assert (lim.last_cut >= cut).all()


def test_default_returns_everything():
    chunks, codec = chunks_of(frames_of("dense", np.uint16))
    c = b.chunk2sparse(MASK, np.uint16, codec)
    assert c.howmany is None
    npx, (v, i) = c.multi(chunks, 0)
    assert (c.last_nabove == npx).all() and (c.last_cut == 0).all()
    assert (npx > NPIX // 2).all()           # the dense frames really are dense


def test_default_thresholds():
    assert _toppix.default_thresholds(0) == [1 * 2 ** i for i in range(16)]
    assert _toppix.default_thresholds(1) == [2 ** i for i in range(16)]
    assert _toppix.default_thresholds(5) == [5 * 2 ** i for i in range(16)]
    assert _toppix.default_thresholds(0.5)[0] == 1          # max(cut, 1)


def test_cut_zero_still_reduces():
    """ImageD11's cut * 2**i is all zeros for cut 0: it never reduced the frame."""
    chunks, codec = chunks_of(frames_of("dense", np.uint16, n=1))
    c = b.chunk2sparse(MASK, np.uint16, codec, howmany=NPIX // 4)
    npx, (v, i) = c.multi(chunks, 0)
    assert c.last_nabove[0] > NPIX // 4
    assert c.last_cut[0] >= 1 and npx[0] <= NPIX // 4
    assert (v[0, :npx[0]] > c.last_cut[0]).all()


def test_explicit_thresholds():
    chunks, codec = chunks_of(frames_of("dense", np.uint16, n=1))
    full = b.chunk2sparse(MASK, np.uint16, codec)
    n0, (v0, i0) = full.multi(chunks, 0)
    v0, i0 = v0.copy(), i0.copy()
    for thr in ([3, 6, 9], [100], [], [9, 9, 9], [0, 2]):
        c = b.chunk2sparse(MASK, np.uint16, codec, howmany=1000, thresholds=thr)
        n, (v, i) = c.multi(chunks, 0)
        if len(thr):
            wv, wi = oracle(v0[0], i0[0], int(n0[0]), 1000, [float(t) for t in thr])
        else:                                                # no thresholds: the first howmany
            wv, wi = v0[0, :1000], i0[0, :1000]
        assert n[0] == len(wv) and (v[0, :n[0]] == wv).all() and (i[0, :n[0]] == wi).all(), thr


def test_validation():
    for bad in ({"howmany": -1}, {"howmany": 1.5}, {"thresholds": [2, 1]},
                {"thresholds": list(range(17))}):
        with pytest.raises(ValueError):
            b.chunk2sparse(MASK, np.uint16, b.CODEC_LZ4, **bad)
    b.chunk2sparse(MASK, np.uint16, b.CODEC_LZ4, howmany=0, thresholds=list(range(16)))


def histogram_matrix(nbins=20):
    bins = np.arange(NPIX) % nbins
    return sp.csc_matrix((np.ones(NPIX, np.float32), (bins, np.arange(NPIX))), shape=(nbins, NPIX))


def test_csc_powder_does_not_depend_on_howmany():
    chunks, codec = chunks_of(frames_of("dense", np.uint16))
    M = histogram_matrix()
    ref = b.chunk2sparseCSC(MASK, M, np.uint16, codec)
    lim = b.chunk2sparseCSC(MASK, M, np.uint16, codec, howmany=100)
    n0, (v0, i0), p0 = ref.multi(chunks, 2)
    p0 = p0.copy()
    n1, (v1, i1), p1 = lim.multi(chunks, 2)
    assert (p1 == p0).all()
    assert (n1 <= 100).all() and (n0 > 100).all()
    plain = b.chunk2sparse(MASK, np.uint16, codec, howmany=100)
    n2, (v2, i2) = plain.multi(chunks, 2)
    assert (n1 == n2).all() and (lim.last_cut == plain.last_cut).all()
    for f in range(len(chunks)):
        assert (v1[f, :n1[f]] == v2[f, :n2[f]]).all() and (i1[f, :n1[f]] == i2[f, :n2[f]]).all()


# the four situations of interest
def test_sparse_image_low_threshold_is_untouched():
    chunks, codec = chunks_of(frames_of("sparse", np.uint16))
    c = b.chunk2sparse(MASK, np.uint16, codec, howmany=100000)
    full = b.chunk2sparse(MASK, np.uint16, codec)
    n, (v, i) = c.multi(chunks, 0)
    n0, (v0, i0) = full.multi(chunks, 0)
    assert (n == n0).all() and (c.last_cut == 0).all()
    for f in range(len(chunks)):
        assert (v[f, :n[f]] == v0[f, :n0[f]]).all() and (i[f, :n[f]] == i0[f, :n0[f]]).all()


def test_dense_image_high_threshold_returns_sparse_list():
    frames = frames_of("dense", np.uint16)
    chunks, codec = chunks_of(frames)
    c = b.chunk2sparse(MASK, np.uint16, codec, howmany=100000)
    n, (v, i) = c.multi(chunks, 12)
    want = [np.flatnonzero((frames[f].ravel() > 12) & (MASK.ravel() > 0)) for f in range(len(chunks))]
    assert 0 < n.max() < NPIX // 10                      # genuinely sparse, no top_pixels needed
    for f in range(len(chunks)):
        assert (i[f, :n[f]] == want[f]).all()
    assert (c.last_cut == 12).all() and (c.last_nabove == n).all()


def test_sparse_image_high_threshold_gives_no_pixels_but_the_powder():
    chunks, codec = chunks_of(frames_of("sparse", np.uint16))
    M = histogram_matrix()
    c = b.chunk2sparseCSC(MASK, M, np.uint16, codec, howmany=100)
    n_lo, _, p_lo = c.multi(chunks, 0)
    n_lo, p_lo = n_lo.copy(), p_lo.copy()           # the next call overwrites these
    n_hi, _, p_hi = c.multi(chunks, 65534)
    assert (n_lo > 0).all() and (n_hi == 0).all()
    assert (p_hi == p_lo).all() and p_hi.sum() > 0


def test_dense_image_low_threshold_is_capped_and_reported():
    chunks, codec = chunks_of(frames_of("dense", np.uint16))
    c = b.chunk2sparse(MASK, np.uint16, codec, howmany=500, thresholds=[1])
    n, (v, i) = c.multi(chunks, 0)
    assert (c.last_nabove > 500).all()                   # the caller can see the frame was not sparse
    assert (n <= 500).all() and (n > 0).all()
    assert (c.last_cut == 1).all()                       # no threshold left fewer than howmany: the last


def test_matches_imaged11():
    lima = pytest.importorskip("ImageD11.sinograms.lima_segmenter")
    rng = np.random.default_rng(5)
    for trial in range(200):
        nnz = int(rng.integers(0, 400))
        val = rng.integers(1, 40, nnz).astype(np.uint16)
        idx = np.sort(rng.choice(5000, nnz, replace=False)).astype(np.uint32)
        howmany = int(rng.integers(0, 300))
        thr = sorted(rng.integers(0, 30, int(rng.integers(1, 8))).tolist())
        row, col, v = idx.copy(), idx.copy(), val.copy()
        n = lima.top_pixels(nnz, row, col, v, howmany, np.array(thr, np.float64))
        wv, wi = val.copy(), idx.copy()
        got, _ = _toppix.top_pixels(wv, wi, nnz, howmany, [float(t) for t in thr])
        assert n == got and (v[:n] == wv[:n]).all() and (row[:n] == wi[:n]).all(), trial
