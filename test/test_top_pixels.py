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
        ds = f.create_dataset(
            "d",
            data=frames,
            chunks=(1,) + frames.shape[1:],
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
        )
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))], b.detect_codec(
            ds
        )


MASK = np.ones(SHAPE, np.uint8)
MASK[3, 5:9] = 0
DTYPES = (np.uint8, np.uint16, np.uint32, np.float32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["sparse", "dense"])
@pytest.mark.parametrize("cut", [0, 1, 3])
@pytest.mark.parametrize("howmany", [0, 1, 50, 400, 10**9])
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
        assert (v[f, : npx[f]] == wv).all() and (i[f, : npx[f]] == wi).all()
        assert npx[f] <= howmany
        assert (np.diff(i[f, : npx[f]].astype(np.int64)) > 0).all()
    for f in range(
        len(chunks)
    ):  # last_nabove: the pixels above the cut the frame ended with, before truncation
        assert lim.last_nabove[f] == np.count_nonzero(v0[f, : npx0[f]] > lim.last_cut[f]), f
    assert (lim.last_cut >= cut).all()


def test_default_returns_everything():
    chunks, codec = chunks_of(frames_of("dense", np.uint16))
    c = b.chunk2sparse(MASK, np.uint16, codec)
    assert c.howmany is None
    npx, (v, i) = c.multi(chunks, 0)
    assert (c.last_nabove == npx).all() and (c.last_cut == 0).all()
    assert (npx > NPIX // 2).all()  # the dense frames really are dense


def test_default_thresholds():
    assert _toppix.default_thresholds(0) == [1 * 2**i for i in range(16)]
    assert _toppix.default_thresholds(1) == [2**i for i in range(16)]
    assert _toppix.default_thresholds(5) == [5 * 2**i for i in range(16)]
    assert _toppix.default_thresholds(0.5)[0] == 1  # max(cut, 1)


def test_cut_zero_still_reduces():
    """ImageD11's cut * 2**i is all zeros for cut 0: it never reduced the frame."""
    chunks, codec = chunks_of(frames_of("dense", np.uint16, n=1))
    c = b.chunk2sparse(MASK, np.uint16, codec, howmany=NPIX // 4)
    npx, (v, i) = c.multi(chunks, 0)
    assert c.last_cut[0] >= 1 and npx[0] <= NPIX // 4
    assert (
        c.last_nabove[0] == npx[0]
    )  # a level was found with fewer than howmany: nothing truncated
    assert (v[0, : npx[0]] > c.last_cut[0]).all()


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
        else:  # no thresholds: the first howmany
            wv, wi = v0[0, :1000], i0[0, :1000]
        assert n[0] == len(wv) and (v[0, : n[0]] == wv).all() and (i[0, : n[0]] == wi).all(), thr


def test_validation():
    for bad in (
        {"howmany": -1},
        {"howmany": 1.5},
        {"thresholds": [2, 1]},
        {"thresholds": list(range(17))},
    ):
        with pytest.raises(ValueError):
            b.chunk2sparse(MASK, np.uint16, b.CODEC_LZ4, **bad)
    b.chunk2sparse(MASK, np.uint16, b.CODEC_LZ4, howmany=0, thresholds=list(range(16)))


def histogram_matrix(nbins=20, npix=NPIX):
    bins = np.arange(npix) % nbins
    return sp.csc_matrix((np.ones(npix, np.float32), (bins, np.arange(npix))), shape=(nbins, npix))


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
        assert (v1[f, : n1[f]] == v2[f, : n2[f]]).all() and (i1[f, : n1[f]] == i2[f, : n2[f]]).all()


# the four situations of interest
def test_sparse_image_low_threshold_is_untouched():
    chunks, codec = chunks_of(frames_of("sparse", np.uint16))
    c = b.chunk2sparse(MASK, np.uint16, codec, howmany=100000)
    full = b.chunk2sparse(MASK, np.uint16, codec)
    n, (v, i) = c.multi(chunks, 0)
    n0, (v0, i0) = full.multi(chunks, 0)
    assert (n == n0).all() and (c.last_cut == 0).all()
    for f in range(len(chunks)):
        assert (v[f, : n[f]] == v0[f, : n0[f]]).all() and (i[f, : n[f]] == i0[f, : n0[f]]).all()


def test_dense_image_high_threshold_returns_sparse_list():
    frames = frames_of("dense", np.uint16)
    chunks, codec = chunks_of(frames)
    c = b.chunk2sparse(MASK, np.uint16, codec, howmany=100000)
    n, (v, i) = c.multi(chunks, 12)
    want = [
        np.flatnonzero((frames[f].ravel() > 12) & (MASK.ravel() > 0)) for f in range(len(chunks))
    ]
    assert 0 < n.max() < NPIX // 10  # genuinely sparse, no top_pixels needed
    for f in range(len(chunks)):
        assert (i[f, : n[f]] == want[f]).all()
    assert (c.last_cut == 12).all() and (c.last_nabove == n).all()


def test_sparse_image_high_threshold_gives_no_pixels_but_the_powder():
    chunks, codec = chunks_of(frames_of("sparse", np.uint16))
    M = histogram_matrix()
    c = b.chunk2sparseCSC(MASK, M, np.uint16, codec, howmany=100)
    n_lo, _, p_lo = c.multi(chunks, 0)
    n_lo, p_lo = n_lo.copy(), p_lo.copy()  # the next call overwrites these
    n_hi, _, p_hi = c.multi(chunks, 65534)
    assert (n_lo > 0).all() and (n_hi == 0).all()
    assert (p_hi == p_lo).all() and p_hi.sum() > 0


def test_dense_image_low_threshold_is_capped_and_reported():
    chunks, codec = chunks_of(frames_of("dense", np.uint16))
    c = b.chunk2sparse(MASK, np.uint16, codec, howmany=500, thresholds=[1])
    n, (v, i) = c.multi(chunks, 0)
    assert (c.last_nabove > 500).all()  # the caller can see the frame was not sparse
    assert (n <= 500).all() and (n > 0).all()
    assert (c.last_cut == 1).all()  # no threshold left fewer than howmany: the last


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
        got = _toppix.top_pixels(wv, wi, nnz, howmany, [float(t) for t in thr])[0]
        assert n == got and (v[:n] == wv[:n]).all() and (row[:n] == wi[:n]).all(), trial


@pytest.mark.parametrize("with_matrix", [False, True])
def test_streaming_random_matches_oracle(with_matrix):
    """Frames of 15 blocks, random densities, cuts, howmany and thresholds: the in-loop selection of
    the C driver equals 'all pixels above the cut, then top_pixels'."""
    shape = (160, 400)
    npix = shape[0] * shape[1]
    mask = np.ones(shape, np.uint8)
    mask[::7, 11:19] = 0
    M = histogram_matrix(40, npix) if with_matrix else None
    rng = np.random.default_rng(11 + with_matrix)
    for trial in range(36):
        dtype = [np.uint8, np.uint16, np.uint32, np.int16, np.float32][trial % 5]
        top = 120 if np.dtype(dtype).itemsize == 1 else 3000
        frames = np.stack(
            [
                np.minimum(rng.poisson(lam, shape), top) + (rng.random(shape) > 0.995) * 40
                for lam in rng.choice([0.01, 0.3, 3.0, 20.0], 3)
            ]
        ).astype(dtype)
        chunks, codec = chunks_of(frames)
        cut = int(rng.choice([0, 1, 3, 10]))
        howmany = int(rng.choice([0, 1, 7, 100, 2000, 20000, 10**6]))
        kind = trial % 3
        thr = (
            None
            if kind == 0
            else sorted(int(x) for x in rng.integers(0, 60, int(rng.integers(0, 17))))
        )
        if kind == 2 and dtype == np.float32:
            thr = [t + 0.5 for t in thr]  # a float pixel with a non-integer threshold: numpy path
        mk = (
            (lambda **k: b.chunk2sparseCSC(mask, M, dtype, codec, **k))
            if with_matrix
            else (lambda **k: b.chunk2sparse(mask, dtype, codec, **k))
        )
        full, lim = mk(), mk(howmany=howmany, thresholds=thr)
        r0, r1 = full.multi(chunks, cut), lim.multi(chunks, cut)
        npx0, v0, i0 = r0[0].copy(), r0[1][0].copy(), r0[1][1].copy()
        npx, v, i = r1[0], r1[1][0], r1[1][1]
        thresholds = _toppix.default_thresholds(cut) if thr is None else [float(t) for t in thr]
        for f in range(len(frames)):
            wv, wi = oracle(v0[f], i0[f], int(npx0[f]), howmany, thresholds)
            assert npx[f] == len(wv), (trial, dtype, cut, howmany, thr, f, npx[f], len(wv))
            assert (v[f, : npx[f]] == wv).all() and (i[f, : npx[f]] == wi).all(), (trial, f)
            assert lim.last_nabove[f] == np.count_nonzero(v0[f, : npx0[f]] > lim.last_cut[f]), (
                trial,
                f,
            )
        if with_matrix:
            assert (r1[2] == r0[2]).all()


# ---- the back half alone: uncompressed frames (from_frames) give what the chunks give ----


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(96, 130), (160, 400), (7, 9), (64, 128)])
@pytest.mark.parametrize("with_matrix", [False, True])
def test_from_frames_equals_chunks(dtype, shape, with_matrix):
    npix = shape[0] * shape[1]
    mask = np.ones(shape, np.uint8)
    mask.ravel()[::13] = 0
    frames = np.stack(
        [
            _fit(frames_of(kind, dtype, n=1, seed=s)[0], shape)
            for kind, s in (("sparse", 1), ("dense", 2), ("sparse", 3))
        ]
    )
    chunks, codec = chunks_of(frames)
    M = histogram_matrix(30, npix) if with_matrix else None
    for kw in ({}, {"howmany": 40}, {"howmany": 500, "thresholds": [2, 5, 9]}):
        mk = (
            (lambda: b.chunk2sparseCSC(mask, M, dtype, codec, **kw))
            if with_matrix
            else (lambda: b.chunk2sparse(mask, dtype, codec, **kw))
        )
        a, c = mk(), mk()
        for cut in (0, 3):
            ra = a.multi(chunks, cut)
            na, va, ia = ra[0].copy(), ra[1][0].copy(), ra[1][1].copy()
            rc = c.from_frames(frames, cut)
            assert (rc[0] == na).all(), (kw, cut)
            for f in range(len(frames)):
                assert (rc[1][0][f, : na[f]] == va[f, : na[f]]).all() and (
                    rc[1][1][f, : na[f]] == ia[f, : na[f]]
                ).all()
            assert (c.last_cut == a.last_cut).all() and (c.last_nabove == a.last_nabove).all()
            if with_matrix:
                assert np.allclose(rc[2], ra[2], rtol=1e-6, atol=0)
            # flat 2D input and a non-contiguous input give the same
            rf = c.from_frames(frames.reshape(len(frames), -1)[:, ::1], cut)
            assert (rf[0] == na).all()


def _fit(img, shape):
    out = np.zeros(shape, img.dtype)
    h, w = min(shape[0], img.shape[0]), min(shape[1], img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


# ---- thresholds the C driver cannot compare exactly use numpy, which does ----


def _one(dtype, vals, **kw):
    fr = np.zeros((1, 16, 16), dtype)
    fr.ravel()[: len(vals)] = np.array(vals, dtype)
    chunks, codec = chunks_of(fr)
    c = b.chunk2sparse(np.ones((16, 16), np.uint8), dtype, codec, **kw)
    n, (v, i) = c.multi(chunks, 0)
    return int(n[0]), c


def test_float_threshold_not_exact_in_the_pixel_type():
    # 16777219 is not a float32: C would round it to 16777220 and keep no pixel
    n, c = _one(np.float32, [16777220.0] * 2, howmany=1, thresholds=[16777219.0])
    assert n == 1 and c.last_nabove[0] == 2


def test_threshold_above_two_to_the_62():
    n, c = _one(np.uint64, [2**62 + 5] * 5, howmany=4, thresholds=[1e30])
    assert n == 0 and c.last_nabove[0] == 0


def test_infinite_nan_and_huge_howmany():
    n, c = _one(np.uint16, [5] * 8, howmany=2, thresholds=[float("inf")])
    assert n == 0 and c.last_cut[0] == float("inf")
    n, c = _one(np.float32, [5] * 8, howmany=2, thresholds=[float("inf")])
    assert n == 0
    n, c = _one(np.uint16, [5] * 8, howmany=2, thresholds=[float("-inf")])
    assert n == 2 and c.last_cut[0] == float("-inf")
    with pytest.raises(ValueError, match="NaN"):
        b.chunk2sparse(
            np.ones((16, 16), np.uint8), np.uint16, b.CODEC_LZ4, thresholds=[float("nan")]
        )
    n, c = _one(np.uint16, [5] * 8, howmany=2**40)  # more than a frame holds: no limit
    assert n == 8
