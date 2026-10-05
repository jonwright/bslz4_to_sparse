"""Every matrix-dot kernel on real pyFAI matrices.

pyFAI's CSC engine for integrate1d is built with each pixel-splitting
scheme -- "no" (a histogram: one bin per pixel, weight 1), "bbox" and
"full" (pixel area split over neighbouring bins) -- at two bin widths:
about one bin per pixel (radial bins as wide as a pixel) and about five
(each pixel split over several bins).  Every dot that accepts the matrix
must match numpy's M @ (img * mask) on both routes, and the per-pixel dots
must accept what they are meant for: csc-nosplit the histogram, csc-run
every split (each pixel reaches one run of consecutive bins).
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
pyFAI = pytest.importorskip("pyFAI")
from pyFAI.method_registry import IntegrationMethod  # noqa: E402

import bslz4_to_sparse as b  # noqa: E402

ROUTES = (("dense", 1e9), ("sparse", 0.0))
SPLITS = ("no", "bbox", "full")
BINS_PER_PIXEL = (1, 5)


def expected(integ, frames, ref):
    """(reference powder, relative tolerance) for this integrator's dot.  The
    fixed-point dots sum exact integers, so they are checked against numpy
    with the same rounded weights; csc-run-moment multiplies by q itself
    instead of using the stored float32 w*q, so it agrees to ~1e-7."""
    v = getattr(integ, "variant", None)
    if v is not None and v.scale is not None:
        nm = integ._nm
        Mq = sp.csc_matrix((v.data.astype(np.float64) * v.scale, nm.indices, nm.indptr),
                           shape=nm.shape)
        return np.array([Mq @ f.ravel().astype(np.float64) for f in frames]), 1e-12
    if integ.dot == "csc-run-moment":
        return ref, 1e-6
    return ref, 1e-9


def make_ai():
    det = pyFAI.detector_factory("Eiger_500k")          # 514 x 1030, module gap masked
    ai = pyFAI.load({"detector": det.name, "dist": 0.1, "poni1": 0.02, "poni2": 0.035,
                     "rot1": 0.01, "rot2": 0.02, "rot3": 0.03, "wavelength": 2.85e-11})
    return ai


def radial_pixels(ai):
    """Radial extent of the detector in pixels: npt for ~1 bin per pixel."""
    r = ai.rArray() / ai.detector.pixel1
    return int(np.ceil(r.max() - r.min()))


def pyfai_matrix(ai, split, npt):
    method = IntegrationMethod.select_one_available((split, "csc", "cython"), dim=1)
    ai.integrate1d(np.ones(ai.detector.shape, np.float32), npt, unit="r_mm", method=method)
    e = ai.engines[method].engine
    npix = len(e.indptr) - 1
    return sp.csc_matrix((np.asarray(e.data, np.float32), np.asarray(e.indices, np.int32),
                          np.asarray(e.indptr, np.int32)), shape=(npt, npix))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    ai = make_ai()
    shape = ai.detector.shape
    mask = (1 - ai.detector.mask).astype(np.uint8)
    rng = np.random.default_rng(11)
    frames = rng.poisson(0.5, (3,) + shape).astype(np.uint16)
    path = tmp_path_factory.mktemp("pyfai") / "f.h5"
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + shape,
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        codec = b.detect_codec(ds)
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]
    return ai, mask, frames, chunks, codec


@pytest.mark.parametrize("bpp", BINS_PER_PIXEL)
@pytest.mark.parametrize("split", SPLITS)
def test_every_dot_on_pyfai_matrix(setup, split, bpp):
    ai, mask, frames, chunks, codec = setup
    M = pyfai_matrix(ai, split, radial_pixels(ai) * bpp)
    keep = mask.ravel().astype(np.float64)
    ref = np.array([M @ (f.ravel().astype(np.float64) * keep) for f in frames])
    scale = max(np.abs(ref).max(), 1.0)
    seen = set()
    saved = b.get_dense_sparse_threshold()
    try:
        for dot in b.available_dots():
            for route, thr in ROUTES:
                b.set_dense_sparse_threshold(thr)
                try:
                    integ = b.chunk2sparseCSCmulti(mask, M, dtype=np.uint16, codec=codec, dot=dot)
                except ValueError:
                    continue
                _npx, _sparse, p = integ(chunks, 0)
                seen.add(dot)
                r, tol = expected(integ, frames * keep.reshape(frames.shape[1:]), ref)
                err = np.abs(p - r).max() / scale
                assert err < tol, (dot, route, split, bpp, err)
    finally:
        b.set_dense_sparse_threshold(saved)
    assert "csc-run" in seen
    for dot in ("csc-run-u16", "csc-run-delta", "csc-int", "csc-run-int", "csc-run-int16"):
        assert dot in seen, dot
    assert "csc-tile" not in seen                # 1D: every pixel is one run
    for dot in ("csc-nosplit-dump", "csc-nosplit-u16", "csc-nosplit-delta", "csc-nosplit-walk"):
        assert (dot in seen) == (split == "no"), dot
    assert ("csc-nosplit" in seen) == (split == "no")
    assert ("bsb-csr-nosplit" in seen) == (split == "no")


def ring_matrices(ai, split):
    """Rings summed from pyFAI's 2D matrix: G (4 rings x NA azimuth) and the
    first moment beside it, interleaved [I, qI, ...] and blocked [I.., qI..]."""
    na, nr = 72, radial_pixels(ai)
    method = IntegrationMethod.select_one_available((split, "csc", "cython"), dim=2)
    res = ai.integrate2d(np.ones(ai.detector.shape, np.float32), nr, na, unit="q_nm^-1",
                         method=method)
    e = ai.engines[method].engine
    M2 = sp.csc_matrix((np.asarray(e.data, np.float64), np.asarray(e.indices),
                        np.asarray(e.indptr)), shape=(nr * na, len(e.indptr) - 1))
    q = res.radial
    edges = np.quantile(q, [0.2, 0.25, 0.4, 0.45, 0.6, 0.65, 0.8, 0.85]).reshape(4, 2)
    W = np.array([(q >= lo) & (q < hi) for lo, hi in edges], dtype=np.float64)
    G = (sp.kron(sp.csr_matrix(W), sp.identity(na), format="csr") @ M2).tocsc()
    GQ = G @ sp.diags(ai.qArray().ravel().astype(np.float64))
    n = G.shape[0]
    perm = np.empty(2 * n, np.int64)
    perm[0::2] = np.arange(n)
    perm[1::2] = n + np.arange(n)
    both = sp.vstack([G, GQ]).tocsr()

    def f32(M):
        M = M.tocsc()
        M.sort_indices()
        return sp.csc_matrix((M.data.astype(np.float32), M.indices.astype(np.int32),
                              M.indptr.astype(np.int32)), shape=M.shape)
    return {"inter": f32(both[perm]), "block": f32(both)}


@pytest.mark.parametrize("order", ["inter", "block"])
@pytest.mark.parametrize("split", ["no", "bbox"])
def test_ring_first_moment(setup, split, order):
    """sum(I) and sum(q I) per (ring, azimuth): plain extra matrix rows, so
    every dot that accepts the matrix must match numpy.  csc-nosplit-moment
    takes exactly the no-split interleaved case; csc-run every interleaved one
    (a pixel's entries stay one run of consecutive bins)."""
    ai, mask, frames, chunks, codec = setup
    M = ring_matrices(ai, split)[order]
    keep = mask.ravel().astype(np.float64)
    ref = np.array([M @ (f.ravel().astype(np.float64) * keep) for f in frames])
    scale = max(np.abs(ref).max(), 1.0)
    seen = set()
    saved = b.get_dense_sparse_threshold()
    try:
        for dot in b.available_dots():
            for route, thr in ROUTES:
                b.set_dense_sparse_threshold(thr)
                try:
                    integ = b.chunk2sparseCSCmulti(mask, M, dtype=np.uint16, codec=codec, dot=dot)
                except ValueError:
                    continue
                _npx, _sparse, p = integ(chunks, 0)
                seen.add(dot)
                r, tol = expected(integ, frames * keep.reshape(frames.shape[1:]), ref)
                err = np.abs(p - r).max() / scale
                assert err < tol, (dot, route, split, order, err)
    finally:
        b.set_dense_sparse_threshold(saved)
    assert ("csc-nosplit-moment" in seen) == (split == "no" and order == "inter")
    assert ("csc-nosplit-moment-dump" in seen) == (split == "no" and order == "inter")
    assert ("csc-run-moment" in seen) == (order == "inter")
    assert ("csc-run" in seen) == (order == "inter")
    assert "csc-nosplit" not in seen


def run_dots(mask, M, frames, chunks, codec, dots):
    """{dot: max relative error} on both routes for each dot that accepts M."""
    keep = mask.ravel().astype(np.float64)
    ref = np.array([M @ (f.ravel().astype(np.float64) * keep) for f in frames])
    scale = max(np.abs(ref).max(), 1.0)
    seen = {}
    saved = b.get_dense_sparse_threshold()
    try:
        for dot in dots:
            for route, thr in ROUTES:
                b.set_dense_sparse_threshold(thr)
                try:
                    integ = b.chunk2sparseCSCmulti(mask, M, dtype=np.uint16, codec=codec, dot=dot)
                except ValueError:
                    continue
                _npx, _sparse, p = integ(chunks, 0)
                r, tol = expected(integ, frames * mask, ref)
                err = np.abs(p - r).max() / scale
                assert err < tol, (dot, route, err)
                seen[dot] = max(seen.get(dot, 0.0), err)
    finally:
        b.set_dense_sparse_threshold(saved)
    return seen


@pytest.mark.parametrize("split", ["no", "bbox"])
def test_2d_tile_and_walk(setup, split):
    """2D (radial x azimuth, radial-major): the walk stream takes the
    histogram, the tile takes the split (each pixel one rectangle, azimuth
    wrapping), and both match numpy on both routes."""
    ai, mask, frames, chunks, codec = setup
    na, nr = 72, radial_pixels(ai)
    method = IntegrationMethod.select_one_available((split, "csc", "cython"), dim=2)
    ai.integrate2d(np.ones(ai.detector.shape, np.float32), nr, na, unit="q_nm^-1", method=method)
    e = ai.engines[method].engine
    M = sp.csc_matrix((np.asarray(e.data, np.float32), np.asarray(e.indices, np.int32),
                       np.asarray(e.indptr, np.int32)), shape=(nr * na, len(e.indptr) - 1))
    seen = run_dots(mask, M, frames, chunks, codec,
                    ("csc", "csc-tile", "csc-nosplit-walk", "csc-nosplit-delta", "csc-int"))
    assert "csc-int" in seen
    assert ("csc-nosplit-walk" in seen) == (split == "no")
    assert ("csc-tile" in seen) == (split == "bbox")


def test_stream_built_for_another_block_size_is_refused(setup):
    """The delta stream has a per-block table: a stream whose header names
    another block size must give -109, never a wrong powder."""
    ai, mask, frames, chunks, codec = setup
    M = pyfai_matrix(ai, "bbox", radial_pixels(ai))
    integ = b.chunk2sparseCSCmulti(mask, M, dtype=np.uint16, codec=codec, dot="csc-run-delta")
    integ(chunks, 0)
    h = integ.variant.indices.view(np.uint32)
    h[2] += 1
    integ._rebuild_layout = lambda be: None     # keep the corrupted stream
    with pytest.raises(Exception, match="-109"):
        integ(chunks, 0)


def test_fazit_permutation(setup):
    """FAZIT (P. Boesecke's fast radial regrouping, as in ImageD11): the
    unmasked pixels sorted by (radial bin, azimuth), each to its own slot.
    csc-permute writes the image in that order in the pixel dtype; it must
    equal numpy's reordering exactly."""
    ai, mask, frames, chunks, codec = setup
    valid = (mask > 0).ravel()
    rbin = np.floor(ai.rArray().ravel() / ai.detector.pixel1).astype(np.int64)
    chi = ai.chiArray().ravel()
    pix = np.flatnonzero(valid)
    order = np.lexsort((chi[pix], rbin[pix]))
    adr = np.empty(pix.size, np.int64)
    adr[order] = np.arange(pix.size)
    indptr = np.concatenate(([0], np.cumsum(valid))).astype(np.int32)
    M = sp.csc_matrix((np.ones(pix.size, np.float32), adr.astype(np.int32), indptr),
                      shape=(pix.size, valid.size))
    seen = run_dots(mask, M, frames, chunks, codec,
                    ("csc", "csc-nosplit", "csc-permute", "csc-nosplit-delta", "bsb-csr-nosplit"))
    assert "csc-permute" in seen and "csc-nosplit" in seen
    integ = b.chunk2sparseCSCmulti(mask, M, dtype=np.uint16, codec=codec, dot="csc-permute")
    _npx, _sparse, p = integ(chunks, 0)
    assert p.dtype == np.uint16
    want = np.array([f.ravel()[pix[order]] for f in frames])
    assert (p == want).all()


@pytest.mark.parametrize("fast", ["az", "rad"])
def test_tiled_fazit_runs(setup, fast):
    """Approximate FAZIT: bins of ~4x4 pixels, stable sort, so row segments
    become runs of consecutive slots; csc-permute-runs copies whole runs and
    must equal numpy's reordering exactly, as must csc-permute."""
    ai, mask, frames, chunks, codec = setup
    valid = (mask > 0).ravel()
    r = ai.rArray().ravel() / ai.detector.pixel1
    chi = ai.chiArray().ravel()
    pix = np.flatnonzero(valid)
    rbin = np.floor(r[pix] / 4).astype(np.int64)
    if fast == "az":
        abin = np.floor((chi[pix] + np.pi) * (rbin + 0.5) * 4 / 4).astype(np.int64)
        order = np.lexsort((abin, rbin))
    else:
        sect = np.floor((chi[pix] + np.pi) / (2 * np.pi) * 90).astype(np.int64)
        order = np.lexsort((rbin, sect))
    adr = np.empty(pix.size, np.int64)
    adr[order] = np.arange(pix.size)
    indptr = np.concatenate(([0], np.cumsum(valid))).astype(np.int32)
    M = sp.csc_matrix((np.ones(pix.size, np.float32), adr.astype(np.int32), indptr),
                      shape=(pix.size, valid.size))
    want = np.array([f.ravel()[pix[order]] for f in frames])
    saved = b.get_dense_sparse_threshold()
    try:
        for dot in ("csc-permute", "csc-permute-runs"):
            for route, thr in ROUTES:
                b.set_dense_sparse_threshold(thr)
                integ = b.chunk2sparseCSCmulti(mask, M, dtype=np.uint16, codec=codec, dot=dot)
                _npx, _sparse, p = integ(chunks, 0)
                assert (p == want).all(), (dot, route)
    finally:
        b.set_dense_sparse_threshold(saved)
