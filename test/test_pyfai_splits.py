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
from bslz4_to_sparse import _pipeline, _matrix  # noqa: E402

ROUTES = ("dense", "sparse")
SPLITS = ("no", "bbox", "full")
BINS_PER_PIXEL = (1, 5)


def expected(integ, frames, ref):
    """(reference powder, relative tolerance) for this integrator's dot.  The
    fixed-point dots sum exact integers, so they are checked against numpy
    with the same rounded weights (the layout's data, in the order of the
    mask-folded matrix)."""
    lay = integ._layout
    if lay.scale is not None:
        nm = integ._nm
        Mq = sp.csc_matrix((lay.args()[0].astype(np.float64) * lay.scale, nm.indices, nm.indptr),
                           shape=nm.shape)
        return np.array([Mq @ f.ravel().astype(np.float64) for f in frames]), 1e-12
    return ref, 1e-9


def available_dots():
    """The dot names this CPU/build can run."""
    return [n for i, n in enumerate(_pipeline.NAMES["dot"])
            if i and _pipeline.available("dot", i) == 1]


def make_integ(mask, M, codec, dot, route=None):
    return b.chunk2sparseCSC(mask, M, dtype=np.uint16, codec=codec,
                             pipeline={"dot": dot, "route": route})


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
    for dot in available_dots():
        for route in ROUTES:
            try:
                integ = make_integ(mask, M, codec, dot, route)
            except ValueError:
                continue
            _npx, _sparse, p = integ.multi(chunks, 0)
            seen.add(dot)
            r, tol = expected(integ, frames * keep.reshape(frames.shape[1:]), ref)
            err = np.abs(p - r).max() / scale
            assert err < tol, (dot, route, split, bpp, err)
    assert "csc-run" in seen                     # 1D: every pixel is one run
    for dot in ("csc-int", "csc-run-int", "csc-run-int16"):
        assert dot in seen, dot
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
    for dot in available_dots():
        for route in ROUTES:
            try:
                integ = make_integ(mask, M, codec, dot, route)
            except ValueError:
                continue
            _npx, _sparse, p = integ.multi(chunks, 0)
            seen.add(dot)
            r, tol = expected(integ, frames * keep.reshape(frames.shape[1:]), ref)
            err = np.abs(p - r).max() / scale
            assert err < tol, (dot, route, split, order, err)
    assert ("csc-nosplit-moment" in seen) == (split == "no" and order == "inter")
    assert ("csc-run" in seen) == (order == "inter")
    assert "csc-nosplit" not in seen


def run_dots(mask, M, frames, chunks, codec, dots):
    """{dot: max relative error} on both routes for each dot that accepts M."""
    keep = mask.ravel().astype(np.float64)
    ref = np.array([M @ (f.ravel().astype(np.float64) * keep) for f in frames])
    scale = max(np.abs(ref).max(), 1.0)
    seen = {}
    for dot in dots:
        for route in ROUTES:
            try:
                integ = make_integ(mask, M, codec, dot, route)
            except ValueError:
                continue
            _npx, _sparse, p = integ.multi(chunks, 0)
            r, tol = expected(integ, frames * mask, ref)
            err = np.abs(p - r).max() / scale
            assert err < tol, (dot, route, err)
            seen[dot] = max(seen.get(dot, 0.0), err)
    return seen


@pytest.mark.parametrize("split", ["no", "bbox"])
def test_2d_matrix(setup, split):
    """2D (radial x azimuth, radial-major): every dot that accepts the matrix
    matches numpy on both routes; the histogram dots take only split='no'."""
    ai, mask, frames, chunks, codec = setup
    na, nr = 72, radial_pixels(ai)
    method = IntegrationMethod.select_one_available((split, "csc", "cython"), dim=2)
    ai.integrate2d(np.ones(ai.detector.shape, np.float32), nr, na, unit="q_nm^-1", method=method)
    e = ai.engines[method].engine
    M = sp.csc_matrix((np.asarray(e.data, np.float32), np.asarray(e.indices, np.int32),
                       np.asarray(e.indptr, np.int32)), shape=(nr * na, len(e.indptr) - 1))
    seen = run_dots(mask, M, frames, chunks, codec, available_dots())
    assert "csc" in seen and "csc-int" in seen and "bsb-csr" in seen
    assert ("csc-nosplit" in seen) == (split == "no")
    assert ("bsb-csr-nosplit" in seen) == (split == "no")


@pytest.mark.parametrize("dot", ["padded", "bsb-csr"])
def test_layout_built_for_another_block_size_is_refused(setup, dot):
    """padded and bsb-csr have per-block tables: a layout built for another
    block size must give -109, never a wrong powder."""
    ai, mask, frames, chunks, codec = setup
    M = pyfai_matrix(ai, "bbox", radial_pixels(ai))
    integ = make_integ(mask, M, codec, dot)
    integ.multi(chunks, 0)
    lay = integ._layout
    # the C entry gets arrays for 2048-pixel blocks; the Python side still
    # believes they are for this dataset's block size, so does not rebuild
    lay._args = _matrix.build(dot, integ._nm, lay.block_elems // 2, np.uint16).args()
    with pytest.raises(Exception, match="-109"):
        integ.multi(chunks, 0)


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
                    ("csc", "csc-nosplit", "csc-permute", "bsb-csr-nosplit"))
    assert "csc-permute" in seen and "csc-nosplit" in seen
    integ = make_integ(mask, M, codec, "csc-permute")
    _npx, _sparse, p = integ.multi(chunks, 0)
    assert p.dtype == np.uint16
    want = np.array([f.ravel()[pix[order]] for f in frames])
    assert (p == want).all()


@pytest.mark.parametrize("fast", ["az", "rad"])
def test_tiled_fazit_runs(setup, fast):
    """Approximate FAZIT: bins of ~4x4 pixels, stable sort, so row segments
    become runs of consecutive slots; csc-permute must equal numpy's
    reordering exactly."""
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
    for route in ROUTES:
        integ = make_integ(mask, M, codec, "csc-permute", route)
        _npx, _sparse, p = integ.multi(chunks, 0)
        assert (p == want).all(), route
