"""Integrator: a pyFAI matrix and normalisation kept in a file, intensities recovered from the
exact int64 sums of the decoder."""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pytest

pyFAI = pytest.importorskip("pyFAI")
pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import conveyor as cv  # noqa: E402
from pyFAI.detectors import Detector  # noqa: E402
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # pyFAI >= 2024.10
except ImportError:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # older pyFAI
from bslz4_to_sparse.conveyor._integrator import sparse_method  # noqa: E402

SHAPE = (96, 112)


def make_ai():
    return AzimuthalIntegrator(
        dist=0.1,
        poni1=0.004,
        poni2=0.0045,
        wavelength=3e-11,
        detector=Detector(75e-6, 75e-6, max_shape=SHAPE),
    )


def frame():
    rng = np.random.RandomState(4)
    return rng.randint(0, 300, SHAPE).astype(np.uint16)


CORRECTIONS = {
    "none": dict(correctSolidAngle=False),
    "solid angle": dict(correctSolidAngle=True),
    "polarization": dict(correctSolidAngle=False, polarization_factor=0.9),
    "all": dict(correctSolidAngle=True, polarization_factor=0.95, normalization_factor=2.5),
}


def reference(ai, img, npt, npt_azim=None, split="bbox", unit="q_nm^-1", **kw):
    """pyFAI's (sum_signal, intensity, result) for img, with the method from_pyfai uses.  A pyFAI
    that gives no sum_normalization (0.17) defines its sum_signal and intensity with corrections
    otherwise (the sum and the mean of value / correction): from it, sum_signal is its sum with no
    corrections, and the intensity None unless there are none."""
    method = sparse_method(split, 1 if npt_azim is None else 2)

    def run(**k):
        if npt_azim is None:
            return ai.integrate1d(img, npt, unit=unit, method=method, **k)
        return ai.integrate2d(img, npt, npt_azim, unit=unit, method=method, **k)

    res = run(**kw)
    if getattr(res, "sum_normalization", None) is not None:
        return np.asarray(res.sum_signal), res.intensity, res
    off = dict(correctSolidAngle=False, polarization_factor=None, normalization_factor=1.0)
    plain = run(flat=np.ones(img.shape, np.float32), **dict(kw, **off))
    corrected = (
        kw.get("correctSolidAngle", True)
        or kw.get("polarization_factor") is not None
        or kw.get("normalization_factor", 1.0) != 1.0
        or kw.get("flat") is not None
        or ai.detector.flatfield is not None
    )
    return np.asarray(plain.sum, np.float64), None if corrected else res.intensity, res


def masked_ai(flat=False):
    """a detector with masked pixels (pyFAI: 1 = masked) and optionally a flat field"""
    ai = make_ai()
    rng = np.random.RandomState(6)
    ai.detector.mask = (rng.random_sample(SHAPE) < 0.03).astype(np.int8)
    if flat:
        ai.detector.flatfield = 0.8 + 0.4 * rng.random_sample(SHAPE)
    return ai


@pytest.mark.parametrize("name", ["dummy", "delta_dummy"])
def test_dummy_and_delta_dummy_are_refused(name):
    """pyFAI masks pixels equal to dummy (within delta_dummy) frame by frame; a fixed matrix cannot."""
    with pytest.raises(ValueError, match=name + " is not supported"):
        cv.Integrator.from_pyfai(make_ai(), 60, unit="q_nm^-1", **{name: 1.0})


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("corrections", sorted(CORRECTIONS))
@pytest.mark.parametrize("split", ["no", "bbox", "full"])
def test_the_intensity_equals_pyfais(tmpdir, split, corrections, flat):
    """pyFAI's intensity, from the decoder's exact sums and the Integrator's normalisation: with
    pixel splitting, solid angle, polarization, a normalisation factor, a flat field and a mask."""
    kw = CORRECTIONS[corrections]
    ai = masked_ai(flat)
    integ = cv.Integrator.from_pyfai(ai, 60, unit="q_nm^-1", split=split, **kw)
    path = os.path.join(str(tmpdir), "integrator.h5")
    integ.save(path)
    back = cv.Integrator.load(path)
    assert back.unit == "q_nm^-1" and back.kind == "powder1d" and back.shape == SHAPE
    assert (back.csc != integ.csc).nnz == 0 and np.array_equal(back.norm, integ.norm)
    assert np.allclose(back.radial, integ.radial) and "Distance: 0.1" in back.poni
    assert abs(back.ai().dist - 0.1) < 1e-12

    fr = frame()
    mask = (1 - ai.detector.mask).astype(np.uint8)  # the decoder's: 1 = use the pixel
    d = b.chunk2sparseCSC(mask, b.fixed_point_matrix(back.csc, mask, np.uint16), np.uint16)
    _, _, powder = d.from_frames(fr[None], 0)
    assert powder.dtype == np.int64
    got = back.intensity(powder, d.powder_scale)[0]
    ss, intensity, ref = reference(ai, fr, 60, split=split, **kw)
    # the sums and the normalisation are pyFAI's own (float64 sums of float32 weights): measured
    # relative differences are below 5e-11 and 0.  Its intensity is stored as float32, whose
    # rounding is 6e-8 (measured 5.7e-8 at most), the only reason the intensity differs.
    assert np.allclose(powder[0] * (1.0 / d.powder_scale), ss, rtol=1e-9, atol=0)
    if getattr(ref, "sum_normalization", None) is not None:  # else from_pyfai checked its own
        assert np.array_equal(back.norm, np.asarray(ref.sum_normalization).ravel())
    if intensity is not None:
        assert intensity.dtype == np.float32
        assert np.allclose(got, intensity, rtol=1.2e-7, atol=0)
    # the float powder and the integer one give the same intensity
    pf = b.chunk2sparseCSC(mask, back.csc, np.uint16).from_frames(fr[None], 0)[2]
    assert np.allclose(back.intensity(pf)[0], got, rtol=1e-9, atol=0)


def test_2d_rows_are_radial_then_azimuth(tmpdir):
    ai = make_ai()
    integ = cv.Integrator.from_pyfai(ai, 20, unit="q_nm^-1", npt_azim=8)
    assert integ.kind == "powder2d" and integ.nbins == 20 * 8 and len(integ.azimuthal) == 8
    fr = frame()
    ones = np.ones(SHAPE, np.uint8)
    d = b.chunk2sparseCSC(ones, b.fixed_point_matrix(integ.csc, ones, np.uint16), np.uint16)
    powder = d.from_frames(fr[None], 0)[2]
    got = integ.intensity(powder, d.powder_scale)[0].reshape(20, 8).T
    ss, intensity, _ = reference(ai, fr, 20, npt_azim=8)
    sums = (powder[0] * (1.0 / d.powder_scale)).reshape(20, 8).T
    assert np.allclose(sums, ss, rtol=1e-9, atol=0)  # pyFAI's (azimuth, radial)
    if intensity is not None:
        assert np.allclose(got, intensity, rtol=1.2e-7, atol=0)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("split", ["no", "bbox", "full"])
def test_sparse_method_is_exactly_the_one_asked_for(split, dim):
    """pyFAI's own selection falls back to another splitting or algorithm without saying so."""
    make_ai()  # fills pyFAI's registry
    try:
        m = sparse_method(split, dim)
    except ValueError as e:  # pyFAI 0.18 has no full-split CSR in 2D
        assert repr(split) in str(e)
        return
    assert (m.dimension, m.split_lower, m.algo_lower, m.impl_lower) in (
        (dim, split, "csc", "cython"),
        (dim, split, "csr", "cython"),
    )


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("split", ["bbox", "full"])
def test_the_csr_route_equals_the_csc_one(monkeypatch, split, dim):
    """The CSR engine (pyFAI 0.17 and 0.18 have no CSC), converted, is the CSC one."""
    from bslz4_to_sparse.conveyor import _integrator

    make_ai()
    csc_method = sparse_method(split, dim)
    if csc_method.algo_lower != "csc":
        pytest.skip("this pyFAI has no CSC integrator to compare with")
    kw = dict(npt_azim=None if dim == 1 else 8, split=split, polarization_factor=0.9)
    want = cv.Integrator.from_pyfai(masked_ai(True), 20, **kw)
    from pyFAI.method_registry import IntegrationMethod

    csr = [m for m in IntegrationMethod.select_method(dim, split, "csr", "cython")
           if (m.split_lower, m.algo_lower) == (split, "csr")]
    monkeypatch.setattr(_integrator, "sparse_method", lambda split, dim: csr[0])
    got = cv.Integrator.from_pyfai(masked_ai(True), 20, **kw)
    assert (got.csc != want.csc).nnz == 0
    # pyFAI sums sum_normalization with each engine: measured 1.0e-8 apart at most
    assert np.allclose(got.norm, want.norm, rtol=1e-7, atol=0)


def test_a_matrix_of_another_splitting_is_never_reused():
    """pyFAI 0.17 reuses its one CSR engine whatever the splitting asked for (its intensity is then
    that of the other splitting): the matrix here is always built for the one asked for."""
    ai = make_ai()
    no = cv.Integrator.from_pyfai(ai, 30, correctSolidAngle=False, split="no")
    bbox = cv.Integrator.from_pyfai(ai, 30, correctSolidAngle=False, split="bbox")
    fresh = cv.Integrator.from_pyfai(make_ai(), 30, correctSolidAngle=False, split="bbox")
    assert (bbox.csc != fresh.csc).nnz == 0 and bbox.csc.nnz > no.csc.nnz


def test_2d_full_splitting_works_or_stops():
    """pyFAI 0.17 raises NotImplementedError, 0.18 builds no engine (ValueError here); newer ones
    give the matrix, whose sums are pyFAI's."""
    ai = make_ai()
    try:
        integ = cv.Integrator.from_pyfai(ai, 20, npt_azim=8, split="full")
    except (NotImplementedError, ValueError) as e:
        assert "full" in str(e).lower()
        return
    fr = frame().astype(np.float64)
    ss = reference(ai, fr, 20, npt_azim=8, split="full")[0]
    assert np.allclose(integ.csc.dot(fr.ravel()).reshape(20, 8).T, ss, rtol=1e-6, atol=0)


def textom_process_one_frame(ai, image, nrad, naz, ranges, bg, **kw):
    """The sum over each peak window of the normalised cake, per azimuth, as the texture notebook's
    process_one_frame computes it from pyFAI's integrate2d (a cake minus a background per radial
    bin, summed over the bins of each window)."""
    res = ai.integrate2d(image, nrad, naz, unit="2th_deg", **kw)
    cake = res.intensity - bg[np.newaxis, :]
    return np.stack(
        [cake[:, (res.radial >= lo) & (res.radial < hi)].sum(axis=-1) for lo, hi in ranges], axis=-1
    )


@pytest.mark.parametrize("background", [False, True])
def test_rings_equal_the_summed_normalised_cake(tmpdir, background):
    """What a texture pipeline computes frame by frame with pyFAI (peak windows of the cake, per
    azimuth), from the decoder's matrix product and an Integrator.rings, saved and loaded."""
    nrad, naz = 70, 12
    kw = dict(polarization_factor=0.95, radial_range=(0.3, 3.0))
    ai = masked_ai()
    integ = cv.Integrator.from_pyfai(ai, nrad, unit="2th_deg", npt_azim=naz, **kw)
    ranges = [(0.8, 1.0), (1.3, 1.45), (2.1, 2.5)]
    bg = np.linspace(0.5, 1.5, nrad) if background else np.zeros(nrad)
    rings = integ.rings(ranges, bg if background else None)
    rings.save(os.path.join(str(tmpdir), "rings.h5"))
    rings = cv.Integrator.load(os.path.join(str(tmpdir), "rings.h5"))
    assert rings.kind == "rings" and rings.bins_shape == (naz, 3) and rings.nbins == naz * 3

    fr = frame()
    if reference(ai, fr, nrad, npt_azim=naz, unit="2th_deg", **kw)[1] is None:
        pytest.skip("this pyFAI's intensity with corrections is not sum_signal / sum_normalization")
    mask = (1 - ai.detector.mask).astype(np.uint8)
    d = b.chunk2sparseCSC(mask, rings.csc, np.uint16)  # float: the weights are 1/norm
    powder = d.from_frames(fr[None], 0)[2]
    got = rings.intensity(powder)[0].reshape(naz, 3)
    ref = textom_process_one_frame(ai, fr, nrad, naz, ranges, bg, **kw)
    # measured: 4.5e-8 at most, the float32 rounding of pyFAI's cake (zero bins agree exactly)
    assert np.allclose(got, ref, rtol=1.2e-7, atol=0)


def test_first_moment_is_the_mean_q_of_the_signal():
    """The stacked q-weighted matrix gives pyFAI's sum_signal of the q-weighted image, and its ratio
    to the plain sum is the mean q of each bin."""
    ai = masked_ai()
    integ = cv.Integrator.from_pyfai(ai, 50, unit="q_nm^-1", split="bbox")
    mom = integ.first_moment()
    assert mom.kind == "moment" and mom.nbins == 100 and mom.bins_shape == (2, 50)
    fr = frame()
    mask = (1 - ai.detector.mask).astype(np.uint8)
    p = b.chunk2sparseCSC(mask, mom.csc, np.uint16).from_frames(fr[None], 0)[2]
    q = ai.array_from_unit(shape=SHAPE, typ="center", unit="q_nm^-1")
    ss, intensity, _ = reference(ai, fr, 50)
    ssq = reference(ai, fr * q, 50)[0]
    assert np.allclose(p[0, :50], ss, rtol=1e-9, atol=0)  # measured: identical
    # pyFAI's q-weighted image is float32: measured differences are 3.3e-8 at most
    assert np.allclose(p[0, 50:], ssq, rtol=1e-7, atol=0)
    got = mom.mean_q(p)[0]
    ok = ss > 0
    assert np.allclose(got[ok], ssq[ok] / ss[ok], rtol=1e-7, atol=0)
    if intensity is not None:
        assert np.allclose(mom.intensity(p)[0], intensity, rtol=1.2e-7, atol=0)


def test_one_pass_for_everything(tmpdir):
    """A 1D powder, ring windows and the first moment, as one stacked matrix and one product: each
    block equals what it gives alone, through a save and a load of the stack."""
    ai = masked_ai()
    kw = dict(polarization_factor=0.95)
    one = cv.Integrator.from_pyfai(ai, 50, unit="q_nm^-1", **kw)
    cake = cv.Integrator.from_pyfai(ai, 40, unit="q_nm^-1", npt_azim=8, radial_range=(5, 40), **kw)
    rng = cake.radial[[5, 20]]
    rings = cake.rings([(rng[0] - 0.4, rng[0] + 0.4), (rng[1] - 0.4, rng[1] + 0.4)])
    stack = cv.IntegratorStack(
        [("powder1d", one), ("rings", rings), ("moment", one.first_moment())]
    )
    path = os.path.join(str(tmpdir), "stack.h5")
    stack.save(path)
    stack = cv.IntegratorStack.load(path)
    assert stack.nbins == 50 + 16 + 100 and list(stack.slices) == ["powder1d", "rings", "moment"]
    fr = frame()
    mask = (1 - ai.detector.mask).astype(np.uint8)
    d = b.chunk2sparseCSC(mask, stack.csc, np.uint16)
    powder = d.from_frames(fr[None], 0)[2]
    out = stack.split(powder)
    alone = {
        k: b.chunk2sparseCSC(mask, blk.csc, np.uint16).from_frames(fr[None], 0)[2]
        for k, blk in stack.blocks.items()
    }
    for k, blk in stack.blocks.items():
        assert np.allclose(out[k], blk.intensity(alone[k]), rtol=1e-12, atol=0), k  # same sums
    ss, intensity, _ = reference(ai, fr, 50, **kw)
    assert np.allclose(alone["powder1d"][0], ss, rtol=1e-9, atol=0)
    if intensity is not None:
        assert np.allclose(out["powder1d"][0], intensity, rtol=1.2e-7, atol=0)
