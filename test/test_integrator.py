"""Integrator: a pyFAI matrix and normalisation kept in a file, intensities recovered from the
exact int64 sums of the decoder."""

import numpy as np
import pytest

pyFAI = pytest.importorskip("pyFAI")
pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import conveyor as cv  # noqa: E402
from pyFAI.detectors import Detector  # noqa: E402
from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: E402
from pyFAI.method_registry import IntegrationMethod  # noqa: E402

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
    rng = np.random.default_rng(4)
    return rng.integers(0, 300, SHAPE).astype(np.uint16)


CORRECTIONS = {
    "none": dict(correctSolidAngle=False),
    "solid angle": dict(correctSolidAngle=True),
    "polarization": dict(correctSolidAngle=False, polarization_factor=0.9),
    "all": dict(correctSolidAngle=True, polarization_factor=0.95, normalization_factor=2.5),
}


def masked_ai(flat=False):
    """a detector with masked pixels (pyFAI: 1 = masked) and optionally a flat field"""
    ai = make_ai()
    rng = np.random.default_rng(6)
    ai.detector.mask = (rng.random(SHAPE) < 0.03).astype(np.int8)
    if flat:
        ai.detector.flatfield = 0.8 + 0.4 * rng.random(SHAPE)
    return ai


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("corrections", sorted(CORRECTIONS))
@pytest.mark.parametrize("split", ["no", "bbox", "full"])
def test_the_intensity_equals_pyfais(tmp_path, split, corrections, flat):
    """pyFAI's intensity, from the decoder's exact sums and the Integrator's normalisation: with
    pixel splitting, solid angle, polarization, a normalisation factor, a flat field and a mask."""
    kw = CORRECTIONS[corrections]
    ai = masked_ai(flat)
    integ = cv.Integrator.from_pyfai(ai, 60, unit="q_nm^-1", split=split, **kw)
    path = str(tmp_path / "integrator.h5")
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
    method = IntegrationMethod.select_one_available((split, "csc", "cython"), dim=1)
    ref = ai.integrate1d(fr, 60, unit="q_nm^-1", method=method, **kw)
    # the sums and the normalisation are pyFAI's own (float64 sums of float32 weights): measured
    # relative differences are below 5e-11 and 0.  Its intensity is stored as float32, whose
    # rounding is 6e-8 (measured 5.7e-8 at most), the only reason the intensity differs.
    ss = np.asarray(ref.sum_signal)
    assert np.allclose(powder[0] * (1.0 / d.powder_scale), ss, rtol=1e-9, atol=0)
    assert np.array_equal(back.norm, np.asarray(ref.sum_normalization).ravel())
    assert ref.intensity.dtype == np.float32
    assert np.allclose(got, ref.intensity, rtol=1.2e-7, atol=0)
    # the float powder and the integer one give the same intensity
    pf = b.chunk2sparseCSC(mask, back.csc, np.uint16).from_frames(fr[None], 0)[2]
    assert np.allclose(back.intensity(pf)[0], got, rtol=1e-9, atol=0)


def test_2d_rows_are_radial_then_azimuth(tmp_path):
    ai = make_ai()
    integ = cv.Integrator.from_pyfai(ai, 20, unit="q_nm^-1", npt_azim=8)
    assert integ.kind == "powder2d" and integ.nbins == 20 * 8 and len(integ.azimuthal) == 8
    fr = frame()
    ones = np.ones(SHAPE, np.uint8)
    d = b.chunk2sparseCSC(ones, b.fixed_point_matrix(integ.csc, ones, np.uint16), np.uint16)
    powder = d.from_frames(fr[None], 0)[2]
    got = integ.intensity(powder, d.powder_scale)[0].reshape(20, 8).T
    method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=2)
    ref = ai.integrate2d(fr, 20, 8, unit="q_nm^-1", method=method)
    assert np.allclose(got, ref.intensity, rtol=1.2e-7, atol=0)


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
def test_rings_equal_the_summed_normalised_cake(tmp_path, background):
    """What a texture pipeline computes frame by frame with pyFAI (peak windows of the cake, per
    azimuth), from the decoder's matrix product and an Integrator.rings, saved and loaded."""
    nrad, naz = 70, 12
    kw = dict(polarization_factor=0.95, radial_range=(0.3, 3.0))
    ai = masked_ai()
    integ = cv.Integrator.from_pyfai(ai, nrad, unit="2th_deg", npt_azim=naz, **kw)
    ranges = [(0.8, 1.0), (1.3, 1.45), (2.1, 2.5)]
    bg = np.linspace(0.5, 1.5, nrad) if background else np.zeros(nrad)
    rings = integ.rings(ranges, bg if background else None)
    rings.save(str(tmp_path / "rings.h5"))
    rings = cv.Integrator.load(str(tmp_path / "rings.h5"))
    assert rings.kind == "rings" and rings.bins_shape == (naz, 3) and rings.nbins == naz * 3

    fr = frame()
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
    method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
    ref = ai.integrate1d(fr, 50, unit="q_nm^-1", method=method)
    refq = ai.integrate1d(fr * q, 50, unit="q_nm^-1", method=method)
    assert np.allclose(p[0, :50], ref.sum_signal, rtol=1e-9, atol=0)  # measured: identical
    # pyFAI's q-weighted image is float32: measured differences are 3.3e-8 at most
    assert np.allclose(p[0, 50:], refq.sum_signal, rtol=1e-7, atol=0)
    got = mom.mean_q(p)[0]
    ok = ref.sum_signal > 0
    assert np.allclose(got[ok], refq.sum_signal[ok] / ref.sum_signal[ok], rtol=1e-7, atol=0)
    assert np.allclose(mom.intensity(p)[0], ref.intensity, rtol=1.2e-7, atol=0)


def test_one_pass_for_everything(tmp_path):
    """A 1D powder, ring windows and the first moment, as one stacked matrix and one product: each
    block equals what it gives alone, through a save and a load of the stack."""
    ai = masked_ai()
    kw = dict(polarization_factor=0.95)
    one = cv.Integrator.from_pyfai(ai, 50, unit="q_nm^-1", **kw)
    cake = cv.Integrator.from_pyfai(ai, 40, unit="q_nm^-1", npt_azim=8, radial_range=(5, 40), **kw)
    rng = cake.radial[[5, 20]]
    rings = cake.rings([(rng[0] - 0.4, rng[0] + 0.4), (rng[1] - 0.4, rng[1] + 0.4)])
    stack = cv.IntegratorStack({"powder1d": one, "rings": rings, "moment": one.first_moment()})
    path = str(tmp_path / "stack.h5")
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
    method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
    ref = ai.integrate1d(fr, 50, unit="q_nm^-1", method=method, **kw)
    assert np.allclose(out["powder1d"][0], ref.intensity, rtol=1.2e-7, atol=0)
