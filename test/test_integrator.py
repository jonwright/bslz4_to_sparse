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
    d = b.chunk2sparseCSC(mask, back.csc, np.uint16, exact=True)
    _, _, powder = d.from_frames(fr[None], 0)
    assert powder.dtype == np.int64
    got = back.intensity(powder, d.powder_scale)[0]
    method = IntegrationMethod.select_one_available((split, "csc", "cython"), dim=1)
    ref = ai.integrate1d(fr, 60, unit="q_nm^-1", method=method, **kw)
    assert np.allclose(got, ref.intensity, rtol=1e-4, atol=1e-4)
    # and the float powder gives the same intensity (the fixed-point sums lose nothing visible)
    pf = b.chunk2sparseCSC(mask, back.csc, np.uint16).from_frames(fr[None], 0)[2]
    assert np.allclose(back.intensity(pf)[0], got, rtol=1e-5, atol=1e-6)


def test_2d_rows_are_radial_then_azimuth(tmp_path):
    ai = make_ai()
    integ = cv.Integrator.from_pyfai(ai, 20, unit="q_nm^-1", npt_azim=8)
    assert integ.kind == "powder2d" and integ.nbins == 20 * 8 and len(integ.azimuthal) == 8
    fr = frame()
    d = b.chunk2sparseCSC(np.ones(SHAPE, np.uint8), integ.csc, np.uint16, exact=True)
    powder = d.from_frames(fr[None], 0)[2]
    got = integ.intensity(powder, d.powder_scale)[0].reshape(20, 8).T
    method = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=2)
    ref = ai.integrate2d(fr, 20, 8, unit="q_nm^-1", method=method)
    assert np.allclose(got, ref.intensity, rtol=1e-4, atol=1e-4)
