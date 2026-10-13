"""chunk2sparseCSC(exact=True): the powder is int64 sums and a power-of-two scale, and equals the
float powder."""

import numpy as np
import pytest
import scipy.sparse as sp

import bslz4_to_sparse as b

SHAPE = (40, 50)
NBINS = 17


def matrix(weights):
    npix = SHAPE[0] * SHAPE[1]
    rng = np.random.default_rng(2)
    rows = rng.integers(0, NBINS, npix)
    data = np.ones(npix, np.float32) if not weights else rng.random(npix).astype(np.float32)
    return sp.csc_matrix((data, (rows, np.arange(npix))), (NBINS, npix)), rows


def frames(n=6, dtype=np.uint16):
    rng = np.random.default_rng(3)
    return rng.integers(0, 60000, (n,) + SHAPE).astype(dtype)


def test_unit_weights_are_the_exact_photon_sums():
    m, rows = matrix(False)
    mask = np.ones(SHAPE, np.uint8)
    fr = frames()
    c = b.chunk2sparseCSC(mask, m, np.uint16, exact=True)
    _, _, powder = c.from_frames(fr, 0)
    assert powder.dtype == np.int64 and c.powder_scale is not None
    assert np.log2(c.powder_scale) == int(np.log2(c.powder_scale))
    want = np.stack([np.bincount(rows, weights=f.ravel(), minlength=NBINS) for f in fr])
    # integer weights are round(1 * 2**b): the sums are exact multiples of 2**b
    assert np.array_equal(powder.astype(np.float64) * c.powder_scale, want)


def test_weights_match_the_float_powder():
    m, _ = matrix(True)
    mask = np.ones(SHAPE, np.uint8)
    fr = frames()
    ce = b.chunk2sparseCSC(mask, m, np.uint16, exact=True)
    cf = b.chunk2sparseCSC(mask, m, np.uint16)
    pe = ce.from_frames(fr, 0)[2].astype(np.float64) * ce.powder_scale
    pf = cf.from_frames(fr, 0)[2]
    assert np.allclose(pe, pf, rtol=1e-5)
    assert cf.powder_scale is None


def test_exact_refuses_float_pixels_and_float_dots():
    m, _ = matrix(False)
    mask = np.ones(SHAPE, np.uint8)
    with pytest.raises(ValueError, match="integer pixels"):
        b.chunk2sparseCSC(mask, m, np.float32, exact=True)
    with pytest.raises(ValueError, match="integer dot"):
        b.chunk2sparseCSC(mask, m, np.uint16, exact=True, pipeline={"dot": "csc"})
