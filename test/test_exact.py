"""chunk2sparseCSC(exact=...): the powder is int64 sums and a power-of-two scale, and equals the
float powder; which kind a call gives by default, and how coarse the weights are."""

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
    cf = b.chunk2sparseCSC(mask, m, np.uint16, exact=False)
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


def test_the_default_is_exact_for_16_bit_pixels_and_float_otherwise():
    m, _ = matrix(False)
    mask = np.ones(SHAPE, np.uint8)
    for dt in (np.uint8, np.int16, np.uint16):
        fr = frames(dtype=np.uint16).astype(dt) if dt != np.uint8 else (frames() % 200).astype(dt)
        c = b.chunk2sparseCSC(mask, m, dt)
        p = c.from_frames(fr, 0)[2]
        assert p.dtype == np.int64 and c.powder_scale is not None, dt
    for dt in (np.uint32, np.int32, np.float32):  # float unless maxval says how large they get
        c = b.chunk2sparseCSC(mask, m, dt)
        assert (
            c.powder_scale is None and c.from_frames(frames().astype(dt), 0)[2].dtype == np.float64
        )
    # asked for: float, or a float dot by name
    assert b.chunk2sparseCSC(mask, m, np.uint16, exact=False).powder_scale is None
    assert b.chunk2sparseCSC(mask, m, np.uint16, pipeline={"dot": "csc"}).powder_scale is None
    # a matrix that cannot be fixed-point (negative weights) falls back to float, silently
    neg = m.copy()
    neg.data[::7] = -neg.data[::7]
    c = b.chunk2sparseCSC(mask, neg, np.uint16)
    assert c.powder_scale is None
    with pytest.raises(ValueError):
        b.chunk2sparseCSC(mask, neg, np.uint16, exact=True)


def test_the_fixed_point_weights_get_fewer_bits_as_the_pixels_get_larger():
    """b = min(32-bit weights, 62 - log2(largest pixel value + 1) - log2(largest sum of weights in a
    bin)): each doubling of maxval costs a bit, and a 32-bit pixel type costs the most."""
    m, _ = matrix(False)  # weight 1 per pixel: the largest sum in a bin is the largest count
    mask = np.ones(SHAPE, np.uint8)
    nmax = np.bincount(m.indices, minlength=NBINS).max()  # pixels in the fullest bin
    scales = {}
    for dt, maxval in (
        (np.uint16, None),
        (np.uint32, None),
        (np.uint32, 2**28),
        (np.uint32, 2**16 - 1),
    ):
        c = b.chunk2sparseCSC(mask, m, dt, exact=True, maxval=maxval)
        bits = -int(np.log2(c.powder_scale))
        top = float(2**32 if maxval is None and dt == np.uint32 else (maxval or 2**16) + 1)
        want = min(31, int(np.floor(62 - np.log2(top) - np.log2(nmax))))
        assert bits == want, (dt, maxval, bits, want)
        scales[(dt, maxval)] = bits
    # the same data type gets more bits when the data cannot reach its largest value
    assert scales[(np.uint32, 2**16 - 1)] > scales[(np.uint32, None)]
    assert scales[(np.uint32, 2**28)] < scales[(np.uint32, 2**16 - 1)]
    # and maxval alone makes a wide integer type exact by default
    assert b.chunk2sparseCSC(mask, m, np.uint32, maxval=2**28).powder_scale is not None
