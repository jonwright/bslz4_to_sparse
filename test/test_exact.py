"""fixed_point_matrix: chunk2sparseCSC given it returns int64 sums and a power-of-two scale that equal
the float powder; a raw matrix gives float64 as released; how coarse the weights are."""

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


def fp(m, mask, dtype, **kw):
    return b.fixed_point_matrix(m, mask, dtype, **kw)


def test_unit_weights_are_the_exact_photon_sums():
    m, rows = matrix(False)
    mask = np.ones(SHAPE, np.uint8)
    fr = frames()
    c = b.chunk2sparseCSC(mask, fp(m, mask, np.uint16), np.uint16)
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
    ce = b.chunk2sparseCSC(mask, fp(m, mask, np.uint16), np.uint16)
    cf = b.chunk2sparseCSC(mask, m, np.uint16)
    pe = ce.from_frames(fr, 0)[2].astype(np.float64) * ce.powder_scale
    pf = cf.from_frames(fr, 0)[2]
    assert np.allclose(pe, pf, rtol=1e-5)
    assert cf.powder_scale is None and pf.dtype == np.float64


def test_the_decoder_does_not_prepare_the_matrix():
    """A raw matrix is a float matrix for every pixel type, as in the released versions."""
    m, _ = matrix(False)
    mask = np.ones(SHAPE, np.uint8)
    for dt in (np.uint8, np.int16, np.uint16, np.uint32, np.float32):
        c = b.chunk2sparseCSC(mask, m, dt)
        assert (
            c.powder_scale is None and c.from_frames(frames().astype(dt), 0)[2].dtype == np.float64
        )
    # an integer dot needs the prepared matrix, and a prepared matrix an integer dot
    with pytest.raises(ValueError, match="fixed_point_matrix"):
        b.chunk2sparseCSC(mask, m, np.uint16, pipeline={"dot": "csc-int"})
    with pytest.raises(ValueError, match="integer dot"):
        b.chunk2sparseCSC(mask, fp(m, mask, np.uint16), np.uint16, pipeline={"dot": "csc"})


def test_fixed_point_matrix_refuses_what_it_cannot_do():
    m, _ = matrix(False)
    mask = np.ones(SHAPE, np.uint8)
    with pytest.raises(ValueError, match="integer pixel"):
        fp(m, mask, np.float32)
    neg = m.copy()
    neg.data[::7] = -neg.data[::7]
    with pytest.raises(ValueError):
        fp(neg, mask, np.uint16)
    # made for 16-bit pixels, used for 32-bit ones: the sums could overflow
    with pytest.raises(ValueError, match="overflow"):
        b.chunk2sparseCSC(mask, fp(m, mask, np.uint16), np.uint32)
    with pytest.raises(ValueError, match="pixels"):
        b.chunk2sparseCSC(np.ones((3, 3), np.uint8), fp(m, mask, np.uint16), np.uint16)
    # and it is integers only: wbits 16 needs the 16-bit dot, which the default dot is for it
    q16 = fp(m, mask, np.uint16, wbits=16)
    c = b.chunk2sparseCSC(mask, q16, np.uint16, pipeline={"dot": "csc-run-int16"})
    assert c.powder_scale == q16.scale
    with pytest.raises(ValueError, match="weights"):
        b.chunk2sparseCSC(mask, q16, np.uint16, pipeline={"dot": "csc-int"})


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
        q = fp(m, mask, dt, maxval=maxval)
        top = float(2**32 if maxval is None and dt == np.uint32 else (maxval or 2**16) + 1)
        want = min(31, int(np.floor(62 - np.log2(top) - np.log2(nmax))))
        assert q.bits == want, (dt, maxval, q.bits, want)
        assert q.scale == 2.0**-q.bits
        scales[(dt, maxval)] = q.bits
    # the same data type gets more bits when the data cannot reach its largest value
    assert scales[(np.uint32, 2**16 - 1)] > scales[(np.uint32, None)]
    assert scales[(np.uint32, 2**28)] < scales[(np.uint32, 2**16 - 1)]


def test_the_backend_policy():
    """The conveyor's CpuBackend prepares the matrix: exact for 16-bit pixels by default."""
    from bslz4_to_sparse import conveyor as cv

    m, _ = matrix(False)
    mask = np.ones(SHAPE, np.uint8)

    def scale(dt, **kw):
        return cv.CpuBackend(mask, dt, csc=m, **kw).make_worker().powder_scale

    assert scale(np.uint16) is not None and scale(np.uint8) is not None
    assert scale(np.uint32) is None and scale(np.uint32, maxval=2**28) is not None
    assert scale(np.uint16, exact=False) is None
    assert scale(np.uint16, pipeline={"dot": "csc"}) is None
    neg = m.copy()
    neg.data[::7] = -neg.data[::7]
    assert cv.CpuBackend(mask, np.uint16, csc=neg).make_worker().powder_scale is None  # falls back
    with pytest.raises(ValueError):
        cv.CpuBackend(mask, np.uint16, csc=neg, exact=True).make_worker()
