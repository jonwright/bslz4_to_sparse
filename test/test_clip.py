"""wtmax afterwards: clipping the exact powder from the sparse pixels equals summing clipped
frames, bit for bit (u16 and u32, split weights, saturated pixels, a mask)."""

import numpy as np
import pytest
import scipy.sparse as sp

import bslz4_to_sparse as b
from bslz4_to_sparse import conveyor as cv

SHAPE = (50, 60)
NBINS = 14


def split_matrix():
    npix = SHAPE[0] * SHAPE[1]
    rng = np.random.default_rng(5)
    r0 = rng.integers(0, NBINS, npix)
    r1 = (r0 + 1 + rng.integers(0, NBINS - 1, npix)) % NBINS
    w = rng.random(npix).astype(np.float32) * 0.8 + 0.1
    cols = np.arange(npix)
    return sp.csc_matrix(
        (np.concatenate([w, 1 - w]), (np.concatenate([r0, r1]), np.concatenate([cols, cols]))),
        (NBINS, npix),
    )


@pytest.mark.parametrize("dtype,wtmax", [(np.uint16, 1000), (np.uint32, 50000)])
def test_clip_afterwards_equals_clipping_first(dtype, wtmax):
    rng = np.random.default_rng(7)
    top = np.iinfo(dtype).max
    fr = rng.poisson(20, (8,) + SHAPE).astype(dtype)
    hot = rng.random(fr.shape) < 0.01
    fr[hot] = rng.integers(wtmax + 1, top, hot.sum(), endpoint=True).astype(dtype)
    fr[:, 0, :3] = top  # saturated pixels
    mask = np.ones(SHAPE, np.uint8)
    mask[3, :] = 0
    m = split_matrix()
    cut = 5  # at most wtmax: every pixel above wtmax is in the sparse output
    e = b.chunk2sparseCSC(mask, b.fixed_point_matrix(m, mask, dtype), dtype)
    npx, (values, indices), powder = e.from_frames(fr, cut)
    powder = powder.copy()
    values, indices = values.copy(), indices.copy()
    W, scale = cv.fixed_weights(m, mask, dtype)
    assert scale == e.powder_scale
    got = cv.clip_exact_powder(powder, npx, values, indices, wtmax, W)
    ref = (
        b.chunk2sparseCSC(mask, b.fixed_point_matrix(m, mask, dtype), dtype)
        .from_frames(np.minimum(fr, wtmax).astype(dtype), cut)[2]
        .copy()
    )
    assert np.array_equal(got, ref)
    assert not np.array_equal(got, powder)  # the clip did something
