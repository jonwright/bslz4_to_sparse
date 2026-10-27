"""The integer dots add modulo 2**64: a fixed-point matrix made for a maxval below the saturated
pixels (Eiger u32: 2**32 - 1) gives int64 sums that wrap (u32 pixels, 32-bit weights), and they
are still exact once the sparse pixels above maxval are subtracted again, by clipping them to
maxval or by dropping them.  Checked bit for bit against the same dot on frames clipped (or
zeroed) first, on both routes, for each integer dot and unsigned pixel type."""

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

import bslz4_to_sparse as b

SHAPE = (50, 60)
NBINS = 14


def run_matrix():
    """Each pixel split over two neighbouring bins (a run, so every integer dot takes it)."""
    npix = SHAPE[0] * SHAPE[1]
    rng = np.random.RandomState(5)
    r0 = rng.randint(0, NBINS - 1, npix)
    w = rng.random_sample(npix).astype(np.float32) * 0.8 + 0.1
    cols = np.arange(npix)
    return sp.csc_matrix(
        (np.concatenate([w, 1 - w]), (np.concatenate([r0, r0 + 1]), np.concatenate([cols, cols]))),
        (NBINS, npix),
    )


def exact_sums(W, frames):
    """W @ frame for each frame in Python integers (no overflow), (nframes, nbins) of objects."""
    W = W.tocsc()
    out = np.zeros((len(frames), W.shape[0]), object)
    for f, fr in enumerate(frames):
        v = fr.ravel()
        for p in np.flatnonzero(v):
            k0, k1 = W.indptr[p], W.indptr[p + 1]
            for k in range(k0, k1):
                out[f, W.indices[k]] += int(W.data[k]) * int(v[p])
    return out


def subtract(powder, npx, values, indices, W, excess):
    """powder minus W @ excess(values) over each frame's sparse pixels, modulo 2**64."""
    out = powder.copy().view(np.uint64)
    W = W.tocsc()
    for f in range(len(npx)):
        k = int(npx[f])
        v, idx = values[f, :k], indices[f, :k].astype(np.int64)
        e = excess(v.astype(np.uint64))
        for p, x in zip(idx, e):
            k0, k1 = W.indptr[p], W.indptr[p + 1]
            np.subtract.at(out[f], W.indices[k0:k1], W.data[k0:k1].astype(np.uint64) * x)
    return out.view(np.int64)


@pytest.mark.parametrize("route", ["dense", "sparse"])
@pytest.mark.parametrize(
    "dot,wbits", [("csc-int", 32), ("csc-run-int", 32), ("csc-run-int16", 16)]
)
@pytest.mark.parametrize("dtype,maxval", [(np.uint8, 100), (np.uint16, 1000), (np.uint32, 1000)])
def test_wrapped_sums_correct_exactly(dtype, maxval, dot, wbits, route):
    rng = np.random.RandomState(7)
    top = np.iinfo(dtype).max
    fr = rng.poisson(20, (4,) + SHAPE).astype(dtype)
    fr[:, 0, :5] = top  # saturated
    fr[1, 10, 20:40] = top
    mask = np.ones(SHAPE, np.uint8)
    mask[3, :] = 0
    fr[:, 3, :] = top  # masked: no entries in the matrix, nothing to subtract
    M = b.fixed_point_matrix(run_matrix(), mask, dtype, maxval=maxval, wbits=wbits)
    W = M.to_scipy()

    def powder_of(frames, cut):
        c = b.chunk2sparseCSC(mask, M, dtype, pipeline={"dot": dot, "route": route})
        npx, (values, indices), powder = c.from_frames(frames, cut)
        return npx.copy(), values.copy(), indices.copy(), powder.copy()

    npx, values, indices, powder = powder_of(fr, maxval)
    exact = exact_sums(W, fr)
    assert powder.dtype == np.int64
    if dtype == np.uint32 and wbits == 32:  # u8, u16 and 16-bit weights cannot reach 2**63
        assert (exact >= 2**63).any()
    wrapped = np.array([[int(x) % 2**64 for x in row] for row in exact], np.uint64).view(np.int64)
    assert np.array_equal(powder, wrapped)

    clipped = subtract(powder, npx, values, indices, W, lambda v: v - np.uint64(maxval))
    ref = powder_of(np.minimum(fr, maxval).astype(dtype), maxval)[3]
    assert np.array_equal(clipped, ref)

    dropped = subtract(powder, npx, values, indices, W, lambda v: v)
    ref = powder_of(np.where(fr > maxval, 0, fr).astype(dtype), maxval)[3]
    assert np.array_equal(dropped, ref)
