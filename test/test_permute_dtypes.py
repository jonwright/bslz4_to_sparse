"""A permutation matrix (FAZIT: weight 1, one bin per pixel, each bin reached once) on the CPU:
the automatic dot is csc-permute and the powder is the pixels in bin order, in the pixel dtype,
for every pixel dtype, on sparse and dense frames (both routes), with negative pixels for the
signed types.  Some bins are reached by no pixel (0).  An explicit dot keeps its accumulator."""

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

import bslz4_to_sparse as b  # noqa: E402

DTYPES = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32]


def chunks_of(path, frames):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]


def permutation(mask, extra, rng):
    pix = np.flatnonzero(mask.ravel())
    nbins = pix.size + extra
    bins = rng.permutation(nbins)[: pix.size]
    n = np.zeros(mask.size, np.int64)
    n[pix] = 1
    M = sp.csc_matrix((np.ones(pix.size, np.float32), bins.astype(np.int32),
                       np.concatenate(([0], np.cumsum(n))).astype(np.int32)),
                      shape=(nbins, mask.size))
    return M, pix, bins


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("lam", [0.02, 30.0])   # sparse route / dense route
def test_permutation_keeps_the_pixel_dtype(tmp_path, dt, lam):
    shape = (61, 83)
    rng = np.random.default_rng(5)
    mask = (rng.random(shape) > 0.2).astype(np.uint8)
    a = rng.poisson(lam, (4,) + shape)
    if np.dtype(dt).kind == "i":
        a = a - rng.poisson(lam, a.shape)
    if np.dtype(dt).itemsize == 1:
        a = np.clip(a, -100, 100)
    frames = a.astype(dt)
    M, pix, bins = permutation(mask, 7, rng)
    chunks = chunks_of(tmp_path / "f.h5", frames)
    want = np.zeros((len(frames), M.shape[0]), dt)
    want[:, bins] = frames.reshape(len(frames), -1)[:, pix]
    c = b.chunk2sparseCSC(mask, M, dtype=dt)
    for cut in (0, 3):
        npx, sparse, p = c.multi(chunks, cut)
        assert c.dot == "csc-permute", c.dot
        assert p.dtype == np.dtype(dt), p.dtype
        assert (p == want).all(), (cut, int((p != want).sum()))
    c = b.chunk2sparseCSC(mask, M, dtype=dt, pipeline={"dot": "csc-nosplit"})
    npx, sparse, p = c.multi(chunks, 0)
    assert p.dtype == np.float64 and (p == want).all()
