"""The cut is a threshold of the pixel type, not of a C int: 32-bit pixels take cuts up to 2**32 - 1
and above (nothing is above them), and 16-bit pixels take any cut too."""

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

import bslz4_to_sparse as b  # noqa: E402

SHAPE = (24, 32)


def chunks_of(frames, dtype):
    with h5py.File("cut_range.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset(
            "d",
            data=frames.astype(dtype),
            chunks=(1,) + SHAPE,
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        return [bytes(ds.id.read_direct_chunk((i, 0, 0))[1]) for i in range(len(frames))]


@pytest.mark.parametrize(
    "cut", [0, 7, 2**31 - 1, 2**31, 3_000_000_000, 2**32 - 2, 2**32 - 1, 2**40]
)
def test_32_bit_pixels_with_any_cut(cut):
    rng = np.random.default_rng(1)
    big = rng.integers(0, 2**32, (3,) + SHAPE, dtype=np.uint64)
    frames = np.where(rng.random((3,) + SHAPE) < 0.3, big, 0).astype(np.uint64)
    frames[0, 0, :3] = [2**31 - 1, 2**31, 2**32 - 1]
    chunks = chunks_of(frames, np.uint32)
    c = b.chunk2sparse(np.ones(SHAPE, np.uint8), np.uint32)
    npx, (values, indices) = c.multi(chunks, cut)
    for i in range(3):
        ref = np.flatnonzero(frames[i].ravel() > cut)
        assert npx[i] == len(ref) and (indices[i, : npx[i]] == ref).all()
        assert (values[i, : npx[i]] == frames[i].ravel()[ref]).all()


@pytest.mark.parametrize("cut", [0, 1000, 65535, 70000, 2**31])
def test_16_bit_pixels_with_any_cut(cut):
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 65536, (2,) + SHAPE)
    c = b.chunk2sparse(np.ones(SHAPE, np.uint8), np.uint16)
    npx, (values, indices) = c.multi(chunks_of(frames, np.uint16), cut)
    assert [int(n) for n in npx] == [int((f > cut).sum()) for f in frames]


def test_a_cut_with_a_matrix_too():
    import scipy.sparse as sp

    rng = np.random.default_rng(3)
    frames = rng.integers(0, 2**32, (2,) + SHAPE, dtype=np.uint64)
    npix = SHAPE[0] * SHAPE[1]
    m = sp.csc_matrix(
        (np.ones(npix, np.float32), (np.arange(npix) % 5, np.arange(npix))), (5, npix)
    )
    c = b.chunk2sparseCSC(np.ones(SHAPE, np.uint8), m, np.uint32)
    cut = 3_000_000_000
    npx, _, powder = c.multi(chunks_of(frames, np.uint32), cut)
    assert [int(n) for n in npx] == [int((f > cut).sum()) for f in frames]
    want = np.stack(
        [
            np.bincount(np.arange(npix) % 5, weights=f.ravel().astype(np.float64), minlength=5)
            for f in frames
        ]
    )
    assert np.allclose(powder, want, rtol=1e-6)


def test_a_negative_cut_is_still_refused():
    c = b.chunk2sparse(np.ones(SHAPE, np.uint8), np.uint16)
    chunk = chunks_of(np.ones((1,) + SHAPE), np.uint16)[0]
    with pytest.raises(Exception, match="negative"):
        c(chunk, -1)


@pytest.mark.parametrize("cut", [0, 0.5, 2.5, 100.25, 1e10])
def test_float_pixels_with_a_fractional_cut(cut):
    rng = np.random.default_rng(4)
    frames = (rng.random((2,) + SHAPE) * 200).astype(np.float32)
    frames[0, 0, :3] = [2.5, np.float32(100.25), np.float32(100.26)]
    c = b.chunk2sparse(np.ones(SHAPE, np.uint8), np.float32)
    npx, (values, indices) = c.multi(chunks_of(frames, np.float32), cut)
    for i in range(2):
        ref = np.flatnonzero(frames[i].ravel() > np.float32(cut))
        assert npx[i] == len(ref) and (indices[i, : npx[i]] == ref).all()
        assert (values[i, : npx[i]] == frames[i].ravel()[ref]).all()


def test_an_integer_cut_is_floored_and_clipped():
    rng = np.random.default_rng(5)
    frames = rng.integers(0, 10, (2,) + SHAPE)
    c = b.chunk2sparse(np.ones(SHAPE, np.uint8), np.uint8)
    chunks = chunks_of(frames, np.uint8)
    for cut, eff in [(2.5, 2), (np.array(3), 3), (300, 255), (1e30, 255)]:
        npx, _ = c.multi(chunks, cut)
        assert [int(n) for n in npx] == [int((f > eff).sum()) for f in frames]
