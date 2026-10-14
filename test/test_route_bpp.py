"""route "bpp": a block takes the sparse route when it is compressed below 2 bits per pixel, for
every pixel width (the "ratio" rule's 8x is 2 bits per pixel for u16 only).  The powder does not
depend on the route."""
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

import bslz4_to_sparse as b
from bslz4_to_sparse import _pipeline as P, _testing

SHAPE = (128, 256)          # whole 8 kB blocks for every width; the empty tail counts as one sparse block


def routes(ran):
    """(dense blocks, sparse blocks without the empty tail)"""
    return ran.get("dense", 0), ran.get("sparse", 0) - 1


def encode(frame):
    kw = hdf5plugin.Bitshuffle(nelems=8192 // frame.dtype.itemsize, cname="lz4")
    with h5py.File("t.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("d", data=frame[None], chunks=(1,) + frame.shape, **kw)
        return [ds.id.read_direct_chunk((0, 0, 0))[1]]


def blocks_per_route(dtype, frame, route):
    npix = frame.size
    M = sp.csc_matrix((np.ones(npix, np.float32), (np.arange(npix) % 7, np.arange(npix))), shape=(7, npix))
    c = b.chunk2sparseCSC(np.ones(SHAPE, np.uint8), M, dtype=dtype, codec=b.CODEC_LZ4,
                          pipeline={"dot": "csc-nosplit", "route": route})
    chunks = encode(frame)
    with _testing.counting():
        powder = np.array(c.multi(chunks, 1)[2])
        ran = _testing.counters()["route"]
    return ran, powder


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32))
def test_threshold_is_bits_per_pixel(dtype):
    rng = np.random.default_rng(5)
    for frac, expect in ((0.002, "sparse"), (0.9, "dense")):    # compressed far below / far above 2 bpp
        frame = np.zeros(SHAPE, dtype)
        hit = rng.random(SHAPE) < frac
        frame[hit] = rng.integers(1, 3, hit.sum())
        ran, _ = blocks_per_route(dtype, frame, "bpp")
        assert (routes(ran)[0] == 0) == (expect == "sparse"), (dtype, frac, ran)
        assert (routes(ran)[1] == 0) == (expect == "dense"), (dtype, frac, ran)


def test_u32_between_the_two_rules():
    """About 3 bits per pixel: dense by the bpp rule; the ratio rule's 8x sends u32 to sparse."""
    rng = np.random.default_rng(6)
    frame = (rng.random(SHAPE) < 0.2).astype(np.uint32) * rng.integers(1, 8, SHAPE).astype(np.uint32)
    chunks = encode(frame)
    bpp = 8.0 * (len(chunks[0]) - 12) / frame.size
    assert 2.6 < bpp < 3.8, bpp
    d, s = routes(blocks_per_route(np.uint32, frame, "bpp")[0])
    assert d > 0 and s == 0
    d, s = routes(blocks_per_route(np.uint32, frame, "ratio")[0])
    assert d == 0 and s > 0


@pytest.mark.parametrize("dtype", (np.uint16, np.uint32))
def test_same_powder_on_every_route(dtype):
    rng = np.random.default_rng(7)
    frame = ((rng.random(SHAPE) < 0.15) * rng.integers(1, 40, SHAPE)).astype(dtype)
    ref = blocks_per_route(dtype, frame, "dense")[1]
    for route in ("sparse", "ratio", "bpp"):
        assert np.array_equal(blocks_per_route(dtype, frame, route)[1], ref)


def test_auto_route_is_bpp():
    c = b.chunk2sparseCSC(np.ones((16, 64), np.uint8),
                          sp.csc_matrix(np.ones((1, 1024), np.float32)), dtype=np.uint32, codec=b.CODEC_LZ4)
    assert b.describe(c.pipeline)["route"] == "bpp"
