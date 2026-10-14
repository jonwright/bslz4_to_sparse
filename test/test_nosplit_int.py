"""csc-nosplit-int (the automatic dot of a histogram of integer pixels): csc-nosplit's histogram summed in
int64 and returned as float64.

It must be a drop-in for the float csc-nosplit: the same float64 powder (the sums are exact for these
sizes) on both routes, for every integer pixel type of at most 32 bits, with a frame that ends in a
partial tile and a partial block, and with garbage at the masked pixels.  A sum above 2**53 is
rounded once, in the cast at the end; wider pixels, float pixels and a matrix that splits pixels are
refused.
"""
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

DOT = "csc-nosplit-int"


def encode(frames, block=8192):
    shape = frames.shape[1:]
    kw = hdf5plugin.Bitshuffle(nelems=block // frames.dtype.itemsize, cname="lz4")
    with h5py.File("t.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + shape, **kw)
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]


def histogram(shape, nbins, seed=3):
    npix = shape[0] * shape[1]
    # sorted: neighbouring pixels share a bin, in runs, as radial bins give
    bins = np.sort(np.random.default_rng(seed).integers(0, nbins, npix))
    return sp.csc_matrix((np.ones(npix, np.float32), (bins, np.arange(npix))), shape=(nbins, npix))


def run(mask, M, dtype, chunks, dot, route):
    c = b.chunk2sparseCSC(mask, M, dtype=dtype, codec=b.CODEC_LZ4,
                          pipeline={"dot": dot, "route": route})
    npx, (v, i), powder = c.multi(chunks, 1)
    return c, [(v[k, :npx[k]].copy(), i[k, :npx[k]].copy()) for k in range(len(npx))], np.array(powder)


@pytest.mark.parametrize("route", ("dense", "sparse", "ratio"))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32, np.int16, np.int32))
def test_equals_the_float_histogram(dtype, route):
    shape = (97, 211)                                    # 20467 pixels: a partial block and tile
    mask = np.ones(shape, np.uint8)
    mask[10:13, :] = 0
    mask[np.random.default_rng(1).random(shape) < 0.02] = 0
    info = np.iinfo(dtype)
    rng = np.random.default_rng(2)
    frames = rng.integers(0, min(info.max, 3000), (3,) + shape).astype(dtype)
    frames[:, 50:60] = rng.integers(0, info.max, (3, 10, shape[1])).astype(dtype)   # dense, large
    frames[:, mask == 0] = info.max                      # what a detector writes at masked pixels
    M = histogram(shape, 40)
    chunks = encode(frames)
    cf, sparse_f, powder_f = run(mask, M, dtype, chunks, "csc-nosplit", route)
    ci, sparse_i, powder_i = run(mask, M, dtype, chunks, DOT, route)
    assert powder_i.dtype == np.float64 and ci.powder_scale is None
    assert powder_i.shape == powder_f.shape
    assert float(np.abs(powder_f).max()) < 2 ** 53 and np.array_equal(powder_i, powder_f)
    for (vf, if_), (vi, ii) in zip(sparse_f, sparse_i):
        assert np.array_equal(vf, vi) and np.array_equal(if_, ii)


def test_sum_above_2_53_is_rounded_once():
    shape = (2048, 2048)
    npix = shape[0] * shape[1]
    mask = np.ones(shape, np.uint8)
    v = 4000000001
    frames = np.full((1,) + shape, v, np.uint32)
    M = sp.csc_matrix((np.ones(npix, np.float32), (np.zeros(npix, np.int64), np.arange(npix))),
                      shape=(1, npix))
    _, _, powder = run(mask, M, np.uint32, encode(frames), DOT, "dense")
    assert npix * v > 2 ** 53
    assert powder.dtype == np.float64 and powder[0, 0] == float(npix * v)   # int64 sum, one rounding


@pytest.mark.parametrize("dtype", (np.uint64, np.int64, np.float32))
def test_refuses_other_pixel_types(dtype):
    mask = np.ones((16, 64), np.uint8)
    with pytest.raises(ValueError, match="32 bits|integer pixels"):
        b.chunk2sparseCSC(mask, histogram((16, 64), 4), dtype=dtype, codec=b.CODEC_LZ4,
                          pipeline={"dot": DOT})


def test_refuses_a_matrix_that_splits_pixels():
    shape = (16, 64)
    npix = 16 * 64
    pix = np.arange(npix)
    split = sp.csc_matrix((np.full(2 * npix, 0.5, np.float32),
                           (np.concatenate([pix % 5, pix % 5 + 1]), np.concatenate([pix, pix]))),
                          shape=(6, npix))
    with pytest.raises(ValueError, match="cannot represent this matrix"):
        b.chunk2sparseCSC(np.ones(shape, np.uint8), split, dtype=np.uint16, codec=b.CODEC_LZ4,
                          pipeline={"dot": DOT})


@pytest.mark.parametrize("dtype,want", ((np.uint8, DOT), (np.uint16, DOT), (np.int32, DOT), (np.uint32, DOT),
                                        (np.uint64, "bsb-csr-nosplit"), (np.int64, "bsb-csr-nosplit"),
                                        (np.float32, "bsb-csr-nosplit")))
def test_automatic_dot(dtype, want):
    """A histogram of integer pixels of at most 32 bits takes it; wider and float pixels do not."""
    mask = np.ones((16, 64), np.uint8)
    c = b.chunk2sparseCSC(mask, histogram((16, 64), 4), dtype=dtype, codec=b.CODEC_LZ4)
    assert c.dot == want


def test_automatic_dot_for_a_split_matrix():
    shape = (16, 64)
    npix = 16 * 64
    pix = np.arange(npix)
    split = sp.csc_matrix((np.full(2 * npix, 0.5, np.float32),
                           (np.concatenate([pix % 5, pix % 5 + 1]), np.concatenate([pix, pix]))),
                          shape=(6, npix))
    c = b.chunk2sparseCSC(np.ones(shape, np.uint8), split, dtype=np.uint16, codec=b.CODEC_LZ4)
    assert c.dot != DOT


def test_repeated_calls_and_batch_sizes():
    shape = (64, 100)
    mask = np.ones(shape, np.uint8)
    M = histogram(shape, 9)
    rng = np.random.default_rng(4)
    frames = rng.integers(0, 500, (5,) + shape).astype(np.uint16)
    chunks = encode(frames)
    ci = b.chunk2sparseCSC(mask, M, dtype=np.uint16, codec=b.CODEC_LZ4, pipeline={"dot": DOT})
    cf = b.chunk2sparseCSC(mask, M, dtype=np.uint16, codec=b.CODEC_LZ4, pipeline={"dot": "csc-nosplit"})
    for sel in (slice(0, 2), slice(0, 5), slice(3, 4), slice(0, 5)):
        got = np.array(ci.multi(chunks[sel], 1)[2])
        ref = np.array(cf.multi(chunks[sel], 1)[2])
        assert got.dtype == np.float64 and np.array_equal(got, ref)
