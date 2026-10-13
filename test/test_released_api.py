"""The names of the released versions (0.0.15, 0.0.16) keep working: the calls below are the ones
of those releases' tests and README, unchanged.  chunk2sparseCSC is the one exception: since 0.0.22
its default for 8 and 16-bit pixels is an exact int64 powder (see exact=), so the old float
powder is asked for with exact=False.
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

SHAPE = (97, 211)
DTYPES = (np.uint8, np.uint16, np.uint32)


def make(tmp_path, dtype, lz4=True):
    rng = np.random.default_rng(7)
    frames = rng.poisson(0.2, (3,) + SHAPE).astype(dtype)
    name = str(tmp_path / ("f%d.h5" % np.dtype(dtype).itemsize))
    with h5py.File(name, "w") as f:
        f.create_dataset(
            "d", data=frames, chunks=(1,) + SHAPE, compression=32008, compression_opts=(0, 2)
        )  # bitshuffle + lz4, as the old tests
    return name, frames


def reference(frame, mask, cut):
    sel = (frame * mask > cut).ravel()
    return frame.ravel()[sel], np.nonzero(sel)[0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_bslz4_to_sparse_and_chunk2sparse(tmp_path, dtype):
    name, frames = make(tmp_path, dtype)
    mask = np.ones(SHAPE, np.uint8)
    mask[5, 7] = 0
    with h5py.File(name, "r") as f:
        ds = f["d"]
        obj = b.chunk2sparse(mask, ds.dtype)  # the dtype of the dataset, as the old tests
        for num in range(len(frames)):
            for cut in (0, 1, 2):
                pv, pi = reference(frames[num], mask, cut)
                npx, (cv, ci) = b.bslz4_to_sparse(ds, num, cut, mask.ravel())
                assert npx == len(pv) and (cv[:npx] == pv).all() and (ci[:npx] == pi).all()
                _, chunk = ds.id.read_direct_chunk((num, 0, 0))
                npx, (cv, ci) = obj(chunk, cut)
                assert npx == len(pv) and (cv[:npx] == pv).all() and (ci[:npx] == pi).all()
                n, row, col, val = obj.coo(chunk, cut)
                assert n == len(pv) and (val == pv).all()
                assert (row.astype(np.int64) * SHAPE[1] + col == pi).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunk2sparseCSC(tmp_path, dtype):
    name, frames = make(tmp_path, dtype)
    mask = np.ones(SHAPE, np.uint8)
    mat = sp.random(30, mask.size, density=0.1, format="csc", random_state=3, dtype=np.float32)
    mat2 = type(
        "pyFAIlike", (), dict(data=mat.data, indices=mat.indices, indptr=mat.indptr, bins=30)
    )()
    with h5py.File(name, "r") as f:
        ds = f["d"]
        for csc in (mat, mat2):
            obj = b.chunk2sparseCSC(mask, csc, np.dtype(ds.dtype), exact=False)
            for num in range(len(frames)):
                _, chunk = ds.id.read_direct_chunk((num, 0, 0))
                npx, (cv, ci), powder = obj(chunk, 1)
                pv, pi = reference(frames[num], mask, 1)
                want = mat @ frames[num].ravel().astype(np.float64)
                assert npx == len(pv) and (cv[:npx] == pv).all() and (ci[:npx] == pi).all()
                assert powder.dtype == np.float64
                assert np.abs(powder - want).max() < 1e-3 * max(1.0, np.abs(want).max())
                n, row, col, val, p2 = obj.coo(chunk, 1)
                assert n == len(pv) and (val == pv).all() and (p2 == powder).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_raw_functions_and_helpers(tmp_path, dtype):
    name, frames = make(tmp_path, dtype)
    mask = np.ones(SHAPE, np.uint8)
    mask[3, 4] = 0
    mat = sp.random(30, mask.size, density=0.1, format="csc", random_state=3, dtype=np.float32)
    raw = getattr(b, "bslz4_%s_t" % np.dtype(dtype).name)
    rawcsc = getattr(b, "bslz4_csc_%s_t" % np.dtype(dtype).name)
    with h5py.File(name, "r") as f:
        ds = f["d"]
        _, chunk = ds.id.read_direct_chunk((1, 0, 0))
        assert isinstance(chunk, bytes)  # read only: npbuf makes it a writable-typed uint8 array
        cmp = b.npbuf(chunk)
        assert cmp.dtype == np.uint8 and b.npbuf(cmp) is cmp
        assert (b.npbuf(memoryview(chunk)) == cmp).all()
        values = np.empty(mask.size, dtype)
        indices = np.empty(mask.size, np.uint32)
        npx = raw(cmp, mask.ravel(), values, indices, 0)
        pv, pi = reference(frames[1], mask, 0)
        assert npx == len(pv) and (values[:npx] == pv).all() and (indices[:npx] == pi).all()
        powder = np.empty(30, np.float64)
        npx = rawcsc(
            cmp,
            mask.ravel(),
            values,
            indices,
            0,
            powder,
            mat.data,
            mat.indices.astype(np.uint32),
            mat.indptr.astype(np.uint32),
        )
        assert npx == len(pv) and (values[:npx] == pv).all() and (indices[:npx] == pi).all()
        want = mat @ (frames[1] * mask).ravel().astype(np.float64)
        assert np.abs(powder - want).max() < 1e-3 * max(1.0, np.abs(want).max())


def test_buffer_from_memory():
    buf = b"abcdef"
    if not hasattr(b.buffer_from_memory, "argtypes"):
        pytest.skip("python 2.7 path")
    import ctypes

    mv = b.buffer_from_memory(ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p), len(buf), 0x200)
    assert bytes(mv) == buf
