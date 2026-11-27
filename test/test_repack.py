"""tools/example_data.py (repack): another dtype, codec and block size, the same pixels."""

from __future__ import absolute_import, division, print_function

from conftest import bitshuffle, needs_zstd_writer
import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
)
import example_data as rp  # noqa: E402

import bslz4_to_sparse as b  # noqa: E402


@pytest.mark.parametrize(
    "dtype,codec,level,block_kb",
    [("uint16", "lz4", None, 8),
     pytest.param("uint32", "zstd", 2, 256, marks=needs_zstd_writer),
     pytest.param("uint16", "zstd", 2, 8, marks=needs_zstd_writer)],
)
def test_repack_keeps_the_pixels(tmpdir, dtype, codec, level, block_kb):
    rng = np.random.RandomState(1)
    frames = (rng.random_sample((6, 40, 50)) < 0.05) * rng.randint(1, 5000, (6, 40, 50))
    src, dst = os.path.join(str(tmpdir), "s.h5"), os.path.join(str(tmpdir), "d.h5")
    with h5py.File(src, "w") as h:
        h.create_dataset(
            rp.DSET,
            data=frames.astype(np.uint32),
            chunks=(1, 40, 50),
            **bitshuffle(nelems=0, cname="lz4")
        )
        h.create_dataset("entry_0000/header/x", data=7)
        h.create_dataset("entry_0000/ESRF-ID11/eiger/time_of_frame", data=np.arange(6.0))
    n, top, nbytes = rp.repack(src, dst, (1, 5), dtype, codec, level, block_kb)
    assert n == 4 and top == frames[1:5].max() and nbytes > 0
    with h5py.File(dst, "r") as h:
        ds = h[rp.DSET]
        assert ds.dtype == np.dtype(dtype) and ds.shape == (4, 40, 50) and ds.chunks == (1, 40, 50)
        assert np.array_equal(ds[:], frames[1:5])
        assert h["entry_0000/header/x"][()] == 7
        assert list(h["entry_0000/ESRF-ID11/eiger/time_of_frame"][:]) == [1.0, 2.0, 3.0, 4.0]
        # our decoder reads the chunks too, whatever the codec and block size
        c = b.chunk2sparse(np.ones((40, 50), np.uint8), np.dtype(dtype), b.detect_codec(ds))
        chunk = b.read_direct_chunk(ds, (2, 0, 0))[1]
        npx, (v, ix) = c(chunk, 0)
        assert npx == np.count_nonzero(frames[3]) and (v[:npx] == frames[3].ravel()[ix[:npx]]).all()


def test_a_dtype_that_cannot_hold_the_values_is_refused(tmpdir):
    src, dst = os.path.join(str(tmpdir), "s.h5"), os.path.join(str(tmpdir), "d.h5")
    with h5py.File(src, "w") as h:
        h.create_dataset(
            rp.DSET,
            data=np.full((2, 8, 8), 70000, np.uint32),
            chunks=(1, 8, 8),
            **bitshuffle(nelems=0, cname="lz4")
        )
    with pytest.raises(ValueError, match="too large"):
        rp.repack(src, dst, None, "uint16")


@pytest.mark.parametrize("dtype", ["uint16", "uint32"])
def test_mask_sets_the_masked_pixels_to_minus_one(tmpdir, dtype):
    rng = np.random.RandomState(2)
    frames = rng.randint(0, 3000, (3, 16, 24)).astype(np.uint32)
    mask = (rng.random_sample((16, 24)) < 0.1).astype(np.uint8)
    src, dst = os.path.join(str(tmpdir), "s.h5"), os.path.join(str(tmpdir), "d.h5")
    with h5py.File(src, "w") as h:
        h.create_dataset(
            rp.DSET, data=frames, chunks=(1, 16, 24), **bitshuffle(nelems=0, cname="lz4")
        )
    rp.repack(src, dst, dtype=dtype, mask=mask)
    with h5py.File(dst, "r") as h:
        got = h[rp.DSET][:]
    want = frames.astype(dtype)
    want[:, mask != 0] = np.iinfo(dtype).max
    assert np.array_equal(got, want) and (got[:, mask != 0] == np.array(-1).astype(dtype)).all()
