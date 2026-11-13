"""Bytes after the end of a chunk's data (a writer may append, for example, a copy of the block
offsets) are ignored, as by the HDF5 filter: the raw pixels that do not fill a group of 8 are read
right after the last block, not from the end of the chunk."""

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

import bslz4_to_sparse as b
from test_decompress import frames_of, write

EXTRA = bytes(bytearray(range(101, 165)))  # 64 bytes, none of them zero


def chunks_and_padded(path, frames, cname):
    """The chunks as written, and the same with EXTRA appended (rewritten into the file and read
    back through the HDF5 filter, which must give the frames)."""
    with h5py.File(write(path, frames, cname), "a") as h:
        ds = h["d"]
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]
        for i, c in enumerate(chunks):
            ds.id.write_direct_chunk((i, 0, 0), c + EXTRA)
        padded = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]
        assert (ds[()] == frames).all()  # the filter ignores the extra bytes
    assert all(p == c + EXTRA for c, p in zip(chunks, padded))
    return chunks, padded


@pytest.mark.parametrize("cname", ["lz4", "zstd"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.float32])
@pytest.mark.parametrize("shape", [(96, 130), (7, 9), (1, 1), (50, 163)])  # 7x9, 1x1, 50x163: raw
def test_extra_bytes_are_ignored(tmp_path, cname, dtype, shape):
    frames = frames_of(dtype, shape, n=3)
    chunks, padded = chunks_and_padded(tmp_path / "f.h5", frames, cname)
    assert (b.decompress(padded, dtype).reshape(frames.shape) == frames).all()
    assert (b.block_starts(padded, dtype) == b.block_starts(chunks, dtype)).all()
    mask = np.ones(shape, np.uint8)
    c = b.chunk2sparse(mask, dtype)
    n0, (v0, i0) = c.multi(chunks, 0)
    n1, (v1, i1) = c.multi(padded, 0)
    assert (n1 == n0).all()
    for f in range(len(frames)):
        want = np.flatnonzero(frames[f].ravel() > 0)
        assert n1[f] == len(want) and (i1[f, : n1[f]] == want).all()
        assert (v1[f, : n1[f]] == frames[f].ravel()[want]).all()


def test_a_truncated_raw_tail_still_raises(tmp_path):
    frames = frames_of(np.uint16, (50, 163), n=2)  # 8150 pixels: 6 stored raw
    chunks, _ = chunks_and_padded(tmp_path / "f.h5", frames, "lz4")
    short = [chunks[0], chunks[1][:-1]]
    with pytest.raises(Exception, match=r"chunks \[1\]"):
        b.block_starts(short, np.uint16)
    with pytest.raises(Exception, match="-107"):
        b.decompress(short, np.uint16)
    with pytest.raises(Exception, match="-107"):
        b.chunk2sparse(np.ones((50, 163), np.uint8)).multi(short, 0)
