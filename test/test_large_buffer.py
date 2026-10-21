"""from_buffer / block_starts_from_buffer with the chunks beyond 2 GiB in one buffer (an mmap of a
large file): the same result as the chunks in a small buffer.  Wrappers before c2py23 0.5.10
refused any buffer above INT_MAX bytes."""

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
from test_block_starts import chunks_of
from test_decompress import frames_of, write

if sys.maxsize < 2**32:
    pytest.skip("needs a 64-bit Python", allow_module_level=True)


def test_chunks_beyond_2gib(tmp_path):
    frames = frames_of(np.uint16, (96, 130), n=4)
    chunks = chunks_of(write(tmp_path / "f.h5", frames, "lz4"), 4)
    lens = np.array([len(c) for c in chunks], np.int64)
    offs = np.concatenate(([0], np.cumsum(lens)[:-1]))
    small = np.frombuffer(b"".join(chunks), np.uint8)
    big = np.zeros(2**31 + 2**20, np.uint8)  # only the pages written below are touched
    boffs = 2**31 + offs
    for o, c in zip(boffs, chunks):
        big[o : o + len(c)] = np.frombuffer(c, np.uint8)
    mask = np.ones(frames.shape[1:], np.uint8)
    c = b.chunk2sparse(mask, np.uint16)
    n0, (v0, i0) = c.from_buffer(small, offs, lens, 0)
    n0, v0, i0 = n0.copy(), v0.copy(), i0.copy()
    n1, (v1, i1) = c.from_buffer(big, boffs, lens, 0)
    assert list(n1) == [np.count_nonzero(f) for f in frames] and (n1 == n0).all()
    for k in range(len(frames)):
        assert (v1[k, : n1[k]] == v0[k, : n0[k]]).all() and (i1[k, : n1[k]] == i0[k, : n0[k]]).all()
    s0 = b.block_starts_from_buffer(small, offs, lens, np.uint16)
    assert (b.block_starts_from_buffer(big, boffs, lens, np.uint16) == s0).all()
