"""block_starts / block_starts_from_buffer: the block index found on the CPU without decoding, lz4 and
zstd, default and large blocks, partial tail blocks and raw tails, against a walk in Python."""

import os
import struct
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


def chunks_of(path, n):
    with h5py.File(path, "r") as h:
        return [h["d"].id.read_direct_chunk((i, 0, 0))[1] for i in range(n)]


def walk(chunk, itemsize):
    """The block starts of one chunk, from its header, in Python."""
    total, bs = struct.unpack_from(">QI", chunk, 0)
    bs = bs or 8192
    full, rem = divmod(total, bs)
    tail = 8 * itemsize * (rem // (8 * itemsize))
    pos, out = 12, []
    for _ in range(full + (tail > 0)):
        out.append(pos)
        pos += 4 + struct.unpack_from(">I", chunk, pos)[0]
    assert pos + rem - tail == len(chunk)
    return out


@pytest.mark.parametrize("cname", ["lz4", "zstd"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.float32, np.uint64])
@pytest.mark.parametrize("shape", [(96, 130), (7, 9), (1, 1), (64, 128), (50, 163)])
@pytest.mark.parametrize("nelems", [0, 2048, 32768])
def test_starts_match_the_headers(tmp_path, cname, dtype, shape, nelems):
    frames = frames_of(dtype, shape, n=3)
    chunks = chunks_of(write(tmp_path / "f.h5", frames, cname, nelems), 3)
    want = np.array([walk(c, np.dtype(dtype).itemsize) for c in chunks], np.uint32)
    got = b.block_starts(chunks, dtype)
    assert got.dtype == np.uint32 and got.shape == want.shape and (got == want).all()
    blob = b"".join(chunks)
    lens = np.array([len(c) for c in chunks])
    offs = np.concatenate(([0], np.cumsum(lens)[:-1]))
    got2 = b.block_starts_from_buffer(np.frombuffer(blob, np.uint8), offs, lens, dtype)
    assert (got2 == want).all()
    out = np.zeros(want.size, np.uint32)
    assert b.block_starts(chunks, dtype, out=out).base is not None and (out == want.ravel()).all()


def test_errors(tmp_path):
    frames = frames_of(np.uint16, (96, 130), n=3)
    chunks = chunks_of(write(tmp_path / "f.h5", frames, "lz4"), 3)
    with pytest.raises(Exception):
        b.block_starts([], np.uint16)
    with pytest.raises(Exception, match="-103"):
        b.block_starts(chunks, np.uint16, out=np.zeros(3, np.uint32))
    bad = bytearray(chunks[1])
    bad[12:16] = struct.pack(">I", len(bad))  # the first block runs past the chunk
    with pytest.raises(Exception, match=r"chunks \[1\]"):
        b.block_starts([chunks[0], bytes(bad), chunks[2]], np.uint16)
    with pytest.raises(Exception, match=r"chunks \[2\]"):  # truncated: does not end with its tail
        b.block_starts([chunks[0], chunks[1], chunks[2][:-3]], np.uint16)
    with pytest.raises(Exception, match=r"chunks \[0\]"):  # a byte too many
        b.block_starts([chunks[0] + b"\0", chunks[1]], np.uint16)
    other = chunks_of(write(tmp_path / "g.h5", frames_of(np.uint16, (50, 163), n=1), "lz4"), 1)
    with pytest.raises(Exception, match="-106"):  # not the same total size
        b.block_starts([chunks[0], other[0]], np.uint16)
