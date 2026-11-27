"""decompress / decompress_from_buffer: the front half alone, lz4 and zstd, with and without the
unshuffle, against what h5py reads through the HDF5 filter."""

from __future__ import absolute_import, division, print_function

from conftest import bitshuffle, needs_zstd_writer
import os
import struct
import sys
import threading

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

import bslz4_to_sparse as b


def write(path, frames, cname, nelems=0):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "d",
            data=frames,
            chunks=(1,) + frames.shape[1:],
            **bitshuffle(nelems=nelems, cname=cname)
        )
    return str(path)


def frames_of(dtype, shape, n=3, seed=0):
    rng = np.random.RandomState(seed)
    hi = 100 if np.dtype(dtype).itemsize == 1 else 5000
    out = np.where(rng.random_sample((n,) + shape) < 0.3, rng.randint(0, hi, (n,) + shape), 0)
    return out.astype(dtype)


def _packbits_little(a, axis):
    """np.packbits(a, axis, bitorder="little") as numpy 1.16 can: byte j of a row is the bits at
    columns 8j..8j+7, the first column the least-significant bit.  numpy 1.17 added axis/bitorder."""
    a = np.asarray(a, dtype=np.uint8)
    if a.shape[axis] % 8:
        raise ValueError("the packed length must be a multiple of 8")
    b = np.moveaxis(a, axis, -1).reshape(a.shape[:axis] + (a.shape[axis] // 8, 8))
    out = (b * (1 << np.arange(8))).sum(axis=-1, dtype=np.uint8)
    return np.moveaxis(out, -1, axis)


@pytest.mark.parametrize("cname", ["lz4", pytest.param("zstd", marks=needs_zstd_writer)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.int16, np.float32, np.uint64])
@pytest.mark.parametrize("shape", [(96, 130), (7, 9), (1, 1), (64, 128), (50, 163)])
def test_unshuffled_equals_hdf5(tmpdir, cname, dtype, shape):
    frames = frames_of(dtype, shape)
    path = write(os.path.join(str(tmpdir), "f.h5"), frames, cname)
    with h5py.File(path, "r") as h:
        ds = h["d"]
        chunks = [b.read_direct_chunk(ds, (i, 0, 0))[1] for i in range(len(frames))]
        codec = b.detect_codec(ds)
        got = b.decompress(chunks, dtype, codec=codec)
        assert got.shape == (len(frames), shape[0] * shape[1]) and got.dtype == np.dtype(dtype)
        assert (got.reshape(frames.shape) == frames).all()
        assert (
            b.decompress(chunks, dtype).reshape(frames.shape) == frames
        ).all()  # codec found from the data


@pytest.mark.parametrize("cname", ["lz4", pytest.param("zstd", marks=needs_zstd_writer)])
def test_shuffled_is_what_the_filter_decoder_yields(tmpdir, cname):
    """unshuffle=False + the bit unshuffle of bitshuffle (via h5py reading with the filter) agree
    block by block: re-shuffling the pixel result block by block reproduces it."""
    frames = frames_of(np.uint16, (96, 130))
    path = write(os.path.join(str(tmpdir), "f.h5"), frames, cname)
    with h5py.File(path, "r") as h:
        ds = h["d"]
        chunks = [b.read_direct_chunk(ds, (i, 0, 0))[1] for i in range(len(frames))]
    raw = b.decompress(chunks, np.uint16, unshuffle=False)
    pix = b.decompress(chunks, np.uint16, unshuffle=True)
    assert raw.dtype == np.uint8 and raw.shape == (len(frames), frames[0].size * 2)
    for f in range(len(frames)):
        # the first full block: 4096 pixels = 8192 bytes, bit-shuffled: byte j of bit-plane k holds
        # bit k of pixels 8j..8j+7
        px = pix[f, :4096]
        bits = ((px[:, None] >> np.arange(16)) & 1).astype(np.uint8)  # (4096, 16)
        planes = _packbits_little(bits.T, 1).reshape(
            -1
        )  # bit-plane k, then byte j: pixels 8j..8j+7
        assert (raw[f, :8192] == planes).all()


def test_offsets_out_and_threads(tmpdir):
    frames = frames_of(np.uint16, (96, 130), n=6)
    path = write(os.path.join(str(tmpdir), "f.h5"), frames, "lz4")
    with h5py.File(path, "r") as h:
        ds = h["d"]
        chunks = [b.read_direct_chunk(ds, (i, 0, 0))[1] for i in range(6)]
    blob = b"".join(chunks)
    lens = np.array([len(c) for c in chunks])
    offs = np.concatenate(([0], np.cumsum(lens)[:-1]))
    out = np.empty((6, 96 * 130), np.uint16)
    r = b.decompress_from_buffer(np.frombuffer(blob, np.uint8), offs, lens, np.uint16, out=out)
    assert r is out and (out.reshape(frames.shape) == frames).all()
    # threads decompress in parallel, each its own output
    res = [None] * 3

    def work(k):
        res[k] = b.decompress(chunks[2 * k : 2 * k + 2], np.uint16)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all((res[k].reshape(2, 96, 130) == frames[2 * k : 2 * k + 2]).all() for k in range(3))


def test_errors(tmpdir):
    frames = frames_of(np.uint16, (96, 130), n=2)
    path = write(os.path.join(str(tmpdir), "f.h5"), frames, "lz4")
    with h5py.File(path, "r") as h:
        chunks = [b.read_direct_chunk(h["d"], (i, 0, 0))[1] for i in range(2)]
    with pytest.raises(Exception):
        b.decompress([], np.uint16)
    with pytest.raises(ValueError):
        b.decompress(chunks, np.uint16, out=np.empty(10, np.uint16))
    with pytest.raises(ValueError):
        b.decompress(chunks, np.dtype("S3"))
    bad = bytearray(chunks[0])
    bad[16:40] = bytes(bytearray(range(24)))  # garbage inside the first block
    with pytest.raises(Exception, match="Error decoding"):
        b.decompress([bytes(bad)], np.uint16)
    with pytest.raises(Exception, match="Error decoding"):
        b.decompress([chunks[0][: len(chunks[0]) // 2]], np.uint16)
    with pytest.raises(Exception):
        b.decompress([chunks[0], chunks[0][:12] + chunks[0][12:200]], np.uint16)


@pytest.mark.parametrize("cname", ["lz4", pytest.param("zstd", marks=needs_zstd_writer)])
def test_large_blocks(tmpdir, cname):
    frames = frames_of(np.uint16, (160, 400), n=2)
    path = write(os.path.join(str(tmpdir), ("f%s.h5" % cname)), frames, cname, nelems=32768)
    with h5py.File(path, "r") as h:
        chunks = [b.read_direct_chunk(h["d"], (i, 0, 0))[1] for i in range(2)]
    assert (b.decompress(chunks, np.uint16).reshape(frames.shape) == frames).all()


def test_corrupt_header_does_not_allocate_by_it(tmpdir):
    frames = frames_of(np.uint16, (96, 130), n=1)
    path = write(os.path.join(str(tmpdir), "f.h5"), frames, "lz4")
    with h5py.File(path, "r") as h:
        chunk = bytearray(b.read_direct_chunk(h["d"], (0, 0, 0))[1])
    chunk[8:12] = struct.pack(">I", 2**31 - 8)  # a block size of 2 GB
    with pytest.raises(Exception, match="Error decoding"):
        b.decompress([bytes(chunk)], np.uint16)
    chunk[8:12] = struct.pack(">I", 8192 + 4)  # not a whole number of groups of 8 pixels
    with pytest.raises(Exception, match="Error decoding"):
        b.decompress([bytes(chunk)], np.uint16)
