"""codec=None (the default) finds lz4 vs zstd from the first chunk decoded, so
chunk2sparse(mask, dtype) and chunk2sparseCSC(mask, csc, dtype) read either
kind of file without detect_codec(dataset) or an explicit codec."""

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
from bslz4_to_sparse import _pipeline

SHAPE = (99, 209)
MASK = np.ones(SHAPE, np.uint8)
MASK[:, 100:104] = 0
CODECS = (("lz4", b.CODEC_LZ4), ("zstd", b.CODEC_ZSTD))


def frames_of(dt):
    rng = np.random.default_rng(3)
    f = np.zeros((3,) + SHAPE, dt)
    for i in range(3):
        ij = rng.integers(0, SHAPE[0] * SHAPE[1], 400)
        f[i].ravel()[ij] = rng.integers(1, 300, 400)
    return f


def write(path, frames, cname):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname=cname))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]


def check(c, frames, chunks):
    for f, chunk in zip(frames, chunks):
        n, (vals, adr) = c(chunk, 0)
        want = np.flatnonzero((f.ravel() > 0) & (MASK.ravel() > 0))
        assert n == want.size
        assert (adr[:n] == want).all()
        assert (vals[:n] == f.ravel()[want]).all()


@pytest.mark.parametrize("dt", [np.uint16, np.uint32])
@pytest.mark.parametrize("cname,codec", CODECS)
def test_default_codec(tmp_path, dt, cname, codec):
    frames = frames_of(dt)
    chunks = write(tmp_path / "a.h5", frames, cname)
    c = b.chunk2sparse(MASK, dtype=dt)
    assert c.codec == b.CODEC_LZ4          # provisional, until the first chunk
    check(c, frames, chunks)
    assert c.codec == codec
    # same pipeline as one made with the codec given
    ref = b.chunk2sparse(MASK, dtype=dt, codec=codec)
    assert list(b.describe(c.pipeline).items()) == list(b.describe(ref.pipeline).items())


@pytest.mark.parametrize("cname,codec", CODECS)
def test_multi_and_offsets(tmp_path, cname, codec):
    frames = frames_of(np.uint16)
    path = tmp_path / "m.h5"
    chunks = write(path, frames, cname)
    c = b.chunk2sparse(MASK, dtype=np.uint16)
    npx, (vals, adr) = c.multi(chunks, 0)
    assert c.codec == codec
    ref = b.chunk2sparse(MASK, dtype=np.uint16, codec=codec).multi(chunks, 0)
    assert (npx == ref[0]).all()
    # from_buffer: the first chunk is only seen as a short header
    with h5py.File(str(path), "r") as f:
        offs = b.locate_chunks(f["d"])
    with open(str(path), "rb") as fh:
        data = np.frombuffer(fh.read(), np.uint8)
    o, l = b.offsets_lengths(offs, sorted(offs))
    c2 = b.chunk2sparse(MASK, dtype=np.uint16)
    npx2, _ = c2.from_buffer(data, o, l, 0)
    assert c2.codec == codec
    assert (npx2 == ref[0]).all()


@pytest.mark.parametrize("cname,codec", CODECS)
def test_csc(tmp_path, cname, codec):
    frames = frames_of(np.uint16)
    chunks = write(tmp_path / "c.h5", frames, cname)
    npix = SHAPE[0] * SHAPE[1]
    M = sp.random(50, npix, density=0.02, format="csc", random_state=1, dtype=np.float32)
    c = b.chunk2sparseCSC(MASK, M, dtype=np.uint16, exact=False)
    ref = np.array([M @ (f.ravel().astype(np.float64) * MASK.ravel()) for f in frames])
    n, (vals, adr), p = c.multi(chunks, 0)
    assert c.codec == codec
    assert np.allclose(p, ref, rtol=1e-6, atol=1e-6)


def test_explicit_decode_step_names_the_codec():
    # a pipeline that names the decoder fixes the codec: nothing to detect
    c = b.chunk2sparse(MASK, dtype=np.uint16, pipeline={"decode": "zstd"})
    assert c.codec == b.CODEC_ZSTD and not c._codec_pending
    c = b.chunk2sparse(MASK, dtype=np.uint16, pipeline={"decode": "lz4-stock"})
    assert c.codec == b.CODEC_LZ4 and not c._codec_pending


def test_given_codec_is_not_changed(tmp_path):
    frames = frames_of(np.uint16)
    chunks = write(tmp_path / "g.h5", frames, "lz4")
    c = b.chunk2sparse(MASK, dtype=np.uint16, codec=b.CODEC_LZ4)
    assert not c._codec_pending
    check(c, frames, chunks)
    assert c.codec == b.CODEC_LZ4
