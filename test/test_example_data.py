"""tools/example_data.py: the variants made from a download hold the same pixels, 8 kB lz4 blocks."""

from __future__ import absolute_import, division, print_function

from conftest import bitshuffle, needs_zstd_writer
import os
import struct
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
)
import example_data as ed  # noqa: E402

import bslz4_to_sparse as b  # noqa: E402


@needs_zstd_writer
def test_variants_from_a_download(tmpdir, monkeypatch):
    rng = np.random.RandomState(3)
    shape = (32, 64)
    mask = (rng.random_sample(shape) < 0.1).astype(np.uint8)
    frames = rng.randint(0, 500, (4,) + shape).astype(np.uint32)
    frames[:, mask != 0] = 0
    src, cache = os.path.join(str(tmpdir), "src"), os.path.join(str(tmpdir), "cache")
    os.makedirs(src)
    with h5py.File(os.path.join(src, ed.FILES["ceo2"]), "w") as h:
        h.create_dataset(
            ed.DSET,
            data=frames,
            chunks=(1,) + shape,
            **bitshuffle(nelems=65536, cname="zstd", clevel=2)
        )
    with h5py.File(os.path.join(src, ed.MASK), "w") as h:
        h["mask"] = mask
    monkeypatch.setenv("BSLZ4_EXAMPLES", src)
    monkeypatch.setenv("BSLZ4_BENCH_DIR", cache)
    full, lz4, lz4mask = (ed.path("ceo2", v) for v in ed.VARIANTS)
    assert full == os.path.join(src, ed.FILES["ceo2"]) and os.path.dirname(lz4) == cache
    with h5py.File(lz4, "r") as h:
        ds = h[ed.DSET]
        assert b.detect_codec(ds) == b.CODEC_LZ4 and np.array_equal(ds[:], frames)
        chunk = b.read_direct_chunk(ds, (1, 0, 0))[1]
        assert struct.unpack(">I", chunk[8:12])[0] == 8192  # the block size, bytes
    with h5py.File(lz4mask, "r") as h:
        got = h[ed.DSET][:]
    assert (got[:, mask != 0] == np.iinfo(np.uint32).max).all()
    assert np.array_equal(got[:, mask == 0], frames[:, mask == 0])
    assert np.array_equal(ed.mask(), mask)
    assert ed.path("ceo2", "lz4") == lz4  # made once


def test_download_checks_the_md5(tmpdir, monkeypatch):
    web, dst = os.path.join(str(tmpdir), "web"), os.path.join(str(tmpdir), "dst")
    os.makedirs(web)
    for i, name in enumerate(ed.MD5):
        open(os.path.join(web, name), "wb").write(b"data %d" % i)
    base = ("file://" + os.path.abspath(web) + "/%s")
    monkeypatch.setattr(ed, "MD5", {n: ed.md5(os.path.join(web, n)) for n in ed.MD5})
    assert ed.download(dst, base) == dst
    assert sorted(os.listdir(dst)) == sorted(ed.MD5)
    ed.download(dst, base)  # already there: nothing fetched
    open(os.path.join(web, "mask.h5"), "wb").write(b"changed")  # now the source does not match
    os.remove(os.path.join(dst, "mask.h5"))
    with pytest.raises(SystemExit):
        ed.download(dst, base)
    assert not os.path.exists(os.path.join(dst, "mask.h5")) and \
        not os.path.exists(os.path.join(dst, "mask.h5.part"))
