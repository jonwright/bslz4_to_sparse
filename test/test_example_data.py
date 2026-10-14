"""tools/example_data.py: the variants made from a download hold the same pixels, 8 kB lz4 blocks."""

import os
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


def test_variants_from_a_download(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    shape = (32, 64)
    mask = (rng.random(shape) < 0.1).astype(np.uint8)
    frames = rng.integers(0, 500, (4,) + shape).astype(np.uint32)
    frames[:, mask != 0] = 0
    src, cache = tmp_path / "src", tmp_path / "cache"
    src.mkdir()
    with h5py.File(str(src / ed.FILES["ceo2"]), "w") as h:
        h.create_dataset(
            ed.DSET,
            data=frames,
            chunks=(1,) + shape,
            **hdf5plugin.Bitshuffle(nelems=65536, cname="zstd", clevel=2)
        )
    with h5py.File(str(src / ed.MASK), "w") as h:
        h["mask"] = mask
    monkeypatch.setenv("BSLZ4_EXAMPLES", str(src))
    monkeypatch.setenv("BSLZ4_BENCH_DIR", str(cache))
    full, lz4, lz4mask = (ed.path("ceo2", v) for v in ed.VARIANTS)
    assert full == str(src / ed.FILES["ceo2"]) and os.path.dirname(lz4) == str(cache)
    with h5py.File(lz4, "r") as h:
        ds = h[ed.DSET]
        assert b.detect_codec(ds) == b.CODEC_LZ4 and np.array_equal(ds[:], frames)
        chunk = ds.id.read_direct_chunk((1, 0, 0))[1]
        assert int.from_bytes(chunk[8:12], "big") == 8192  # the block size, bytes
    with h5py.File(lz4mask, "r") as h:
        got = h[ed.DSET][:]
    assert (got[:, mask != 0] == np.iinfo(np.uint32).max).all()
    assert np.array_equal(got[:, mask == 0], frames[:, mask == 0])
    assert np.array_equal(ed.mask(), mask)
    assert ed.path("ceo2", "lz4") == lz4  # made once


def test_download_checks_the_md5(tmp_path, monkeypatch):
    web, dst = tmp_path / "web", tmp_path / "dst"
    web.mkdir()
    for i, name in enumerate(ed.MD5):
        (web / name).write_bytes(b"data %d" % i)
    base = (web.as_uri() + "/%s")
    monkeypatch.setattr(ed, "MD5", {n: ed.md5(str(web / n)) for n in ed.MD5})
    assert ed.download(str(dst), base) == str(dst)
    assert sorted(os.listdir(str(dst))) == sorted(ed.MD5)
    ed.download(str(dst), base)  # already there: nothing fetched
    (web / "mask.h5").write_bytes(b"changed")  # now the source does not match
    (dst / "mask.h5").unlink()
    with pytest.raises(SystemExit):
        ed.download(str(dst), base)
    assert not (dst / "mask.h5").exists() and not (dst / "mask.h5.part").exists()
