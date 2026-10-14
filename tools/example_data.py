#!/usr/bin/env python3
"""The example data every benchmark and timing tool starts from, and the tool that prepares it.

    python3 tools/example_data.py download [--dir WHERE]
    python3 tools/example_data.py prepare [--dir DOWNLOADED] [--cache WHERE]
    python3 tools/example_data.py repack SRC DST [--frames 510:760] [--dtype uint16] \
        [--codec zstd --level 2 --block-kb 256] [--mask mask.h5]

The download (https://doi.org/10.5281/zenodo.23361659; a folder, $BSLZ4_EXAMPLES, else DEFAULT_DIR
if that exists, else ~/.cache/bslz4_to_sparse/examples) holds

    ceo2_u32_zstd2_256k.h5     dense:  CeO2 powder, uint32, 72 frames (mean ~68 counts per pixel)
    silicon_u16_zstd2_256k.h5  sparse: silicon, uint16, 250 frames (~0.4 % of pixels above 0)
    mask.h5                    dataset "mask", pyFAI's convention (non-zero = masked), both files
    (calib.poni, an approximate pyFAI geometry of both files without spatial distortion, is
     tools/calib.poni in the repository, not in the download)

zstd level 2 in 256 kB blocks, to keep the download small.  The decoders are meant for the 8 kB
blocks of bitshuffle-lz4 that the detector software writes, so `prepare` makes, in the cache folder
($BSLZ4_BENCH_DIR, else DEFAULT_CACHE if that exists, else ~/.cache/bslz4_to_sparse/variants), for
each file:

    <key>_lz4.h5      bitshuffle-lz4, 8 kB blocks, the same pixels (masked pixels are 0)
    <key>_lz4mask.h5  the same with the masked pixels set to -1 in the pixel type (65535 or
                      4294967295), as the detector writes them

Benchmarks call path(key, variant), which builds a missing variant on first use.

`repack` is the general tool behind it, for any file with the frames in DSET: another pixel type,
codec, block size or range of frames.  Everything in the file but the frames is copied as it is
(lima writes a header of metadata beside the data), except time_of_frame, which is cut to the same
range; the frames are written one per chunk with the bitshuffle filter: `--codec lz4` (bslz4) or
`zstd` (bszstd, at `--level`), with blocks of `--block-kb` kilobytes (8 is the bitshuffle default).
A pixel type that cannot hold the values is refused, and every frame written is read back and
compared, so the result holds exactly the pixels of the source.  `--mask mask.h5` sets the masked
pixels to -1 in the pixel type, which is what the detector writes for a masked pixel.

The example data came from ID11 files in which the detector's 16-bit counts had been stored as
32-bit by the acquisition software: --dtype uint16 puts them back.
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402
import numpy as np  # noqa: E402

DSET = "/entry_0000/ESRF-ID11/eiger/data"
DEFAULT_DIR = "/data/id11/jon/hdftest/bslz4_to_sparse_examples"
DEFAULT_CACHE = "/data/id11/jon/hdftest/bslz4_to_sparse_variants"  # ESRF: the variants and matrices
FILES = {"ceo2": "ceo2_u32_zstd2_256k.h5", "silicon": "silicon_u16_zstd2_256k.h5"}
MASK = "mask.h5"
PONI = os.path.join(HERE, "calib.poni")
VARIANTS = ("zstd", "lz4", "lz4mask")
ZENODO = "https://zenodo.org/api/records/23361660/files/%s/content"  # of doi 10.5281/zenodo.23361659
MD5 = {
    "ceo2_u32_zstd2_256k.h5": "39872d5b45ea89adead86b11a36022f1",
    "silicon_u16_zstd2_256k.h5": "cb3fe7636eabf8d2dffd9af692b1917e",
    "mask.h5": "52665c08ca33aef9d5897b1d5f04a298",
}


def repack(
    src, dst, frames=None, dtype=None, codec="lz4", level=None, block_kb=8, dset=DSET, mask=None
):
    """Write frames[start:stop] of src as dtype with the given codec and block size into dst, the
    pixels where mask (frame shaped, non-zero = masked) is set to the largest value of dtype;
    returns (the number of frames, the largest pixel value, the compressed bytes written)."""
    with h5py.File(src, "r") as hs, h5py.File(dst, "w") as hd:
        ds = hs[dset]
        start, stop = frames or (0, ds.shape[0])
        dtype = np.dtype(dtype or ds.dtype)
        shape = ds.shape[1:]
        skip = {dset.strip("/")}
        times = dset.rsplit("/", 1)[0].strip("/") + "/time_of_frame"
        skip.add(times)

        def copy(name, obj):
            if name in skip:
                return
            if isinstance(obj, h5py.Group):
                hd.require_group(name).attrs.update(obj.attrs)
            else:
                hs.copy(obj, hd, name=name)

        hs.visititems(copy)
        kw = dict(nelems=block_kb * 1024 // dtype.itemsize, cname=codec)
        if level is not None:
            kw["clevel"] = level
        out = hd.create_dataset(
            dset,
            shape=(stop - start,) + shape,
            dtype=dtype,
            chunks=(1,) + shape,
            **hdf5plugin.Bitshuffle(**kw)
        )
        out.attrs.update(ds.attrs)
        if times in hs:
            hd.create_dataset(times, data=hs[times][start:stop])
        top = 0
        masked = None if mask is None else np.asarray(mask) != 0
        for i in range(start, stop):
            f = ds[i]
            top = max(top, int(f.max()))
            if top > np.iinfo(dtype).max:
                raise ValueError("frame %d has a value of %d, too large for %s" % (i, top, dtype))
            f = f.astype(dtype)
            if masked is not None:
                f[masked] = np.iinfo(dtype).max
            out[i - start] = f
            if not np.array_equal(out[i - start], f):
                raise ValueError("frame %d was not written exactly" % i)
        hd.attrs["repacked_from"] = "%s frames %d:%d" % (os.path.basename(src), start, stop)
        nbytes = sum(out.id.get_chunk_info(i).size for i in range(stop - start))
    return stop - start, top, nbytes


def md5(fname):
    h = hashlib.md5()
    with open(fname, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(dst=None, base=ZENODO):
    """Fetch the files not yet in dst from base % name, checking each against MD5.

    Zenodo is a free public service: do not fetch what we already have.  The files always go to the
    same place (dst, by default example_dir(): $BSLZ4_EXAMPLES, else DEFAULT_DIR, else
    ~/.cache/bslz4_to_sparse/examples), a file whose md5 matches is never fetched again, and an interrupted
    download is resumed from its .part file, not started over.  Anything else in this module reads
    the files from that place and never downloads."""
    dst = dst or example_dir()
    os.makedirs(dst, exist_ok=True)
    for name, want in MD5.items():
        out = os.path.join(dst, name)
        if os.path.exists(out) and md5(out) == want:
            print("%s: there" % name)
            continue
        tmp = out + ".part"
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(base % name)
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        print("%s: downloading%s" % (name, " (resuming at %d bytes)" % have if have else ""))
        with urllib.request.urlopen(req) as r:
            resumed = have and getattr(r, "status", None) == 206  # a server that ignores Range sends it all
            with open(tmp, "ab" if resumed else "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
        if md5(tmp) != want:
            os.remove(tmp)
            sys.exit("%s: the md5 is not %s, not kept" % (name, want))
        os.rename(tmp, out)
    return dst


def example_dir():
    env = os.environ.get("BSLZ4_EXAMPLES")
    if env or os.path.isdir(DEFAULT_DIR):
        return env or DEFAULT_DIR
    return os.path.join(home_dir(), "examples")


def home_dir():
    """Where the files live unless told otherwise: a user cache folder, not /tmp (it is wiped)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "bslz4_to_sparse")


def cache_dir():
    env = os.environ.get("BSLZ4_BENCH_DIR")
    if env or os.path.isdir(DEFAULT_CACHE):
        return env or DEFAULT_CACHE
    return os.path.join(home_dir(), "variants")


def mask(src=None):
    """The detector mask, pyFAI's convention: non-zero = masked."""
    with h5py.File(os.path.join(src or example_dir(), MASK), "r") as h:
        return h["mask"][()]


def path(key, variant="lz4", src=None, cache=None):
    """File of a data set in a variant (VARIANTS); a missing 8 kB variant is made from the download."""
    src, cache = src or example_dir(), cache or cache_dir()
    full = os.path.join(src, FILES[key])
    if not os.path.exists(full):
        sys.exit("%s is not there: the example data is the download of "
                 "https://doi.org/10.5281/zenodo.23361659, a folder %s: python3 tools/example_data.py "
                 "download [--dir %s] (BSLZ4_EXAMPLES says where)" % (FILES[key], src, src))
    if variant == "zstd":
        return full
    out = os.path.join(cache, "%s_%s.h5" % (key, variant))
    if not os.path.exists(out):
        os.makedirs(cache, exist_ok=True)
        tmp = out + ".tmp"
        repack(full, tmp, codec="lz4", block_kb=8,
                  mask=mask(src) if variant == "lz4mask" else None)
        os.rename(tmp, out)
    return out


def geometry_tag(src=None):
    """Names the geometry and mask in a cache key, so that matrices follow a changed poni or mask."""
    src = src or example_dir()
    with open(PONI, "rb") as fh:
        h = hashlib.sha256(fh.read())
    h.update(mask(src).tobytes())
    return h.hexdigest()[:8]


def integrator(src=None):
    """pyFAI AzimuthalIntegrator of calib.poni, with the detector mask of mask.h5."""
    import pyFAI

    src = src or example_dir()
    ai = pyFAI.load(PONI)
    ai.detector.mask = (mask(src) != 0).astype(np.int8)
    return ai


def frames(key, nframes, variant="lz4", src=None):
    """nframes frames spread evenly over the file (all of them if it has fewer) and the frame count
    of the file."""
    with h5py.File(path(key, variant, src), "r") as f:
        ds = f[DSET]
        idx = np.unique(np.linspace(0, ds.shape[0] - 1, nframes).round().astype(int))
        return np.stack([ds[int(i)] for i in idx]), int(ds.shape[0])


def chunks(key, nframes, variant="lz4", src=None):
    """The compressed chunks of nframes frames spread evenly over the file (no file I/O after
    this), the first frame, and {frames_in_file, shape, dtype}."""
    with h5py.File(path(key, variant, src), "r") as f:
        ds = f[DSET]
        idx = np.unique(np.linspace(0, ds.shape[0] - 1, nframes).round().astype(int))
        cs = [ds.id.read_direct_chunk((int(i), 0, 0))[1] for i in idx]
        info = {"frames_in_file": int(ds.shape[0]), "shape": list(ds.shape[1:]),
                "dtype": str(ds.dtype)}
        return cs, ds[int(idx[0])], info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="fetch the example files from Zenodo")
    p.add_argument("--dir", default=None, help="where to (default $BSLZ4_EXAMPLES)")
    p = sub.add_parser("prepare", help="make the lz4 and lz4mask variants of the download")
    p.add_argument("--dir", default=None, help="the download (default $BSLZ4_EXAMPLES)")
    p.add_argument("--cache", default=None, help="where the variants go (default $BSLZ4_BENCH_DIR, else ~/.cache/bslz4_to_sparse)")
    p = sub.add_parser("repack", help="another dtype, codec, block size or frames of a file")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--frames", default=None, help="start:stop")
    p.add_argument("--dtype", default=None)
    p.add_argument("--codec", default="lz4", choices=["lz4", "zstd"])
    p.add_argument("--level", type=int, default=None, help="zstd level")
    p.add_argument("--block-kb", type=int, default=8)
    p.add_argument("--dset", default=DSET)
    p.add_argument("--mask", default=None, help="mask.h5: masked pixels are set to the maximum")
    a = ap.parse_args()
    if a.cmd == "download":
        print(download(a.dir))
        return
    if a.cmd == "prepare":
        for key in FILES:
            for variant in VARIANTS[1:]:
                p = path(key, variant, a.dir, a.cache)
                print("%s  %.1f MB" % (p, os.path.getsize(p) / 1e6))
        return
    mask = None
    if a.mask:
        with h5py.File(a.mask, "r") as h:
            mask = h["mask"][()]
    fr = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    n, top, nbytes = repack(a.src, a.dst, fr, a.dtype, a.codec, a.level, a.block_kb, a.dset, mask)
    print("%d frames, largest value %d, %d bytes of chunks (%.0f per frame)" % (n, top, nbytes, nbytes / n))


if __name__ == "__main__":
    main()
