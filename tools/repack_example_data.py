#!/usr/bin/env python3
"""Re-pack the frames of an HDF5 file: another pixel type, codec, block size, or a range of frames.

    python3 tools/repack_example_data.py silicon.h5 out.h5 --frames 510:760 --dtype uint16 \
        --codec zstd --level 2 --block-kb 256

Everything in the file but the frames is copied as it is (lima writes a header of metadata beside
the data), except time_of_frame, which is cut to the same range; the frame dataset is written one frame per chunk with the bitshuffle filter: `--codec
lz4` (bslz4) or `zstd` (bszstd, at `--level`), with blocks of `--block-kb` kilobytes (8 is the
bitshuffle default).  A pixel type that cannot hold the values is refused, and every frame written
is read back and compared, so the result holds exactly the pixels of the source.

The example data of this project came from ID11 files in which the detector's 16-bit counts had been
stored as 32-bit by the acquisition software: --dtype uint16 puts them back.
"""

import argparse
import sys

import h5py
import hdf5plugin
import numpy as np

DSET = "/entry_0000/ESRF-ID11/eiger/data"


def repack(src, dst, frames=None, dtype=None, codec="lz4", level=None, block_kb=8, dset=DSET):
    """Write frames[start:stop] of src as dtype with the given codec and block size into dst;
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
        for i in range(start, stop):
            f = ds[i]
            top = max(top, int(f.max()))
            if top > np.iinfo(dtype).max:
                raise ValueError("frame %d has a value of %d, too large for %s" % (i, top, dtype))
            out[i - start] = f.astype(dtype)
            if not np.array_equal(out[i - start].astype(np.int64), f.astype(np.int64)):
                raise ValueError("frame %d was not written exactly" % i)
        hd.attrs["repacked_from"] = "%s frames %d:%d" % (src, start, stop)
        nbytes = sum(out.id.get_chunk_info(i).size for i in range(stop - start))
    return stop - start, top, nbytes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--frames", default=None, help="start:stop")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--codec", default="lz4", choices=["lz4", "zstd"])
    ap.add_argument("--level", type=int, default=None, help="zstd level")
    ap.add_argument("--block-kb", type=int, default=8)
    ap.add_argument("--dset", default=DSET)
    a = ap.parse_args()
    fr = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    n, top, nbytes = repack(a.src, a.dst, fr, a.dtype, a.codec, a.level, a.block_kb, a.dset)
    print(
        "%d frames, largest value %d, %d bytes of chunks (%.0f per frame)"
        % (n, top, nbytes, nbytes / n)
    )


if __name__ == "__main__":
    sys.exit(main())
