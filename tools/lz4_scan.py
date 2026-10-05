"""Scan lz4 decompression speed over controlled pixel patterns.

Makes synthetic uint16 frames on the Eiger2 4M and 16M shapes (the real
detector masks, masked pixels 0 as the Eigers are run), compresses each with
hdf5plugin's bitshuffle-LZ4, and hands the chunks to tools/lz4_blocks.c, which
times LZ4_decompress_safe alone and describes the lz4 stream.

Families:
  poisson  lam    Poisson background of mean lam everywhere (density sweep)
  const    c      every pixel = c (fixes which bit-planes are set)
  frac64   f      Poisson(30) clipped to < 64, then one pixel of 100 in a
                  fraction f of the 4096-pixel blocks (how many blocks use
                  bit-plane 6)
  bench    kind   frame 0 of the cached benchmark data (medium / dense)

Usage:
  cc -O2 -Ilz4/lib -o /tmp/lz4_blocks tools/lz4_blocks.c lz4/lib/lz4.c
  python3 tools/lz4_scan.py --exe /tmp/lz4_blocks [--det 4M,16M] [--cpu 7]
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite as bs  # noqa: E402
import bench_perf as bp  # noqa: E402

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import hdf5plugin  # noqa: E402
import pyFAI  # noqa: E402


def compress(frame, tmpdir):
    fn = os.path.join(tmpdir, "c.h5")
    with h5py.File(fn, "w") as h:
        ds = h.create_dataset("d", data=frame[None], chunks=(1,) + frame.shape,
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        return ds.id.read_direct_chunk((0, 0, 0))[1]


def patterns(det, rng):
    d = pyFAI.detector_factory(bs.DETECTORS[det])
    shape, bad = d.shape, d.mask.astype(bool)

    def done(a):
        a = np.minimum(a, 65535).astype(np.uint16)
        a[bad] = 0
        return a
    for lam in (0.001, 0.01, 0.1, 0.5, 1, 3, 10, 30, 100, 300, 1000):
        yield "poisson %g" % lam, done(rng.poisson(lam, shape))
    for c in (1, 30, 63, 64, 100, 127, 128):
        yield "const %d" % c, done(np.full(shape, c))
    base = np.minimum(rng.poisson(30, shape), 63)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        a = base.copy().ravel()
        nb = a.size // 4096
        pick = rng.random(nb) < f
        a[np.flatnonzero(pick) * 4096 + 2000] = 100
        yield "frac64 %.2f" % f, done(a.reshape(shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="one or more lz4_blocks builds, comma separated; "
                    "each is run on the same chunks")
    ap.add_argument("--det", default="4M,16M")
    ap.add_argument("--cpu", type=int, default=7)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        rec = os.path.join(tmp, "records.bin")
        with open(rec, "wb") as out:
            for det in a.det.split(","):
                items = list(patterns(det, rng))
                ai = bs.make_ai(det, "mid")
                for kind in ("medium", "dense"):
                    chunks, _f0 = bs.load_chunks(bp.frames_for(det, "mid", kind, ai))
                    items.append(("bench %s" % kind, chunks[0]))
                for name, frame in items:
                    chunk = frame if isinstance(frame, bytes) else compress(frame, tmp)
                    label = ("%s %s" % (det, name)).encode()
                    out.write(struct.pack("<I", len(label)) + label)
                    out.write(struct.pack("<Q", len(chunk)) + chunk)
        for exe in a.exe.split(","):
            print("### %s" % exe, flush=True)
            subprocess.run(["taskset", "-c", str(a.cpu), exe, rec, str(a.reps)], check=True)


if __name__ == "__main__":
    main()
