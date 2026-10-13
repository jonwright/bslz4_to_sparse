#!/usr/bin/env python3
"""What compressing the results costs, and saves: ratio and CPU per codec on real decoder output.

    BSLZ4_TO_SPARSE_PATH=lib BSLZ4_BENCH_DIR=<matrix cache> python3 tools/bench_codecs.py \
        --files 'GLOB' --index-dir DIR --frames 200 --out codecs.json

Decodes the first --frames frames of the files (sparse pixels above --cut, and a 1D powder with the
pyFAI matrix), then compresses the results in chunks of --chunk elements, one thread, with each
codec, and prints one JSON line per (result, codec):

  ratio          compressed bytes / raw bytes of that result
  mb_s           raw MB per second of one core, compression only
  core_s_per_gb  core seconds spent per GB of compressed *input* (frames' chunks): multiply by the
                 input rate of a node to get the cores the writer needs
  out_over_in    compressed result bytes / compressed input bytes

Results: intensity (pixel dtype), row and col (uint16, the ImageD11 sparse layout), flat (uint32
pixel index), nnz (uint32 per frame), powder (float64 sums per bin).  Codecs: raw (none), lz4, bslz4
(bitshuffle + lz4), zstd1, bszstd1 (bitshuffle + zstd level 1), and "entropy": the zeroth-order
byte entropy as a size estimate (no data written), which is what a cheap guard could use."""

import argparse
import glob
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite as bs  # noqa: E402,F401  (puts lib/ or $BSLZ4_TO_SPARSE_PATH on sys.path)
import bench_ops as bo  # noqa: E402

import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import conveyor  # noqa: E402


def codecs():
    import bitshuffle
    import lz4.block
    import zstandard

    zc = zstandard.ZstdCompressor(level=1)
    return {
        "raw": lambda a: a.tobytes(),
        "lz4": lambda a: lz4.block.compress(a.tobytes(), mode="fast"),
        "bslz4": lambda a: bitshuffle.compress_lz4(a),
        "zstd1": lambda a: zc.compress(a.tobytes()),
        "bszstd1": lambda a: zc.compress(bitshuffle.bitshuffle(a).tobytes()),
    }


def entropy_size(a):
    """bytes a zeroth-order coder would need for a's bytes: the size estimate, not a stream"""
    h = np.bincount(a.view(np.uint8), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return int(np.ceil(-(p * np.log2(p)).sum() * a.nbytes / 8)) + 1


def chunks(a, n):
    for i in range(0, len(a), n):
        yield np.ascontiguousarray(a[i : i + n])


def decode(files, dset, index_dir, nframes, cut, mats, mask):
    """(results dict of 1D arrays, compressed input bytes) for the first nframes frames."""
    units, shape, dtype = conveyor.make_units(files, dset, index_dir=index_dir)
    nj = shape[1]
    parts = {k: [] for k in ("intensity", "row", "col", "flat", "nnz", "powder")}
    got = in_bytes = 0
    be = conveyor.CpuBackend(mask, dtype, csc=mats["1d"], cut=cut)
    fn = be.make_worker()
    for u in units:
        if got >= nframes:
            break
        with open(u.path, "rb") as f:
            f.seek(u.start)
            data = np.frombuffer(f.read(u.span), np.uint8)
        n = min(u.n, nframes - got)
        npx, values, indices, powder = fn(data, u.offsets[:n], u.lengths[:n])
        for i in range(n):
            k = int(npx[i])
            parts["intensity"].append(values[i, :k].copy())
            parts["flat"].append(indices[i, :k].copy())
            parts["row"].append((indices[i, :k] // nj).astype(np.uint16))
            parts["col"].append((indices[i, :k] % nj).astype(np.uint16))
        parts["nnz"].append(npx[:n].astype(np.uint32))
        parts["powder"].append(np.asarray(powder[:n], np.float64).ravel())
        got += n
        in_bytes += int(u.lengths[:n].sum())
    return {k: np.concatenate(v) for k, v in parts.items()}, in_bytes, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--cut", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=65536, help="elements per compressed chunk")
    ap.add_argument("--dset", default=bo.DSET)
    ap.add_argument("--repeat", type=int, default=3, help="timing: best of")
    ap.add_argument("--out", default=None, help="write the JSON lines here too")
    a = ap.parse_args()
    files = sorted(glob.glob(a.files))
    _, shape, dtype = conveyor.make_units(files[:1], a.dset, index_dir=a.index_dir)
    units, _, _ = conveyor.make_units(files[:1], a.dset, 1 << 30, a.index_dir)
    u0 = units[0]  # the mask: pixels at the maximum value in the first frame (as bench_ops.py)
    with open(u0.path, "rb") as fh:
        fh.seek(u0.start + int(u0.offsets[0]))
        f0 = b.decompress([fh.read(int(u0.lengths[0]))], dtype)[0]
    mask = (f0 != np.iinfo(dtype).max).astype(np.uint8).reshape(shape)
    mats = bo.matrices(shape, mask)
    res, in_bytes, nfr = decode(files, a.dset, a.index_dir, a.frames, a.cut, mats, mask)
    cs = codecs()
    lines = []
    for name, arr in res.items():
        if not arr.size:
            continue
        for cname, fn in list(cs.items()) + [("entropy", None)]:
            best, size = None, 0
            for _ in range(1 if cname == "raw" else a.repeat):
                t = time.perf_counter()
                size = sum(
                    entropy_size(c) if fn is None else len(fn(c)) for c in chunks(arr, a.chunk)
                )
                t = time.perf_counter() - t
                best = t if best is None else min(best, t)
            lines.append(
                dict(
                    host=socket.gethostname(),
                    bslz4_to_sparse=b.version,
                    cut=a.cut,
                    frames=nfr,
                    chunk=a.chunk,
                    result=name,
                    dtype=str(arr.dtype),
                    elements=int(arr.size),
                    codec=cname,
                    raw_bytes=int(arr.nbytes),
                    stored_bytes=int(size),
                    ratio=round(size / arr.nbytes, 4),
                    mb_s=round(arr.nbytes / best / 1e6, 1),
                    core_s_per_gb=round(best / in_bytes * 1e9, 3),
                    out_over_in=round(size / in_bytes, 5),
                    in_bytes=in_bytes,
                )
            )
    for ln in lines:
        print(json.dumps(ln), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(lines, f, indent=1)


if __name__ == "__main__":
    main()
