#!/usr/bin/env python3
"""Cores needed to saturate a link: what one core does per stage of the pipeline, on real frames.

    BSLZ4_TO_SPARSE_PATH=lib BSLZ4_BENCH_DIR=<matrix cache> python3 tools/bench_ops.py \
        --files 'GLOB' --index-dir DIR --workers 1,8,16,32 --ops decompress,unshuffle,sparsify,1d,2d,rings,fazit

The mask is the pixels at the dtype maximum in the first frame and the matrices are those of the example
data's mask.h5 and tools/calib.poni (tools/example_data.py), so the files are its lz4mask variants:
`python3 tools/example_data.py prepare` makes them, `--files "$BSLZ4_BENCH_DIR/*_lz4mask.h5"`.

Each worker (a pinned thread) reads one file of its own into its own memory once, then runs the stage
over its frames for --seconds: pure compute, no file system, the same decoder objects and 16-frame
batches the conveyor uses.  Stages, each including the ones before it:
  decompress  lz4/zstd decode only, the blocks left bit-shuffled
  unshuffle   decompress and the bit unshuffle: the pixel values
  sparsify    ... and the pixels above --cut (default 1, so that at most ~1 % of the pixels remain)
  1d          ... and a 1D powder (pyFAI bbox matrix)       2d     ... and a 2D powder (72 azimuth bins)
  rings       ... and 8 ring windows x 72 azimuth bins       fazit  ... and the FAZIT permutation
Prints per stage and worker count: frames/s per worker, GB/s of compressed input per worker, and the
workers needed for --link GB/s (25 GbE is 3.125 GB/s of line rate).  One JSON line per point."""

import argparse
import glob
import json
import os
import socket
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite as bs  # noqa: E402  (puts lib/ or $BSLZ4_TO_SPARSE_PATH on sys.path)
import example_data as ed  # noqa: E402

import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import conveyor  # noqa: E402

DSET = "/entry_0000/ESRF-ID11/eiger/data"
NAZIM = 72


def matrices(shape, mask):
    """{op: csc matrix} for 1d, 2d, rings and fazit (cached like the docs benchmark's)."""
    import bench_docs as bd

    ai = ed.integrator()
    if tuple(shape) != tuple(ai.detector.shape) or ((mask == 0) != (ai.detector.mask != 0)).any():
        sys.exit("the frames are not the example data's shape and mask (tools/example_data.py)")
    m = bd.matrices(ai)
    fz = bs._cached("docs_ex%s_fazit" % ed.geometry_tag(), lambda: bs.fazit_matrix(ai))
    return {"1d": m["1D"][0], "2d": m["2D"][0], "rings": m["2D+rings"][0], "fazit": fz}


def backend(op, dtype, shape, mask, mats, cut, gpu=False, batch=16, acc="f64", dot=None):
    if (
        gpu
    ):  # decompress alone is not an operation of the GPU decoder; fazit is not a matrix product
        if op in ("decompress", "unshuffle", "fazit"):
            return None
        return conveyor.GpuBackend(mask, dtype, csc=mats.get(op), cut=cut, batch=batch, acc=acc)
    if op in ("decompress", "unshuffle"):
        return conveyor.DecompressBackend(dtype, shape, unshuffle=op == "unshuffle")
    if op == "sparsify":
        return conveyor.CpuBackend(mask, dtype, cut=cut)
    pipe = {"dot": "csc-permute"} if op == "fazit" else ({"dot": dot} if dot else None)
    return conveyor.CpuBackend(mask, dtype, csc=mats[op], cut=cut, pipeline=pipe)


def run_point(op, be, units, workers, seconds, cpus, batch):
    """workers threads, each over its own unit, for `seconds`: (frames per worker list, sparse pixels per frame)"""
    barrier, go = threading.Barrier(workers + 1), threading.Event()
    out = [None] * workers
    errs = []

    def work(k):
        try:
            if cpus:
                os.sched_setaffinity(0, [cpus[k % len(cpus)]])
            u = units[k % len(units)]
            buf = np.empty(u.span, np.uint8)
            with open(u.path, "rb") as f:
                f.seek(u.start)
                f.readinto(memoryview(buf))
            fn = be.make_worker()
            fn(buf, u.offsets[:batch], u.lengths[:batch])  # warm up: buffers, matrix in cache
            barrier.wait()
            go.wait()
            t0, frames, px = time.perf_counter(), 0, 0
            while time.perf_counter() - t0 < seconds:
                for f0 in range(0, u.n, batch):
                    n = min(batch, u.n - f0)
                    r = fn(buf, u.offsets[f0 : f0 + n], u.lengths[f0 : f0 + n])
                    px += int(np.asarray(r[0]).sum())
                    frames += n
                    if time.perf_counter() - t0 >= seconds:
                        break
            out[k] = (frames, time.perf_counter() - t0, px)
        except BaseException as e:  # noqa: BLE001
            errs.append(repr(e))
            barrier.abort()

    ts = [threading.Thread(target=work, args=(k,)) for k in range(workers)]
    for t in ts:
        t.start()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        raise RuntimeError(errs[0] if errs else "worker failed")
    go.set()
    for t in ts:
        t.join()
    if errs:
        raise RuntimeError(errs[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--workers", default="1,8")
    ap.add_argument("--ops", default="decompress,unshuffle,sparsify,1d,2d,rings,fazit")
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--cut", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--link", type=float, default=3.125, help="GB/s to saturate (25 GbE line rate)")
    ap.add_argument("--layout", default="interleave")
    ap.add_argument(
        "--backend", default="cpu", choices=["cpu", "gpu"], help="gpu: bslz4_to_sparse.cuda"
    )
    ap.add_argument("--acc", default="f64", help="gpu matrix product accumulator: f64, i64 or f32")
    ap.add_argument("--dot", default=None, help="the dot layout of 1d/2d/rings (default: auto)")
    ap.add_argument("--dset", default=DSET)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    wlist = [int(x) for x in a.workers.split(",")]
    files = sorted(glob.glob(a.files))[a.skip : a.skip + max(wlist)]
    units, shape, dtype = conveyor.make_work_units(files, a.dset, 1 << 30, a.index_dir)
    u0 = units[0]
    with open(u0.path, "rb") as fh:
        fh.seek(u0.start + int(u0.offsets[0]))
        f0 = b.decompress([fh.read(int(u0.lengths[0]))], dtype)[0]
    mask = (f0 != np.iinfo(dtype).max).astype(np.uint8).reshape(shape)
    ops = a.ops.split(",")
    mats = matrices(shape, mask) if set(ops) & {"1d", "2d", "rings", "fazit"} else {}
    npix = int(np.prod(shape))
    for op in ops:
        be = backend(op, dtype, shape, mask, mats, a.cut, a.backend == "gpu", a.batch, a.acc, a.dot)
        if be is None:
            continue
        for W in wlist:
            if W > len(units):
                continue
            cpus = None
            if a.layout != "free":
                nodes = conveyor.numa_nodes()
                cpus = (
                    [n[i] for i in range(max(map(len, nodes))) for n in nodes if i < len(n)]
                    if a.layout == "interleave"
                    else [c for n in nodes for c in n]
                )[:W]
            res = run_point(op, be, units, W, a.seconds, cpus, a.batch)
            fps = [fr / t for fr, t, _ in res]
            mb = [units[k % len(units)].span / units[k % len(units)].n / 1e6 for k in range(W)]
            gbs = [f * m / 1e3 for f, m in zip(fps, mb)]  # compressed GB/s of each worker
            frames = sum(r[0] for r in res)
            px = sum(r[2] for r in res)
            total = sum(gbs)
            per = total / W
            row = dict(
                op=op,
                workers=W,
                fps_per_worker=round(float(np.mean(fps)), 1),
                gbs_per_worker=round(per, 4),
                total_gbs=round(total, 3),
                workers_for_link=round(a.link / per, 1),
                mb_per_frame=round(float(np.mean(mb)), 4),
                sparse_pct=(
                    round(100 * px / frames / npix, 2)
                    if op not in ("decompress", "unshuffle")
                    else None
                ),
                host=socket.gethostname(),
                layout=a.layout,
                tag=a.tag,
                link_gbs=a.link,
                cut=a.cut,
                backend=a.backend,
                dot=a.dot,
                batch=a.batch,
            )
            print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
