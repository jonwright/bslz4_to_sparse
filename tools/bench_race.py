#!/usr/bin/env python3
"""The conveyor drag race shown on the performance page: CPU and GPU decoders on real files.

    BSLZ4_TO_SPARSE_PATH=lib python3 tools/bench_race.py --files GLOB --nfiles 250 --label TEXT \
        --index-dir DIR --cpus 12-23 --out docs/bench/conveyor_<cpu>.json

Every point decodes the same NFILES files (the first of GLOB, sorted), read with the conveyor
(tools/bench_conveyor.py, one process per point) on the CPUs of --cpus, workers and readers
alike.  The order is fixed: the GPU points twice (--gpu-workers each, --ops), the CPU points
(--cpu-workers each, the same ops), then the GPU points twice again, so that whatever a file
system caches shows as a difference between the first and the last GPU passes.  Each point
records frames/s, the compressed GB/s read and the cores the workers used; the JSON adds the
machine, the GPU, the network link speed, the files, and whether they are in memory (in /dev/shm:
copy them there to time the decoders without the file system).  Needs the chunk indexes
(tools/index_chunks.py) and, for the GPU points, cupy.  The matrix operations use matrices built
for the frames' own mask (the pixels at the dtype maximum, tools/bench_conveyor.py) with the
example data's approximate geometry (tools/calib.poni): the work of an integration of this
detector and mask, not a calibrated pattern of these files.  Never run it in CI."""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_docs as bd  # noqa: E402  (machine(); puts lib/ or $BSLZ4_TO_SPARSE_PATH on sys.path)


def cpu_list(text):
    """'12-23,30' -> [12, ..., 23, 30]"""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def link_gbps():
    """The fastest network link of the machine, Gb/s (None if /sys does not say)."""
    best = None
    for p in glob.glob("/sys/class/net/*/speed"):
        try:
            with open(p) as fh:
                v = int(fh.read())
        except (OSError, ValueError):
            continue
        if v > 0 and "/lo/" not in p:
            best = max(best or 0, v)
    return None if best is None else best / 1000.0


def gpu_name():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:  # noqa: BLE001  (no cupy, no GPU)
        return None


def point(a, cpus, backend, op, workers, rep):
    cmd = [sys.executable, os.path.join(HERE, "bench_conveyor.py"), "--files", a.files,
           "--index-dir", a.index_dir, "--points", str(workers),
           "--files-per-worker", repr(a.nfiles / float(workers)), "--backend", backend,
           "--op", op, "--layout", "interleave", "--dset", a.dset, "--example-geometry"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       preexec_fn=lambda: os.sched_setaffinity(0, cpus))
    lines = [ln for ln in r.stdout.decode().splitlines() if ln.startswith("RESULT ")]
    if r.returncode or len(lines) != 1:
        raise SystemExit("point %s %s %d failed:\n%s" % (backend, op, workers, r.stdout.decode()))
    res = json.loads(lines[0][len("RESULT "):])
    keep = ("workers", "frames", "wall_s", "fps", "in_bytes", "in_gbs", "out_over_in",
            "worker_cpu_cores", "busy_frac", "nfiles", "compressed_mb_per_frame")
    rec = dict({k: res[k] for k in keep}, backend=backend, op=op, rep=rep)
    print("%-5s %-3s %-8s %2d workers: %7.0f frames/s  %5.2f GB/s in  %5.1f s"
          % (rep, backend, op, workers, rec["fps"], rec["in_gbs"], rec["wall_s"]), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", required=True, help="glob of the input files")
    ap.add_argument("--nfiles", type=int, default=250)
    ap.add_argument("--dset", default="/entry_0000/ESRF-ID11/eiger/data")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--cpus", required=True, help="CPUs for every point, e.g. 12-23")
    ap.add_argument("--cpu-workers", default="4,8,12")
    ap.add_argument("--gpu-workers", default="1,2,4")
    ap.add_argument("--ops", default="sparsify,1d-nosplit")
    ap.add_argument("--out", required=True, help="docs/bench/conveyor_<cpu>.json")
    ap.add_argument("--label", required=True, help="what the files are, for the page")
    a = ap.parse_args()
    cpus = cpu_list(a.cpus)
    files = sorted(glob.glob(a.files))[: a.nfiles]
    if len(files) < a.nfiles:
        raise SystemExit("%d files match, %d asked" % (len(files), a.nfiles))
    ops = a.ops.split(",")
    gw = [int(w) for w in a.gpu_workers.split(",")] if gpu_name() else []
    cw = [int(w) for w in a.cpu_workers.split(",")]
    out = {"schema": 1, "tool": "tools/bench_race.py",
           "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "machine": dict(bd.machine(), cores_usable=len(cpus)), "gpu": gpu_name(),
           "link_gbps": link_gbps(),
           "data": {"label": a.label, "files": len(files),
                    "bytes": sum(os.path.getsize(f) for f in files),
                    "first": files[0], "last": files[-1], "dataset": a.dset,
                    "in_memory": os.path.realpath(files[0]).startswith("/dev/shm/")},
           "results": []}
    plan = ([("gpu%d" % k, "gpu", op, w) for k in (1, 2) for op in ops for w in gw]
            + [("cpu", "cpu", op, w) for op in ops for w in cw]
            + [("gpu%d" % k, "gpu", op, w) for k in (3, 4) for op in ops for w in gw])
    for rep, backend, op, w in plan:
        out["results"].append(point(a, cpus, backend, op, w, rep))
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
