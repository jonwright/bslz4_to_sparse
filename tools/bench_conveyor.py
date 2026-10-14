#!/usr/bin/env python3
"""Time the conveyor (src/conveyor) on real files: reading, decoding, and the cores they take.

    BSLZ4_TO_SPARSE_PATH=lib python3 tools/bench_conveyor.py --files 'GLOB' --skip 3300 --index-dir tmp/idx \
        --points 8,16,32 --mode threads --layout interleave --tag xeon

--points   numbers of workers, one measurement each; every worker has its own reader thread.
           Every point reads its own files that no other point read (the first call warms the file
           system cache for everything after it), then the list wraps.
--layout   free (the OS places threads), interleave (workers on the CPUs of the list that
           alternates between NUMA nodes) or pack (a node is filled before the next one starts).
--op       decompress, unshuffle, sparsify, 1d, 1d-nosplit, 2d, rings, fazit (tools/bench_ops.py: the same stages and
           pyFAI matrices); --backend gpu for the CUDA decoder.
One JSON line per point, prefixed RESULT.  Never run it in CI."""

import argparse
import glob
import json
import os
import socket
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite  # noqa: E402,F401  (puts lib/ or $BSLZ4_TO_SPARSE_PATH on sys.path)

import bench_ops as bo  # noqa: E402  (the stages and their matrices)
import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import conveyor  # noqa: E402

DSET = "/entry_0000/ESRF-ID11/eiger/data"


def layout(name, workers):
    """worker_cpus for a layout name, from the CPUs this process may use (the readers stay
    unpinned: they wait for the file system nearly all the time)."""
    if name == "free":
        return None
    nodes = conveyor.numa_nodes()
    if name == "interleave":
        cpus = [n[i] for i in range(max(map(len, nodes))) for n in nodes if i < len(n)]
    elif name == "pack":
        cpus = [c for n in nodes for c in n]
    else:
        raise ValueError(name)
    if len(cpus) < workers:
        raise ValueError("%d workers need %d CPUs, %d allowed" % (workers, workers, len(cpus)))
    return cpus[:workers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--points", default="8", help="numbers of workers, comma separated")
    ap.add_argument("--files-per-worker", type=float, default=4)
    ap.add_argument("--mode", default="threads", choices=["threads", "processes"])
    ap.add_argument("--layout", default="interleave")
    ap.add_argument(
        "--op",
        default="sparsify",
        choices=["decompress", "unshuffle", "sparsify", "1d", "1d-nosplit", "2d", "rings", "fazit"],
    )
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--acc", default="f64", help="gpu matrix accumulator")
    ap.add_argument("--unit-mb", type=int, default=256, help="a unit is a whole file when it fits")
    ap.add_argument(
        "--index-dir", default=None, help="sidecar chunk indexes (tools/index_chunks.py)"
    )
    ap.add_argument(
        "--io-mb", type=int, default=0, help="size of each read call (0: the whole unit)"
    )
    ap.add_argument("--slots", type=int, default=2, help="slots per worker")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cut", type=int, default=1)
    ap.add_argument("--dset", default=DSET)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--example-geometry",
        action="store_true",
        help="matrix ops on frames of another run: the example geometry, matrices built for the "
        "frames' own mask",
    )
    a = ap.parse_args()
    allfiles = sorted(glob.glob(a.files))[a.skip :]
    pos = 0
    for point in a.points.split(","):
        nw = int(point)
        nf = max(1, int(round(a.files_per_worker * nw)))
        files = [allfiles[(pos + k) % len(allfiles)] for k in range(nf)]
        pos += nf
        units, shape, dtype = conveyor.make_work_units(files, a.dset, a.unit_mb << 20, a.index_dir)
        u0 = units[0]  # the mask: pixels at the dtype maximum in the first frame; no HDF5 involved
        with open(u0.path, "rb") as fh:
            fh.seek(u0.start + int(u0.offsets[0]))
            chunk0 = fh.read(int(u0.lengths[0]))
        f0 = b.decompress([chunk0], dtype)[0]
        mask = (f0 != np.iinfo(dtype).max).astype(np.uint8).reshape(shape)
        mats = bo.matrices(shape, mask, a.example_geometry) if a.op in ("1d", "1d-nosplit", "2d", "rings", "fazit") else {}
        be = bo.backend(a.op, dtype, shape, mask, mats, a.cut, a.backend == "gpu", a.batch)
        if be is None:
            raise SystemExit("%s is not available on the %s backend" % (a.op, a.backend))
        wcpu = layout(a.layout, nw)
        r = conveyor.run(
            units,
            be,
            nw,
            mode=a.mode,
            batch=a.batch,
            slots_per_worker=a.slots,
            worker_cpus=wcpu,
            io_bytes=a.io_mb << 20,
        )
        r.update(
            host=socket.gethostname(),
            tag=a.tag,
            layout=a.layout,
            op=a.op,
            fps_per_worker_total=round(r["fps"] / nw, 1),
            nfiles=len(files),
            unit_mb=a.unit_mb,
            io_mb=a.io_mb,
            skip=a.skip,
            first_file=files[0],
            last_file=files[-1],
            compressed_mb_per_frame=round(
                sum(u.span for u in units) / sum(u.n for u in units) / 1e6, 4
            ),
        )
        print("RESULT " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
