#!/usr/bin/env python3
"""Time the conveyor (src/conveyor) on real files: reading, decoding, and the cores they take.

    BSLZ4_TO_SPARSE_PATH=lib python3 tools/bench_conveyor.py --files 'GLOB' --skip 3300 --index-dir tmp/idx \
        --points 8,16,32 --mode threads --layout interleave --tag xeon

--points   numbers of workers, one measurement each; every worker has its own reader thread.
           Every point reads its own files that no other point read (the first call warms the file
           system cache for everything after it), then the list wraps.
--layout   free (the OS places threads), interleave (workers on the CPUs of the list that
           alternates between NUMA nodes) or pack (a node is filled before the next one starts).
--op       sparsify, or 1d (a synthetic radial matrix, 2000 bins).
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


def radial_matrix(shape, nbins=2000):
    import scipy.sparse as sp

    ni, nj = shape
    y, x = np.mgrid[0:ni, 0:nj]
    r = np.hypot(y - ni * 0.55, x - nj * 0.45)
    pos = r / r.max() * (nbins - 1)
    lo = np.minimum(np.floor(pos).astype(np.int64), nbins - 2)
    w = (pos - lo).astype(np.float32)
    npix = ni * nj
    rows = np.concatenate([lo.ravel(), lo.ravel() + 1])
    cols = np.concatenate([np.arange(npix), np.arange(npix)])
    data = np.concatenate([(1 - w).ravel(), w.ravel()]).astype(np.float32)
    return sp.csc_matrix((data, (rows, cols)), shape=(nbins, npix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--points", default="8", help="numbers of workers, comma separated")
    ap.add_argument("--files-per-worker", type=float, default=4)
    ap.add_argument("--mode", default="threads", choices=["threads", "processes"])
    ap.add_argument("--layout", default="interleave")
    ap.add_argument("--op", default="sparsify", choices=["sparsify", "1d"])
    ap.add_argument("--unit-mb", type=int, default=256, help="a unit is a whole file when it fits")
    ap.add_argument(
        "--index-dir", default=None, help="sidecar chunk indexes (tools/index_chunks.py)"
    )
    ap.add_argument(
        "--io-mb", type=int, default=0, help="size of each read call (0: the whole unit)"
    )
    ap.add_argument("--slots", type=int, default=2, help="slots per worker")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cut", type=int, default=0)
    ap.add_argument("--dset", default=DSET)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    allfiles = sorted(glob.glob(a.files))[a.skip :]
    pos = 0
    for point in a.points.split(","):
        nw = int(point)
        nf = max(1, int(round(a.files_per_worker * nw)))
        files = [allfiles[(pos + k) % len(allfiles)] for k in range(nf)]
        pos += nf
        units, shape, dtype = conveyor.make_units(files, a.dset, a.unit_mb << 20, a.index_dir)
        u0 = units[0]  # the mask: pixels at the dtype maximum in the first frame; no HDF5 involved
        with open(u0.path, "rb") as fh:
            fh.seek(u0.start + int(u0.offsets[0]))
            chunk0 = fh.read(int(u0.lengths[0]))
        f0 = b.decompress([chunk0], dtype)[0]
        mask = (f0 != np.iinfo(dtype).max).astype(np.uint8).reshape(shape)
        csc = radial_matrix(shape) if a.op == "1d" else None
        be = conveyor.CpuBackend(mask, dtype, csc=csc, cut=a.cut)
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
