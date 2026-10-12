#!/usr/bin/env python3
"""Write the chunk-index sidecar (src/conveyor/_index.py) for HDF5 files: once per file.

    BSLZ4_TO_SPARSE_PATH=lib python3 tools/index_chunks.py --files 'GLOB' --dset /entry_0000/... \
        [--index-dir DIR] [--procs 16] [--force] [--check]

HDF5 (through h5py) is read here and nowhere else: it takes a global lock, so the files are shared
out over forked processes, not threads.  Each file costs a few hundred ms cold (dozens of small
reads of the B-tree nodes spread through it, not the data).  --index-dir keeps the sidecars away
from read-only data directories.  --check verifies existing sidecars against a fresh harvest."""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite  # noqa: E402,F401  (puts lib/ or $BSLZ4_TO_SPARSE_PATH on sys.path)

from bslz4_to_sparse import conveyor  # noqa: E402


def one(args):
    path, dset, index_dir, force, check = args
    try:
        t = time.perf_counter()
        sc = conveyor.sidecar_path(path, index_dir)
        if check:
            have = conveyor.read_index(path, index_dir)
            fresh = conveyor.harvest(path, dset)
            ok = (
                np.array_equal(have.offsets, fresh.offsets)
                and np.array_equal(have.lengths, fresh.lengths)
                and have.shape == fresh.shape
                and have.codec == fresh.codec
            )
            return path, "ok" if ok else "DIFFERS", time.perf_counter() - t
        if os.path.exists(sc) and not force:
            try:
                conveyor.read_index(path, index_dir)
                return path, "kept", 0.0
            except conveyor.StaleIndex:
                pass
        conveyor.write_index(conveyor.harvest(path, dset), path, index_dir)
        return path, "written", time.perf_counter() - t
    except Exception as e:  # noqa: BLE001 (report per file, carry on)
        return path, "ERROR %r" % (e,), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--dset", default="/entry_0000/ESRF-ID11/eiger/data")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files")
    a = ap.parse_args()
    paths = sorted(glob.glob(a.files))
    if a.limit:
        paths = paths[: a.limit]
    t0 = time.perf_counter()
    work = [(p, a.dset, a.index_dir, a.force, a.check) for p in paths]
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(one, work, chunksize=1)
    bad = [r for r in res if r[1] not in ("written", "kept", "ok")]
    from collections import Counter

    print(
        dict(Counter(r[1].split(" ")[0] for r in res)),
        "%.1f s for %d files" % (time.perf_counter() - t0, len(paths)),
    )
    for p, s, _ in bad[:10]:
        print("  ", p, s)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
