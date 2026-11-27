#!/usr/bin/env python3
"""Write the chunk-index sidecar (src/conveyor/_index.py) for HDF5 files: once per file.

    BSLZ4_TO_SPARSE_PATH=lib python3 tools/index_chunks.py --files 'GLOB' --dset /entry_0000/... \
        [--index-dir DIR] [--procs 16] [--force] [--check] [--limit N]

HDF5 (through h5py) is read here and nowhere else: it takes a global lock, so the files are shared
out over plain subprocesses (this script again, each given its share of the files on stdin, one
per line, as argv has a size limit), not threads.  No multiprocessing: its semaphores and temporary
files in /dev/shm and /tmp can collide between jobs on a node.  Nothing comes back but the exit
status: a subprocess exits 0 when every file of its share is done, prints what failed to stderr,
and the sidecars are then found on disk.  Each file
costs a few hundred ms cold (dozens of small reads of the B-tree nodes spread through it, not the
data).  --index-dir keeps the sidecars away from read-only data directories.  --check verifies
existing sidecars against a fresh harvest."""

import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite  # noqa: E402,F401  (puts lib/ or $BSLZ4_TO_SPARSE_PATH on sys.path)

from bslz4_to_sparse import conveyor  # noqa: E402


def fresh(path, index_dir):
    """True when path has a sidecar that matches the file as it is now; False when it has none, or
    one that is stale, truncated, corrupt or of another version."""
    try:
        conveyor.read_index(path, index_dir)
        return True
    except Exception:  # noqa: BLE001 (any sidecar that cannot be read is rebuilt)
        return False


def one(path, dset, index_dir, check):
    """Write (or with check, verify) the sidecar of one file; raises when that fails."""
    if check:
        have, new = conveyor.read_index(path, index_dir), conveyor.harvest(path, dset)
        if not (
            np.array_equal(have.offsets, new.offsets)
            and np.array_equal(have.lengths, new.lengths)
            and have.shape == new.shape
            and have.codec == new.codec
        ):
            raise ValueError("the sidecar differs from a fresh harvest")
    else:
        conveyor.write_index(conveyor.harvest(path, dset), path, index_dir)


def worker(argv):
    """A subprocess: argv is dset, index_dir ("" for none), check ("0"/"1"); its files come on
    stdin, one per line.  Exits 0 when all of them are done, else 1."""
    dset, index_dir, check = argv[0], argv[1] or None, argv[2] == "1"
    paths = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    status = 0
    for p in paths:
        try:
            one(p, dset, index_dir, check)
        except Exception as e:  # noqa: BLE001 (report per file, carry on)
            print("index_chunks: %s: %r" % (p, e), file=sys.stderr)
            status = 1
    return status


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
    todo = paths if (a.check or a.force) else [p for p in paths if not fresh(p, a.index_dir)]
    n = max(1, min(a.procs, len(todo)))
    head = [sys.executable, os.path.abspath(__file__), "--worker", a.dset, a.index_dir or "", "1" if a.check else "0"]
    shares = [todo[i::n] for i in range(n) if todo[i::n]]
    procs = [subprocess.Popen(head, stdin=subprocess.PIPE, universal_newlines=True) for _ in shares]
    for p, share in zip(procs, shares):  # all started first: each reads its share after its imports
        p.stdin.write("".join(f + "\n" for f in share))
        p.stdin.close()
    failed = sum(p.wait() != 0 for p in procs)
    bad = [p for p in todo if not fresh(p, a.index_dir)]
    done = "checked" if a.check else "written"
    print(
        {"kept": len(paths) - len(todo), done: len(todo) - len(bad), "missing": len(bad)},
        "%.1f s for %d files in %d processes" % (time.perf_counter() - t0, len(paths), len(procs)),
        "; %d processes failed (errors above)" % failed if failed else "",
    )
    for p in bad[:10]:
        print("  ", p)
    return 1 if (bad or failed) else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        sys.exit(worker(sys.argv[2:]))
    sys.exit(main())
