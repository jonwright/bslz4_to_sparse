"""
Record and compare decode timings across machines, builds and pipelines.

    times, results = readbench.time_interleaved(runs, repeats=5)
    readbench.record('timings.jsonl', times, nframes=nf, bytes_per_frame=...,
                     cases={...}, data={...}, pipeline_used=integ.pipeline)
    readbench.report('timings.jsonl')

    with readbench.pinned() as cpu:   # like `taskset -c cpu` for the timed block
        ...

record() appends one JSON line per (reader, case) for the current machine
and build, replacing any earlier line with the same machine, sources,
pipeline, data, case and reader (so rerunning a notebook does not pile up
duplicates). report() prints every machine/build/pipeline in the file as its
own table, followed by what a network link would need: the decode here is
single threaded, so with reads overlapped with compute one core keeps up with
the link only if it decodes faster than frames arrive.
"""
from __future__ import print_function

import contextlib
import json
import os
import platform
import socket
import time

import numpy as np

import bslz4_to_sparse as b



@contextlib.contextmanager
def pinned(cpu=None):
    """Run the block on one CPU, like `taskset -c cpu`: the decode is single
    threaded, and pinning stops the scheduler migrating it between cores (and
    caches) in the middle of a timing.  cpu=None takes the highest-numbered
    CPU this process may use (core 0 tends to service more interrupts).
    Restores the previous affinity afterwards.  Yields the CPU, or None where
    the affinity cannot be set (not Linux)."""
    if not hasattr(os, "sched_setaffinity"):
        yield None
        return
    old = os.sched_getaffinity(0)
    cpu = max(old) if cpu is None else int(cpu)
    os.sched_setaffinity(0, {cpu})
    try:
        yield cpu
    finally:
        os.sched_setaffinity(0, old)


def time_interleaved(runs, repeats=5):
    """Run each of {key: fn} `repeats` times, round-robin so slow drift of a
    shared machine hits every key alike. Returns ({key: best seconds},
    {key: result of the last call})."""
    times = dict((k, []) for k in runs)
    results = {}
    for _ in range(repeats):
        for k, run in runs.items():
            t0 = time.perf_counter()
            results[k] = run()
            times[k].append(time.perf_counter() - t0)
    return dict((k, min(v)) for k, v in times.items()), results


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def machine():
    """Where and with what code the timings were taken."""
    info = b.build_info()
    return {
        "host": socket.gethostname(),
        "cpu": _cpu_model(),
        "cores_visible": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                         else os.cpu_count(),
        "version": info["version"],
        "git": info["git"],
        "src_sha256": info["src_sha256"],
        "compiler": info["compiler"],
    }


def pipeline(p, dtype=np.uint16):
    """The step names of a resolved pipeline (an object's .pipeline)."""
    d = {"dtype": np.dtype(dtype).name}
    d.update((k, v) for k, v in b.describe(p).items() if v is not None)
    return d


def _key(r):
    return (r["machine"]["host"], r["machine"]["src_sha256"],
            json.dumps(r["pipeline"], sort_keys=True), r["data"]["name"],
            r["batch"], r["case"], r["reader"])


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record(path, times, nframes, bytes_per_frame, cases, data, pipeline_used, batch=25,
           dtype=np.uint16):
    """Append the timings {(reader, case): best seconds for nframes} of this
    machine to path. cases = {case: description}; data = {"name": ...,
    anything else describing the frames}; pipeline_used = the .pipeline of
    the object that was timed, or {case: .pipeline} when the cases used
    different objects; recorded by step name. Returns the new records."""
    m = machine()
    if isinstance(pipeline_used, dict):
        ppc = dict((c, pipeline(p, dtype)) for c, p in pipeline_used.items())
    else:
        ppc = dict((c, pipeline(pipeline_used, dtype)) for c in cases)
    when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new = [{"date_utc": when, "machine": m, "pipeline": ppc[case], "data": data,
            "bytes_per_frame": float(bytes_per_frame), "nframes": int(nframes),
            "batch": batch, "case": case, "case_desc": cases[case], "reader": reader,
            "ms_per_frame": 1e3 * t / nframes}
           for (reader, case), t in times.items()]
    keys = set(_key(r) for r in new)
    kept = [r for r in load(path) if _key(r) not in keys]
    with open(path, "w") as f:
        for r in kept + new:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return new


def _steps(p):
    """'decode lz4-band, mask none, ...' from the steps present in p (old
    records say decompress for decode and carry a dense_sparse_threshold)."""
    q = dict(("decode" if k == "decompress" else k, v) for k, v in p.items())
    order = ["decode", "mask", "untranspose", "collect", "route", "dot"]
    return ", ".join("%s %s" % (k, q[k]) for k in order if q.get(k) is not None)


def _caption(r):
    m, p = r["machine"], r["pipeline"]
    return ("%s, %s, one core (%d visible)\n"
            "  bslz4_to_sparse %s, git %s, sources %s, %s\n"
            "  pipeline %s: %s\n"
            "  data %s, %.0f kB/frame compressed, batches of %d, %d frames"
            % (m["host"], m["cpu"], m["cores_visible"],
               m["version"], m["git"], m["src_sha256"][:12], m["compiler"],
               p["dtype"], _steps(p),
               r["data"]["name"], r["bytes_per_frame"] / 1e3, r["batch"], r["nframes"]))


def report(path, links_gbps=(10, 25)):
    """Print each machine/build/pipeline/data group in path as a table of
    ms/frame (readers x cases), then the network budget for the fastest
    reader of each case."""
    groups = {}
    for r in load(path):
        g = (r["machine"]["host"], r["machine"]["src_sha256"],
             json.dumps(r["pipeline"], sort_keys=True), r["data"]["name"], r["batch"])
        groups.setdefault(g, []).append(r)
    for rows in groups.values():
        cases = list(dict.fromkeys(r["case"] for r in rows))
        readers = list(dict.fromkeys(r["reader"] for r in rows))
        ms = dict(((r["reader"], r["case"]), r["ms_per_frame"]) for r in rows)
        print(_caption(rows[0]))
        print("  %-34s" % "ms/frame" + "".join("%9s" % c for c in cases))
        for rd in readers:
            print("  %-34s" % rd + "".join(
                "%9.2f" % ms[rd, c] if (rd, c) in ms else "%9s" % "-" for c in cases))
        nbytes = rows[0]["bytes_per_frame"]
        print("  network, nominal line rate, reads overlapped with compute,"
              " fastest reader per case:")
        best = dict((c, min(v for (rd, cc), v in ms.items() if cc == c)) for c in cases)
        print("  %-34s" % "frames/s on one core" + "".join("%9.0f" % (1e3 / best[c]) for c in cases))
        for gbps in links_gbps:
            link_fps = gbps * 1e9 / 8 / nbytes
            print("  %-34s" % ("cores for %d GbE (%.0f frames/s)" % (gbps, link_fps)) + "".join(
                "%9.1f" % (link_fps * best[c] / 1e3) for c in cases))
        print()
