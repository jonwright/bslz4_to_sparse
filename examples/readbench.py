"""
Record and compare decode timings across machines, builds and pipelines.

    times, results = readbench.time_interleaved(runs, repeats=5)
    readbench.record('timings.jsonl', times, nframes=nf, bytes_per_frame=...,
                     cases={...}, data={...})
    readbench.report('timings.jsonl')

record() appends one JSON line per (reader, case) for the current machine
and build, replacing any earlier line with the same machine, sources,
pipeline, data, case and reader (so rerunning a notebook does not pile up
duplicates). report() prints every machine/build/pipeline in the file as its
own table, followed by what a network link would need: the decode here is
single threaded, so with reads overlapped with compute one core keeps up with
the link only if it decodes faster than frames arrive.
"""
from __future__ import print_function

import json
import os
import platform
import socket
import time

import numpy as np

import bslz4_to_sparse as b

_DECOMPRESS = {b.CODEC_LZ4: "lz4", b.CODEC_ZSTD: "zstd"}
_UNTRANSPOSE = dict((v, k) for k, v in b._BACKEND_TO_ID.items())
_DOT = {0: "csc", 1: "csc-fused"}


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


def pipeline(dtype=np.uint16, codec=b.CODEC_LZ4, dot=0):
    """The stages a default chunk2sparseCSCmulti(dtype=dtype) runs (no
    explicit pipeline=), by name."""
    tier = b._collect_tier_for_suffix(b._suffix_for_dtype(dtype))
    dec, unt, col, dt, _opt = b._pipeline_for(codec, tier, dot).tolist()
    return {
        "dtype": np.dtype(dtype).name,
        "decompress": _DECOMPRESS.get(dec, dec),
        "untranspose": _UNTRANSPOSE.get(unt, unt),
        "collect": b._COLLECT_ID_TO_NAME.get(col, col),
        "dot": _DOT.get(dt, dt),
        "dense_sparse_threshold": b.get_dense_sparse_threshold(),
    }


def _key(r):
    return (r["machine"]["host"], r["machine"]["src_sha256"],
            json.dumps(r["pipeline"], sort_keys=True), r["data"]["name"],
            r["batch"], r["case"], r["reader"])


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record(path, times, nframes, bytes_per_frame, cases, data, batch=25,
           dtype=np.uint16, codec=b.CODEC_LZ4):
    """Append the timings {(reader, case): best seconds for nframes} of this
    machine to path. cases = {case: description}; data = {"name": ...,
    anything else describing the frames}. Returns the new records."""
    m, p = machine(), pipeline(dtype, codec)
    when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new = [{"date_utc": when, "machine": m, "pipeline": p, "data": data,
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


def _caption(r):
    m, p = r["machine"], r["pipeline"]
    return ("%s, %s, one core (%d visible)\n"
            "  bslz4_to_sparse %s, git %s, sources %s, %s\n"
            "  pipeline %s: %s decompress, %s untranspose, %s collect, %s dot, "
            "dense/sparse threshold %g\n"
            "  data %s, %.0f kB/frame compressed, batches of %d, %d frames"
            % (m["host"], m["cpu"], m["cores_visible"],
               m["version"], m["git"], m["src_sha256"][:12], m["compiler"],
               p["dtype"], p["decompress"], p["untranspose"], p["collect"], p["dot"],
               p["dense_sparse_threshold"],
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
