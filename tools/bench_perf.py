"""Hardware-counter and profile measurements of a few matrix-dot kernels.

Data sets (per detector and beam centre, cached like tools/bench_suite.py):
  sparse  < 0.1 % of pixels active: a weak background (Poisson 2e-4) plus
          ~50 Bragg spots per frame (2 px wide, peak ~1000 counts)
  medium  bench_suite.py's frames: LaB6 rings with texture, ~7-9 % active
  dense   every pixel counts: background ~30 plus the rings (x20), so the
          compression is low and the default route is dense

Each configuration (data, matrix case, dot, cut) runs in its own process:
the frames and the matrix are loaded and the integrator built and warmed up
first, and only then is perf counting switched on (perf --control fifo), so
the counters cover the timed decode+dot loop and nothing else.

  cut "max"  : threshold above the dtype maximum -> no sparse output (powder only)
  cut "on"   : the sparse output at a level that makes sense for the data
               (sparse: every non-zero pixel; medium: > 4; dense: > 200)
  dot "none" : the plain sparsify (no matrix) -- decompress + untranspose
               (+ the collect when cut is "on")

Usage:
  python3 tools/bench_perf.py sweep [--det 4M] [--centre mid] [--cpu 7]
        [--out examples/perf_suite.jsonl]   # perf stat over the grid
  python3 tools/bench_perf.py profile ...   # perf record: time per symbol
  python3 tools/bench_perf.py run ...       # one configuration (used by the above)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite as bs  # noqa: E402  (also puts lib/ on sys.path)

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import hdf5plugin  # noqa: E402
import bslz4_to_sparse as b  # noqa: E402

DATASETS = ("sparse", "medium", "dense")
CASES = ("1D bbox x1", "1D no x1", "rings")
DOTS = {"1D bbox x1": ("csc", "csc-run", "csc-run-int", "padded-sse2", "padded-avx2", "bsb-csr"),
        "1D no x1": ("csc", "csc-nosplit", "padded-sse2", "bsb-csr-nosplit"),
        "rings": ("csc", "padded-sse2", "padded-avx2", "bsb-csr")}
CUT_ON = {"sparse": 0, "medium": 4, "dense": 200}
EVENTS = ("cycles:u", "instructions:u", "ex_ret_brn_misp:u",
          "ls_any_fills_from_sys.local_l2:u", "ls_any_fills_from_sys.local_ccx:u",
          "ls_any_fills_from_sys.dram_io_near:u", "ls_any_fills_from_sys.dram_io_far:u")


def frames_for(det, centre, kind, ai):
    if kind == "medium":
        return bs.frames_file(det, centre, ai)
    fname = os.path.join(bs.CACHE, "frames_%s_%s_%s.h5" % (det, centre, kind))
    if os.path.exists(fname):
        return fname
    shape = ai.detector.shape
    bad = ai.detector.mask.astype(bool)
    rng = np.random.default_rng(1)
    if kind == "dense":
        from pyFAI.calibrant import get_calibrant
        cal = get_calibrant("LaB6")
        cal.wavelength = ai.wavelength
        qarr, chi = ai.qArray(), ai.chiArray()
        mean = np.full(shape, 30.0)
        for k, qk in enumerate(cal.get_peaks("q_nm^-1")[:60]):
            mean += 20 * 4.0 / (1 + 0.3 * k) * (1 + 0.5 * np.cos(2 * chi + 0.4 * k)) * \
                np.exp(-0.5 * ((qarr - qk) / 0.15) ** 2)
    tmp = fname + ".tmp"
    with h5py.File(tmp, "w") as h5f:
        ds = h5f.create_dataset("data", shape=(bs.NFRAMES[det],) + shape, dtype=np.uint16,
                                chunks=(1,) + shape,
                                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        yy, xx = np.mgrid[-4:5, -4:5]
        spot = np.exp(-0.5 * (xx ** 2 + yy ** 2) / 2.0 ** 2)
        for i in range(bs.NFRAMES[det]):
            if kind == "dense":
                frame = rng.poisson(mean)
            else:
                frame = rng.poisson(2e-4, shape)
                for _ in range(50):
                    ci = rng.integers(5, shape[0] - 5)
                    cj = rng.integers(5, shape[1] - 5)
                    frame[ci - 4:ci + 5, cj - 4:cj + 5] += rng.poisson(1000 * spot)
            frame = np.minimum(frame, 65535).astype(np.uint16)
            frame[bad] = 0
            ds[i] = frame
    os.rename(tmp, fname)
    return fname


def matrix_for(ai, det, centre, case):
    if case.startswith("1D "):
        _, split, bpp = case.split()
        return bs._cached("m1d_%s_%s_%s_%s" % (det, centre, split, bpp[1:]), None)
    return dict(bs.build_cases(ai, det, centre, {"rings"}))[case]


def cmd_run(a):
    os.sched_setaffinity(0, {a.cpu})
    ai = bs.make_ai(a.det, a.centre)
    valid = np.ascontiguousarray((1 - ai.detector.mask).astype(np.uint8))
    chunks, f0 = bs.load_chunks(frames_for(a.det, a.centre, a.data, ai))
    chunks = chunks[:a.frames]
    cut = 70000 if a.cut == "max" else CUT_ON[a.data]
    if a.dot == "none":
        integ = b.chunk2sparse(valid, dtype=np.uint16).multi
    else:
        integ = b.chunk2sparseCSC(valid, matrix_for(ai, a.det, a.centre, a.case),
                                  dtype=np.uint16, pipeline={"dot": a.dot}).multi
    nf = len(chunks)
    integ(chunks[:bs.BATCH], cut)                       # warmup
    ctl = open(a.ctl, "w") if a.ctl else None
    ack = open(a.ack, "r") if a.ack else None
    if ctl:
        ctl.write("enable\n")
        ctl.flush()
        ack.readline()
    t0 = time.perf_counter()
    npx = 0
    reps = 0
    while True:
        for s in range(0, nf, bs.BATCH):
            r = integ(chunks[s:s + bs.BATCH], cut)
            npx += int(r[0].sum())
        reps += 1
        if time.perf_counter() - t0 > a.seconds:
            break
    t = time.perf_counter() - t0
    if ctl:
        ctl.write("disable\n")
        ctl.flush()
        ack.readline()
    comp = sum(len(c) for c in chunks) / nf
    print(json.dumps({"det": a.det, "centre": a.centre, "data": a.data, "case": a.case,
                      "dot": a.dot, "cut": a.cut, "frames": nf * reps,
                      "ms_per_frame": t * 1e3 / (nf * reps),
                      "nonzero_frac": float(np.count_nonzero(f0) / f0.size),
                      "compression": float(f0.nbytes / comp),
                      "sparse_out_frac": npx / (nf * reps) / f0.size}))


def _perf(a, conf, mode, outfile=None):
    """Run one configuration under perf stat/record with counting only in the
    timed loop.  Returns (the run's JSON dict, perf output text)."""
    d = tempfile.mkdtemp()
    ctl, ack = os.path.join(d, "ctl"), os.path.join(d, "ack")
    os.mkfifo(ctl)
    os.mkfifo(ack)
    run = [sys.executable, os.path.abspath(__file__), "run", "--det", a.det, "--centre", a.centre,
           "--cpu", str(a.cpu), "--frames", str(a.frames), "--seconds", str(a.seconds),
           "--data", conf["data"], "--case", conf["case"], "--dot", conf["dot"], "--cut", conf["cut"],
           "--ctl", ctl, "--ack", ack]
    if mode == "stat":
        perf = ["perf", "stat", "-x,", "-D", "-1", "--control", "fifo:%s,%s" % (ctl, ack),
                "-e", ",".join(EVENTS), "--"]
    else:
        perf = ["perf", "record", "-q", "-e", "cycles:u", "-F", "2000", "-D", "-1",
                "--control", "fifo:%s,%s" % (ctl, ack), "-o", outfile, "--"]
    p = subprocess.run(perf + run, capture_output=True, text=True)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            res = json.loads(line)
    if res is None:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return res, p.stderr


def _grid(a):
    for data in a.data.split(","):
        for case in a.cases.split(","):
            for dot in ("none",) + DOTS[case]:
                if a.dots and dot not in a.dots.split(","):
                    continue
                if dot == "none" and case != a.cases.split(",")[0]:
                    continue
                for cut in ("max", "on"):
                    yield {"data": data, "case": "-" if dot == "none" else case,
                           "dot": dot, "cut": cut}


def cmd_sweep(a):
    out = open(a.out, "a")
    print("%-6s %-11s %-16s %-5s %8s %6s %7s %7s %8s %8s %8s %6s" % (
        "data", "case", "dot", "cut", "ms/frm", "IPC", "brmis/k", "L2fl/k", "L3fl/k", "DRAMfl/k",
        "GB/s mem", "comp"))
    for conf in _grid(a):
        conf["case"] = CASES[0] if conf["case"] == "-" else conf["case"]
        res, err = _perf(a, conf, "stat")
        ev = {}
        for line in err.splitlines():
            f = line.split(",")
            if len(f) > 3 and f[2] in EVENTS:
                try:
                    ev[f[2]] = float(f[0])
                except ValueError:
                    ev[f[2]] = float("nan")
        cyc, ins = ev.get("cycles:u", 1), ev.get("instructions:u", 0)
        kins = ins / 1e3 if ins else 1
        dram = ev.get("ls_any_fills_from_sys.dram_io_near:u", 0) + \
            ev.get("ls_any_fills_from_sys.dram_io_far:u", 0)
        secs = res["ms_per_frame"] * res["frames"] / 1e3
        res.update({"events": ev, "ipc": ins / cyc,
                    "dram_gbs": dram * 64 / secs / 1e9})
        if conf["dot"] == "none":
            res["case"] = "-"
        print("%-6s %-11s %-16s %-5s %8.2f %6.2f %7.2f %7.1f %8.1f %8.2f %8.2f %6.1f" % (
            res["data"], res["case"][:11], res["dot"], res["cut"], res["ms_per_frame"],
            res["ipc"], ev.get("ex_ret_brn_misp:u", 0) / kins,
            ev.get("ls_any_fills_from_sys.local_l2:u", 0) / kins,
            ev.get("ls_any_fills_from_sys.local_ccx:u", 0) / kins, dram / kins,
            res["dram_gbs"], res["compression"]))
        sys.stdout.flush()
        out.write(json.dumps(res) + "\n")
        out.flush()


GROUPS = (("lz4", ("LZ4_",)),
          ("untranspose", ("bitshuf", "bshuf", "kcb", "untranspose")),
          ("collect", ("collect_nz", "collect_gt", "bslz4_sparse_u", "bslz4_sparse_i",
                       "bslz4_sparse_f")),
          ("dot", ("bslz4_sparse_dot_", "bslz4v::", "padded", "bsbcsr", "nosplit_", "run_dense",
                   "run_sparse")),
          ("driver", ("bslz4_driver", "bslz4_sparsify")),
          ("memset/memcpy", ("memset", "memcpy", "memmove")))


def _symbols(datafile):
    """[(percent, full symbol name)] from perf report, one row per symbol."""
    rep = subprocess.run(["perf", "report", "-i", datafile, "--stdio", "--sort", "sym",
                          "-F", "overhead,sym", "-t", "|", "--percent-limit", "0.1"],
                         capture_output=True, text=True).stdout
    out = []
    for line in rep.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split("|", 1)
        if len(f) != 2:
            continue
        try:
            pct = float(f[0].strip().rstrip("%"))
        except ValueError:
            continue
        sym = f[1].strip()
        if sym.startswith("[.] ") or sym.startswith("[k] "):
            sym = sym[4:]
        out.append((pct, sym))
    return out


def cmd_profile(a):
    for conf in _grid(a):
        conf["case"] = CASES[0] if conf["case"] == "-" else conf["case"]
        fd, data = tempfile.mkstemp(suffix=".perf")
        os.close(fd)
        res, _err = _perf(a, conf, "record", outfile=data)
        top = _symbols(data)
        os.unlink(data)
        share = {g: 0.0 for g, _ in GROUPS}
        share["other"] = 0.0
        for pct, sym in top:
            for g, keys in GROUPS:
                if any(k in sym for k in keys):
                    share[g] += pct
                    break
            else:
                share["other"] += pct
        ms = res["ms_per_frame"]
        print("%-6s %-11s %-16s %-4s %6.2f ms | " % (conf["data"], "-" if conf["dot"] == "none"
                                                     else conf["case"][:11], conf["dot"],
                                                     conf["cut"], ms)
              + "  ".join("%s %.2f" % (g, ms * p / 100) for g, p in share.items() if p > 0.5))
        if a.verbose:
            for pct, sym in top[:8]:
                print("        %5.1f%% %s" % (pct, sym[:110]))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "sweep", "profile"))
    ap.add_argument("--det", default="4M")
    ap.add_argument("--centre", default="mid")
    ap.add_argument("--cpu", type=int, default=7)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=2.0, help="minimum timed seconds")
    ap.add_argument("--data", default=",".join(DATASETS))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--case", default=CASES[0])
    ap.add_argument("--dot", default="csc")
    ap.add_argument("--cut", default="max")
    ap.add_argument("--ctl")
    ap.add_argument("--ack")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dots", default="", help="limit the grid to these dots (comma list)")
    ap.add_argument("--out", default=os.path.join(bs.REPO, "examples", "perf_suite.jsonl"))
    a = ap.parse_args()
    {"run": cmd_run, "sweep": cmd_sweep, "profile": cmd_profile}[a.mode](a)


if __name__ == "__main__":
    main()
