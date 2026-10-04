#!/usr/bin/env python3
"""
Compiler suite: build the extension with gcc, clang and zig, each with and
without profile-guided optimisation (PGO), test every build, and time them
on the same real/synthetic frames, interleaved, on one pinned core.

The question it answers: are the timings a property of the code, or of the
compiler (and of where the code happens to land in memory)?  A large spread
between compilers, or between PGO and not, on a case says that case's C is
sensitive to how it is compiled; a small one says the result is intrinsic.

Everything goes to $BSLZ4_SUITE_DIR/<host>-<arch>/ (default
/tmp_14_days/$USER/bslz4_compiler_suite), so several machines that see the
same disk each keep their own results:

  libs/<variant>/bslz4_to_sparse/   the six builds (gcc, gcc-pgo, clang,
                                    clang-pgo, zig, zig-pgo); a compiler that
                                    is missing is skipped and reported
  obj/<variant>/, pgo/<variant>/    objects and profiles
  test_<variant>.log                pytest output per build
  bench.jsonl, summary.txt          timings (one JSON line per case/build/round)

Usage (one line, from any machine that sees this repo and the data; the
launcher picks the jupyter-slurm python of the machine's architecture from
cvmfs -- x86_64 2025.04.6, ppc64le 2023.10.11 -- no module load needed):

  sh /home/esrf/wright/git/bslz4_to_sparse_llm/tools/compiler_suite.sh all

  sh tools/compiler_suite.sh build   [--variants gcc,clang-pgo,...]
  sh tools/compiler_suite.sh test
  sh tools/compiler_suite.sh bench   [--rounds 3] [--secs 1.0] [--cpu N]
  sh tools/compiler_suite.sh summary

Data: $BSLZ4_SUITE_DATA (default /tmp_14_days/wright/bslz4_bench_shared, see
its README.txt).  Tools: $BSLZ4_SUITE_TOOLS (default /tmp_14_days/wright/
tools) holds zig 0.16 (zig-<arch>-linux-0.16.0/, used through tools/zig-cc
and tools/zig-c++) and, for zig PGO on x86_64, LLVM 21's llvm-profdata and
profile runtime (LLVM-21.1.0-Linux-X64/).  The PGO training run is
the benchmark workload itself (one short pass of every case), so the PGO
numbers are a best case for that workload.
"""
import argparse
import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
USER = os.environ.get("USER", "user")
SUITE = os.environ.get("BSLZ4_SUITE_DIR", "/tmp_14_days/%s/bslz4_compiler_suite" % USER)
DATA = os.environ.get("BSLZ4_SUITE_DATA", "/tmp_14_days/wright/bslz4_bench_shared")
TOOLS = os.environ.get("BSLZ4_SUITE_TOOLS", "/tmp_14_days/wright/tools")
HOST = "%s-%s" % (socket.gethostname().split(".")[0], platform.machine())
OUT = os.path.join(SUITE, HOST)
ALL_VARIANTS = ("gcc", "gcc-pgo", "clang", "clang-pgo", "zig", "zig-pgo")


# ------------------------------------------------------------------ compilers

def _which(cmd):
    return shutil.which(cmd) if not os.path.isabs(cmd) else (cmd if os.access(cmd, os.X_OK) else None)


def _version(cmd):
    try:
        return subprocess.check_output([cmd, "--version"], stderr=subprocess.STDOUT
                                       ).decode("utf-8", "replace").splitlines()[0].strip()
    except Exception:
        return None


def _llvm_profdata_for(clang):
    """llvm-profdata matching a clang's major version, or None."""
    v = _version(clang) or ""
    major = None
    for tok in v.replace("(", " ").split():
        if tok[:1].isdigit() and "." in tok:
            major = tok.split(".")[0]
            break
    cands = []
    if major:
        cands += ["llvm-profdata-%s" % major, "/usr/lib/llvm-%s/bin/llvm-profdata" % major]
    cands.append(os.path.join(os.path.dirname(os.path.realpath(_which(clang) or clang)), "llvm-profdata"))
    for c in cands:
        p = _which(c)
        if p:
            return p
    return None


def _cxx_ok(cxx, flags):
    """1 if `cxx flags` compiles a file that includes <limits>."""
    import tempfile
    d = tempfile.mkdtemp()
    src = os.path.join(d, "t.cpp")
    open(src, "w").write("#include <limits>\nint f(){return std::numeric_limits<int>::max();}\n")
    r = subprocess.call([cxx] + flags + ["-c", src, "-o", os.path.join(d, "t.o")],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(d, ignore_errors=True)
    return r == 0


def _clang_cflags(clangxx, gcc):
    """clang may pick a gcc install without C++ headers: try the system gcc
    installs (newest first), then the gcc on PATH."""
    if _cxx_ok(clangxx, []):
        return []
    cands = sorted(glob.glob("/usr/lib/gcc/*-linux-gnu/*"), reverse=True,
                   key=lambda p: [int(x) if x.isdigit() else 0 for x in os.path.basename(p).split(".")])
    if gcc:
        cands.append(os.path.dirname(subprocess.check_output([gcc, "-print-libgcc-file-name"]).decode().strip()))
    for c in cands:
        f = ["--gcc-install-dir=%s" % c]
        if _cxx_ok(clangxx, f):
            return f
    return None


def toolchains():
    """{family: dict(cc, cxx, ldflags, profdata, prof_rt, why_not)}"""
    tc = {}
    gcc, gxx = _which(os.environ.get("SUITE_GCC", "gcc")), _which(os.environ.get("SUITE_GXX", "g++"))
    tc["gcc"] = dict(cc=gcc, cxx=gxx, ldflags=[], why_not=None if gcc and gxx else "gcc/g++ not found")
    clang, clangxx = _which(os.environ.get("SUITE_CLANG", "clang")), _which(os.environ.get("SUITE_CLANGXX", "clang++"))
    cfl = _clang_cflags(clangxx, gcc) if clang and clangxx else None
    tc["clang"] = dict(cc=clang, cxx=clangxx, ldflags=[], cflags=cfl or [],
                       profdata=_llvm_profdata_for(clang) if clang else None,
                       why_not=None if clang and clangxx and cfl is not None else
                       "clang/clang++ not found" if not (clang and clangxx) else
                       "clang++ finds no C++ standard headers")
    zcc, zxx = os.path.join(HERE, "zig-cc"), os.path.join(HERE, "zig-c++")
    zok = os.access(zcc, os.X_OK) and _version(zcc) is not None
    ld = []
    if platform.machine() == "x86_64" and gcc:
        # zig's compiler_rt has no __cpu_model (__builtin_cpu_supports):
        # take it from libgcc.a
        ld.append(subprocess.check_output([gcc, "-print-libgcc-file-name"]).decode().strip())
    llvm21 = os.path.join(TOOLS, "LLVM-21.1.0-Linux-X64")
    prt = glob.glob(os.path.join(llvm21, "lib", "clang", "*", "lib", "*", "libclang_rt.profile.a"))
    tc["zig"] = dict(cc=zcc if zok else None, cxx=zxx if zok else None, ldflags=ld,
                     profdata=_which(os.path.join(llvm21, "bin", "llvm-profdata")),
                     prof_rt=prt[0] if prt else None,
                     why_not=None if zok else "zig not found (tools/zig-cc looks in %s)" % TOOLS)
    return tc


def variant_plan(variant, tc):
    """(family, pgo, problem-or-None)"""
    fam, pgo = variant.split("-")[0], variant.endswith("-pgo")
    t = tc[fam]
    if t["why_not"]:
        return fam, pgo, t["why_not"]
    if pgo and fam in ("clang", "zig") and not t.get("profdata"):
        return fam, pgo, "no llvm-profdata matching %s" % _version(t["cc"])
    if pgo and fam == "zig" and not t.get("prof_rt"):
        return fam, pgo, "no LLVM profile runtime for zig (x86_64 only, see TOOLS)"
    return fam, pgo, None


# ------------------------------------------------------------------ build

def _libdir(variant):
    return os.path.join(OUT, "libs", variant)


def _build_once(variant, fam, tc, cflags, ldflags, log):
    t = tc[fam]
    pkg = os.path.join(_libdir(variant), "bslz4_to_sparse")
    os.makedirs(pkg, exist_ok=True)
    for f in glob.glob(os.path.join(REPO, "src", "*.py")):
        shutil.copy(f, pkg)
    env = dict(os.environ, CC=t["cc"], CXX=t["cxx"],
               BSLZ4_EXTRA_CFLAGS=" ".join(list(t.get("cflags", [])) + list(cflags)),
               BSLZ4_EXTRA_LDFLAGS=" ".join(list(t["ldflags"]) + list(ldflags)))
    cmd = [sys.executable, os.path.join(HERE, "build_extension.py"), "--out", pkg,
           "--build-dir", os.path.join(OUT, "obj", variant)]
    with open(log, "a") as fh:
        fh.write("\n$ CC=%s CXX=%s BSLZ4_EXTRA_CFLAGS='%s' BSLZ4_EXTRA_LDFLAGS='%s' %s\n"
                 % (env["CC"], env["CXX"], env["BSLZ4_EXTRA_CFLAGS"], env["BSLZ4_EXTRA_LDFLAGS"],
                    " ".join(cmd)))
        fh.flush()
        subprocess.check_call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)


def build(variants, rounds_train=1):
    tc = toolchains()
    os.makedirs(OUT, exist_ok=True)
    built = []
    for v in variants:
        fam, pgo, problem = variant_plan(v, tc)
        log = os.path.join(OUT, "build_%s.log" % v)
        open(log, "w").close()
        if problem:
            print("%-10s skipped: %s" % (v, problem))
            continue
        shutil.rmtree(_libdir(v), ignore_errors=True)
        pgodir = os.path.join(OUT, "pgo", v)
        try:
            if not pgo:
                _build_once(v, fam, tc, [], [], log)
            else:
                shutil.rmtree(pgodir, ignore_errors=True)
                os.makedirs(pgodir)
                if fam == "gcc":
                    gen = ["-fprofile-generate", "-fprofile-update=single", "-fprofile-dir=%s" % pgodir]
                    _build_once(v, fam, tc, gen, ["-fprofile-generate"], log)
                    train(v, log, {})
                    use = ["-fprofile-use", "-fprofile-dir=%s" % pgodir, "-fprofile-partial-training",
                           "-Wno-missing-profile", "-Wno-coverage-mismatch"]
                    _build_once(v, fam, tc, use, [], log)
                else:
                    gen = ["-fprofile-instr-generate"]
                    lgen = ["-fprofile-instr-generate"]
                    if fam == "zig":
                        # zig links no profile runtime: LLVM 21's, kept alive
                        lgen = [tc["zig"]["prof_rt"], "-Wl,-u,__llvm_profile_runtime"]
                    _build_once(v, fam, tc, gen, lgen, log)
                    train(v, log, {"LLVM_PROFILE_FILE": os.path.join(pgodir, "raw-%p.profraw")})
                    raws = glob.glob(os.path.join(pgodir, "raw-*.profraw"))
                    if not raws:
                        raise RuntimeError("training wrote no .profraw")
                    merged = os.path.join(pgodir, "merged.profdata")
                    subprocess.check_call([tc[fam]["profdata"], "merge", "-o", merged] + raws)
                    use = ["-fprofile-instr-use=%s" % merged, "-Wno-profile-instr-unprofiled",
                           "-Wno-profile-instr-out-of-date", "-Wno-backend-plugin"]
                    _build_once(v, fam, tc, use, [], log)
            info = _child_info(v)
            print("%-10s built: %s | %s" % (v, info.get("compiler"), info.get("git")))
            built.append(v)
        except Exception as e:
            print("%-10s FAILED: %s (see %s)" % (v, e, log))
    with open(os.path.join(OUT, "toolchains.json"), "w") as fh:
        json.dump({k: dict(v, version=_version(v["cc"]) if v.get("cc") else None)
                   for k, v in tc.items()}, fh, indent=1)
    return built


def _child_info(variant):
    out = subprocess.check_output([sys.executable, "-c",
                                   "import sys,json; sys.path.insert(0,%r); import bslz4_to_sparse as b; "
                                   "print(json.dumps(b.build_info()))" % _libdir(variant)])
    return json.loads(out.decode().strip().splitlines()[-1])


def train(variant, log, env_extra):
    """The PGO training run: every bench case once, briefly."""
    env = dict(os.environ, **env_extra)
    with open(log, "a") as fh:
        fh.write("\n# PGO training\n")
        fh.flush()
        subprocess.check_call([sys.executable, os.path.abspath(__file__), "_child", _libdir(variant),
                               "--secs", "0", "--cpu", "-1"], env=env, stdout=fh, stderr=subprocess.STDOUT)


def present():
    return [v for v in ALL_VARIANTS
            if glob.glob(os.path.join(_libdir(v), "bslz4_to_sparse", "_bslz4_to_sparse*"))]


# ------------------------------------------------------------------ test

def test(variants):
    ok = True
    for v in variants:
        log = os.path.join(OUT, "test_%s.log" % v)
        env = dict(os.environ, BSLZ4_TO_SPARSE_PATH=_libdir(v))
        with open(log, "w") as fh:
            r = subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                 os.path.join(REPO, "test")], env=env, stdout=fh,
                                stderr=subprocess.STDOUT, cwd=REPO)
        last = open(log).read().strip().splitlines()[-1:]
        print("%-10s %s %s" % (v, "PASS" if r == 0 else "FAIL", last[0] if last else ""))
        ok &= r == 0
    return ok


# ------------------------------------------------------------------ bench

# (dataset, case, dot, matrix); dot None = plain sparsify
DATASETS = ("dense", "mid", "sparse", "WAu0008", "WAu0012")
CASES = (("sparsify cut0", None, None, 0), ("sparsify cut2", None, None, 2),
         ("1D bbox csc", "csc", "1D_bbox_x1", 0), ("1D bbox csc-run", "csc-run", "1D_bbox_x1", 0),
         ("1D no csc-nosplit", "csc-nosplit", "1D_no_x1", 0), ("2D bbox csc", "csc", "2D_bbox", 0),
         ("rings bsb-csr", "bsb-csr", "rings", 0))


def child(libdir, secs, cpu, rnd, variant, out):
    """Time every case with one build (run in its own process)."""
    sys.path.insert(0, libdir)
    import numpy as np
    import h5py
    import hdf5plugin  # noqa: F401
    import scipy.sparse as sp
    import bslz4_to_sparse as b
    if cpu >= 0:
        os.sched_setaffinity(0, {cpu})
    info = b.build_info()
    mats = {}
    for n in set(c[2] for c in CASES if c[2]):
        z = np.load(os.path.join(DATA, n + ".npz"))
        mats[n] = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    detvalid = (np.load(os.path.join(DATA, "detmask.npy")) == 0).astype(np.uint8)
    fh = open(out, "a") if out else None
    for ds in DATASETS:
        if ds.startswith("WAu"):
            num = ds[3:]
            fn = os.path.join(DATA, "WAu5um_eiger%s_f400-599_maskzero.h5" % num)
            mask = np.load(os.path.join(DATA, "WAu5um_eiger%s_mask.npy" % num))
        else:
            fn = os.path.join(DATA, "frames_4M_mid%s.h5" % ("" if ds == "mid" else "_" + ds))
            mask = detvalid
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        with h5py.File(fn, "r") as h:
            ch = [h["data"].id.read_direct_chunk((k, 0, 0))[1] for k in range(100)]
        for name, dot, mat, cut in CASES:
            if dot is None:
                integ = b.chunk2sparseMulti(mask, dtype=np.uint16)
            else:
                integ = b.chunk2sparseCSCmulti(mask, mats[mat], dtype=np.uint16, dot=dot)
            for s in range(0, 100, 25):                    # warmup pass
                integ(ch[s:s + 25], cut)
            t0 = time.perf_counter()
            n = 0
            while True:
                for s in range(0, 100, 25):
                    integ(ch[s:s + 25], cut)
                n += 100
                if time.perf_counter() - t0 >= secs:
                    break
            ms = (time.perf_counter() - t0) * 1e3 / n
            rec = {"host": HOST, "variant": variant, "round": rnd, "dataset": ds, "case": name,
                   "ms": ms, "frames": n, "compiler": info.get("compiler"), "git": info.get("git"),
                   "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            print("%-8s %-20s %8.3f" % (ds, name, ms), flush=True)
            del integ


def bench(variants, rounds, secs, cpu):
    out = os.path.join(OUT, "bench.jsonl")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    if os.path.exists(out):
        os.rename(out, out.replace(".jsonl", "_%s.jsonl" % stamp))
    for r in range(rounds):
        for v in variants:                              # interleaved: v1 v2 ... v1 v2 ...
            print("round %d/%d  %s" % (r + 1, rounds, v), flush=True)
            subprocess.check_call([sys.executable, os.path.abspath(__file__), "_child", _libdir(v),
                                   "--secs", str(secs), "--cpu", str(cpu), "--round", str(r),
                                   "--variant", v, "--out", out], stdout=subprocess.DEVNULL)
    summary()


def summary():
    out = os.path.join(OUT, "bench.jsonl")
    rows = [json.loads(l) for l in open(out)]
    variants = [v for v in ALL_VARIANTS if any(r["variant"] == v for r in rows)]
    keys = []
    for r in rows:
        k = (r["dataset"], r["case"])
        if k not in keys:
            keys.append(k)
    best = {}
    noise = {}
    for k in keys:
        for v in variants:
            t = sorted(r["ms"] for r in rows if (r["dataset"], r["case"]) == k and r["variant"] == v)
            if t:
                best[k, v] = t[0]
                noise[k, v] = (t[-1] - t[0]) / t[0] if len(t) > 1 else 0.0
    lines = ["host %s | %d rounds | ms/frame, min over rounds; [x] = (max-min)/min over rounds"
             % (HOST, max(r["round"] for r in rows) + 1),
             "spread = (slowest - fastest)/fastest over the builds; best = fastest build", ""]
    lines.append("%-8s %-20s" % ("data", "case") + "".join("%15s" % v for v in variants)
                 + "   spread  best")
    spreads = []
    for k in keys:
        vals = [best.get((k, v)) for v in variants]
        got = [x for x in vals if x is not None]
        sp_ = (max(got) - min(got)) / min(got) if got else 0
        spreads.append(sp_)
        bv = variants[vals.index(min(got))] if got else "-"
        lines.append("%-8s %-20s" % k + "".join(
            "%15s" % ("-" if x is None else "%.3f [%2.0f%%]" % (x, 100 * noise[k, v]))
            for x, v in zip(vals, variants)) + "  %5.1f%%  %s" % (100 * sp_, bv))
    lines.append("")
    lines.append("geometric mean vs gcc: " + "  ".join(
        "%s %.3f" % (v, _gmean([best[k, v] / best[k, "gcc"] for k in keys
                                if (k, v) in best and (k, "gcc") in best]))
        for v in variants if v != "gcc" and "gcc" in variants))
    lines.append("median noise over rounds: %.1f%%   median spread over builds: %.1f%%"
                 % (100 * _median(list(noise.values())), 100 * _median(spreads)))
    text = "\n".join(lines)
    open(os.path.join(OUT, "summary.txt"), "w").write(text + "\n")
    print(text)


def compare(hosts=None):
    """All hosts under SUITE side by side: per host the CPU, the geometric
    mean of each build vs gcc, noise and spread; per case the gcc time on
    each host and which build was fastest there."""
    dirs = sorted(d for d in glob.glob(os.path.join(SUITE, "*"))
                  if os.path.exists(os.path.join(d, "bench.jsonl")))
    if hosts:
        dirs = [d for d in dirs if any(h in os.path.basename(d) for h in hosts)]
    data, cpu, complete = {}, {}, {}
    for d in dirs:
        h = os.path.basename(d)
        rows = [json.loads(l) for l in open(os.path.join(d, "bench.jsonl"))]
        try:
            m = json.load(open(os.path.join(d, "machine.json")))
            cpu[h] = m.get("cpu_model_name") or m.get("cpu_cpu") or "?"
        except Exception:
            cpu[h] = "?"
        nv = len(set(r["variant"] for r in rows))
        complete[h] = "%d builds, %d/%d rows" % (nv, len(rows), nv * 3 * len(DATASETS) * len(CASES))
        best = {}
        for r in rows:
            k = (r["dataset"], r["case"], r["variant"])
            best[k] = min(best.get(k, 1e99), r["ms"])
        data[h] = best
    lines = ["hosts (ms/frame = min over rounds; ratios < 1 are faster than gcc on that host)", ""]
    for h in data:
        b = data[h]
        keys = set((k[0], k[1]) for k in b)
        g = []
        for v in ALL_VARIANTS[1:]:
            r = [b[k + (v,)] / b[k + ("gcc",)] for k in keys if k + (v,) in b and k + ("gcc",) in b]
            g.append("%s %.3f" % (v, _gmean(r)) if r else "%s -" % v)
        lines.append("%-24s %-45s %s" % (h, cpu[h][:45], complete[h]))
        lines.append("%24s geomean vs gcc: %s" % ("", "  ".join(g)))
    lines.append("")
    hs = list(data)
    lines.append("%-8s %-20s" % ("data", "case") + "".join("%22s" % h[:21] for h in hs))
    lines.append("%-29s" % "" + "".join("%22s" % "gcc ms  best build" for h in hs))
    for ds in DATASETS:
        for c in CASES:
            k = (ds, c[0])
            cells = []
            for h in hs:
                b = data[h]
                vals = {v: b[k + (v,)] for v in ALL_VARIANTS if k + (v,) in b}
                if not vals:
                    cells.append("%22s" % "-")
                    continue
                bv = min(vals, key=vals.get)
                g = vals.get("gcc")
                cells.append("%22s" % ("%s %s %+.0f%%" % ("%.3f" % g if g else "-", bv,
                                                           100 * (vals[bv] / g - 1) if g else 0)))
            lines.append("%-8s %-20s" % k + "".join(cells))
    text = "\n".join(lines)
    open(os.path.join(SUITE, "compare.txt"), "w").write(text + "\n")
    print(text)


def _gmean(x):
    import math
    return math.exp(sum(math.log(a) for a in x) / len(x)) if x else float("nan")


def _median(x):
    x = sorted(x)
    return x[len(x) // 2] if x else float("nan")


# ------------------------------------------------------------------ main

def write_machine():
    """OUT/machine.json: what the host name does not say (CPU model etc.)."""
    m = {"host": HOST, "node": platform.node(), "machine": platform.machine(),
         "kernel": platform.release(), "python": platform.python_version(),
         "ncpu": os.cpu_count(), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        info = open("/proc/cpuinfo").read().splitlines()
        for key in ("model name", "cpu", "machine", "flags"):
            for line in info:
                if line.split(":")[0].strip() == key:
                    v = line.split(":", 1)[1].strip()
                    m["cpu_" + key.replace(" ", "_")] = v if key != "flags" else \
                        " ".join(f for f in v.split() if f.startswith(("avx", "gfni", "vbmi", "sse4")))
                    break
    except OSError:
        pass
    try:
        m["lscpu"] = subprocess.check_output(["lscpu"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "machine.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["all", "build", "test", "bench", "summary", "compare", "_child"])
    ap.add_argument("lib", nargs="?", help=argparse.SUPPRESS)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--secs", type=float, default=1.0, help="minimum seconds per case per round")
    ap.add_argument("--cpu", type=int, default=None, help="core to pin (default: the last one)")
    ap.add_argument("--round", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--variant", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    cpu = a.cpu if a.cpu is not None else max(os.sched_getaffinity(0))
    if a.cmd == "_child":
        child(a.lib, a.secs, cpu, a.round, a.variant, a.out)
        return
    if a.cmd == "compare":
        compare([h for h in a.variants.split(",") if h] if a.variants != ",".join(ALL_VARIANTS) else None)
        return
    print("bslz4_to_sparse compiler suite: %s -> %s (data %s)" % (HOST, OUT, DATA))
    try:
        m = write_machine()
        print("cpu: %s" % (m.get("cpu_model_name") or m.get("cpu_cpu") or m.get("machine")))
    except Exception as e:                      # never stop a run over this
        print("machine.json not written: %s" % e)
    wanted = [v for v in a.variants.split(",") if v]
    if a.cmd in ("all", "build"):
        build(wanted)
    have = [v for v in wanted if v in present()]
    if a.cmd in ("all", "test"):
        test(have)
    if a.cmd in ("all", "bench"):
        bench(have, a.rounds, a.secs, cpu)
    if a.cmd == "summary":
        summary()


if __name__ == "__main__":
    main()
