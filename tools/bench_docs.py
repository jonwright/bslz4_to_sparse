#!/usr/bin/env python3
"""Measure the speed figures shown on the documentation site.

Three real ID11 Eiger data sets (CASES below), each through four operations:

  sparsify   chunk2sparse: decode, mask, threshold to (values, indices)
  1D         chunk2sparseCSC with a pyFAI 1D bbox matrix (r_mm, one bin per
             pixel width)
  1D no-split  the same with pyFAI's no-split 1D matrix: a histogram, every
             pixel in one bin with weight 1
  1D fine    pyFAI 1D bbox with NFINE (6000) bins: a fine split, several
             bins per pixel (the padded rows the AVX2 kernels vectorise)
  2D         chunk2sparseCSC with a pyFAI 2D bbox matrix (q_nm^-1, one radial
             bin per pixel width x NAZIM azimuth)
  2D+rings   8 ring windows x NAZIM azimuth summed from the 2D matrix, the ring
             widths chosen so the rings hold ~8 % of the unmasked pixels

The descriptions written into the JSON (OP_TEXT) are what the documentation
shows, so they are built from the same constants as the matrices.

The files carry no calibration, so every case uses one stated generic
geometry (GEOMETRY below, beam at the detector centre); the matrices
describe the shape of the work, not a refinement of these samples.  The
mask is every pixel holding 65535 in the first frame (the module gaps and
bad pixels of these files).  Pipelines are the automatic choice.

Timing: FRAMES frames spread evenly over each file are read into memory
first (no file I/O is timed).  Each operation is built, warmed up on one
batch, then timed over all frames in batches of BATCH; the best of REPEATS
passes is kept.  Reported: frames/s, compressed GB/s (the bitshuffle-LZ4
bytes consumed) and pixel GB/s (frames x pixels x bytes per pixel), GB =
1e9 bytes.  Run it on the machine the numbers are for, on one core
(`taskset -c N` or a one-core job), and commit the JSON, one per machine
(docs/bench/real_data_<cpu>.json; the documentation shows them side by
side):

    BSLZ4_TO_SPARSE_PATH=lib python3 tools/bench_docs.py \\
        --out docs/bench/real_data_xeon-gold-6248.json

Never run it in CI: shared runners give meaningless timings.
tools/generate_docs.py renders the JSONs; they are the only source of the
numbers on the site.  pyFAI matrices are cached in $BSLZ4_BENCH_DIR.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite as bs  # noqa: E402  (puts lib/ or $BSLZ4_TO_SPARSE_PATH on sys.path)
from bench_docs_hash import py_sha256  # noqa: E402

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import pyFAI  # noqa: E402
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: E402  (pyFAI >= 2024)
except ImportError:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import bslz4_to_sparse as b  # noqa: E402

CASES = [
    {"key": "eiger4m_u32", "label": "very sparse, uint32",
     "file": "/data/id11/jon/hdftest/eiger4m_u32.h5",
     "dataset": "/entry_0000/ESRF-ID11/eiger/data", "cut": 0},
    {"key": "WAu5um_DT3", "label": "sparse photon counting, uint16",
     "file": "/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5",
     "dataset": "/entry_0000/ESRF-ID11/eiger/data", "cut": 0},
    {"key": "kevlar", "label": "dense fibre pattern, uint16",
     "file": "/data/id11/jon/hdftest/kevlar.h5",
     "dataset": "/entry/data/data", "cut": 5},
]
OPS = ("sparsify", "1D", "1D no-split", "1D fine", "2D", "2D+rings")
NFINE = 6000                   # 1D bins of the fine split (rows <= 8 bins here)
NAZIM = 72                     # azimuthal bins of the 2D and ring matrices
OP_TEXT = {
    "sparsify": "chunk2sparse: decode, mask and threshold to (values, indices)",
    "1D": "chunk2sparseCSC, pyFAI 1D bbox matrix (r_mm, one bin per pixel width)",
    "1D no-split": "chunk2sparseCSC, pyFAI 1D no-split matrix: a histogram, every pixel "
                   "in one bin with weight 1 (r_mm, one bin per pixel width)",
    "1D fine": "chunk2sparseCSC, pyFAI 1D bbox matrix with %d bins (r_mm): a fine split, "
               "several bins per pixel" % NFINE,
    "2D": "chunk2sparseCSC, pyFAI 2D bbox matrix (q_nm^-1, one radial bin per pixel "
          "width x %d azimuthal bins)" % NAZIM,
    "2D+rings": "chunk2sparseCSC, %d LaB6 ring windows x %d azimuthal bins summed from the "
                "2D matrix, the windows sized to hold ~%d %% of the unmasked pixels"
                % (bs.NRINGS, NAZIM, round(100 * bs.RING_FRACTION)),
}
MASK_VALUE = 65535
GEOMETRY = {"pixel_m": 75e-6, "dist_m": 0.25, "wavelength_m": 2.85e-11,
            "poni": "detector centre", "rot_rad": [0.0, 0.0, 0.0]}
FRAMES = 100
BATCH = 25
REPEATS = 3


def read_case(case, nframes):
    with h5py.File(case["file"], "r") as f:
        ds = f[case["dataset"]]
        idx = np.linspace(0, ds.shape[0] - 1, nframes).round().astype(int)
        chunks = [ds.id.read_direct_chunk((int(i), 0, 0))[1] for i in idx]
        frame0 = ds[int(idx[0])]
        info = {"frames_in_file": int(ds.shape[0]), "shape": list(ds.shape[1:]),
                "dtype": str(ds.dtype)}
    return chunks, frame0, info


def make_ai(shape, masked):
    det = pyFAI.detectors.Detector(GEOMETRY["pixel_m"], GEOMETRY["pixel_m"], max_shape=shape)
    det.mask = masked.astype(np.int8)
    return AzimuthalIntegrator(dist=GEOMETRY["dist_m"], poni1=0.5 * shape[0] * det.pixel1,
                               poni2=0.5 * shape[1] * det.pixel2, detector=det,
                               wavelength=GEOMETRY["wavelength_m"])


def matrices(ai, key):
    """{op: csc_matrix} for the three matrix operations, cached per data set."""
    rp = bs.radial_pixels(ai)
    m1 = bs._cached("docs_%s_1d" % key, lambda: bs._engine_csc(ai, "bbox", 1, rp, "r_mm"))
    m1n = bs._cached("docs_%s_1d_no" % key, lambda: bs._engine_csc(ai, "no", 1, rp, "r_mm"))
    m1f = bs._cached("docs_%s_1d_fine%d" % (key, NFINE),
                     lambda: bs._engine_csc(ai, "bbox", 1, NFINE, "r_mm"))
    m2 = bs._cached("docs_%s_2d_%d" % (key, NAZIM),
                    lambda: bs._engine_csc(ai, "bbox", 2, rp, "q_nm^-1", NAZIM))

    def rings():
        method = bs.IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=2)
        qc = ai.integrate2d(np.ones(ai.detector.shape, np.float32), rp, NAZIM,
                            unit="q_nm^-1", method=method).radial
        pick, h = bs.ring_windows(ai)
        W = np.array([np.abs(qc - p) < h for p in pick], dtype=np.float64)
        S = sp.kron(sp.csr_matrix(W), sp.identity(NAZIM), format="csr")
        return bs._f32(S @ m2)

    mr = bs._cached("docs_%s_rings_%d" % (key, NAZIM), rings)
    return {"1D": (m1, [rp]), "1D no-split": (m1n, [rp]), "1D fine": (m1f, [NFINE]),
            "2D": (m2, [rp, NAZIM]),
            "2D+rings": (mr, [mr.shape[0] // NAZIM, NAZIM])}


def matrix_info(M, bins_shape, unmasked):
    """Size and shape of a matrix: bins, entries, the fraction of unmasked
    pixels with an entry, and bins per pixel (mean over those, maximum)."""
    n = np.diff(M.indptr)
    used = n > 0
    return {"bins": int(M.shape[0]), "bins_shape": bins_shape, "nnz": int(M.nnz),
            "pixel_frac": float(used.sum()) / unmasked,
            "bins_per_pixel_mean": float(n[used].mean()) if used.any() else 0.0,
            "bins_per_pixel_max": int(n.max()) if n.size else 0}


def time_op(fn, chunks):
    fn(chunks[:BATCH])                                   # warmup: buffers, matrix in cache
    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for s in range(0, len(chunks), BATCH):
            fn(chunks[s:s + BATCH])
        t = time.perf_counter() - t0
        best = t if best is None else min(best, t)
    return best


def machine():
    info = b.build_info()
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"cpu": cpu, "cores_usable": cores,               # no host name: it is published
            "python": platform.python_version(), "numpy": np.__version__,
            "pyFAI": pyFAI.version, "version": b.version, "git": info["git"],
            "src_sha256": info["src_sha256"],
            "py_sha256": py_sha256(os.path.dirname(b.__file__)), "compiler": info["compiler"],
            "platform": info["platform"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="JSON file to write (docs/bench/real_data_<cpu>.json)")
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--cases", default=",".join(c["key"] for c in CASES))
    args = ap.parse_args()

    out = {"schema": 1, "tool": "tools/bench_docs.py",
           "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "machine": machine(),
           "ops": OP_TEXT,
           "method": {"frames": args.frames, "batch": BATCH, "repeats": REPEATS,
                      "mask": "pixels == %d in the first timed frame" % MASK_VALUE,
                      "geometry": GEOMETRY, "gb": 1e9},
           "cases": [], "results": []}
    m = out["machine"]
    print("%s | %d core(s) | bslz4_to_sparse %s (%s, src %s)"
          % (m["cpu"], m["cores_usable"], m["version"], m["git"], m["src_sha256"][:12]))
    keys = args.cases.split(",")
    for case in [c for c in CASES if c["key"] in keys]:
        chunks, frame0, info = read_case(case, args.frames)
        masked = frame0 == MASK_VALUE
        mask = (~masked).astype(np.uint8)
        dtype = np.dtype(info["dtype"])
        npix = int(np.prod(info["shape"]))
        cbytes = sum(len(c) for c in chunks)
        pbytes = len(chunks) * npix * dtype.itemsize
        codec = b.CODEC_LZ4
        crec = dict(case, **info)
        crec.update({"frames_timed": len(chunks), "masked_pixels": int(masked.sum()),
                     "compression": pbytes / cbytes})
        out["cases"].append(crec)
        print("\n%s: %s %s, %d frames timed, compression %.1f, %d masked, cut %d"
              % (case["key"], info["dtype"], info["shape"], len(chunks), crec["compression"],
                 masked.sum(), case["cut"]))
        unmasked = float(mask.sum())
        npx, _ = b.chunk2sparse(mask, dtype=dtype, codec=codec).multi(chunks, case["cut"])
        crec["kept_frac"] = float(np.asarray(npx, np.float64).mean()) / unmasked
        print("  pixels above the cut: %.4f %% of the unmasked pixels" % (100 * crec["kept_frac"]))
        ai = make_ai(info["shape"], masked)
        mats = matrices(ai, case["key"])
        for op in OPS:
            if op == "sparsify":
                obj = b.chunk2sparse(mask, dtype=dtype, codec=codec)
                mat = None
            else:
                M, bins_shape = mats[op]
                obj = b.chunk2sparseCSC(mask, M, dtype=dtype, codec=codec)
                mat = matrix_info(M, bins_shape, unmasked)
            cut = case["cut"]
            t = time_op(lambda ch: obj.multi(ch, cut), chunks)
            rec = {"case": case["key"], "op": op, "matrix": mat,
                   "pipeline": {k: v for k, v in b.describe(obj.pipeline).items() if v},
                   "seconds": t, "fps": len(chunks) / t,
                   "compressed_gbs": cbytes / t / 1e9, "pixel_gbs": pbytes / t / 1e9}
            out["results"].append(rec)
            print("  %-9s %8.1f frames/s  %6.3f GB/s compressed  %6.2f GB/s pixels"
                  % (op, rec["fps"], rec["compressed_gbs"], rec["pixel_gbs"]))

    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, args.out)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
