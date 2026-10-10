#!/usr/bin/env python3
"""Each pipeline step on real data, one step at a time.

Two data sets of tools/bench_docs.py: WAu5um_DT3 (sparse) and kevlar
(dense, cropped to the same 2162 x 2068 pixels so both use one set of
pyFAI matrices).  FRAMES frames of each are decoded and re-encoded in memory
(bitshuffle-LZ4 and bitshuffle-zstd level 2, with 8 kB and 256 kB blocks;
DATASETS says which encodings each data set and table uses).  The C code
composes a pipeline at run time (a switch per step per block, templates
only over the pixel dtype), so the steps are scanned one at a time, not as
a product: starting from the automatic pipeline, each step's values are
timed with the other steps held there.

  sparsify   the decode, mask, untranspose and collect axes (chunk2sparse),
             on the masked frames (masked pixels at 65535, with the mask)
             and on the same frames not masked (masked pixels set to 0, no
             mask: where the mask step 'none' applies).  The collect axis
             holds untranspose at kcb: the low-planes values fuse the
             collect, so beside them the collect value is not used.
  matrices   the dot axis (route automatic) and the route axis (dot
             automatic) on the matrices of tools/bench_docs.py: 1D, 1D
             no-split (a pyFAI histogram), 1D fine, 2D and 2D+rings.

A value the library refuses (the data, the block size or the matrix does
not allow it) is recorded with the reason, not timed.  Every timed result
is checked against the automatic pipeline: the sparse output must be
identical, the powder is compared as max |difference| / max |powder| (the
fixed-point dots round).  The check pass over all frames is also the
warmup; then the best of REPEATS timed passes.  Frames are in memory (no
file I/O timed).  Sized to finish within about ten minutes on one core.
Run on one core of the machine the numbers are for and commit the JSON:

    BSLZ4_TO_SPARSE_PATH=lib taskset -c N python3 tools/bench_kernels.py \\
        --out docs/bench/kernels_<cpu>.json

Never run in CI.  tools/generate_docs.py renders docs/bench/kernels_*.json.
"""
import argparse
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_docs as bd  # noqa: E402  (bench_suite, lib/ on sys.path, pyFAI)

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import hdf5plugin  # noqa: E402
import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import _pipeline as P  # noqa: E402

GEOMETRY_CASE = "WAu5um_DT3"          # its shape and mask define the matrices
ENC = {  # key: (codec name, zstd level, block bytes)
    "lz4 8k": ("lz4", None, 8192),
    "lz4 256k": ("lz4", None, 262144),
    "zstd-2 8k": ("zstd", 2, 8192),
    "zstd-2 256k": ("zstd", 2, 262144),
}
DATASETS = [  # (case key, encodings for sparsify, encodings for the matrices)
    ("WAu5um_DT3", ["lz4 8k", "lz4 256k", "zstd-2 8k", "zstd-2 256k"], ["lz4 8k", "zstd-2 256k"]),
    ("kevlar", ["lz4 8k", "zstd-2 256k"], ["lz4 8k"]),
]
SPARSIFY_STEPS = ("decode", "mask", "untranspose", "collect")
FRAMES = 25
BATCH = 25
REPEATS = 2


def encode(frames, enc):
    cname, clevel, block = ENC[enc]
    kw = hdf5plugin.Bitshuffle(nelems=block // frames.dtype.itemsize, cname=cname,
                               **({"clevel": clevel} if clevel is not None else {}))
    with h5py.File("variant.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("data", data=frames, chunks=(1,) + frames.shape[1:], **kw)
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]


def usable(step):
    return [n for i, n in enumerate(P.NAMES[step]) if i and P.available(step, i) == 1]


def names(obj):
    return {k: v for k, v in b.describe(obj.pipeline).items() if v}


def run_all(obj, chunks, cut):
    """One pass over all frames: per frame (values, indices) copies, and the
    powders (None without a matrix).  Also the warmup."""
    sparse, powders = [], []
    for s in range(0, len(chunks), BATCH):
        r = obj.multi(chunks[s:s + BATCH], cut)
        npx, (v, i) = r[0], r[1]
        for k in range(len(npx)):
            sparse.append((v[k, :npx[k]].copy(), i[k, :npx[k]].copy()))
        if len(r) > 2:
            powders.append(np.array(r[2], np.float64))
    return sparse, (np.concatenate(powders) if powders else None)


def best_time(obj, chunks, cut):
    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for s in range(0, len(chunks), BATCH):
            obj.multi(chunks[s:s + BATCH], cut)
        t = time.perf_counter() - t0
        best = t if best is None else min(best, t)
    return best


def same_sparse(a, ref):
    return len(a) == len(ref) and all(np.array_equal(va, vr) and np.array_equal(ia, ir)
                                      for (va, ia), (vr, ir) in zip(a, ref))


def load(case, nframes, shape):
    """nframes frames spread over the file, cropped to shape."""
    with h5py.File(case["file"], "r") as f:
        ds = f[case["dataset"]]
        idx = np.linspace(0, ds.shape[0] - 1, nframes).round().astype(int)
        frames = np.stack([ds[int(i)][:shape[0], :shape[1]] for i in idx])
        return frames, int(ds.shape[0]), list(ds.shape[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="docs/bench/kernels_<cpu>.json")
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args()
    t_start = time.perf_counter()
    cases = {c["key"]: c for c in bd.CASES}

    geo, _, _ = load(cases[GEOMETRY_CASE], 1, (10 ** 6, 10 ** 6))
    geo_masked = geo[0] == bd.MASK_VALUE
    shape = list(geo.shape[1:])
    ai = bd.make_ai(shape, geo_masked)
    mats = bd.matrices(ai, GEOMETRY_CASE)

    out = {"schema": 2, "tool": "tools/bench_kernels.py",
           "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "machine": bd.machine(),
           "method": {"frames": args.frames, "batch": BATCH, "repeats": REPEATS, "gb": 1e9,
                      "geometry": bd.GEOMETRY, "shape": shape,
                      "mask": "pixels == %d in the first frame" % bd.MASK_VALUE,
                      "no_mask": "the same frames with the masked pixels set to 0 and an "
                                 "all-valid mask"},
           "ops": {k: bd.OP_TEXT[k] for k in bd.OPS},
           "matrices": {op: bd.matrix_info(M, s, float((~geo_masked).sum()))
                        for op, (M, s) in mats.items()},
           "datasets": [], "auto": {}, "results": []}
    m = out["machine"]
    print("%s | %d core(s) | %s src %s" % (m["cpu"], m["cores_usable"], m["git"],
                                           m["src_sha256"][:12]))

    for key, sp_encs, mat_encs in DATASETS:
        case = cases[key]
        frames, nfile, orig_shape = load(case, args.frames, shape)
        masked = frames[0] == bd.MASK_VALUE
        mask = (~masked).astype(np.uint8)
        ones = np.ones_like(mask)
        frames_nomask = frames.copy()
        frames_nomask[:, masked] = 0
        dtype, cut = frames.dtype, case["cut"]
        pbytes = frames.size * dtype.itemsize
        dset = {"key": key, "label": case["label"], "file": case["file"],
                "dataset": case["dataset"], "frames_in_file": nfile, "shape": orig_shape,
                "dtype": str(dtype), "cut": cut, "masked_pixels": int(masked.sum()),
                "sparsify_encodings": sp_encs, "matrix_encodings": mat_encs, "encodings": []}
        out["datasets"].append(dset)
        out["auto"][key] = {}
        for enc in sp_encs:
            codec = b.CODEC_ZSTD if ENC[enc][0] == "zstd" else b.CODEC_LZ4
            chunks = encode(frames, enc)
            chunks_nomask = encode(frames_nomask, enc)
            dset["encodings"].append({"key": enc, "codec": ENC[enc][0], "clevel": ENC[enc][1],
                                      "block_bytes": ENC[enc][2],
                                      "compression": pbytes / sum(len(c) for c in chunks)})
            auto_here = out["auto"][key].setdefault(enc, {})
            print("\n== %s, %s: compression %.1f (%.0f s)"
                  % (key, enc, dset["encodings"][-1]["compression"], time.perf_counter() - t_start))

            def record(kind, axis, pipeline, obj=None, ref=None, err=None, use=None):
                use = chunks if use is None else use
                rec = {"data": key, "variant": enc, "kind": kind, "axis": axis,
                       "pipeline": pipeline}
                if err is None:
                    try:
                        got_sparse, got_powder = run_all(obj, use, cut)   # check + warmup
                        t = best_time(obj, use, cut)
                    except Exception as e:                   # a decode error from C
                        err = e
                if err is not None:
                    rec["refused"] = str(err)
                    out["results"].append(rec)
                    return
                rec.update({"pipeline": names(obj), "fps": len(use) / t,
                            "compressed_gbs": sum(len(c) for c in use) / t / 1e9,
                            "pixel_gbs": pbytes / t / 1e9,
                            "sparse_ok": same_sparse(got_sparse, ref[0])})
                if got_powder is not None:
                    scale = float(np.abs(ref[1]).max()) or 1.0
                    rec["powder_rel"] = float(np.abs(got_powder - ref[1]).max()) / scale
                out["results"].append(rec)

            # sparsify axes, masked and not
            decodes = [d for d in usable("decode") if (d == "zstd") == (ENC[enc][0] == "zstd")]
            for kind, kmask, kchunks in (("sparsify", mask, chunks),
                                         ("sparsify, no mask", ones, chunks_nomask)):
                auto = b.chunk2sparse(kmask, dtype=dtype, codec=codec)
                ref = run_all(auto, kchunks, cut)       # the first frame settles an auto mask
                base = {k: names(auto)[k] for k in SPARSIFY_STEPS}
                auto_here[kind] = dict(base)
                del auto
                for axis in SPARSIFY_STEPS:
                    held = dict(base, untranspose="kcb") if axis == "collect" else base
                    for val in (decodes if axis == "decode" else usable(axis)):
                        p = dict(held, **{axis: val})
                        try:
                            obj = b.chunk2sparse(kmask, dtype=dtype, codec=codec, pipeline=p)
                        except ValueError as e:
                            record(kind, axis, p, err=e, use=kchunks)
                            continue
                        record(kind, axis, p, obj, ref, use=kchunks)
                        del obj

            if enc not in mat_encs:
                continue
            # matrices: the dot axis (route automatic), the route axis (dot automatic)
            for op, (M, _s) in mats.items():
                auto = b.chunk2sparseCSC(mask, M, dtype=dtype, codec=codec)
                ref = run_all(auto, chunks, cut)        # settles the dot for this block size
                base = {k: names(auto)[k] for k in ("dot", "route")}
                auto_here[op] = dict(base)
                for route in usable("route"):           # the route is read on every call
                    auto.pipeline[P.ROUTE] = P.NAMES["route"].index(route)
                    record(op, "route", dict(base, route=route), auto, ref)
                del auto
                for dot in usable("dot"):
                    p = dict(base, dot=dot)
                    try:
                        obj = b.chunk2sparseCSC(mask, M, dtype=dtype, codec=codec, pipeline=p)
                    except ValueError as e:
                        record(op, "dot", p, err=e)
                        continue
                    record(op, "dot", p, obj, ref)
                    del obj

    out["method"]["seconds"] = round(time.perf_counter() - t_start)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, args.out)
    t = [r for r in out["results"] if "fps" in r]
    print("\n%d timed, %d refused, %d sparse outputs differ; %.0f s"
          % (len(t), len(out["results"]) - len(t), sum(not r["sparse_ok"] for r in t),
             time.perf_counter() - t_start))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
