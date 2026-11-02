"""Integrate a Bliss mesh scan into an xrdmap/diffmap-viewer-compatible HDF5.

Writes a pyFAI-diffmap-layout file (see _diffmapio.py) from the raw Eiger frames,
using the bslz4 sparse matrix product against a pyFAI geometry.  Saturated pixels
(value >= exposure*1e7) are clipped (exact) and listed.

Run as ``bs_diffmap_integrate`` (after install) or
``python -m bslz4_to_sparse.conveyor.diffmap_integrate``::

  bs_diffmap_integrate [MASTER] OUTPUT --entry 2.1 --poni PONI [--n-bins N]
      [--split full] [--mask MASK] [--max-frames N] [--input-file F]
  bs_diffmap_integrate OUTPUT --plan PLAN       # reuse a recipe: fills in all args

MASTER: read-only Bliss master (taken from PLAN if omitted).  OUTPUT: the .h5 to
write (put in test/).  A "--plan" recipe stores the geometry, the pyFAI CSC matrix,
the detector mask and the worker config, so a later run needs only OUTPUT --plan.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(prog="bs_diffmap_integrate", description=__doc__)
    p.add_argument("master", nargs="?", type=Path, help="read-only Bliss master (from --plan if omitted)")
    p.add_argument("output", nargs="?", type=Path, help=".h5 to write (required)")
    p.add_argument("--entry", default="2.1", help="scan entry in the master (e.g. 1.1/2.1/3.1)")
    p.add_argument("--poni", type=Path, default=None, help="pyFAI .poni geometry (from --plan if reused)")
    p.add_argument("--n-bins", type=int, default=2000, help="radial bins")
    p.add_argument("--split", choices=["no", "bbox", "full"], default="full",
        help="pyFAI pixel splitting (default full: ~2-3 output bins per input pixel)")
    p.add_argument("--unit", default="2th_deg", help="radial unit (2th_deg for xrdmap)")
    p.add_argument("--fast", default=None, help="fast motor name (taken from an amesh/dmesh title if omitted)")
    p.add_argument("--slow", default=None, help="slow motor name (taken from an amesh/dmesh title if omitted)")
    p.add_argument("--map-shape", nargs=2, type=int, default=None, metavar=("rows", "cols"), help="force map grid")
    p.add_argument("--mask", default=None, help="detector mask file (pyFAI: 1 = masked) or 'auto' (default: derive from the first frame)")
    p.add_argument("--max-frames", type=int, default=None, help="integrate only the first N frames (debug)")
    p.add_argument("--input-file", default=None, help="force chunk source file::dataset (auto-detected)")
    p.add_argument("--write-poni", type=Path, default=None, help="where to write the .poni copy")
    p.add_argument("--threads", type=int, default=None,
        help="integration threads (default: min(16, CPU affinity // 2))")
    p.add_argument("--no-pin", action="store_true", help="do not pin each worker thread to a core")
    p.add_argument("--batch", type=int, default=None, help="frames per bslz4_to_sparse multi() call "
        "(default: 2; raise it only on a machine where reading the matrix dominates)")
    p.add_argument("--plan", type=Path, default=None,
        help="a recipe file: stores the geometry, pyFAI CSC matrix, mask and worker config. "
        "When given, all other config args (master/poni/entry/n_bins/split/unit/mask/fast/slow/"
        "map-shape) are taken from it.  Default when --plan is omitted: <poni>_plan.h5 beside the poni.")
    return p.parse_args()


def open_frame_reader(source, ds_path, shape, dtype):
    """Return (offsets, mmap) so frames are read without reopening the file."""
    import mmap

    import bslz4_to_sparse as b
    import h5py

    with h5py.File(source, "r") as h:
        off = b.locate_chunks(h[ds_path])
    fp = open(source, "rb")
    mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
    return off, mm, fp


# Shared, read-only integration context the worker threads use.
_CTX = {}
# Worker thread -> core, for sched_setaffinity; set once in main() (None = don't pin).
_PIN_IT = None


def _pin_worker():
    """Pin this (worker) thread to a core from the affinity mask; no-op if not pinning."""
    if _PIN_IT is None:
        return
    try:
        os.sched_setaffinity(0, {next(_PIN_IT)})
    except (OSError, ValueError, StopIteration):
        pass


def _integrate_indices(indices):
    """Integrate ``indices`` frames; returns (indices, sums, ints, clipped)."""
    from bslz4_to_sparse import conveyor as cv

    ctx = _CTX
    d = ctx["plan"].context()
    n = len(indices)
    sums = np.zeros((n, ctx["n_rad"]), np.int64)
    ints = np.zeros((n, ctx["n_rad"]), np.float64)
    clip = []
    off = ctx["offsets"]
    framing = ctx["framing"]
    batch = ctx["batch"]
    for j0 in range(0, n, batch):
        idxs = indices[j0:j0 + batch]
        chunks = [framing[off[fi][0]:off[fi][0] + off[fi][1]] for fi in idxs]
        # every pixel above the cut is in (values, pixidx), so the exact "clip every
        # pixel to wtmax" of clip_exact_powder applies: hot pixels are kept at full
        # value, and their (weights x excess) is subtracted from the int64 sums.
        npx, (values, pixidx), powder = d.multi(chunks, 0)
        cv.clip_exact_powder(powder, npx, values, pixidx, ctx["wtmax_i"], ctx["weights"], in_place=True)
        ints[j0:j0 + len(idxs)] = ctx["integ"].intensity(powder, d.powder_scale)
        sums[j0:j0 + len(idxs)] = powder
        for i, fi in enumerate(idxs):
            v = values[i, :npx[i]]
            hot = v >= ctx["wtmax_i"]
            if hot.any():
                clip.append((fi, pixidx[i, :npx[i]][hot], v[hot].astype(np.int64)))
    return indices, sums, ints, clip


def _geom_to_json(geom):
    d = {}
    for k, v in geom.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif k == "map_shape":
            d[k] = [int(x) for x in v]
        elif k == "frame_shape":
            d[k] = [int(x) for x in v]
        elif k == "source":
            d[k] = None if v is None else [str(x) for x in v]
        elif k == "dtype":
            d[k] = np.dtype(v).name
        elif isinstance(v, (str, int, float, bool)) or v is None:
            d[k] = v
        else:
            raise TypeError("geometry key %r is not JSON-serializable" % k)
    return d


def _geom_from_json(d):
    g = dict(d)
    for k, t in (("row", np.int64), ("col", np.int64), ("frame_index", np.int32)):
        g[k] = np.asarray(d[k], dtype=t)
    g["slow_values"] = np.asarray(d["slow_values"], dtype=float)
    g["fast_values"] = np.asarray(d["fast_values"], dtype=float)
    g["map_shape"] = tuple(int(x) for x in d["map_shape"])
    g["frame_shape"] = tuple(int(x) for x in d["frame_shape"])
    g["source"] = None if d["source"] is None else tuple(d["source"])
    g["dtype"] = np.dtype(d["dtype"])
    return g


def _setup_dict(args):
    return dict(
        master=str(args.master), entry=args.entry, poni=str(args.poni),
        n_bins=args.n_bins, split=args.split, unit=args.unit,
        mask=args.mask, fast=args.fast, slow=args.slow,
        map_shape=None if args.map_shape is None else [int(x) for x in args.map_shape],
    )


def _save_plan(path, integ, mask, wc, geom, setup):
    """Write the recipe: the Integrator (CSC + pyFAI info), mask, worker config, setup args, geometry."""
    import json

    import h5py

    integ.save(str(path), compression=False)
    with h5py.File(str(path), "a") as H:
        H.create_dataset("mask", data=np.asarray(mask, np.uint8))
        H.create_dataset("worker_config", data=json.dumps(wc).encode("utf-8"))
        H.create_dataset("setup", data=json.dumps(setup).encode("utf-8"))
        H.create_dataset("geometry", data=json.dumps(_geom_to_json(geom)).encode("utf-8"))


def _load_plan(path):
    """(integ, mask, wc, geom, setup) or None if the file is not a usable recipe."""
    import json

    import h5py
    from bslz4_to_sparse import conveyor as cv

    try:
        with h5py.File(str(path), "r") as H:
            setup = json.loads(H["setup"][()].decode("utf-8"))
            geom = _geom_from_json(json.loads(H["geometry"][()].decode("utf-8")))
            mask = H["mask"][()].astype(np.uint8)
            wc = json.loads(H["worker_config"][()].decode("utf-8"))
        integ = cv.Integrator.load(str(path))
        return integ, mask, wc, geom, setup
    except Exception:
        return None


def main():
    import time

    t0 = time.time()
    args = parse_args()

    # A lone positional with --plan is the OUTPUT; the master comes from the plan.
    if args.output is None and args.master is not None:
        if args.plan is not None:
            args.output, args.master = args.master, None
        else:
            raise SystemExit("give MASTER OUTPUT, or OUTPUT --plan PLAN")
    if args.output is None:
        raise SystemExit("give an OUTPUT (.h5 to write)")
    if args.master is None and args.poni is None and args.plan is None:
        raise SystemExit("give a MASTER (or --poni / --plan)")

    import bslz4_to_sparse as b
    from bslz4_to_sparse import conveyor as cv
    from bslz4_to_sparse.conveyor import _diffmapio

    # Locate/reuse a plan recipe; otherwise configure from the CLI args.
    recipe = None
    if args.plan is not None and args.plan.exists():
        recipe = _load_plan(args.plan)
    elif args.plan is None and args.poni is not None:
        cand = Path(args.poni).parent / (Path(args.poni).stem + "_plan.h5")
        if cand.exists():
            args.plan, recipe = cand, _load_plan(cand)

    if recipe is not None:
        integ, mask, wc, geom, setup = recipe
        if args.master is not None and str(args.master) != setup["master"]:
            raise SystemExit("--plan was built for %s, not %s" % (setup["master"], args.master))
        args.master = Path(setup["master"])
        args.entry = setup["entry"]
        args.poni = Path(setup["poni"])
        args.n_bins = setup["n_bins"]
        args.split = setup["split"]
        args.unit = setup["unit"]
        args.mask = setup["mask"]
        args.fast = setup["fast"]
        args.slow = setup["slow"]
        args.map_shape = None if setup["map_shape"] is None else tuple(setup["map_shape"])
        print("plan:", args.plan)
    else:
        geom = _diffmapio.bliss_geometry(
            str(args.master), args.entry,
            fast=args.fast, slow=args.slow,
            map_shape=None if args.map_shape is None else tuple(args.map_shape),
        )
        integ = mask = wc = None

    rows, cols = geom["map_shape"]
    n_frames = geom["n_points"]
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    shape = geom["frame_shape"]
    dtype = geom["dtype"].name
    print(f"map {rows}x{cols}, {n_frames} pts, frame {shape} {dtype}, wtmax={geom['wtmax']:.3e}")

    src_file, src_ds = geom["source"] or (None, None)
    if args.input_file:
        src_file, src_ds = args.input_file.split("::", 1)
    print("chunk source:", src_file, "::", src_ds)

    wtmax_i = int(geom["wtmax"])
    offsets, framing, framing_fp = open_frame_reader(src_file, src_ds, shape, dtype)

    if integ is None:
        import pyFAI
        pyFAI.use_opencl = False
        from pyFAI import load
        ai = load(str(args.poni))
        ai.detector.shape = shape
        if args.mask:
            import fabio
            ai.detector.mask = (fabio.open(str(args.mask)).data > 0).astype(np.int8)

        integ = cv.Integrator.from_pyfai(ai, args.n_bins, unit=args.unit, split=args.split)

        det_mask = None
        if args.mask and args.mask.lower() not in ("", "auto"):
            import fabio
            det_mask = (fabio.open(args.mask).data > 0).astype(np.int8)
        elif ai.detector.mask is not None:
            det_mask = (ai.detector.mask > 0).astype(np.int8)
        else:
            o, s = offsets[0]
            first = b.decompress([framing[o:o + s]], np.dtype(dtype)).reshape(shape)
            dm = ai.detector.dynamic_mask(first)
            det_mask = dm.astype(np.int8) if np.any(dm) else None
        mask = np.ones(shape, np.uint8)
        if det_mask is not None:
            mask = (1 - det_mask).astype(np.uint8)
            print(f"mask: derived {int(np.count_nonzero(det_mask))} masked pixels")

        wc = _diffmapio.worker_config_dict(ai, args.n_bins, shape, unit=args.unit)

        plan_path = args.plan or Path(args.poni).parent / (Path(args.poni).stem + "_plan.h5")
        try:
            _save_plan(plan_path, integ, mask, wc, geom, _setup_dict(args))
            print("plan written:", plan_path)
        except OSError as err:
            print("plan NOT written (%s): %s" % (plan_path, err))
        args.plan = plan_path

    n_rad = integ.nbins
    print("radial bins:", n_rad, "range:", float(integ.radial[0]), float(integ.radial[-1]))
    print(f"split={args.split} CSC nnz={int(integ.csc.nnz)} "
          f"(~{integ.csc.nnz / np.prod(shape):.2f} output entries/input pixel)")

    int_cube = np.zeros((rows, cols, n_rad), np.float64)
    sum_cube = np.zeros((rows, cols, n_rad), np.int64)

    fp = b.fixed_point_matrix(integ.csc, mask, np.dtype(dtype))
    import scipy.sparse as sp
    nm = fp.nm
    weights = sp.csc_matrix(
        (fp.data.astype(np.int64), nm.indices.astype(np.int64), nm.indptr.astype(np.int64)),
        shape=nm.shape,
    )

    # Prepare the matrix (fold the mask, analyse, lay out for the dot) once and share
    # it: each worker's context() reuses the plan's read-only arrays instead of copying
    # the whole CSC per thread.
    plan = b.plan(mask, fp, np.dtype(dtype))

    _CTX.update(
        plan=plan, integ=integ, weights=weights, wtmax_i=wtmax_i,
        offsets=offsets, framing=framing, n_rad=n_rad,
    )

    try:
        default_threads = max(1, len(os.sched_getaffinity(0)) // 2)
    except AttributeError:
        default_threads = max(1, (os.cpu_count() or 1) - 1)
    n_threads = args.threads or min(16, default_threads)
    n_threads = max(1, min(n_threads, n_frames))
    print(f"threads: {n_threads}")

    # Pin each worker to its own core (unless --no-pin): avoids OS migration and
    # keeps the shared matrix / per-worker buffers cache-resident on shared nodes.
    global _PIN_IT
    if args.no_pin or not hasattr(os, "sched_getaffinity"):
        _PIN_IT, init = None, None
    else:
        cores = sorted(os.sched_getaffinity(0))
        _PIN_IT = iter(cores * (n_threads + 1))
        init = _pin_worker
    if args.no_pin:
        print("pin: off")

    per_frame_bytes = mask.size * (np.dtype(dtype).itemsize + 4)   # values + uint32 indices
    batch = args.batch if args.batch is not None else 2
    print(f"batch: {batch} frames (~{per_frame_bytes * batch / 1e6:.0f} MB values+indices/worker)")
    _CTX["batch"] = batch

    print(f"setup: {time.time() - t0:.2f}s")
    t_setup = time.time()

    from concurrent.futures import ThreadPoolExecutor

    grouped = [indices.tolist() for indices in np.array_split(np.arange(n_frames), n_threads)]
    clipped = []
    total_sat = 0
    from tqdm import tqdm

    with tqdm(total=n_frames, unit="frame", desc="integrate", smoothing=0.1) as pbar:
        with ThreadPoolExecutor(max_workers=n_threads, initializer=init) as ex:
            for idxs, sums, ints, clip in ex.map(_integrate_indices, grouped):
                g_row = geom["row"][idxs]
                g_col = geom["col"][idxs]
                sum_cube[g_row, g_col] = sums
                int_cube[g_row, g_col] = ints
                for fi, pixel, value in clip:
                    clipped.append(dict(frame=fi, index=pixel, value=value))
                total_sat += sum(len(c[1]) for c in clip)
                pbar.update(len(idxs))

    t_integ = time.time()
    integ_time = t_integ - t_setup
    print(f"integrate: {integ_time:.2f}s ({n_frames / max(integ_time, 1e-9):.0f} fps)")

    framing.close()
    framing_fp.close()

    # write a poni next to the output for xrdmap + the external-link viewers
    out = args.output
    poni_out = args.write_poni or out.with_suffix(".poni")
    poni_out.write_text(integ.poni)

    _diffmapio.write_diffmap(
        str(out),
        geometry=geom,
        worker_config=wc,
        radial=integ.radial,
        intensity_cube=int_cube,
        sum_cube=sum_cube,
        poni_path=str(poni_out),
        frame_shape=shape,
        clipped=clipped,
        unit=args.unit,
    )

    print(f"shutdown/save: {time.time() - t_integ:.2f}s")
    print(f"wrote {out}")
    print(f"clipped/saturated pixels total: {total_sat}")
    print(f"total: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
