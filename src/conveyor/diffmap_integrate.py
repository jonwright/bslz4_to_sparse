"""Integrate a Bliss mesh scan into an xrdmap/diffmap-viewer-compatible HDF5.

Writes a pyFAI-diffmap-layout file (see _diffmapio.py) from the raw Eiger frames,
using the bslz4 sparse matrix product against a pyFAI geometry.  Saturated pixels
(value > wtmax = exposure*1e7) are clipped to wtmax (exactly) and listed.

Run as ``bs_diffmap_integrate`` (after install) or
``python -m bslz4_to_sparse.conveyor.diffmap_integrate``::

  bs_diffmap_integrate MASTER OUTPUT --poni PONI [--entry E] [--n-bins N]
      [--split full] [--mask MASK] [--max-frames N] [--input-file F] [--plan PLAN]
  bs_diffmap_integrate OUTPUT --plan PLAN       # reuse a recipe: fills in all args

MASTER: read-only Bliss master (taken from PLAN if omitted).  OUTPUT: the .h5 to
write.  A recipe ("plan") stores the geometry, the pyFAI CSC matrix, the detector
mask and the worker config, so a later run needs only OUTPUT --plan.  Every run
that builds one writes it: to PLAN if given, else to <OUTPUT stem>_plan.h5.  A
recipe is read only when --plan names an existing file; a setting given on the
command line that differs from the recipe's is refused.
"""

import argparse
import os
import threading
from pathlib import Path

import numpy as np

# The settings a recipe stores, with their defaults.  They default to None on the
# command line, so that a setting given there can be told from one that was not.
CONFIG = dict(poni=None, entry="2.1", n_bins=2000, split="full", unit="2th_deg",
              mask=None, fast=None, slow=None, map_shape=None)


def parse_args():
    p = argparse.ArgumentParser(prog="bs_diffmap_integrate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("master", nargs="?", type=Path, help="read-only Bliss master (from --plan if omitted)")
    p.add_argument("output", nargs="?", type=Path, help=".h5 to write (required)")
    p.add_argument("--entry", default=None, help="scan entry in the master (default 2.1)")
    p.add_argument("--poni", type=Path, default=None, help="pyFAI .poni geometry")
    p.add_argument("--n-bins", type=int, default=None, help="radial bins (default 2000)")
    p.add_argument("--split", choices=["no", "bbox", "full"], default=None,
        help="pyFAI pixel splitting (default full)")
    p.add_argument("--unit", default=None, help="radial unit (default 2th_deg, for xrdmap)")
    p.add_argument("--fast", default=None, help="fast motor name (from an amesh/dmesh title or fscan_parameters if omitted)")
    p.add_argument("--slow", default=None, help="slow motor name (from an amesh/dmesh title or fscan_parameters if omitted)")
    p.add_argument("--map-shape", nargs=2, type=int, default=None, metavar=("rows", "cols"),
        help="number of slow and fast motor positions (an amesh/dmesh title or fscan_parameters give them)")
    p.add_argument("--mask", default=None, help="detector mask file (pyFAI: 1 = masked), or 'auto' "
        "(the same as omitting it): the detector's own mask, else one derived from the first frame")
    p.add_argument("--max-frames", type=int, default=None, help="integrate only the first N frames (debug)")
    p.add_argument("--input-file", default=None, help="force the chunk source file::dataset (auto-detected)")
    p.add_argument("--write-poni", type=Path, default=None, help="where to write the .poni copy")
    p.add_argument("--threads", type=int, default=None,
        help="integration threads (default: min(16, CPU affinity // 2))")
    p.add_argument("--no-pin", action="store_true", help="do not pin each worker thread to a core")
    p.add_argument("--batch", type=int, default=None, help="frames per bslz4_to_sparse multi() call "
        "(default: 2)")
    p.add_argument("--plan", type=Path, default=None,
        help="a recipe file: read if it exists (the settings come from it), else written. "
        "Default: written to <OUTPUT stem>_plan.h5, never read.")
    return p.parse_args()


def open_frames(sources, n_frames):
    """(maps, files, table): one read-only mmap per source, its open file, and for each frame
    0..n_frames-1 (map number, byte offset, byte size) of its chunk."""
    import mmap

    import bslz4_to_sparse as b
    import h5py

    table = np.full((n_frames, 3), -1, np.int64)
    maps, files = [], []
    for fname, dset, v0, s0, count in sources:
        stop = min(v0 + count, n_frames)
        if v0 >= stop:
            continue
        with h5py.File(fname, "r") as h:
            off = b.locate_chunks(h[dset])
        fp = open(fname, "rb")
        files.append(fp)
        maps.append(mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ))
        for v in range(v0, stop):
            if s0 + v - v0 not in off:
                raise ValueError("%s::%s has no chunk for frame %d" % (fname, dset, s0 + v - v0))
            o, s = off[s0 + v - v0]
            table[v] = (len(maps) - 1, o, s)
    missing = np.flatnonzero(table[:, 0] < 0)
    if missing.size:
        raise ValueError("%d frames (first %d) are in no source file" % (missing.size, missing[0]))
    return maps, files, table


# Shared, read-only integration context the worker threads use; each thread keeps its
# own decoder (plan.context()) in _TLS.
_CTX = {}
_TLS = threading.local()
# Worker thread -> core, for sched_setaffinity; set once in main() (None = don't pin).
_PIN_IT = None
_PIN_LOCK = threading.Lock()


def _pin_worker():
    """Pin this (worker) thread to a core from the affinity mask; no-op if not pinning."""
    if _PIN_IT is None:
        return
    try:
        with _PIN_LOCK:
            core = next(_PIN_IT)
        os.sched_setaffinity(0, {core})
    except (OSError, ValueError, StopIteration):
        pass


def _integrate_indices(indices):
    """Integrate frames ``indices`` into the cubes of _CTX; returns (count, clipped)."""
    from bslz4_to_sparse import conveyor as cv

    ctx = _CTX
    d = getattr(_TLS, "decoder", None)
    if d is None:
        d = _TLS.decoder = ctx["plan"].context()
    maps, table, wtmax = ctx["maps"], ctx["table"], ctx["wtmax_i"]
    clip = []
    for j0 in range(0, len(indices), ctx["batch"]):
        idxs = indices[j0:j0 + ctx["batch"]]
        chunks = [maps[table[fi, 0]][table[fi, 1]:table[fi, 1] + table[fi, 2]] for fi in idxs]
        # The cut is wtmax: (values, pixidx) hold exactly the pixels above it, which is what
        # the exact "clip every pixel to wtmax" of clip_exact_powder needs: their
        # (weights x excess) is subtracted from the int64 sums.
        npx, (values, pixidx), powder = d.multi(chunks, wtmax)
        cv.clip_exact_powder(powder, npx, values, pixidx, wtmax, ctx["weights"], in_place=True)
        r, c = ctx["row"][idxs], ctx["col"][idxs]
        ctx["sum_cube"][r, c] = powder
        ctx["int_cube"][r, c] = ctx["integ"].intensity(powder, d.powder_scale)
        for i, fi in enumerate(idxs):
            k = int(npx[i])
            if k:
                clip.append((fi, pixidx[i, :k].copy(), values[i, :k].astype(np.int64)))
    return len(indices), clip


def _geom_to_json(geom):
    d = {}
    for k, v in geom.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif k in ("map_shape", "frame_shape"):
            d[k] = [int(x) for x in v]
        elif k == "source":
            d[k] = [[str(f), str(ds), int(v0), int(s0), int(n)] for f, ds, v0, s0, n in v]
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
    src = d["source"]
    if src is not None and len(src) == 2 and isinstance(src[0], str):
        src = [[src[0], src[1], 0, 0, d["n_points"]]]   # a recipe of the first version: one file
    g["source"] = [tuple(s) for s in src]
    g["dtype"] = np.dtype(d["dtype"])
    return g


def _setup_value(k, v):
    """A setting as the recipe stores it."""
    if v is None:
        return None
    if k in ("poni", "mask"):
        return str(v)
    if k == "map_shape":
        return [int(x) for x in v]
    return v


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
    """(integ, mask, wc, geom, setup) of a recipe file."""
    import json

    import h5py
    from bslz4_to_sparse import conveyor as cv

    with h5py.File(str(path), "r") as H:
        setup = json.loads(H["setup"][()].decode("utf-8"))
        geom = _geom_from_json(json.loads(H["geometry"][()].decode("utf-8")))
        mask = H["mask"][()].astype(np.uint8)
        wc = json.loads(H["worker_config"][()].decode("utf-8"))
    integ = cv.Integrator.load(str(path))
    return integ, mask, wc, geom, setup


def _detector_mask(ai, mask_arg, first_frame):
    """pyFAI's convention (1 = masked), or None: the file mask_arg, else the detector's own mask,
    else the pixels pyFAI's dynamic_mask finds in the first frame."""
    if mask_arg and mask_arg.lower() != "auto":
        import fabio

        return (fabio.open(mask_arg).data > 0).astype(np.int8)
    if ai.detector.mask is not None:
        return (ai.detector.mask > 0).astype(np.int8)
    dm = ai.detector.dynamic_mask(first_frame())
    return dm.astype(np.int8) if np.any(dm) else None


class _NoBar(object):
    """Stands in for tqdm when it is not installed."""

    def __init__(self, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n):
        pass


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

    import bslz4_to_sparse as b
    from bslz4_to_sparse import conveyor as cv
    from bslz4_to_sparse.conveyor import _diffmapio

    given = dict((k, _setup_value(k, getattr(args, k))) for k in CONFIG
                 if getattr(args, k) is not None)
    if args.plan is not None and args.plan.exists():
        integ, mask, wc, geom, setup = _load_plan(args.plan)
        if args.master is not None and str(args.master) != setup["master"]:
            raise SystemExit("--plan was built for %s, not %s" % (setup["master"], args.master))
        differ = ["--%s %s (the plan has %s)" % (k.replace("_", "-"), v, setup[k])
                  for k, v in sorted(given.items()) if v != setup[k]]
        if differ:
            raise SystemExit("%s was built with other settings: %s" % (args.plan, "; ".join(differ)))
        args.master = Path(setup["master"])
        for k in CONFIG:
            setattr(args, k, setup[k])
        args.poni = Path(setup["poni"])
        print("plan:", args.plan)
    else:
        if args.master is None or args.poni is None:
            raise SystemExit("give MASTER and --poni (or --plan with an existing recipe)")
        for k, v in CONFIG.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
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
    dtype = np.dtype(geom["dtype"])
    print("map %dx%d, %d pts, frame %s %s, wtmax=%.3e" % (rows, cols, n_frames, shape, dtype.name,
                                                          geom["wtmax"]))

    sources = geom["source"]
    if args.input_file:
        f, ds = args.input_file.split("::", 1)
        sources = [(f, ds, 0, 0, geom["n_points"])]
    for f, ds, v0, s0, n in sources:
        print("chunk source: frames %d..%d from %s :: %s" % (v0, v0 + n - 1, f, ds))

    wtmax_i = int(geom["wtmax"])
    maps, files, table = open_frames(sources, n_frames)

    if integ is None:
        import pyFAI
        pyFAI.use_opencl = False
        from pyFAI import load
        ai = load(str(args.poni))
        ai.detector.shape = shape
        file_mask = args.mask and args.mask.lower() != "auto"
        if file_mask:
            ai.detector.mask = _detector_mask(ai, args.mask, None)

        integ = cv.Integrator.from_pyfai(ai, args.n_bins, unit=args.unit, split=args.split)

        def first_frame():
            m, o, s = table[0]
            return b.decompress([maps[m][o:o + s]], dtype).reshape(shape)

        det_mask = ai.detector.mask if file_mask else _detector_mask(ai, args.mask, first_frame)
        mask = np.ones(shape, np.uint8)
        if det_mask is not None:
            mask = (1 - det_mask).astype(np.uint8)
            print("mask: %d masked pixels" % int(np.count_nonzero(det_mask)))

        wc = _diffmapio.worker_config_dict(ai, args.n_bins, shape, unit=args.unit)

        plan_path = args.plan or args.output.parent / (args.output.stem + "_plan.h5")
        setup = dict((k, _setup_value(k, getattr(args, k))) for k in CONFIG)
        setup["master"] = str(args.master)
        try:
            _save_plan(plan_path, integ, mask, wc, geom, setup)
            print("plan written:", plan_path)
        except OSError as err:
            print("plan NOT written (%s): %s" % (plan_path, err))

    n_rad = integ.nbins
    print("radial bins:", n_rad, "range:", float(integ.radial[0]), float(integ.radial[-1]))
    print("split=%s CSC nnz=%d (~%.2f output entries/input pixel)"
          % (args.split, int(integ.csc.nnz), integ.csc.nnz / np.prod(shape)))

    # Prepare the matrix (fold the mask, analyse, lay out for the dot) once and share
    # it: each worker's context() reuses the plan's read-only arrays.
    fp = b.fixed_point_matrix(integ.csc, mask, dtype)
    plan = b.plan(mask, fp, dtype)

    # The workers write their frames' cells straight into these (no two frames share a cell).
    _CTX.update(
        plan=plan, integ=integ, weights=fp.to_scipy(), wtmax_i=wtmax_i,
        maps=maps, table=table, row=geom["row"], col=geom["col"],
        int_cube=np.zeros((rows, cols, n_rad), np.float32),
        sum_cube=np.zeros((rows, cols, n_rad), np.int64),
    )

    try:
        default_threads = max(1, len(os.sched_getaffinity(0)) // 2)
    except AttributeError:
        default_threads = max(1, (os.cpu_count() or 1) // 2)
    n_threads = args.threads or min(16, default_threads)
    n_threads = max(1, min(n_threads, n_frames))
    print("threads: %d" % n_threads)

    # Pin each worker to its own core (unless --no-pin): avoids OS migration and
    # keeps the shared matrix / per-worker buffers cache-resident on shared nodes.
    global _PIN_IT
    if args.no_pin or not hasattr(os, "sched_getaffinity"):
        _PIN_IT, init = None, None
    else:
        _PIN_IT = iter(sorted(os.sched_getaffinity(0)))
        init = _pin_worker
    if args.no_pin:
        print("pin: off")

    batch = args.batch if args.batch is not None else 2
    per_frame_bytes = mask.size * (dtype.itemsize + 4)   # values + uint32 indices
    print("batch: %d frames (~%.0f MB values+indices/worker)" % (batch, per_frame_bytes * batch / 1e6))
    _CTX["batch"] = batch

    print("setup: %.2fs" % (time.time() - t0))
    t_setup = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from tqdm import tqdm as bar
    except ImportError:
        bar = _NoBar

    # Small groups, taken by whichever thread is free: the load balances itself.
    group = batch * max(1, 32 // batch)
    groups = [list(range(i, min(i + group, n_frames))) for i in range(0, n_frames, group)]
    clipped = []
    with bar(total=n_frames, unit="frame", desc="integrate", smoothing=0.1) as pbar:
        with ThreadPoolExecutor(max_workers=n_threads, initializer=init) as ex:
            for fut in as_completed([ex.submit(_integrate_indices, g) for g in groups]):
                count, clip = fut.result()
                for fi, pixel, value in clip:
                    clipped.append(dict(frame=fi, index=pixel, value=value))
                pbar.update(count)
    clipped.sort(key=lambda c: c["frame"])
    total_sat = sum(len(c["index"]) for c in clipped)

    t_integ = time.time()
    integ_time = t_integ - t_setup
    print("integrate: %.2fs (%.0f fps)" % (integ_time, n_frames / max(integ_time, 1e-9)))

    for m in maps:
        m.close()
    for fp_ in files:
        fp_.close()

    # write a poni next to the output for xrdmap + the external-link viewers
    out = args.output
    poni_out = args.write_poni or out.with_suffix(".poni")
    poni_out.write_text(integ.poni)

    _diffmapio.write_diffmap(
        str(out),
        geometry=geom,
        worker_config=wc,
        radial=integ.radial,
        intensity_cube=_CTX["int_cube"],
        sum_cube=_CTX["sum_cube"],
        poni_path=str(poni_out),
        frame_shape=shape,
        clipped=clipped,
        unit=args.unit,
    )
    _CTX.clear()

    print("shutdown/save: %.2fs" % (time.time() - t_integ))
    print("wrote %s" % out)
    print("clipped/saturated pixels total: %d" % total_sat)
    print("total: %.2fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
