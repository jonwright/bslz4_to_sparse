"""Matrix-dot kernels on real pyFAI geometries.

The cases follow examples/pyfai_csc_integration.ipynb (synthetic LaB6 powder
rings with texture and Poisson noise, ~9 % non-zero pixels, uint16,
bitshuffle-LZ4), extended to:

  detectors     Eiger2 4M and Eiger2 16M
  beam centre   mid (near the detector centre) and corner (near pixel (40, 40))
  1D            pyFAI split "no" (histogram), "bbox", "full", each at ~1 and ~5
                radial bins per pixel width (unit r_mm, so bins are uniform in
                radius and "bins per pixel" means what it says)
  2D            (radial x 360 azimuth) with split "no" and "bbox", q_nm^-1,
                ~1 radial bin per pixel width
  rings         8 LaB6 rings x 360 azimuth, the ring windows sized so the rings
                hold ~8 % of the unmasked pixels, summed from the 2D bbox matrix
  fazit         FAZIT (P. Boesecke's fast radial regrouping, as in ImageD11):
                unmasked pixels sorted by (radial bin of one pixel width,
                azimuth), each to its own slot -- a permutation; csc-permute
                writes it in the pixel dtype ("packed")
  rings+moment  the rings with a second output per (ring, azimuth): sum(q*I)
                beside sum(I), q = the pixel-centre q.  "inter" puts them
                side by side [I, qI, I, qI, ...]; "block" puts all I then all qI.
                Rings, and rings+moment, are built from the 2D bbox matrix and
                again from the 2D no-split matrix (" no" in the case name).

Every kernel in b.available_dots() is run on every matrix it accepts, on the
forced-dense route and at the default per-block routing, pinned to one core,
100 frames (in batches of 25) read into memory once (no I/O in the timing).
Each kernel is timed on its own: built once, one untimed warmup batch, then
one pass over all frames on each route, before the next kernel starts; kernels
and cases are never interleaved, so a kernel's matrix is as warm in L2/L3 as
in a long run.  The
default is Eiger2 4M and a core set of dots (--det 4M,16M and --dots all for
the full sweep).  Each result is checked against the csc kernel.  A matrix-structure line
(entries per pixel, consecutive runs, active pixels) is printed per case so
the timings can be read against what the matrix looks like.

Generated frames and pyFAI matrices are cached in $BSLZ4_BENCH_DIR (default
/tmp/$USER/bslz4_bench).  Results are appended as JSON lines to --out.

Usage:
  python3 tools/bench_suite.py [--det 4M,16M] [--centre mid,corner]
        [--cases 1d,2d,rings] [--dots csc,csc-run,...|all] [--frames 25] [--cpu 7]
        [--out examples/dot_suite.jsonl]
"""
import argparse
import datetime
import json
import os
import platform
import socket
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.environ.get("BSLZ4_TO_SPARSE_PATH", os.path.join(REPO, "lib"))
sys.path.insert(0, path)

import numpy as np
import scipy.sparse as sp
import h5py
import hdf5plugin
import pyFAI
from pyFAI.calibrant import get_calibrant
from pyFAI.method_registry import IntegrationMethod
import bslz4_to_sparse as b

CACHE = os.environ.get("BSLZ4_BENCH_DIR", "/tmp/%s/bslz4_bench" % os.environ.get("USER", "u"))
DETECTORS = {"4M": "Eiger2_4M", "16M": "Eiger2_16M"}
NFRAMES = {"4M": 100, "16M": 50}   # generated and cached; --frames picks how many are timed
DEFAULT_CASES = ("1D no x1", "1D bbox x1", "1D bbox x5",
                 "2D no", "2D bbox",
                 "rings", "rings+mom inter", "rings no", "rings+mom inter no", "fazit")
FAZIT_DOTS = ("csc-nosplit", "csc-permute", "csc-permute-runs")
CORE_DOTS = ("csc", "csc-run", "csc-nosplit", "csc-nosplit-moment", "padded-sse2",
             "bsb-csr", "bsb-csr-nosplit",
             # experimental CSC-entry dots (src/_csc_variants.py)
             "csc-nosplit-dump", "csc-nosplit-moment-dump", "csc-run-u16", "csc-nosplit-u16",
             "csc-run-delta", "csc-nosplit-delta", "csc-nosplit-walk", "csc-tile",
             "csc-run-moment", "csc-int", "csc-run-int", "csc-run-int16", "csc-permute",
             "csc-permute-runs")
BATCH = 25
NAZIM = 360
NRINGS = 8
RING_FRACTION = 0.08
WAVELENGTH = 2.85e-11
DIST = 0.25


# ---------------------------------------------------------------- geometry

def make_ai(det, centre):
    d = pyFAI.detector_factory(DETECTORS[det])
    ni, nj = d.shape
    if centre == "mid":
        c1, c2 = 0.5 * ni, 0.5 * nj
    else:
        c1, c2 = 40.0, 40.0
    # small tilts as in the notebook; the poni is near the beam centre
    return pyFAI.load({"detector": d.name, "dist": DIST,
                       "poni1": c1 * d.pixel1, "poni2": c2 * d.pixel2,
                       "rot1": 0.01, "rot2": 0.02, "rot3": 0.03,
                       "wavelength": WAVELENGTH})


def radial_pixels(ai):
    """Radial extent of the unmasked detector in pixel widths."""
    r = ai.rArray()[ai.detector.mask == 0] / ai.detector.pixel1
    return int(np.ceil(r.max() - r.min()))


# ---------------------------------------------------------------- data

def frames_file(det, centre, ai):
    fname = os.path.join(CACHE, "frames_%s_%s.h5" % (det, centre))
    if os.path.exists(fname):
        return fname
    os.makedirs(CACHE, exist_ok=True)
    shape = ai.detector.shape
    cal = get_calibrant("LaB6")
    cal.wavelength = ai.wavelength
    qarr, chi = ai.qArray(), ai.chiArray()
    mean = np.full(shape, 0.001)
    for k, qk in enumerate(cal.get_peaks("q_nm^-1")[:60]):
        amp = 4.0 / (1 + 0.3 * k)
        texture = 1 + 0.5 * np.cos(2 * chi + 0.4 * k)
        mean += amp * texture * np.exp(-0.5 * ((qarr - qk) / 0.15) ** 2)
    rng = np.random.default_rng(0)
    bad = ai.detector.mask.astype(bool)
    tmp = fname + ".tmp"
    with h5py.File(tmp, "w") as h5f:
        ds = h5f.create_dataset("data", shape=(NFRAMES[det],) + shape, dtype=np.uint16,
                                chunks=(1,) + shape,
                                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        for i in range(NFRAMES[det]):
            frame = rng.poisson(mean).astype(np.uint16)
            frame[bad] = 0
            ds[i] = frame
    os.rename(tmp, fname)
    return fname


def load_chunks(fname):
    with h5py.File(fname, "r") as h5f:
        ds = h5f["data"]
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(ds.shape[0])]
        f0 = ds[0]
    return chunks, f0


# ---------------------------------------------------------------- matrices

def _engine_csc(ai, split, dim, npt, unit, npt_azim=None):
    method = IntegrationMethod.select_one_available((split, "csc", "cython"), dim=dim)
    ones = np.ones(ai.detector.shape, np.float32)
    if dim == 1:
        ai.integrate1d(ones, npt, unit=unit, method=method)
        nbins = npt
    else:
        ai.integrate2d(ones, npt, npt_azim, unit=unit, method=method)
        nbins = npt * npt_azim
    e = ai.engines[method].engine
    npix = len(e.indptr) - 1
    return sp.csc_matrix((np.asarray(e.data, np.float32), np.asarray(e.indices, np.int32),
                          np.asarray(e.indptr, np.int32)), shape=(nbins, npix))


def _cached(key, build):
    fname = os.path.join(CACHE, key + ".npz")
    if os.path.exists(fname):
        z = np.load(fname)
        return sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    M = build()
    os.makedirs(CACHE, exist_ok=True)
    np.savez(fname + ".tmp.npz", data=M.data, indices=M.indices, indptr=M.indptr,
             shape=np.array(M.shape))
    os.rename(fname + ".tmp.npz", fname)
    return M


def _f32(M):
    M = M.tocsc()
    M.sort_indices()
    return sp.csc_matrix((M.data.astype(np.float32), M.indices.astype(np.int32),
                          M.indptr.astype(np.int32)), shape=M.shape)


def ring_windows(ai):
    """8 LaB6 rings spread over the detector's q range, with one common
    half-width (q_nm^-1) chosen so the rings hold ~RING_FRACTION of the
    unmasked pixels."""
    q = ai.qArray()[ai.detector.mask == 0]
    lo, hi = np.percentile(q, [2, 98])
    cal = get_calibrant("LaB6")
    cal.wavelength = ai.wavelength
    peaks = np.array([p for p in cal.get_peaks("q_nm^-1") if lo < p < hi])
    pick = peaks[np.unique(np.linspace(0, len(peaks) - 1, NRINGS).round().astype(int))]
    # the half-width must keep neighbouring windows apart
    gap = np.diff(pick).min() / 2 if len(pick) > 1 else hi - lo
    a, c = 0.0, gap
    for _ in range(50):
        h = 0.5 * (a + c)
        frac = np.any(np.abs(q[:, None] - pick[None, :]) < h, axis=1).mean() \
            if q.size < 5e6 else _frac_chunked(q, pick, h)
        if frac < RING_FRACTION:
            a = h
        else:
            c = h
    return pick, 0.5 * (a + c)


def _frac_chunked(q, pick, h):
    n = 0
    for s in range(0, q.size, 2_000_000):
        n += np.any(np.abs(q[s:s + 2_000_000, None] - pick[None, :]) < h, axis=1).sum()
    return n / q.size


def build_cases(ai, det, centre, which):
    """[(name, csc_matrix)] for this geometry."""
    rp = radial_pixels(ai)
    tag = "%s_%s" % (det, centre)
    out = []
    if "1d" in which:
        for split in ("no", "bbox", "full"):
            for bpp in (1, 5):
                npt = rp * bpp
                M = _cached("m1d_%s_%s_%d" % (tag, split, bpp),
                            lambda: _engine_csc(ai, split, 1, npt, "r_mm"))
                out.append(("1D %s x%d" % (split, bpp), M))
    if "2d" in which or "rings" in which:
        M2 = {}
        for split in ("no", "bbox"):
            M2[split] = _cached("m2d_%s_%s" % (tag, split),
                                lambda: _engine_csc(ai, split, 2, rp, "q_nm^-1", NAZIM))
        if "2d" in which:
            out.append(("2D no", M2["no"]))
            out.append(("2D bbox", M2["bbox"]))
    if "fazit" in which:
        for name, tile in FAZIT_TILES.items():
            out.append((name, fazit_matrix(ai, tile)))
    if "rings" in which:
        r2 = ai.integrate2d(np.ones(ai.detector.shape, np.float32), rp, NAZIM, unit="q_nm^-1",
                            method=IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=2))
        qc = r2.radial
        pick, h = ring_windows(ai)
        W = np.array([np.abs(qc - p) < h for p in pick], dtype=np.float64)
        S = sp.kron(sp.csr_matrix(W), sp.identity(NAZIM), format="csr")
        qpix = ai.qArray().ravel().astype(np.float64)
        print("  rings: %d LaB6 rings at q = %s nm^-1, half-width %.3g nm^-1"
              % (len(pick), np.round(pick, 2).tolist(), h))
        for split in ("bbox", "no"):
            G = (S @ M2[split]).tocsc()
            GQ = G @ sp.diags(qpix)
            nG = G.shape[0]
            # interleaved: row 2k = sum I of output k, row 2k+1 = sum qI
            perm = np.empty(2 * nG, np.int64)
            perm[0::2] = np.arange(nG)
            perm[1::2] = nG + np.arange(nG)
            inter = sp.vstack([G, GQ]).tocsr()[perm]
            sfx = "" if split == "bbox" else " no"
            out.append(("rings" + sfx, _f32(G)))
            out.append(("rings+mom inter" + sfx, _f32(inter)))
            out.append(("rings+mom block" + sfx, _f32(sp.vstack([G, GQ]))))
    return out


FAZIT_TILES = {"fazit": None, "fazit 4x4 az": (4, 4, "az"), "fazit 4x4 rad": (4, 4, "rad"),
               "fazit 2x8 az": (2, 8, "az"), "fazit 1x32 az": (1, 32, "az")}


def fazit_matrix(ai, tile=None):
    """FAZIT (P. Boesecke's fast radial regrouping, as in ImageD11) as a
    matrix: the unmasked pixels sorted by (radial bin, azimuth), each to its
    own output slot (a permutation, weight 1).  The output is the image in
    that order.

    tile=None: radial bins of one pixel width, exact azimuth order.
    tile=(wr, wa, fast): approximate -- output bins of wr pixels radially by
    about wa pixels of arc; the sort is stable, so inside a bin the pixels
    keep image (raster) order and a row segment becomes a run of consecutive
    output slots.  fast="az": bins ordered (radial, azimuth); fast="rad":
    fixed angular sectors (about wa pixels of arc at the median radius),
    ordered (sector, radial)."""
    valid = (ai.detector.mask == 0).ravel()
    r = ai.rArray().ravel() / ai.detector.pixel1
    chi = ai.chiArray().ravel()
    pix = np.flatnonzero(valid)
    if tile is None:
        order = np.lexsort((chi[pix], np.floor(r[pix])))
    else:
        wr, wa, fast = tile
        rbin = np.floor(r[pix] / wr).astype(np.int64)
        if fast == "az":
            rc = (rbin + 0.5) * wr
            abin = np.floor((chi[pix] + np.pi) * rc / wa).astype(np.int64)
            order = np.lexsort((abin, rbin))
        else:
            nsect = max(1, int(round(2 * np.pi * np.median(r[pix]) / wa)))
            sect = np.floor((chi[pix] + np.pi) / (2 * np.pi) * nsect).astype(np.int64)
            order = np.lexsort((rbin, sect))
    adr = np.empty(pix.size, np.int64)
    adr[order] = np.arange(pix.size)
    indptr = np.zeros(valid.size + 1, np.int64)
    np.cumsum(valid, out=indptr[1:])
    return sp.csc_matrix((np.ones(pix.size, np.float32), adr.astype(np.int32),
                          indptr.astype(np.int32)), shape=(pix.size, valid.size))


def structure(M, valid):
    """One line describing the matrix as the kernels see it (after the mask
    is folded): active pixels, entries per active pixel, run structure."""
    indptr = M.indptr.astype(np.int64)
    n = np.diff(indptr) * (valid.ravel() > 0)
    act = n > 0
    hist = np.bincount(n[act], minlength=7)
    tail = hist[6:].sum()
    pix = np.repeat(np.arange(M.shape[1]), np.diff(indptr))
    ind = M.indices.astype(np.int64)
    same = pix[1:] == pix[:-1]
    broken = np.unique(pix[1:][same & (np.diff(ind) != 1)]).size
    nnz = int(n.sum())
    return {"npix": int(M.shape[1]), "nbins": int(M.shape[0]), "nnz": nnz,
            "active": float(act.mean()), "mean_len": float(n[act].mean()) if act.any() else 0.0,
            "len_hist": [int(x) for x in hist[1:6]] + [int(tail)],
            "max_len": int(n.max()), "nonrun_pixels": int(broken)}


# ---------------------------------------------------------------- timing

def time_dot(integ, chunks):
    """ms/frame over all of `chunks` in batches, after one untimed warmup
    batch: the warmup brings this kernel's matrix (indices, weights) into
    L2/L3 the way a long run would have it, and sizes the work buffers."""
    nf = len(chunks)
    integ(chunks[:BATCH], 0)
    res = []
    t0 = time.perf_counter()
    for s in range(0, nf, BATCH):
        res.append(integ(chunks[s:s + BATCH], 0)[2].copy())
    t = time.perf_counter() - t0
    return t * 1e3 / nf, np.concatenate(res)


def machine():
    info = b.build_info()
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    return {"host": socket.gethostname(), "cpu": cpu, "git": info["git"],
            "src_sha256": info["src_sha256"][:12], "compiler": info["compiler"]}


def main():
    global BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="4M")
    ap.add_argument("--centre", default="mid,corner")
    ap.add_argument("--cases", default="default",
                    help="'default' (%s), 'all', or a comma list of case names"
                         % ", ".join(DEFAULT_CASES))
    ap.add_argument("--dots", default=",".join(CORE_DOTS),
                    help="comma list, or 'all' for every available dot")
    ap.add_argument("--frames", type=int, default=100, help="frames timed (default 100)")
    ap.add_argument("--batch", type=int, default=BATCH, help="frames per decode call (default 25)")
    ap.add_argument("--cpu", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(REPO, "examples", "dot_suite.jsonl"))
    a = ap.parse_args()
    BATCH = a.batch
    dots = list(b.available_dots()) if a.dots == "all" else \
        [d for d in a.dots.split(",") if d in b.available_dots()]
    which = {"1d", "2d", "rings", "fazit"}
    keep = None if a.cases == "all" else \
        set(DEFAULT_CASES if a.cases == "default" else a.cases.split(","))
    os.sched_setaffinity(0, {a.cpu})
    mach = machine()
    print("%s | %s | git %s | pinned cpu %d | cache %s" % (mach["host"], mach["cpu"], mach["git"],
                                                         a.cpu, CACHE))
    routes = (("dense", 1e9), ("default", b.get_dense_sparse_threshold()))
    saved = b.get_dense_sparse_threshold()
    for det in a.det.split(","):
        for centre in a.centre.split(","):
            ai = make_ai(det, centre)
            valid = np.ascontiguousarray((1 - ai.detector.mask).astype(np.uint8))
            fname = frames_file(det, centre, ai)
            chunks, f0 = load_chunks(fname)
            chunks = chunks[:a.frames]
            comp = sum(len(c) for c in chunks) / len(chunks)
            nzf = np.count_nonzero(f0) / f0.size
            print("\n##### Eiger2 %s, beam %s: %d frames, %.0f kB/frame (x%.1f), %.1f %% non-zero, "
                  "radial extent %d px" % (det, centre, len(chunks), comp / 1e3,
                                           f0.nbytes / comp, 100 * nzf, radial_pixels(ai)))
            cases = [(n, M) for n, M in build_cases(ai, det, centre, which)
                     if keep is None or n in keep]
            stats = {}
            for name, M in cases:
                st = stats[name] = structure(M, valid)
                print("  %-34s bins %8d nnz %9d active %5.1f %% len/px %.2f hist(1..5,6+) %s "
                      "max %d non-run px %d" % (name, st["nbins"], st["nnz"], 100 * st["active"],
                                                st["mean_len"], st["len_hist"], st["max_len"],
                                                st["nonrun_pixels"]))
            print("  columns: " + ", ".join("%d=%s" % (i, d) for i, d in enumerate(dots)))
            print("  ms/frame" + "".join("%7d" % i for i in range(len(dots))))
            for name, M in cases:
                # one kernel at a time, both routes back to back, so its matrix
                # stays warm; never interleave kernels or cases
                times = {r: {} for r, _ in routes}
                ref = {}
                notes = []
                for dot in ["csc"] + [d for d in dots if d != "csc"]:
                    if name.startswith("fazit") and dot not in ("csc",) + FAZIT_DOTS:
                        for r, _ in routes:
                            times[r][dot] = None
                        continue
                    try:
                        integ = b.chunk2sparseCSCmulti(valid, M, dtype=np.uint16, dot=dot)
                    except ValueError:
                        for r, _ in routes:
                            times[r][dot] = None
                        continue
                    v = getattr(integ, "variant", None)
                    if v is not None and v.note:
                        notes.append("%s: %s" % (dot, v.note))
                    # exact re-layouts match csc to rounding; fixed-point
                    # weights and csc-run-moment's own w*q do not, by design
                    rtol = 1e-9
                    if v is not None and v.scale is not None:
                        rtol = 4 * v.scale * max(1, int(np.diff(M.indptr).max()))
                    elif dot == "csc-run-moment":
                        rtol = 1e-6
                    for route, th in routes:
                        b.set_dense_sparse_threshold(th)
                        t, p = time_dot(integ, chunks)
                        if route not in ref:
                            ref[route] = p
                        else:
                            r0 = ref[route]
                            err = np.abs(p - r0).max() / max(1.0, np.abs(r0).max())
                            assert err < rtol, (det, centre, name, route, dot, err, rtol)
                        times[route][dot] = t
                    del integ
                    b.set_dense_sparse_threshold(saved)
                for route, _ in routes:
                    tr = times[route]
                    print("  %-8s" % route + "".join("%7s" % ("-" if tr.get(d) is None else "%.2f" % tr[d])
                                                     for d in dots) + "   " + name)
                    with open(a.out, "a") as fh:
                        fh.write(json.dumps({
                            "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "machine": mach, "pinned_cpu": a.cpu, "detector": det, "centre": centre,
                            "case": name, "route": route, "nframes": len(chunks), "batch": BATCH,
                            "kb_per_frame": comp / 1e3, "nonzero_frac": nzf,
                            "structure": stats[name], "ms_per_frame": tr}) + "\n")
                for note in notes:
                    print("           . %s" % note)
                sys.stdout.flush()
            b.set_dense_sparse_threshold(saved)


if __name__ == "__main__":
    main()
