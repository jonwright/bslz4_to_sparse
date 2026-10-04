"""How small do the matrix arrays get under lz4 / bitshuffle?

For the cached pyFAI matrices of tools/bench_suite.py (mask folded as the
integrator does), each array a kernel reads is written to an in-memory HDF5
dataset with hdf5plugin's filters and its stored size compared with the raw
size:
  lz4         hdf5plugin.LZ4()
  bslz4       hdf5plugin.Bitshuffle(nelems=0, cname="lz4")   (the frame format)
  bszstd      hdf5plugin.Bitshuffle(nelems=0, cname="zstd")

Arrays: the CSC (indptr u32, indices u32, data f32), and the per-pixel
variants of the new dots: run starts (u32, u16), start deltas from pixel to
pixel (i16), the histogram bin per pixel (u32), and fixed-point weights
(u32 / u16, as csc-run-int / csc-run-int16 build them).

Usage: python3 tools/compress_matrices.py [--det 4M,16M] [--centre mid]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_suite as bs  # noqa: E402

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import hdf5plugin  # noqa: E402
import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import _csc_variants as V  # noqa: E402

FILTERS = {"lz4": hdf5plugin.LZ4(),
           "bslz4": hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
           "bszstd": hdf5plugin.Bitshuffle(nelems=0, cname="zstd")}
CHUNK = 1 << 20     # elements per HDF5 chunk


def stored(arr, filt):
    arr = np.ascontiguousarray(arr)
    with h5py.File("mem.h5", "w", driver="core", backing_store=False) as h:
        ds = h.create_dataset("a", data=arr, chunks=(min(CHUNK, arr.size),), **filt)
        return sum(ds.id.get_chunk_info(i).size for i in range(ds.id.get_num_chunks()))


def arrays(nm):
    """name -> array, for this (mask-folded) matrix."""
    out = {"csc indptr u32": nm.indptr, "csc indices u32": nm.indices, "csc data f32": nm.data}
    # delta codings for plain lz4: indptr steps (entries per pixel), and the
    # flat indices array as differences from the previous entry
    out["csc indptr delta u8"] = np.diff(nm.indptr.astype(np.int64)).clip(0, 255).astype(np.uint8)
    di = np.diff(nm.indices.astype(np.int64), prepend=0)
    out["csc indices delta i32"] = di.astype(np.int32)
    if np.abs(di[1:]).max() < 32768:
        out["csc indices delta i16"] = di.astype(np.int16)
    n = np.diff(nm.indptr.astype(np.int64))
    try:
        starts, _ = V._runs(nm, "x")
        out["run start u32"] = starts.astype(np.uint32)
        if nm.nbins <= 65536:
            out["run start u16"] = starts.astype(np.uint16)
        d = np.diff(starts, prepend=0)
        if np.abs(d).max() < 32768:
            out["run start delta i16"] = d.astype(np.int16)
    except ValueError:
        pass
    if (n <= 1).all() and (nm.data == 1).all():
        bins = np.where(n == 1, 0, 0xFFFFFFFF).astype(np.int64)
        bins[n == 1] = nm.indices[nm.indptr[:-1][n == 1]]
        out["nosplit bin u32"] = bins.astype(np.uint32)
    else:
        for wtype, bits in ((np.uint32, 32), (np.uint16, 16)):
            try:
                fb = V._fixed_point_bits(nm, np.uint16, bits)
                out["weights fixed u%d" % bits] = V._quantise(nm, fb, wtype)
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="4M,16M")
    ap.add_argument("--centre", default="mid")
    a = ap.parse_args()
    names = ("1D no x1", "1D bbox x1", "1D bbox x5", "2D no", "2D bbox", "rings")
    for det in a.det.split(","):
        for centre in a.centre.split(","):
            ai = bs.make_ai(det, centre)
            mask = (1 - ai.detector.mask).astype(np.uint8).ravel()
            cases = dict(bs.build_cases(ai, det, centre, {"1d", "2d", "rings"}))
            print("\n##### Eiger2 %s, beam %s: stored size / raw size (MB raw)" % (det, centre))
            print("  %-12s %-22s %9s %8s %8s %8s" % ("case", "array", "raw MB", "lz4", "bslz4", "bszstd"))
            for name in names:
                nm = b._fold_mask(b.normalise_matrix(cases[name], npix=mask.size), mask)
                for an, arr in arrays(nm).items():
                    raw = arr.nbytes
                    r = [stored(arr, FILTERS[f]) / raw for f in FILTERS]
                    print("  %-12s %-22s %9.1f %8.3f %8.3f %8.3f" % (name, an, raw / 1e6, *r))
                sys.stdout.flush()


if __name__ == "__main__":
    main()
