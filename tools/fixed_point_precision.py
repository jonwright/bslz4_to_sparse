#!/usr/bin/env python3
"""How coarse are the fixed-point (exact) sums of your matrix and your pixel type?

    python3 tools/fixed_point_precision.py --h5 eiger_0000.h5 --dset /entry_0000/ESRF-ID11/eiger/data \
        --poni calib.poni --mask mask.h5 [--npt 500] [--frames 8]

fixed_point_matrix rounds the weights to integers, round(w * 2**b), stored in 32 bits, and
chunk2sparseCSC given it sums pixel x weight in int64.  The number of fractional bits b is the smaller of

  * the weights' width:        b <= floor(log2((2**32 - 1) / the largest weight))
  * the accumulator's headroom: b <= floor(62 - log2(the largest pixel value + 1)
                                           - log2(the largest sum of weights in one bin))

so it falls by one for each doubling of the largest pixel value that the sums must hold: 16-bit
pixels are limited by the width of the weights (b = 31 or 32), 32-bit pixels by the headroom (about
17 for a 1D matrix of an Eiger 4M, whose fullest bin holds a few thousand pixels).  Rounding the
weights moves a bin sum by about 2**-b relative to the weights, so this tool prints, for your
matrix: b for several largest pixel values, and what rounding to b bits does to the bin sums of a real
frame of your data (the largest and the median relative difference from float64 sums with the same
weights).  maxval= tells fixed_point_matrix the largest value the data can reach, to get more bits.

Needs pyFAI, h5py, hdf5plugin and scipy.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--dset", required=True)
    ap.add_argument("--poni", required=True)
    ap.add_argument(
        "--mask", required=True, help="HDF5 file with a dataset 'mask' (non-zero = masked)"
    )
    ap.add_argument("--npt", type=int, default=500, help="radial bins of the 1D matrix")
    ap.add_argument(
        "--frames", type=int, default=8, help="how many frames to look at for the largest value"
    )
    a = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401  (the bitshuffle filter)
    import pyFAI
    import scipy.sparse as sp

    from bslz4_to_sparse import _matrix
    from bslz4_to_sparse.conveyor import Integrator

    with h5py.File(a.mask, "r") as h:
        bad = h["mask"][()] != 0
    with h5py.File(a.h5, "r") as h:
        ds = h[a.dset]
        frames = ds[: a.frames]
    dtype = frames.dtype
    top = int(frames.max())
    integ = Integrator.from_pyfai(pyFAI.load(a.poni), a.npt, unit="q_nm^-1", split="bbox", mask=bad)
    nm = _matrix._fold_mask(_matrix.normalise_matrix(integ.csc), (~bad).ravel().astype(np.uint8))
    w = np.abs(nm.data.astype(np.float64))
    wsum = np.bincount(nm.indices, weights=w, minlength=nm.nbins).max()
    print(
        "matrix: %d bins x %d pixels, largest weight %.3f, largest sum of weights in a bin %.1f"
        % (nm.nbins, nm.npix, w.max(), wsum)
    )
    print(
        "data: %s pixels, the largest value in the first %d frames is %d (2**%.1f)"
        % (dtype, a.frames, top, np.log2(top + 1))
    )
    print("\nfractional bits b, by the largest pixel value the sums must hold:")
    cases = [("%s maximum" % dtype, np.iinfo(dtype).max), ("the data's largest value", top)]
    if dtype.itemsize > 2:
        cases.insert(0, ("uint16 maximum", 2**16 - 1))
    for label, vmax in cases:
        b = _matrix._fixed_point_bits(nm, dtype, 32, maxval=vmax)
        print(
            "  %-26s vmax %12d  b = %2d  (weight step 2**-%d = %.1e)" % (label, vmax, b, b, 2.0**-b)
        )
    f = np.where(bad, 0, frames[0]).astype(np.int64).ravel()
    M = sp.csc_matrix((nm.data.astype(np.float64), nm.indices, nm.indptr), shape=nm.shape)
    ref = M @ f.astype(np.float64)
    ok = ref > 0
    print(
        "\nwhat rounding the weights to b bits does to the bin sums of frame 0 (%d bins have counts):"
        % ok.sum()
    )
    print("  %4s %12s %18s %18s" % ("b", "step", "largest rel. diff", "median rel. diff"))
    for b in (12, 15, 17, 20, 24, 28, 31):
        q = _matrix._quantise(nm, b, np.uint32).astype(np.int64)
        got = (sp.csc_matrix((q, nm.indices, nm.indptr), shape=nm.shape) @ f).astype(
            np.float64
        ) * 2.0**-b
        rel = np.abs(got - ref)[ok] / ref[ok]
        print("  %4d %12.1e %18.2e %18.2e" % (b, 2.0**-b, rel.max(), np.median(rel)))


if __name__ == "__main__":
    main()
