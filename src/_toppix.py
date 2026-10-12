"""top_pixels: limit the pixels returned per frame (the rules of
ImageD11.sinograms.lima_segmenter.top_pixels), shared by the CPU and GPU
classes and used as the oracle in the tests.

A frame that has more than `howmany` pixels above the cut keeps the pixels
above the first of the sorted `thresholds` that leaves fewer than `howmany`
(else above the last one), and then at most the first `howmany` of those, in
pixel order.  howmany=None keeps every pixel above the cut.

Default thresholds: max(cut, 1) * 2**i, i = 0..MAXTHR-1.  (ImageD11 itself
uses cut * 2**i, which for cut == 0 is all zeros and never reduces anything.)
"""
import numpy as np

MAXTHR = 16


def default_thresholds(cut):
    return [max(cut, 1) * 2 ** i for i in range(MAXTHR)]


def check(howmany, thresholds):
    """Validated (howmany, thresholds): howmany None or an int >= 0;
    thresholds None (the default doubling from the cut) or at most MAXTHR
    values, sorted increasing."""
    if howmany is not None:
        if int(howmany) != howmany or howmany < 0:
            raise ValueError("howmany must be None or an integer >= 0, got %r" % (howmany,))
        howmany = int(howmany)
    if thresholds is not None:
        thresholds = [float(t) for t in np.asarray(thresholds).ravel()]
        if len(thresholds) > MAXTHR or any(b < a for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError("thresholds: at most %d values, sorted increasing" % MAXTHR)
    return howmany, thresholds


def top_pixels(values, indices, nnz, howmany, thresholds):
    """In place on values[:nnz] and indices[:nnz] (pixels above the cut, in
    pixel order): returns (n kept, the cut applied or None if all were kept)."""
    if nnz <= howmany:
        return nnz, None
    v = values[:nnz]
    tcut = thresholds[-1] if len(thresholds) else None
    for t in thresholds:
        if np.count_nonzero(v > np.float64(t)) < howmany:
            tcut = t
            break
    if tcut is None:                      # no thresholds at all: the first howmany
        keep = np.arange(howmany)
    else:
        keep = np.flatnonzero(v > np.float64(tcut))[:howmany]
    n = len(keep)
    values[:n] = v[keep]
    indices[:n] = indices[:nnz][keep]
    return n, tcut


def apply(npx, values, indices, cut, howmany, thresholds):
    """top_pixels on every frame of a batch (npx modified in place).
    Returns (last_nabove, last_cut): per frame, the number of pixels above
    the call's cut and the cut that frame ended up with."""
    nabove = npx.astype(np.int64)
    used = np.full(len(npx), float(cut))
    if howmany is not None:
        thr = default_thresholds(cut) if thresholds is None else thresholds
        for i in np.flatnonzero(nabove > howmany):
            n, t = top_pixels(values[i], indices[i], int(npx[i]), howmany, thr)
            npx[i] = n
            if t is not None:
                used[i] = t
    return nabove, used
