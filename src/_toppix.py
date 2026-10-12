"""top_pixels: limit the pixels returned per frame (the rules of
ImageD11.sinograms.lima_segmenter.top_pixels), shared by the CPU and GPU
classes and used as the oracle in the tests.

A frame that has more than `howmany` pixels above the cut keeps the pixels
above the first of the sorted `thresholds` that leaves fewer than `howmany`
(else above the last one), and then at most the first `howmany` of those, in
pixel order.  howmany=None keeps every pixel above the cut.

Default thresholds: max(cut, 1) * 2**i, i = 0..MAXTHR-1.  (ImageD11 itself
uses cut * 2**i, which for cut == 0 is all zeros and never reduces anything.)

A frame's count after the selection, `last_nabove`, is the number of pixels above the cut it ended up with (`last_cut`),
before truncation to howmany.

The CPU classes run this inside the block loop of the C driver (src/pipeline/driver.c): a frame whose list passes the
limit moves to the next threshold and its list is filtered in place, which gives the same answer as selecting afterwards.
Float pixels with a non-integer threshold cannot be represented there and use the numpy version below.
"""
import math

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
        if any(t != t for t in thresholds):
            raise ValueError("thresholds must not be NaN")
        if len(thresholds) > MAXTHR or any(b < a for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError("thresholds: at most %d values, sorted increasing" % MAXTHR)
    return howmany, thresholds


def _above(v, t):
    """v > t, compared as float64 (numpy 1.x would compare a float32 array with a float64 scalar in float32)."""
    return np.greater(v, t, signature="dd->?", casting="unsafe")


def top_pixels(values, indices, nnz, howmany, thresholds):
    """In place on values[:nnz] and indices[:nnz] (pixels above the cut, in
    pixel order): returns (n kept, the cut applied or None if all were kept,
    the number of pixels above that cut before truncation)."""
    if nnz <= howmany:
        return nnz, None, nnz
    v = values[:nnz]
    tcut = thresholds[-1] if len(thresholds) else None
    for t in thresholds:
        if np.count_nonzero(_above(v, t)) < howmany:
            tcut = t
            break
    if tcut is None:                      # no thresholds at all: the first howmany
        keep = np.arange(howmany)
        nabove = nnz
    else:
        above = np.flatnonzero(_above(v, tcut))
        nabove = len(above)
        keep = above[:howmany]
    n = len(keep)
    values[:n] = v[keep]
    indices[:n] = indices[:nnz][keep]
    return n, tcut, nabove


EMPTY_TOP = np.zeros(0, np.int64)
EMPTY_OUT = np.zeros(0, np.int32)


def _effective(thr, cut, dtype):
    """The thresholds as the C driver compares them, an int64 per level: max(cut, floor(t)).  None when C cannot compare
    exactly as numpy does: a float pixel type that cannot hold the integer exactly, anything beyond 2**62, or an infinite
    threshold for float and 64-bit pixels (smaller integer types can never exceed one, so it is clamped)."""
    big = 2 ** 62
    out = []
    for t in thr:
        if t == float("-inf"):
            out.append(int(cut))
            continue
        if t == float("inf"):
            if dtype.kind == "f" or dtype.itemsize >= 8:
                return None
            out.append(big)
            continue
        f = math.floor(t)
        if dtype.kind == "f" and (f != t or float(dtype.type(f)) != f):
            return None                                    # not an integer, or not exact in the pixel type
        if f > big:
            if dtype.kind == "f" or dtype.itemsize >= 8:
                return None
            f = big                                        # above every value of the type: nothing passes
        out.append(max(int(cut), int(f)))
    return out


class Request(object):
    """One call's top_pixels: .top and .out are the arrays the C entry points take (empty when there is no howmany or the
    numpy version is used); finish() gives (last_nabove, last_cut) after the call, doing the selection in numpy if C did not."""

    def __init__(self, howmany, thresholds, cut, dtype, nframes):
        self.howmany, self.cut, self.nframes = howmany, cut, nframes
        self.thr = default_thresholds(cut) if thresholds is None else thresholds
        self.top, self.out, self.native = EMPTY_TOP, EMPTY_OUT, False
        if howmany is not None:
            eff = _effective(self.thr, cut, dtype)
            if eff is not None:
                self.native = True
                self.top = np.array([min(howmany, 2 ** 31 - 1), len(eff)] + eff, np.int64)   # more than a frame holds: no limit
                self.out = np.zeros(2 * nframes, np.int32)

    def finish(self, npx, values, indices):
        n = self.nframes
        if self.howmany is None:
            return npx.astype(np.int64), np.full(n, float(self.cut))
        if self.native:
            level = self.out[:n]
            thr = np.array(self.thr, np.float64)
            used = np.full(n, float(self.cut))
            if len(thr):
                used = np.where(level >= 0, thr[np.clip(level, 0, len(thr) - 1)], used)
            return self.out[n:].astype(np.int64), used
        return apply(npx, values, indices, self.cut, self.howmany, self.thr)


def apply(npx, values, indices, cut, howmany, thresholds):
    """top_pixels in numpy on every frame of a batch (npx modified in place).
    Returns (last_nabove, last_cut) as Request.finish."""
    nabove = npx.astype(np.int64)
    used = np.full(len(npx), float(cut))
    if howmany is not None:
        for i in np.flatnonzero(nabove > howmany):
            n, t, na = top_pixels(values[i], indices[i], int(npx[i]), howmany, thresholds)
            npx[i] = n
            nabove[i] = na
            if t is not None:
                used[i] = t
    return nabove, used
