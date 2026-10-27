"""The matrix of a chunk2sparseCSC object: normalised once, analysed (what
the auto dot is chosen from), mask-folded, and laid out for the chosen dot.

Every layout is exact (or, for the fixed-point dots, within their stated
rounding) and is refused, with the reason, for a matrix it cannot
represent.  The C side of each layout is in src/steps/dot_*.hpp.
"""
import threading

import numpy as np

_PADDED_MAX_WIDTH = 64
_NO_BIN = np.uint32(0xFFFFFFFF)


class _NormalMatrix(object):
    """A csc-like object with normalised f32 data and u32 indices/indptr."""

    def __init__(self, data, indices, indptr, nbins, npix):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = (int(nbins), int(npix))
        self.nbins = int(nbins)
        self.npix = int(npix)


def normalise_matrix(csc, npix=None):
    """
    Accept a pyFAI CSC engine, a scipy.sparse.csc_matrix, or any duck-typed
    object with .indptr/.indices/.data plus .shape or .bins; other scipy
    formats (whatever offers .tocsc()) are converted.  Converts to float32 /
    uint32 once and validates the contents:

      len(indptr) == npix + 1 (when npix is given), indptr[0] == 0,
      indptr nondecreasing, indptr[-1] == len(indices), every indices[k] < nbins.

    Returns a _NormalMatrix; raises ValueError on bad contents.
    """
    # Duck-typed conversion to CSC.  A scipy CSR/BSR (or any sparse matrix with
    # a row-wise indptr) has an indptr but it is over rows, so is actually
    # converted; only a true CSC (or an already-CSC pyFAI engine, which has no
    # .format) is left alone.
    if hasattr(csc, "tocsc") and (
        not hasattr(csc, "indptr")
        or getattr(csc, "format", None) not in ("csc", None)
    ):
        csc = csc.tocsc()
    indptr = np.asarray(csc.indptr)
    indices = np.asarray(csc.indices)
    data = np.asarray(csc.data)
    if hasattr(csc, "shape"):
        nbins = int(np.prod(csc.shape[0]))
    elif hasattr(csc, "bins"):
        nbins = int(np.prod(csc.bins))
    else:
        raise ValueError("csc matrix has no shape or bins attribute")
    if npix is not None and len(indptr) != npix + 1:
        raise ValueError("csc indptr has %d entries, expected %d (npix + 1)"
                         % (len(indptr), npix + 1))
    if indptr.size and indptr[0] != 0:
        raise ValueError("csc indptr[0] must be 0")
    if np.any(np.diff(indptr.astype(np.int64)) < 0):
        raise ValueError("csc indptr is not nondecreasing")
    if indptr[-1] != len(indices):
        raise ValueError("csc indptr[-1] (%d) != len(indices) (%d)"
                         % (indptr[-1], len(indices)))
    if indices.size and np.any(indices.astype(np.int64) >= nbins):
        raise ValueError("a csc index is >= nbins (%d)" % nbins)
    return _NormalMatrix(
        data.astype(np.float32),
        indices.astype(np.uint32),
        indptr.astype(np.uint32),
        nbins,
        len(indptr) - 1,
    )


def _copy_normal(nm):
    return _NormalMatrix(nm.data.copy(), nm.indices.copy(), nm.indptr.copy(), nm.nbins, nm.npix)


def _fold_mask(nm, mask):
    """Apply the integrator's 0/1 mask to a _NormalMatrix by dropping the
    entries of masked pixels (mask == 0) so their columns become empty.  The
    matvec kernels then need no mask test (a masked pixel contributes nothing),
    and the `>cut` collect uses the raw mask separately.  Zero, never NaN."""
    m = np.asarray(mask).reshape(-1) > 0
    if m.size != nm.npix:
        raise ValueError("mask has %d pixels, the matrix %d" % (m.size, nm.npix))
    n = np.diff(nm.indptr.astype(np.int64))
    keep = np.repeat(m, n)
    data = nm.data[keep]
    indices = nm.indices[keep]
    new_n = np.where(m, n, 0)
    indptr = np.concatenate(([0], np.cumsum(new_n))).astype(np.uint32)
    return _NormalMatrix(data, indices, indptr, nm.nbins, nm.npix)


class _PaddedLayout(object):
    """A pixel -> bin matrix as one first bin plus a fixed number of weights
    per row (the padding the name refers to).  row == pixel when listed==0,
    else a row per pixel in `pixels`.  `row_ptr` splits the rows by decode
    block (nblocks+1)."""

    def __init__(self, base, weights, pixels, rowmap, row_ptr, width, listed,
                 block_elems, nbins, npix):
        self.base = base
        self.weights = weights
        self.pixels = pixels
        self.rowmap = rowmap
        self.row_ptr = row_ptr
        self.width = width
        self.listed = listed
        self.block_elems = block_elems
        self.nbins = nbins
        self.npix = npix
        self.nrows = base.shape[0]

    @property
    def _weights_flat(self):
        return self.weights.reshape(-1)

    @property
    def _pixels_arg(self):
        return self.pixels if self.listed else np.zeros(1, dtype=np.int32)

    @property
    def _rowmap_arg(self):
        return self.rowmap if self.listed else np.zeros(1, dtype=np.int32)


class _BsbCSR(object):
    """A bit-shuffle-block-sized CSR: per decode block the active (non-empty)
    bins, and per bin the (in-block pixel index, weight) entries."""

    def __init__(self, blk_ptr, bins, bin_ptr, idx, data, block_elems):
        self.blk_ptr = blk_ptr
        self.bins = bins
        self.bin_ptr = bin_ptr
        self.idx = idx
        self.data = data
        self.block_elems = block_elems


def _padded_from_csc(nm, block_elems):
    """Build a _PaddedLayout from a _NormalMatrix, exactly (no weight is
    changed, only re-laid-out).  The integrator passes the mask-folded matrix
    (_fold_mask), so a masked pixel gets an all-zero row (implicit layout) or
    no row (listed layout), and the kernels make no mask test.  Raises
    ValueError naming how many pixels and why if the matrix cannot be padded
    (a pixel reaches bins that are not one run of consecutive bins, or the
    width exceeds the limit)."""
    indptr = nm.indptr.astype(np.int64)
    indices = nm.indices.astype(np.int64)
    data = nm.data
    npix = nm.npix
    nbins = nm.nbins
    n = np.diff(indptr)
    used = int(n.max()) if npix else 0
    width = max(used, 1)
    if width > _PADDED_MAX_WIDTH:
        raise ValueError("a pixel reaches %d bins (padded takes at most %d)"
                         % (width, _PADDED_MAX_WIDTH))
    pix = np.repeat(np.arange(npix, dtype=np.int64), n)
    if pix.size > 1:
        broken = (pix[1:] == pix[:-1]) & (np.diff(indices) != 1)
        nb = np.unique(pix[1:][broken]).size
        if nb:
            raise ValueError("%d of %d pixels with several bins reach bins that are not one "
                             "ascending run of consecutive bins" % (nb, int((n > 1).sum())))
    has = n > 0
    first = indptr[:-1]
    base = np.zeros(npix, dtype=np.int64)
    if indices.size:
        base[has] = indices[np.minimum(first[has], indices.size - 1)]
    new_base = np.minimum(base, max(nbins - width, 0)).astype(np.int32)
    shift = base - new_base.astype(np.int64)
    w = np.zeros((npix, width), dtype=np.float32)
    for k in range(used):
        sel = n > k
        w[sel, shift[sel] + k] = data[first[sel] + k]
    listed = bool(has.mean() < 0.5) if npix else False
    nblocks = (npix + block_elems - 1) // block_elems
    cb = np.minimum(np.arange(nblocks + 1) * block_elems, npix).astype(np.int32)
    if listed:
        rows = np.flatnonzero(has).astype(np.int32)
        rowmap = np.full(npix, -1, dtype=np.int32)
        rowmap[rows] = np.arange(rows.shape[0], dtype=np.int32)
        row_ptr = np.searchsorted(rows, cb).astype(np.int32)
        return _PaddedLayout(new_base[rows], w[rows], rows, rowmap, row_ptr,
                             width, 1, block_elems, nbins, npix)
    return _PaddedLayout(new_base, w, None, None, cb, width, 0, block_elems, nbins, npix)


def _run_starts(nm):
    """indices for the csc-run dot (start + length): the first bin of each
    pixel (npix entries; 0 for an empty column).  The pixel's entries stay
    data[indptr[p]:indptr[p+1]], for the consecutive bins start, start+1, ...
    Raises ValueError if some pixel's bins are not one ascending run."""
    indptr = nm.indptr.astype(np.int64)
    indices = nm.indices.astype(np.int64)
    n = np.diff(indptr)
    pix = np.repeat(np.arange(nm.npix, dtype=np.int64), n)
    if pix.size > 1:
        broken = (pix[1:] == pix[:-1]) & (np.diff(indices) != 1)
        nb = np.unique(pix[1:][broken]).size
        if nb:
            raise ValueError("%d of %d pixels with several bins reach bins that are not one "
                             "ascending run of consecutive bins" % (nb, int((n > 1).sum())))
    starts = np.zeros(nm.npix, dtype=np.uint32)
    has = n > 0
    starts[has] = indices[indptr[:-1][has]]
    return starts


def _nosplit_bins(nm):
    """indices for the csc-nosplit dot (a histogram): the one bin of each pixel
    (npix entries, _NO_BIN for an empty column).  data and indptr are not
    read, so every weight must be exactly 1.  Raises ValueError otherwise."""
    n = np.diff(nm.indptr.astype(np.int64))
    if (n > 1).any():
        raise ValueError("%d pixels reach more than one bin" % int((n > 1).sum()))
    if nm.data.size and not (nm.data == 1).all():
        raise ValueError("%d weights are not 1" % int((nm.data != 1).sum()))
    bins = np.full(nm.npix, _NO_BIN, dtype=np.uint32)
    bins[n == 1] = nm.indices[nm.indptr[:-1][n == 1]]
    return bins


def _bsb_csr_from_csc(nm, block_elems):
    """Build a _BsbCSR from a _NormalMatrix: CSC entries sorted by
    (pixel//block_elems, bin), grouped per (block, bin).  General (works for
    any matrix, 2D/FAZIT included), unlike padded."""
    indptr = nm.indptr.astype(np.int64)
    indices = nm.indices.astype(np.uint32)
    data = nm.data.astype(np.float32)
    npix = nm.npix
    nbins = nm.nbins
    n = np.diff(indptr)
    pix = np.repeat(np.arange(npix, dtype=np.int64), n)
    nz = pix.size
    nblocks = (npix + block_elems - 1) // block_elems
    if nz == 0:
        return _BsbCSR(np.zeros(nblocks + 1, np.uint32), np.zeros(0, np.uint32),
                       np.zeros(1, np.uint32), np.zeros(0, np.uint16),
                       np.zeros(0, np.float32), block_elems)
    block = pix // block_elems
    order = np.lexsort((pix, indices, block))
    block_s = block[order]
    bin_s = indices[order]
    pix_s = pix[order]
    data_s = data[order]
    key = block_s.astype(np.int64) * (nbins + 1) + bin_s.astype(np.int64)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
    block_of_group = block_s[starts]
    bins_g = bin_s[starts]
    bin_ptr = np.concatenate((starts, [nz])).astype(np.uint32)
    idx = (pix_s % block_elems).astype(np.uint16)
    blk_ptr = np.searchsorted(block_of_group, np.arange(nblocks + 1)).astype(np.uint32)
    return _BsbCSR(blk_ptr, bins_g, bin_ptr, idx, data_s.copy(), block_elems)


def _counts(nm):
    return np.diff(nm.indptr.astype(np.int64))


def _runs(nm):
    """First bin per pixel (0 for empty), refusing pixels whose bins are not
    one ascending run of consecutive bins."""
    indptr = nm.indptr.astype(np.int64)
    n = np.diff(indptr)
    ind = nm.indices.astype(np.int64)
    pix = np.repeat(np.arange(nm.npix, dtype=np.int64), n)
    if pix.size > 1:
        broken = (pix[1:] == pix[:-1]) & (np.diff(ind) != 1)
        nb = np.unique(pix[1:][broken]).size
        if nb:
            raise ValueError("%d pixels with several bins reach bins that are not one "
                             "ascending run of consecutive bins" % nb)
    starts = np.zeros(nm.npix, np.int64)
    has = n > 0
    starts[has] = ind[indptr[:-1][has]]
    return starts, n


def _histogram_bins(nm):
    """The one bin of each pixel (-1 for none); weights must be exactly 1."""
    n = _counts(nm)
    if (n > 1).any():
        raise ValueError("%d pixels reach more than one bin" % int((n > 1).sum()))
    if nm.data.size and not (nm.data == 1).all():
        raise ValueError("%d weights are not 1" % int((nm.data != 1).sum()))
    bins = np.full(nm.npix, -1, np.int64)
    bins[n == 1] = nm.indices[nm.indptr[:-1][n == 1]]
    return bins


def _fixed_point_bits(nm, dtype, wbits, maxval=None):
    """Fractional bits b for weights stored as round(w * 2**b) in a wbits-bit
    unsigned integer, keeping every int64 bin sum of w * pixel exact.  Two
    limits: the weights must fit wbits bits (b <= log2((2**wbits - 1) / the
    largest weight)), and the sum of w * pixel over the pixels of a bin must
    fit int64 (b <= 62 - log2(the largest pixel value) - log2(the largest sum
    of the weights of a bin)).  The largest pixel value is the dtype's unless
    maxval says what the data can reach: pixels above maxval can then make a
    sum wrap (the dots add modulo 2**64)."""
    dt = np.dtype(dtype)
    if dt.kind not in "ui":
        raise ValueError("integer weights need an integer pixel dtype, not %s" % dt)
    pbits = dt.itemsize * 8 - (1 if dt.kind == "i" else 0)
    if maxval is not None:
        pbits = np.log2(float(int(maxval) + 1))
    w = np.abs(nm.data.astype(np.float64))
    wmax = float(w.max()) if w.size else 1.0
    rowsum = np.bincount(nm.indices, weights=w, minlength=nm.nbins)
    rmax = float(rowsum.max()) if rowsum.size else 1.0
    b_fit = int(np.floor(np.log2((2.0 ** wbits - 1) / max(wmax, 1e-30))))
    b_sum = int(np.floor(62 - pbits - np.log2(max(rmax, 1e-30))))
    b = min(b_fit, b_sum, 40)
    if b < 4:
        raise ValueError("no room for fixed-point weights (%d fractional bits)" % b)
    return b


def _quantise(nm, b, wtype):
    """round(w * 2**b), then the rounding residual of each pixel's column put
    on its largest weight, so a pixel's integer weights sum exactly to
    round(sum(w) * 2**b): the intensity it hands out is preserved to 2**-b.
    Negative weights within 1e-6 of the largest (pyFAI's "full" split has a
    few at ~-3e-9, area round-off) are taken as 0; larger ones are refused."""
    d = nm.data.astype(np.float64)
    if d.size and d.min() < 0:
        if d.min() < -1e-6 * np.abs(d).max():
            raise ValueError("integer weights are unsigned; the matrix has weights down to %g"
                             % d.min())
        d = np.maximum(d, 0.0)
    w = d * 2.0 ** b
    q = np.rint(w)
    n = _counts(nm)
    has = n > 0
    if q.size:
        first = nm.indptr[:-1][has].astype(np.int64)
        target = np.rint(np.add.reduceat(w, first))
        got = np.add.reduceat(q, first)
        pix = np.repeat(np.arange(nm.npix), n)
        order = np.lexsort((-w, pix))
        start = np.zeros(nm.npix + 1, np.int64)
        np.cumsum(n, out=start[1:])
        biggest = order[start[:-1][has]]
        q[biggest] += target - got
    if (q < 0).any():
        raise ValueError("%d weights are negative after rounding; integer weights are "
                         "unsigned" % int((q < 0).sum()))
    if q.size and q.max() > np.iinfo(wtype).max:
        raise ValueError("a fixed-point weight does not fit %s" % np.dtype(wtype))
    return q.astype(wtype)




INT_DOTS = ("csc-int", "csc-run-int", "csc-run-int16")


class FixedPointMatrix(object):
    """A csc matrix prepared for the integer dots: the mask folded in and the weights stored as
    round(w * scale) integers (data, uint32 or uint16), so that a chunk2sparseCSC given it
    returns the int64 sums of integer weight times pixel, exact; powder / scale is the float sum.
    scale is an integer power of two, 2**b.  Made by fixed_point_matrix; the decoder uses it as
    it is."""

    def __init__(self, nm, data, nbits, dtype, maxval):
        self.nm = nm  # the mask-folded float32 matrix the weights were made from
        self.data = data
        self.scale = 1 << int(nbits)
        self.nbins = nm.nbins
        self.npix = nm.npix
        self.dtype = np.dtype(dtype)
        self.maxval = maxval

    def _snapshot(self):
        """A copy that shares no array with this one."""
        c = FixedPointMatrix.__new__(FixedPointMatrix)
        c.__dict__.update(self.__dict__)
        c.nm = _copy_normal(self.nm)
        c.data = self.data.copy()
        return c

    def to_scipy(self):
        """The integer weights as a scipy.sparse.csc_matrix (nbins, npix) of int64, the mask folded
        in: M @ frame (integer pixels) is the int64 powder of a chunk2sparseCSC given this matrix,
        exactly, so the powder of the sparse pixels can be subtracted from it with nothing left
        over.  Divide by scale for float.  A copy: changing it changes nothing here."""
        import scipy.sparse
        return scipy.sparse.csc_matrix(
            (self.data.astype(np.int64), self.nm.indices.copy(), self.nm.indptr.copy()),
            shape=(self.nbins, self.npix),
        )

    def check(self, dtype, npix):
        """ValueError unless this matrix can be used for pixels of dtype in frames of npix."""
        dtype = np.dtype(dtype)
        if dtype.kind not in "ui":
            raise ValueError("a fixed_point_matrix needs integer pixels, not %s" % dtype)
        if npix != self.npix:
            raise ValueError("the mask has %d pixels, the matrix %d" % (npix, self.npix))
        if dtype.itemsize > self.dtype.itemsize:
            raise ValueError(
                "the matrix was made for %s pixels: its sums could overflow for %s" % (self.dtype, dtype)
            )


def fixed_point_matrix(csc, mask, dtype, maxval=None, wbits=32):
    """
    Prepare a csc matrix (scipy csc, pyFAI CSC engine, ...) for exact integer sums: fold the mask
    in (masked pixels add nothing), then store the weights as fixed-point integers of wbits (32
    or 16) bits.  Give the result to chunk2sparseCSC in place of the matrix.

    dtype: the pixel dtype of the data (integer).  The weights get b fractional bits, b <= wbits
    (their width) and b <= 62 - log2(maxval + 1) - log2(the largest sum of weights in a bin), so
    that the int64 sums cannot overflow; each doubling of maxval costs a bit.  maxval is the
    dtype's maximum unless given: the largest value the data can hold.  For 16-bit pixels b is
    about 31, for 32-bit pixels at their maximum about 17 for a 1D matrix (weight step 2**-17: a
    relative error of a few 1e-6 in a bin sum).  Pixels above maxval (saturated ones) can make a
    sum wrap: the dots add modulo 2**64, so the sum is exact again once those pixels, which are
    in the sparse output when the cut is at most maxval, are clipped or subtracted from it
    (test/test_wrap.py).  tools/fixed_point_precision.py prints b for a matrix and data.

    Raises ValueError for a matrix that cannot be fixed-point (negative weights beyond rounding
    noise, no room for 4 fractional bits).  The result has .scale (the integer 2**b to divide the
    int64 sums by) and .nbins.
    """
    mask = np.asarray(mask)
    dtype = np.dtype(dtype)
    if wbits not in (16, 32):
        raise ValueError("wbits must be 16 or 32")
    nm = _fold_mask(normalise_matrix(csc, npix=mask.size), mask.ravel())
    b = _fixed_point_bits(nm, dtype, wbits, maxval)
    data = _quantise(nm, b, np.uint16 if wbits == 16 else np.uint32)
    return FixedPointMatrix(nm, data, b, dtype, maxval)


def _permute_bins(nm):
    """FAZIT: every pixel reaches its own bin (a permutation of the unmasked
    pixels), weight 1; the output is the pixel values in bin order."""
    bins = _histogram_bins(nm)
    used = bins[bins >= 0]
    if np.unique(used).size != used.size:
        raise ValueError("%d bins are reached by more than one pixel"
                         % int(used.size - np.unique(used).size))
    bins[bins < 0] = 0xFFFFFFFF
    return bins.astype(np.uint32)


# ---- what the matrix is (the auto dot) -----------------------------------

def analyse(nm):
    """Properties of a (mask-folded) matrix that decide which dots fit it."""
    n = np.diff(nm.indptr.astype(np.int64))
    info = {"max_bins": int(n.max()) if n.size else 0,
            "occupancy": float((n > 0).mean()) if n.size else 0.0,
            "unit_weights": bool(nm.data.size == 0 or (nm.data == 1).all())}
    info["nosplit"] = info["max_bins"] <= 1 and info["unit_weights"]
    try:
        _run_starts(nm)
        info["runs"] = True
    except ValueError:
        info["runs"] = False
    return info


BSB_MAX_BLOCK_ELEMS = 65536   # bsb-csr's in-block pixel index is uint16


def auto_dot(info, padded_avx2, block_elems=None, int_pixels=False):
    """The float dot to use for a matrix, from the kernels measured by tools/bench_kernels.py (the
    Every pipeline page of the documentation): a histogram -> csc-nosplit-int for integer pixels of at
    most 32 bits (u32: 5 to 30 % over bsb-csr-nosplit on 1 core of an EPYC and of a Xeon, u16 on the
    sparse route a tie), else bsb-csr-nosplit; one run of <= 8
    bins per pixel -> padded-avx2 (on dense frames) or padded; else few occupied pixels (rings) ->
    bsb-csr; runs of 9-64 bins -> padded for decode blocks of <= 8192 pixels, csc for larger ones
    (csc-run was never the best); else csc.  Decode blocks over BSB_MAX_BLOCK_ELEMS pixels rule
    the bsb-csr dots out (csc-nosplit, csc instead).  Never csc-permute (it changes the output) or
    the fixed-point integer dots, which need a matrix from fixed_point_matrix.  int_pixels: the pixels
    are integers of at most 32 bits."""
    bsb = block_elems is None or block_elems <= BSB_MAX_BLOCK_ELEMS
    if info["nosplit"]:
        if int_pixels:
            return "csc-nosplit-int"
        return "bsb-csr-nosplit" if bsb else "csc-nosplit"
    if info["runs"] and info["max_bins"] <= 8:
        return "padded-avx2" if padded_avx2 else "padded"
    if info["occupancy"] < 0.5 and bsb:
        return "bsb-csr"
    if info["runs"] and info["max_bins"] <= _PADDED_MAX_WIDTH and \
            (block_elems is None or block_elems <= 8192):
        return "padded"
    return "csc"


# ---- layouts --------------------------------------------------------------

class Layout(object):
    """The arrays one dot reads, ready for its C entry ("csc", "padded" or
    "bsb-csr"): args() are the layout's arrays, which follow the powder in the call
    for "csc" and are split around the workspace arguments for the others.  nout is the
    powder length, powder_dtype its type (an int64 powder divided by scale
    is float; scale is None for the float ones), block_elems the decode block size
    it was built for (None when it does not depend on it)."""

    def __init__(self, dot, entry, args, nout, powder_dtype=np.float64, scale=None,
                 block_elems=None):
        self.dot = dot
        self.entry = entry
        self._args = args
        self.nout = nout
        self.powder_dtype = powder_dtype
        self.scale = scale
        self.block_elems = block_elems

    def args(self):
        return self._args


def build(dot, nm, block_elems, dtype, fixed=None):
    """The Layout of a mask-folded _NormalMatrix for one dot (the integer dots take the
    FixedPointMatrix made from it).  Raises
    ValueError (without the step context) if the dot cannot represent it."""
    nbins = nm.nbins
    if fixed is not None and dot not in INT_DOTS:
        raise ValueError("a fixed_point_matrix needs an integer dot (csc-int, ...), not %s" % dot)
    if dot == "csc":
        return Layout(dot, "csc", (nm.data, nm.indices, nm.indptr), nbins)
    if dot == "csc-run":
        return Layout(dot, "csc", (nm.data, _run_starts(nm), nm.indptr), nbins)
    if dot == "csc-nosplit":
        return Layout(dot, "csc", (nm.data, _nosplit_bins(nm), nm.indptr), nbins)
    if dot == "csc-nosplit-int":
        return Layout(dot, "csc", (nm.data, _nosplit_bins(nm), nm.indptr), nbins)
    if dot == "csc-permute":
        return Layout(dot, "csc", (nm.data, _permute_bins(nm), nm.indptr), nbins,
                      powder_dtype=np.dtype(dtype))
    if dot in INT_DOTS:
        if fixed is None:
            raise ValueError("%s needs the matrix prepared by fixed_point_matrix" % dot)
        wtype = np.uint16 if dot == "csc-run-int16" else np.uint32
        if fixed.data.dtype != wtype:
            raise ValueError(
                "%s takes %s weights, the matrix has %s (fixed_point_matrix wbits)"
                % (dot, np.dtype(wtype), fixed.data.dtype)
            )
        idx = nm.indices if dot == "csc-int" else _runs(nm)[0].astype(np.uint32)
        return Layout(dot, "csc", (fixed.data, idx, nm.indptr), nbins,
                      powder_dtype=np.int64, scale=fixed.scale)
    if dot in ("padded", "padded-avx2"):
        p = _padded_from_csc(nm, block_elems)
        return Layout(dot, "padded", (p.base, p._weights_flat, p._pixels_arg, p._rowmap_arg,
                                      p.row_ptr, p.width, int(p.listed), p.block_elems),
                      nbins, block_elems=block_elems)
    if dot in ("bsb-csr", "bsb-csr-nosplit"):
        if block_elems > BSB_MAX_BLOCK_ELEMS:
            raise ValueError("the decode blocks have %d pixels and bsb-csr takes at most %d "
                             "(its in-block index is uint16)" % (block_elems, BSB_MAX_BLOCK_ELEMS))
        q = _bsb_csr_from_csc(nm, block_elems)
        if dot == "bsb-csr-nosplit":
            # the sparse route walks the nested csc as csc-nosplit does (one
            # bin per pixel); csc_data is not read (its length must match)
            bins = _nosplit_bins(nm)
            csc = (np.ones(bins.size, np.float32), bins, nm.indptr)
        else:
            csc = (nm.data, nm.indices, nm.indptr)
        return Layout(dot, "bsb-csr", (q.blk_ptr, q.bins, q.bin_ptr, q.idx, q.data) + csc +
                      (q.block_elems,), nbins, block_elems=block_elems)
    raise ValueError("unknown dot %r" % (dot,))


# ---- what sibling chunk2sparseCSC objects share ------------------------------

_BLOCK_DOTS = ("padded", "padded-avx2", "bsb-csr", "bsb-csr-nosplit")   # layouts that depend on the block size


class _Family(object):
    """What the chunk2sparseCSC objects of one plan share: the plan's own copy of the matrix (nm,
    mask-folded, and fixed: the FixedPointMatrix it came from, copied, or None) and of the mask,
    the analysis of the matrix and the layouts built so far, each built once (one lock; a layout
    already built is read without it).  Read by the objects, changed only by adding a layout.  Every
    array it holds is read-only (flags.writeable False), so a stray write raises."""

    def __init__(self, csc, mask):
        mask = np.array(mask)               # the family's own copy
        self.mask = mask.reshape(-1)
        self.shape = mask.shape
        if isinstance(csc, FixedPointMatrix):
            self.fixed = csc._snapshot()
            self.nm = self.fixed.nm
            if self.nm.npix != self.mask.size:
                raise ValueError("the mask has %d pixels, the matrix %d" % (self.mask.size, self.nm.npix))
        else:
            self.fixed = None
            self.nm = _fold_mask(normalise_matrix(csc, npix=self.mask.size), self.mask)
        self.nbins = self.nm.nbins
        self._info = None
        self._layouts = {}
        self._lock = threading.Lock()
        self._freeze()

    def _freeze(self):
        arrays = [self.mask, self.nm.data, self.nm.indices, self.nm.indptr]
        if self.fixed is not None:
            arrays += [self.fixed.data, self.fixed.nm.data, self.fixed.nm.indices, self.fixed.nm.indptr]
        for lay in self._layouts.values():
            arrays += [a for a in lay.args() if isinstance(a, np.ndarray)]
        for a in arrays:
            a.flags.writeable = False

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._freeze()           # unpickled arrays are writeable

    @property
    def info(self):
        """analyse(nm), computed once"""
        info = self._info
        if info is None:
            with self._lock:
                if self._info is None:
                    self._info = analyse(self.nm)
                info = self._info
        return info

    def layout(self, dot, block_elems, dtype):
        """The Layout for a dot, built on the first request and the same object after that: the
        block size only matters to the padded and bsb-csr dots, the pixel dtype to csc-permute.
        ValueError (not kept) if the dot cannot represent the matrix."""
        dtype = np.dtype(dtype)
        key = (dot, block_elems if dot in _BLOCK_DOTS else None,
               dtype.str if dot == "csc-permute" else None)
        lay = self._layouts.get(key)
        if lay is None:
            with self._lock:
                lay = self._layouts.get(key)
                if lay is None:
                    lay = build(dot, self.nm, block_elems, dtype, self.fixed)
                    for a in lay.args():
                        if isinstance(a, np.ndarray):
                            a.flags.writeable = False
                    self._layouts[key] = lay
        return lay
