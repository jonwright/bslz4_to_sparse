"""The matrix of a chunk2sparseCSC object: normalised once, analysed (what
the auto dot is chosen from), mask-folded, and laid out for the chosen dot.

Every layout is exact (or, for the fixed-point dots, within their stated
rounding) and is refused, with the reason, for a matrix it cannot
represent.  The C side of each layout is in src/steps/dot_*.hpp.
"""
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
    block (nblocks+1), replacing the pre-refactor per-block cursor."""

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
    """indices for dot='csc-run' (start + length): the first bin of each
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
    """indices for dot='csc-nosplit' (a histogram): the one bin of each pixel
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


def _nosplit_moment(nm):
    """(data, indices) for dot='csc-nosplit-moment': a histogram with a first
    moment beside each output, i.e. every pixel reaches either no bin or the
    pair b, b+1 with weights exactly 1 (sum I) and q (sum qI) -- the
    interleaved [I, qI, I, qI, ...] order.  indices[p] = b (_NO_BIN for none),
    data[p] = q (npix entries each).  Raises ValueError otherwise."""
    indptr = nm.indptr.astype(np.int64)
    n = np.diff(indptr)
    if ((n != 0) & (n != 2)).any():
        raise ValueError("%d pixels do not reach exactly 0 or 2 bins"
                         % int(((n != 0) & (n != 2)).sum()))
    has = n == 2
    k0 = indptr[:-1][has]
    b0 = nm.indices[k0].astype(np.int64)
    b1 = nm.indices[k0 + 1].astype(np.int64)
    w0 = nm.data[k0]
    bad = int(((b1 != b0 + 1) | (w0 != 1)).sum())
    if bad:
        raise ValueError("%d pixels are not (bin b, weight 1), (bin b+1, q)" % bad)
    bins = np.full(nm.npix, _NO_BIN, dtype=np.uint32)
    bins[has] = b0
    q = np.zeros(nm.npix, dtype=np.float32)
    q[has] = nm.data[k0 + 1]
    return q, bins


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


def _runs(nm, what):
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


def _histogram_bins(nm, what):
    """The one bin of each pixel (-1 for none); weights must be exactly 1."""
    n = _counts(nm)
    if (n > 1).any():
        raise ValueError("%d pixels reach more than one bin" % int((n > 1).sum()))
    if nm.data.size and not (nm.data == 1).all():
        raise ValueError("%d weights are not 1" % int((nm.data != 1).sum()))
    bins = np.full(nm.npix, -1, np.int64)
    bins[n == 1] = nm.indices[nm.indptr[:-1][n == 1]]
    return bins


def _fixed_point_bits(nm, dtype, wbits):
    """Fractional bits b for weights stored as round(w * 2**b) in a wbits-bit
    unsigned integer, keeping every int64 bin sum of w * pixel exact."""
    dt = np.dtype(dtype)
    if dt.kind not in "ui":
        raise ValueError("integer weights need an integer pixel dtype, not %s" % dt)
    pbits = dt.itemsize * 8 - (1 if dt.kind == "i" else 0)
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




def _permute_bins(nm):
    """FAZIT: every pixel reaches its own bin (a permutation of the unmasked
    pixels), weight 1; the output is the pixel values in bin order."""
    bins = _histogram_bins(nm, "csc-permute")
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
    try:
        _nosplit_moment(nm)
        info["moment"] = True
    except ValueError:
        info["moment"] = False
    return info


def auto_dot(info, padded_avx2):
    """The best guess (measured on 4M frames, 2026-10-06, Zen 4): a histogram
    -> bsb-csr-nosplit (never slower than csc-nosplit, 2x on dense frames with
    few occupied pixels); moment pairs -> csc-nosplit-moment; one run of <= 8
    bins per pixel -> padded (fastest 1D split: 9 % frames 3.1 vs 3.9 ms csc);
    longer runs -> csc-run; few occupied pixels (rings) -> bsb-csr; else csc.
    Never csc-permute or the integer dots: they change the output type."""
    if info["nosplit"]:
        return "bsb-csr-nosplit"
    if info["moment"]:
        return "csc-nosplit-moment"
    if info["runs"] and info["max_bins"] <= 8:
        return "padded-avx2" if padded_avx2 else "padded"
    if info["runs"]:
        return "csc-run"
    if info["occupancy"] < 0.5:
        return "bsb-csr"
    return "csc"


# ---- layouts --------------------------------------------------------------

class Layout(object):
    """The arrays one dot reads, ready for its C entry ("csc", "padded" or
    "bsb-csr"): args() is what follows the powder in the call.  nout is the
    powder length, powder_dtype its type (scale turns an int64 powder back
    into float; None for the float ones), block_elems the decode block size
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


def build(dot, nm, block_elems, dtype):
    """The Layout of a mask-folded _NormalMatrix for one dot.  Raises
    ValueError (without the step context) if the dot cannot represent it."""
    nbins = nm.nbins
    if dot == "csc":
        return Layout(dot, "csc", (nm.data, nm.indices, nm.indptr), nbins)
    if dot == "csc-run":
        return Layout(dot, "csc", (nm.data, _run_starts(nm), nm.indptr), nbins)
    if dot == "csc-nosplit":
        return Layout(dot, "csc", (nm.data, _nosplit_bins(nm), nm.indptr), nbins)
    if dot == "csc-nosplit-moment":
        q, bins = _nosplit_moment(nm)
        return Layout(dot, "csc", (q, bins, nm.indptr), nbins)
    if dot == "csc-permute":
        return Layout(dot, "csc", (nm.data, _permute_bins(nm), nm.indptr), nbins,
                      powder_dtype=np.dtype(dtype))
    if dot in ("csc-int", "csc-run-int", "csc-run-int16"):
        wbits, wtype = (16, np.uint16) if dot == "csc-run-int16" else (32, np.uint32)
        idx = nm.indices if dot == "csc-int" else _runs(nm, dot)[0].astype(np.uint32)
        b = _fixed_point_bits(nm, dtype, wbits)
        return Layout(dot, "csc", (_quantise(nm, b, wtype), idx, nm.indptr), nbins,
                      powder_dtype=np.int64, scale=2.0 ** -b)
    if dot in ("padded", "padded-avx2"):
        p = _padded_from_csc(nm, block_elems)
        return Layout(dot, "padded", (p.base, p._weights_flat, p._pixels_arg, p._rowmap_arg,
                                      p.row_ptr, p.width, int(p.listed), p.block_elems),
                      nbins, block_elems=block_elems)
    if dot in ("bsb-csr", "bsb-csr-nosplit"):
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
