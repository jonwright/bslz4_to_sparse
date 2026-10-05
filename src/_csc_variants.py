"""Builders for the experimental CSC-entry dots (ids 11..22).

Each builder takes the mask-folded matrix (a _NormalMatrix: f32 data, u32
indices/indptr, nbins, npix), the decode block size in pixels and the pixel
dtype, and returns a Variant: the three arrays handed to sparsify_and_dot
(data, indices, indptr, each in whatever element type the dot id reads), how
many hidden output slots follow the real bins (dump bins), and for the
integer-weight dots the scale that turns the int64 powder back into the
float powder.  A builder raises ValueError when its dot cannot represent the
matrix exactly (or, for the fixed-point dots, within their rounding).

The C side is src/bslz4_csc_variants.hpp; the stream header layout is
described there.
"""
import numpy as np

STREAM_MAGIC = 0x58495342       # "BSIX"
STREAM_HEADER_WORDS = 8
DUMP_NONE16 = 0xFFFF


class Variant(object):
    def __init__(self, data, indices, indptr, extra=0, scale=None, block_dependent=False,
                 note="", packed=False):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.extra = int(extra)           # hidden output slots after the nbins bins
        self.scale = scale                # int64 powder * scale = float powder
        self.block_dependent = block_dependent
        self.note = note
        self.packed = packed              # output in the pixel dtype (stores, no sums)


# ---------------------------------------------------------------- helpers

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
            raise ValueError("%d pixels reach bins that are not one ascending run, so "
                             "dot=%r cannot represent this matrix" % (nb, what))
    starts = np.zeros(nm.npix, np.int64)
    has = n > 0
    starts[has] = ind[indptr[:-1][has]]
    return starts, n


def _histogram_bins(nm, what):
    """The one bin of each pixel (-1 for none); weights must be exactly 1."""
    n = _counts(nm)
    if (n > 1).any():
        raise ValueError("%d pixels reach more than one bin, so dot=%r cannot represent "
                         "this matrix" % (int((n > 1).sum()), what))
    if nm.data.size and not (nm.data == 1).all():
        raise ValueError("dot=%r needs every weight to be 1 (%d are not)"
                         % (what, int((nm.data != 1).sum())))
    bins = np.full(nm.npix, -1, np.int64)
    bins[n == 1] = nm.indices[nm.indptr[:-1][n == 1]]
    return bins


def _stream(payload, block_elems, nblocks, stride=0, aux=0, table=None):
    """A headed byte stream (see bslz4_csc_variants.hpp) as a uint8 array."""
    table = np.zeros(nblocks + 1, np.uint32) if table is None else np.asarray(table, np.uint32)
    words = STREAM_HEADER_WORDS + table.size
    head = np.zeros(words, np.uint32)
    head[:6] = (STREAM_MAGIC, words, block_elems, nblocks, stride, aux)
    head[STREAM_HEADER_WORDS:] = table
    payload = np.ascontiguousarray(payload).view(np.uint8).ravel()
    pad = (-payload.size) % 4
    out = np.zeros(head.nbytes + payload.size + pad, np.uint8)
    out[:head.nbytes] = head.view(np.uint8)
    out[head.nbytes:head.nbytes + payload.size] = payload
    return out


def _delta_stream(starts, block_elems):
    """Byte-coded starts: (int8) delta from the previous pixel, or 0x80 and an
    absolute u32; every block starts with an absolute."""
    npix = starts.size
    nblocks = (npix + block_elems - 1) // block_elems
    d = np.empty(npix, np.int64)
    d[0] = 0
    d[1:] = starts[1:] - starts[:-1]
    esc = (d < -127) | (d > 127)
    esc[::block_elems] = True
    size = np.where(esc, 5, 1)
    off = np.zeros(npix + 1, np.int64)
    np.cumsum(size, out=off[1:])
    buf = np.zeros(int(off[-1]), np.uint8)
    ne = ~esc
    buf[off[:-1][ne]] = (d[ne] & 0xFF).astype(np.uint8)
    e = off[:-1][esc]
    buf[e] = 0x80
    absval = starts[esc].astype(np.uint32)
    for k in range(4):
        buf[e + 1 + k] = ((absval >> (8 * k)) & 0xFF).astype(np.uint8)
    table = off[np.minimum(np.arange(nblocks + 1) * block_elems, npix)]
    note = "delta stream %.2f bytes/pixel, %.2f %% escapes" % (buf.size / npix, 100 * esc.mean())
    return _stream(buf, block_elems, nblocks, table=table), note


def _infer_stride(bins, nbins):
    """The 2D azimuth count for a radial-major flat bin index, guessed as the
    stride that lets most neighbouring pixels step by (dr, da) in {-1,0,1}^2."""
    b = bins[bins >= 0]
    d = np.abs(np.diff(b))
    d = d[d > 1]
    if d.size == 0:
        return 1
    vals, cnt = np.unique(d, return_counts=True)
    cands = set()
    for v in vals[np.argsort(cnt)[-8:]]:
        cands.update((int(v) - 1, int(v), int(v) + 1))
    cands = [c for c in cands if 1 < c < nbins]
    dd = np.diff(b)
    best, best_cov = 1, -1
    for s in cands:
        cov = 0
        for dr in (-1, 0, 1):
            cov += int((np.abs(dd - dr * s) <= 1).sum())
        if cov > best_cov:
            best, best_cov = s, cov
    return best


def _walk_stream(bins, block_elems, stride):
    """4-bit 2D steps (dr, da in {0, +1, -1} or escape) and an exception list
    of absolute bins; every block starts with an escape."""
    npix = bins.size
    nblocks = (npix + block_elems - 1) // block_elems
    d = np.zeros(npix, np.int64)
    d[1:] = bins[1:] - bins[:-1]
    code = np.full(npix, 15, np.uint8)
    enc = {0: 0, 1: 1, -1: 2}
    ok = np.zeros(npix, bool)
    for dr in (0, 1, -1):
        da = d - dr * stride
        for a in (0, 1, -1):
            sel = (~ok) & (da == a)
            code[sel] = (enc[dr] << 2) | enc[a]
            ok |= sel
    ok[::block_elems] = False
    code[~ok] = 15
    nib = np.zeros((npix + 1) // 2, np.uint8)
    nib |= code[0::2]
    nib[: code[1::2].size] |= code[1::2] << 4
    exc_pix = np.flatnonzero(~ok)
    exc = bins[exc_pix].astype(np.uint32)
    table = np.searchsorted(exc_pix, np.minimum(np.arange(nblocks + 1) * block_elems, npix))
    aux = nib.size + ((-nib.size) % 4)
    payload = np.zeros(aux + exc.nbytes, np.uint8)
    payload[:nib.size] = nib
    payload[aux:] = exc.view(np.uint8)
    note = "walk stride %d, %.2f bytes/pixel, %.2f %% escapes" % (
        stride, payload.size / npix, 100 * (~ok).mean())
    return _stream(payload, block_elems, nblocks, stride=stride, aux=aux, table=table), note


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


# ---------------------------------------------------------------- builders

def nosplit_dump(nm, block_elems, dtype):
    bins = _histogram_bins(nm, "csc-nosplit-dump")
    bins[bins < 0] = nm.nbins
    return Variant(nm.data, bins.astype(np.uint32), nm.indptr, extra=1)


def nosplit_moment_dump(nm, block_elems, dtype):
    n = _counts(nm)
    if ((n != 0) & (n != 2)).any():
        raise ValueError("dot='csc-nosplit-moment-dump' needs 0 or 2 bins per pixel")
    has = n == 2
    k0 = nm.indptr[:-1][has].astype(np.int64)
    b0 = nm.indices[k0].astype(np.int64)
    if ((nm.indices[k0 + 1] != b0 + 1) | (nm.data[k0] != 1)).any():
        raise ValueError("dot='csc-nosplit-moment-dump' needs pixels (bin b weight 1, bin b+1)")
    bins = np.full(nm.npix, nm.nbins, np.uint32)
    bins[has] = b0
    q = np.zeros(nm.npix, np.float32)
    q[has] = nm.data[k0 + 1]
    return Variant(q, bins, nm.indptr, extra=2)


def run_u16(nm, block_elems, dtype):
    starts, _n = _runs(nm, "csc-run-u16")
    if nm.nbins > 65536:
        raise ValueError("dot='csc-run-u16' needs nbins <= 65536 (got %d)" % nm.nbins)
    return Variant(nm.data, starts.astype(np.uint16), nm.indptr)


def nosplit_u16(nm, block_elems, dtype):
    bins = _histogram_bins(nm, "csc-nosplit-u16")
    if nm.nbins >= DUMP_NONE16:
        raise ValueError("dot='csc-nosplit-u16' needs nbins < 65535 (got %d)" % nm.nbins)
    bins[bins < 0] = DUMP_NONE16
    return Variant(nm.data, bins.astype(np.uint16), nm.indptr)


def run_delta(nm, block_elems, dtype):
    starts, n = _runs(nm, "csc-run-delta")
    # an empty pixel's start is never used: repeat the previous one (delta 0)
    s = starts.copy()
    empty = n == 0
    if empty.any():
        idx = np.where(~empty, np.arange(s.size), 0)
        np.maximum.accumulate(idx, out=idx)
        s = s[idx]
    stream, note = _delta_stream(s, block_elems)
    return Variant(nm.data, stream, nm.indptr, block_dependent=True, note=note)


def nosplit_delta(nm, block_elems, dtype):
    bins = _histogram_bins(nm, "csc-nosplit-delta")
    bins[bins < 0] = nm.nbins
    stream, note = _delta_stream(bins, block_elems)
    return Variant(nm.data, stream, nm.indptr, extra=1, block_dependent=True, note=note)


def nosplit_walk(nm, block_elems, dtype):
    bins = _histogram_bins(nm, "csc-nosplit-walk")
    stride = _infer_stride(bins, nm.nbins)
    bins[bins < 0] = nm.nbins
    stream, note = _walk_stream(bins, block_elems, stride)
    return Variant(nm.data, stream, nm.indptr, extra=1, block_dependent=True, note=note)


def tile(nm, block_elems, dtype):
    """2D: each pixel's bins as one (radial x azimuth) rectangle, azimuth
    wrapping at the stride; missing rectangle entries get weight 0."""
    indptr = nm.indptr.astype(np.int64)
    n = np.diff(indptr)
    ind = nm.indices.astype(np.int64)
    # stride: a row of a rectangle ends at r*S + a0 + l2 - 1 and the next
    # starts at (r+1)*S + a0, so S = jump + (run length) - 1
    pix = np.repeat(np.arange(nm.npix, dtype=np.int64), n)
    brk = np.flatnonzero((pix[1:] == pix[:-1]) & (np.diff(ind) != 1))
    if brk.size == 0:
        raise ValueError("dot='csc-tile' is for 2D matrices; every pixel here is one run "
                         "(use csc-run)")
    runstart = np.zeros(ind.size, bool)
    runstart[0] = True
    runstart[1:] = (pix[1:] != pix[:-1]) | (np.diff(ind) != 1)
    rs = np.flatnonzero(runstart)
    runlen = np.diff(np.append(rs, ind.size))
    # the run that ends at each break
    r_end = np.searchsorted(rs, brk, side="right") - 1
    cand = (ind[brk + 1] - ind[brk]) + runlen[r_end] - 1
    vals, cnt = np.unique(cand[cand > 1], return_counts=True)
    S = int(vals[np.argmax(cnt)])
    r = ind // S
    a = ind % S
    has = n > 0
    first = indptr[:-1][has]
    r0 = np.zeros(nm.npix, np.int64)
    r1 = np.zeros(nm.npix, np.int64)
    r0[has] = np.minimum.reduceat(r, first)
    r1[has] = np.maximum.reduceat(r, first)
    # azimuth: the shorter of the plain and the half-turn-shifted spans
    a_lo = np.zeros(nm.npix, np.int64)
    a_hi = np.zeros(nm.npix, np.int64)
    a_lo[has] = np.minimum.reduceat(a, first)
    a_hi[has] = np.maximum.reduceat(a, first)
    sh = (a + S // 2) % S
    s_lo = np.zeros(nm.npix, np.int64)
    s_hi = np.zeros(nm.npix, np.int64)
    s_lo[has] = np.minimum.reduceat(sh, first)
    s_hi[has] = np.maximum.reduceat(sh, first)
    use_sh = (s_hi - s_lo) < (a_hi - a_lo)
    a0 = np.where(use_sh, (s_lo - S // 2) % S, a_lo)
    l2 = np.where(use_sh, s_hi - s_lo, a_hi - a_lo) + 1
    l1 = r1 - r0 + 1
    l1[~has] = 0
    l2[~has] = 0
    if (l1 > 0xFFFF).any() or (l2 > 0xFFFF).any():
        raise ValueError("dot='csc-tile': a rectangle side exceeds 65535")
    area = l1 * l2
    new_indptr = np.zeros(nm.npix + 1, np.int64)
    np.cumsum(area, out=new_indptr[1:])
    if new_indptr[-1] >= 2 ** 32:
        raise ValueError("dot='csc-tile': padded rectangles exceed 2**32 entries")
    i = r - np.repeat(r0, n)
    j = (a - np.repeat(a0, n)) % S
    pos = np.repeat(new_indptr[:-1], n) + i * np.repeat(l2, n) + j
    if np.unique(pos).size != pos.size:
        raise ValueError("dot='csc-tile': some pixel lists the same bin twice")
    w = np.zeros(int(new_indptr[-1]), np.float32)
    w[pos] = nm.data
    px3 = np.zeros((nm.npix, 3), np.uint32)
    px3[:, 0] = (r0 * S + a0).astype(np.uint32)
    px3[:, 1] = a0.astype(np.uint32)
    px3[:, 2] = ((l1 << 16) | l2).astype(np.uint32)
    note = "tile stride %d, %.2f weights/entry (padding), %.1f %% wrapped" % (
        S, w.size / max(nm.data.size, 1), 100 * (use_sh & has & (a0 + l2 > S)).mean())
    stream = _stream(px3, 0, 0, stride=S)
    return Variant(w, stream, new_indptr.astype(np.uint32), note=note)


def run_moment(nm, block_elems, dtype):
    """From an interleaved moment matrix (rows 2b: sum w I, 2b+1: sum w q I):
    one q at the front of each column, then the weights w of bins b..."""
    starts, n = _runs(nm, "csc-run-moment")
    if (n % 2).any() or (starts % 2).any():
        raise ValueError("dot='csc-run-moment' needs each pixel to reach whole (2b, 2b+1) pairs")
    if nm.nbins % 2:
        raise ValueError("dot='csc-run-moment' needs an even number of bins")
    indptr = nm.indptr.astype(np.int64)
    has = n > 0
    d = nm.data.astype(np.float64)
    even = d[0::2]
    odd = d[1::2]
    npair = n // 2
    # q per pixel from its largest weight, then every pair checked against it
    pairpix = np.repeat(np.arange(nm.npix), npair)
    order = np.lexsort((-even, pairpix))
    firstpair = np.zeros(nm.npix + 1, np.int64)
    np.cumsum(npair, out=firstpair[1:])
    best = order[firstpair[:-1][has]]
    q = np.zeros(nm.npix, np.float64)
    q[has] = odd[best] / even[best]
    qe = np.repeat(q, npair)
    err = np.abs(odd - even * qe)
    if (err > 1e-6 * np.abs(odd) + 1e-30).any():
        raise ValueError("dot='csc-run-moment': %d pairs are not (w, w*q) with one q per pixel"
                         % int((err > 1e-6 * np.abs(odd) + 1e-30).sum()))
    newn = np.where(has, npair + 1, 0)
    new_indptr = np.zeros(nm.npix + 1, np.int64)
    np.cumsum(newn, out=new_indptr[1:])
    data = np.zeros(int(new_indptr[-1]), np.float32)
    data[new_indptr[:-1][has]] = q[has].astype(np.float32)
    pos = np.repeat(new_indptr[:-1] + 1, npair) + (np.arange(even.size) - np.repeat(firstpair[:-1], npair))
    data[pos] = even.astype(np.float32)
    return Variant(data, (starts // 2).astype(np.uint32), new_indptr.astype(np.uint32),
                   note="q from the pixel's largest weight; sum qI agrees to ~1e-7 relative")


def csc_int(nm, block_elems, dtype):
    b = _fixed_point_bits(nm, dtype, 32)
    return Variant(_quantise(nm, b, np.uint32), nm.indices, nm.indptr, scale=2.0 ** -b,
                   note="u32 weights, %d fractional bits" % b)


def run_int(nm, block_elems, dtype):
    starts, _n = _runs(nm, "csc-run-int")
    b = _fixed_point_bits(nm, dtype, 32)
    return Variant(_quantise(nm, b, np.uint32), starts.astype(np.uint32), nm.indptr,
                   scale=2.0 ** -b, note="u32 weights, %d fractional bits" % b)


def run_int16(nm, block_elems, dtype):
    starts, _n = _runs(nm, "csc-run-int16")
    b = _fixed_point_bits(nm, dtype, 16)
    return Variant(_quantise(nm, b, np.uint16), starts.astype(np.uint32), nm.indptr,
                   scale=2.0 ** -b, note="u16 weights, %d fractional bits" % b)


def permute(nm, block_elems, dtype):
    """FAZIT: every pixel reaches its own bin (a permutation of the unmasked
    pixels), weight 1; the output is the pixel values in bin order, in the
    pixel dtype."""
    bins = _histogram_bins(nm, "csc-permute")
    used = bins[bins >= 0]
    if np.unique(used).size != used.size:
        raise ValueError("dot='csc-permute' needs every bin to be reached by at most one pixel")
    bins[bins < 0] = 0xFFFFFFFF
    return Variant(nm.data, bins.astype(np.uint32), nm.indptr, packed=True)


def permute_runs(nm, block_elems, dtype):
    """A permutation cut into runs of pixels that go to consecutive output
    slots (a tiled FAZIT gives runs of about the tile width); runs split at
    the decode-block boundaries.  Masked pixels form runs with no slot."""
    bins = _histogram_bins(nm, "csc-permute-runs")
    used = bins[bins >= 0]
    if np.unique(used).size != used.size:
        raise ValueError("dot='csc-permute-runs' needs every bin to be reached by at most one pixel")
    npix = bins.size
    nblocks = (npix + block_elems - 1) // block_elems
    new = np.ones(npix, bool)
    new[1:] = ~(((bins[1:] == bins[:-1] + 1) & (bins[:-1] >= 0)) |
                ((bins[1:] < 0) & (bins[:-1] < 0)))
    new[::block_elems] = True
    starts = np.flatnonzero(new)
    lens = np.diff(np.append(starts, npix))
    if lens.max() > 0xFFFFFFFF:
        raise ValueError("run too long")
    o = bins[starts]
    runs = np.empty((starts.size, 2), np.uint32)
    runs[:, 0] = np.where(o >= 0, o, 0xFFFFFFFF).astype(np.uint32)
    runs[:, 1] = lens.astype(np.uint32)
    table = np.searchsorted(starts, np.minimum(np.arange(nblocks + 1) * block_elems, npix))
    note = "%d runs, %.2f pixels/run (unmasked: %.2f)" % (
        starts.size, npix / starts.size, used.size / max(int((o >= 0).sum()), 1))
    return Variant(nm.data, _stream(runs, block_elems, nblocks, table=table), nm.indptr,
                   packed=True, block_dependent=True, note=note)


BUILDERS = {
    "csc-nosplit-dump": nosplit_dump,
    "csc-nosplit-moment-dump": nosplit_moment_dump,
    "csc-run-u16": run_u16,
    "csc-nosplit-u16": nosplit_u16,
    "csc-run-delta": run_delta,
    "csc-nosplit-delta": nosplit_delta,
    "csc-nosplit-walk": nosplit_walk,
    "csc-tile": tile,
    "csc-run-moment": run_moment,
    "csc-int": csc_int,
    "csc-run-int": run_int,
    "csc-run-int16": run_int16,
    "csc-permute": permute,
    "csc-permute-runs": permute_runs,
}
