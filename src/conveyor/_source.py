"""Cut files into units: the unit of work of the conveyor.

The byte range of every chunk comes from the sidecar index (_index.py), so reading needs no HDF5.
harvest=True reads missing ones from HDF5 instead, serially (h5py takes a global lock, so threads
would gain nothing)."""

from ._index import read_index, harvest as harvest_index


class Unit(object):
    """Consecutive frames of one file: file[start:start+span] holds frames frame0..frame0+n-1, the
    chunk of frame frame0+i at offsets[i] (relative to start), lengths[i] bytes long."""

    __slots__ = ("file", "path", "frame0", "n", "start", "span", "offsets", "lengths")


def make_units(paths, dset, max_bytes=256 << 20, index_dir=None, harvest=False):
    """(units, frame shape, dtype).  A unit is as many consecutive frames of a file as fit in
    max_bytes of file span, the whole file when it fits: a read is one large contiguous request,
    and a unit is seconds of work, so workers synchronise with the rest about once a second.

    index_dir  where the sidecar indexes are kept (None: beside each file)
    harvest    when a file has no sidecar, read its index from HDF5 instead of raising"""
    idx = []
    for p in paths:
        try:
            ix = read_index(p, index_dir)
        except FileNotFoundError:
            if not harvest:
                raise FileNotFoundError(
                    "no chunk index for %s: build it with tools/index_chunks.py, or pass harvest=True "
                    "(reads it from HDF5)" % p
                )
            ix = harvest_index(p, dset)
        if ix.dataset != dset:
            raise ValueError("%s: the index is for dataset %r, not %r" % (p, ix.dataset, dset))
        idx.append(ix)
    shape, dtype = idx[0].shape[1:], idx[0].dtype
    out = []
    for fi, (p, ix) in enumerate(zip(paths, idx)):
        if ix.shape[1:] != shape or ix.dtype != dtype:
            raise ValueError("%s: shape or dtype differs from the first file" % p)
        off, ln = ix.offsets, ix.lengths
        f0 = 0
        while f0 < len(off):
            n = 1
            while f0 + n < len(off) and off[f0 + n] + ln[f0 + n] - off[f0] <= max_bytes:
                n += 1
            u = Unit()
            u.file, u.path, u.frame0, u.n = fi, p, f0, n
            u.start = int(off[f0])
            u.span = int(off[f0 + n - 1] + ln[f0 + n - 1] - off[f0])
            u.offsets = off[f0 : f0 + n] - off[f0]
            u.lengths = ln[f0 : f0 + n]
            out.append(u)
            f0 += n
    return out, shape, dtype
