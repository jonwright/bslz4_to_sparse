"""What the conveyor agrees to write: sparse pixels, 1D powder and ring sums, and nothing that
would be larger than the data it came from.

A Policy is checked before a run (from the backend and the units) and, through a GuardSink, on the
compressed size of every unit while it runs.  It is a place to evolve the rules, not a file format.

    kinds    "sparse"    pixels above a cut, at most howmany per frame: howmany must be set
             "powder1d"  a 1D powder per frame (needs a matrix)
             "rings"     ring windows x azimuth bins per frame (needs a matrix)
    2D powder and FAZIT images are as large as or larger than the frames, so they have no writer:
    they are returned to memory (see iterate).

A powder result is dense and does not compress, so the check before the run counts its bins at
full size against the compressed frames it is made from.  Sparse pixels do compress, and are
checked as they come (GuardSink).
"""

from ._sinks import GuardSink

KINDS = ("sparse", "powder1d", "rings")
MEMORY_ONLY = ("2d", "fazit", "decompress")


class OutputRefused(ValueError):
    """The requested output is not one the conveyor writes."""


class Policy(object):
    """kinds       one kind or a tuple of them (KINDS), what will be written
    max_howmany  sparse: the largest howmany allowed (None: any, but it must be set)
    limit        largest (output)/(input) ratio, both compressed; 1.0: not larger than the input
    codec        "bslz4" or "bszstd": the codec the output is sized with (and is meant to use)"""

    def __init__(self, kinds, max_howmany=None, limit=1.0, codec="bslz4"):
        self.kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
        for k in self.kinds:
            if k in MEMORY_ONLY:
                raise NotImplementedError(
                    "%r results are as large as the frames or larger and are not written: "
                    "take them in memory with iterate()" % k
                )
            if k not in KINDS:
                raise ValueError("kind %r: one of %r" % (k, KINDS))
        self.max_howmany, self.limit, self.codec = max_howmany, limit, codec

    def check(self, backend, units):
        """Raise OutputRefused when the backend and units would not give an allowed output."""
        mean_in = sum(int(u.lengths.sum()) for u in units) / float(sum(u.n for u in units))
        if "sparse" in self.kinds:
            hm = getattr(backend, "howmany", None)
            if hm is None:
                raise OutputRefused(
                    "sparse output needs howmany set (the most pixels kept per frame; "
                    "ImageD11's lima_segmenter uses 100000)"
                )
            if self.max_howmany is not None and hm > self.max_howmany:
                raise OutputRefused("howmany %d is over the limit %d" % (hm, self.max_howmany))
        if any(k in self.kinds for k in ("powder1d", "rings")):
            csc = getattr(backend, "csc", None)
            if csc is None:
                raise OutputRefused("%s output needs a matrix" % "/".join(self.kinds))
            nbins = _nbins(csc)
            size = nbins * 8  # float64 sums, dense, taken as not compressing
            if size > self.limit * mean_in:
                raise OutputRefused(
                    "%d bins of float64 are %d bytes a frame, over %g x the %d bytes of the "
                    "compressed frame: this is not a reduction" % (nbins, size, self.limit, mean_in)
                )

    def wrap(self, sink_factory):
        """The sink factory with every sink guarded on the compressed size of each unit."""
        return lambda k: GuardSink(sink_factory(k), self.codec, self.limit)


def _nbins(csc):
    from bslz4_to_sparse import _matrix

    return _matrix.normalise_matrix(csc).nbins
