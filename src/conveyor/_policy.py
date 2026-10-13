"""What the conveyor agrees to write: sparse pixels, 1D powder and ring sums, and nothing that
fills the disk with more than the data it came from.

A Policy has two parts, and neither stops a useful job.

  * Sparse pixels are checked as they come (GuardSink): a unit whose pixels, compressed as they
    would be stored, are larger than the compressed frames they came from is not sparse (a cut that
    is too low, howmany None on a dense image).  Its pixels are not written, a warning and the run
    result say so, and the job goes on: the powder and ring sums of those frames are written as
    usual.  The pixels of the other units are written.
  * A powder result (1D, rings) is dense and does not compress, so before the run its bins are
    counted at full size against the compressed frame.  A matrix with so many bins that every frame
    would give more than it read (a 2D cake in disguise) is refused before anything starts.

    kinds    "sparse"    pixels above a cut, at most howmany per frame (howmany None: all of them)
             "powder1d"  a 1D powder per frame (needs a matrix)
             "rings"     ring windows x azimuth bins per frame (needs a matrix)
    2D powder and FAZIT images are as large as or larger than the frames, so they have no writer:
    they are returned to memory (see iterate).
"""

from ._sinks import GuardSink

KINDS = ("sparse", "powder1d", "rings")
MEMORY_ONLY = ("2d", "fazit", "decompress")


class OutputRefused(ValueError):
    """The requested output is not one the conveyor writes."""


class Policy(object):
    """kinds   one kind or a tuple of them (KINDS), what will be written
    limit   largest (output)/(input) ratio, both compressed; 1.0: not larger than the input
    codec   "bslz4" or "bszstd": the codec the output is sized with (and is meant to use)"""

    def __init__(self, kinds, limit=1.0, codec="bslz4"):
        self.kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
        for k in self.kinds:
            if k in MEMORY_ONLY:
                raise NotImplementedError(
                    "%r results are as large as the frames or larger and are not written: "
                    "take them in memory with iterate()" % k
                )
            if k not in KINDS:
                raise ValueError("kind %r: one of %r" % (k, KINDS))
        self.limit, self.codec = limit, codec

    def check(self, backend, units):
        """Raise OutputRefused when the matrix would give more bins than the frame it is made from."""
        mean_in = sum(int(u.lengths.sum()) for u in units) / float(sum(u.n for u in units))
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
        """The sink factory with every sink guarded: its sparse pixels are not written for a unit
        where they are not sparse."""
        return lambda k: GuardSink(sink_factory(k), self.codec, self.limit)


def _nbins(csc):
    from bslz4_to_sparse import _matrix

    return _matrix.normalise_matrix(csc).nbins
