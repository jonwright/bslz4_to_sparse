"""Where results go.  File formats are not the conveyor's business: implement Sink for yours.

A worker calls its sink (one per worker, made by the factory passed to run) once per batch:

    write(file, frame0, npx, values, indices, powder)

file, frame0  which file (index into the paths given to make_units) and the first frame's number
npx           (n,) int32: the number of sparse pixels of each of the n frames
values        (n, npixels) pixel dtype, indices (n, npixels) uint32: row i holds frame i's pixels in
              its first npx[i] entries
powder        (n, nbins) float64, or None without a matrix
The arrays are the decoder's own buffers: they are overwritten after write returns, so a sink that
keeps anything copies it."""

import numpy as np


class Sink(object):
    """Where one worker's results go: subclass it and implement write (and begin, end if needed)."""

    def begin(self, meta):
        """called once in the worker before the first batch; meta: dict (dtype, shape, ...)"""

    def begin_unit(self, unit):
        """called before the batches of a unit (a Unit of make_units: file, frame0, n, lengths)"""

    def write(self, file, frame0, npx, values, indices, powder):
        """Receive one batch of frames (arguments in the module docstring)."""
        raise NotImplementedError

    def end_unit(self, unit):
        """called after the last batch of a unit; may raise OutputTooLarge to refuse it"""

    def end(self):
        """flush and close; return a dict of statistics"""
        return {}


class NullSink(Sink):
    """Counts and throws everything away: measures the reading and decoding alone."""

    def __init__(self):
        self.frames = 0
        self.pixels = 0

    def write(self, file, frame0, npx, values, indices, powder):
        """Count the frames and pixels of the batch."""
        self.frames += len(npx)
        self.pixels += int(npx.sum())

    def end(self):
        """The counts: dict(frames, pixels)."""
        return dict(frames=self.frames, pixels=self.pixels)


class OutputTooLarge(RuntimeError):
    """A unit's results, compressed for writing, are larger than the compressed input they came
    from: the job would inflate the data, so it is refused."""


def stored_size(arrays, codec="bslz4"):
    """Bytes the arrays take compressed with the write codec, each as one stream.
    codec "bslz4" (bitshuffle, lz4) or "bszstd" (bitshuffle, zstd level 1, the fast level): the
    codecs of the inputs.  Whole arrays are shuffled at once, an estimate of what the block-wise
    HDF5 filters store."""
    try:
        import bitshuffle
    except ImportError:
        raise ImportError("the python package bitshuffle is needed to size compressed output")
    if codec == "bslz4":
        comp = bitshuffle.compress_lz4
    elif codec == "bszstd":
        try:
            import zstandard
        except ImportError:
            raise ImportError("the python package zstandard is needed to size bszstd output")
        zc = zstandard.ZstdCompressor(level=1)
        comp = lambda a: zc.compress(bitshuffle.bitshuffle(a).tobytes())  # noqa: E731
    else:
        raise ValueError("codec must be 'bslz4' or 'bszstd'")
    return sum(len(comp(np.ascontiguousarray(a))) for a in arrays if a is not None and a.size)


class GuardSink(Sink):
    """Wraps a sink and refuses to let it write what would be larger than it read.

    A unit's results are kept in memory until the unit is complete, then compressed with the
    write codec as they would be stored (the pixels, their indices, the counts, the powder sums,
    one stream each) and compared with the compressed input of the unit.  Within limit times the
    input they are passed on to the inner sink; otherwise OutputTooLarge stops the run before
    anything was written.  Results of a unit are in RAM meanwhile: units are seconds of work.

    inner  the Sink that writes;  codec  "bslz4" or "bszstd";  limit  largest allowed
    (compressed output) / (compressed input), 1.0: not larger than the input.
    end() adds in_bytes, stored_bytes (what was measured, all units) to the inner statistics."""

    def __init__(self, inner, codec="bslz4", limit=1.0):
        self.inner, self.codec, self.limit = inner, codec, limit
        self.held, self.in_bytes, self.stored_bytes = [], 0, 0

    def begin(self, meta):
        self.inner.begin(meta)

    def begin_unit(self, unit):
        self.held = []
        self.inner.begin_unit(unit)

    def write(self, file, frame0, npx, values, indices, powder):
        """Keep a copy of the batch until its unit is complete."""
        m = int(npx.max()) if len(npx) else 0
        self.held.append(
            (
                file,
                frame0,
                npx.copy(),
                values[:, :m].copy(),
                None if indices is None else indices[:, :m].copy(),
                None if powder is None else powder.copy(),
            )
        )

    def end_unit(self, unit):
        """Compare the unit's compressed results with its compressed input; write or refuse."""
        npx = np.concatenate([h[2] for h in self.held])
        vals = np.concatenate([h[3][i, : h[2][i]] for h in self.held for i in range(len(h[2]))])
        idx = None
        if self.held[0][4] is not None:
            idx = np.concatenate([h[4][i, : h[2][i]] for h in self.held for i in range(len(h[2]))])
        pw = None
        if self.held[0][5] is not None:
            pw = np.concatenate([h[5] for h in self.held])
        out = stored_size([npx, vals, idx, pw], self.codec)
        inp = int(unit.lengths.sum())
        self.in_bytes += inp
        self.stored_bytes += out
        if out > self.limit * inp:
            raise OutputTooLarge(
                "%s frames %d-%d: %d bytes compressed (%s) for %d bytes of input, over %g x"
                % (
                    unit.path,
                    unit.frame0,
                    unit.frame0 + unit.n - 1,
                    out,
                    self.codec,
                    inp,
                    self.limit,
                )
            )
        for h in self.held:
            self.inner.write(*h)
        self.held = []
        self.inner.end_unit(unit)

    def end(self):
        """The inner statistics with in_bytes and stored_bytes."""
        st = dict(self.inner.end())
        st.update(in_bytes=self.in_bytes, stored_bytes=self.stored_bytes)
        return st
