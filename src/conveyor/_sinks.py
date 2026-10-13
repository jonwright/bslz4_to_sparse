"""Where results go.  File formats are not the conveyor's business: implement Sink for yours.

A worker calls its sink (one per worker, made by the factory passed to run) once per batch:

    write(file, frame0, npx, values, indices, powder)

file, frame0  which file (index into the paths given to make_work_units) and the first frame's number
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
        """called before the batches of a unit (a WorkUnit of make_work_units: file, frame0, n, lengths)"""

    def write(self, file, frame0, npx, values, indices, powder):
        """Receive one batch of frames (arguments in the module docstring)."""
        raise NotImplementedError

    def end_unit(self, unit):
        """called after the last batch of a unit"""

    def sparse_skipped(self, unit, reason):
        """called (before the batches of the unit) when the sparse pixels of a unit are not passed
        on because they were not sparse: the batches then carry no pixels (npx is 0).  The powder
        is passed on as always."""

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
    """Wraps a sink and keeps it from writing sparse pixels that are not sparse.

    The pixels of a unit (their values, indices and counts) are kept in memory until the unit is
    complete, then compressed with the write codec as they would be stored and compared with the
    compressed input of the unit.  Within limit times the input they are passed on.  Over it
    (frames that are not sparse, a cut that is too low, howmany None on a dense image) they are
    not: the inner sink gets sparse_skipped(unit, reason) and batches without pixels, and the
    unit goes on into the list in skipped.  The powder sums are always passed on, and the job
    never stops for this.  Results of a unit are in RAM meanwhile: units are seconds of work.

    inner  the Sink that writes;  codec  "bslz4" or "bszstd";  limit  largest (compressed pixels)
    / (compressed input), 1.0: not larger than the input.
    end() adds in_bytes, sparse_stored_bytes (measured, all units) and skipped to the statistics."""

    def __init__(self, inner, codec="bslz4", limit=1.0):
        self.inner, self.codec, self.limit = inner, codec, limit
        self.held, self.in_bytes, self.stored_bytes, self.skipped = [], 0, 0, []

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
        """Size the unit's pixels compressed against its compressed input; pass them on or not."""
        npx = np.concatenate([h[2] for h in self.held])
        out = 0
        if npx.any():
            rows = [(h, i) for h in self.held for i in range(len(h[2]))]
            vals = np.concatenate([h[3][i, : h[2][i]] for h, i in rows])
            idx = None
            if self.held[0][4] is not None:
                idx = np.concatenate([h[4][i, : h[2][i]] for h, i in rows])
            out = stored_size([npx, vals, idx], self.codec)
        inp = int(unit.lengths.sum())
        self.in_bytes += inp
        self.stored_bytes += out
        if out > self.limit * inp:
            reason = (
                "%s frames %d-%d: the pixels are %d bytes compressed (%s) for %d bytes of input, "
                "over %g x: not sparse (set howmany, or a higher cut)"
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
            self.skipped.append(dict(path=unit.path, frame0=unit.frame0, n=unit.n, reason=reason))
            self.inner.sparse_skipped(unit, reason)
            for f, f0, n, v, ix, pw in self.held:
                self.inner.write(
                    f, f0, np.zeros_like(n), v[:, :0], None if ix is None else ix[:, :0], pw
                )
        else:
            for h in self.held:
                self.inner.write(*h)
        self.held = []
        self.inner.end_unit(unit)

    def end(self):
        """The inner statistics with in_bytes, sparse_stored_bytes and skipped."""
        st = dict(self.inner.end())
        st.update(
            in_bytes=self.in_bytes, sparse_stored_bytes=self.stored_bytes, skipped=self.skipped
        )
        return st


class TeeSink(Sink):
    """Several sinks in one: every call goes to each of them, in order (sparse pixels to a pixel
    writer and the powder to a ring writer, say).  end() merges their statistics."""

    def __init__(self, *sinks):
        self.sinks = sinks

    def begin(self, meta):
        for s in self.sinks:
            s.begin(meta)

    def begin_unit(self, unit):
        for s in self.sinks:
            s.begin_unit(unit)

    def sparse_skipped(self, unit, reason):
        for s in self.sinks:
            s.sparse_skipped(unit, reason)

    def write(self, file, frame0, npx, values, indices, powder):
        for s in self.sinks:
            s.write(file, frame0, npx, values, indices, powder)

    def end_unit(self, unit):
        for s in self.sinks:
            s.end_unit(unit)

    def end(self):
        st = {}
        for s in self.sinks:
            st.update(s.end())
        return st
