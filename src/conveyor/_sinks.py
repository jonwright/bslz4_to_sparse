"""Where results go.  File formats are not the conveyor's business: implement Sink for yours.

A worker calls its sink (one per worker, made by the factory passed to run) once per batch:

    write(file, frame0, npx, values, indices, powder)

file, frame0  which file (index into the paths given to make_work_units) and the first frame's number
npx           (n,) int32: the number of sparse pixels of each of the n frames
values        (n, npixels) pixel dtype, indices (n, npixels) uint32: row i holds frame i's pixels in
              its first npx[i] entries
powder        (n, nbins) float64 (int64 sums when the backend is exact: meta['powder_scale'] is the
              integer to divide by for float), or None without a matrix
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


POWDER_MAX_FRACTION = 0.1


def check_powder_size(meta):
    """Refuse, before anything runs, a powder of more than POWDER_MAX_FRACTION of the frame it comes
    from: nbins sums of 8 bytes against the uncompressed frame (shape x pixel size).  The writers
    call it in begin(); a 1D powder or ring windows pass, a cake as large as the image does not.
    meta: the run's meta (shape, dtype, nbins; nbins None or absent: no powder, nothing to check)."""
    nbins = meta.get("nbins")
    if not nbins:
        return
    frame = int(np.prod(meta["shape"])) * np.dtype(meta["dtype"]).itemsize
    if 8 * nbins > POWDER_MAX_FRACTION * frame:
        raise ValueError(
            "a powder of %d bins is %d bytes a frame, more than %g of the %d bytes of the frame: "
            "this is not a reduction, and is not written" % (nbins, 8 * nbins, POWDER_MAX_FRACTION, frame)
        )


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
