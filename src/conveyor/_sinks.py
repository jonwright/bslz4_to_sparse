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


class Sink(object):
    """Where one worker's results go: subclass it and implement write (and begin, end if needed)."""

    def begin(self, meta):
        """called once in the worker before the first batch; meta: dict (dtype, shape, ...)"""

    def write(self, file, frame0, npx, values, indices, powder):
        """Receive one batch of frames (arguments in the module docstring)."""
        raise NotImplementedError

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
