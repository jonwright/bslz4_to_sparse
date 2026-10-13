"""The decoders behind one interface: make_worker() -> fn(buf, offsets, lengths) ->
(npx, values, indices, powder).  Every worker owns one decoder and is the only user of it."""

import numpy as np


class CpuBackend(object):
    """bslz4_to_sparse.chunk2sparse, or chunk2sparseCSC with a matrix (the GIL is released).
    mask, dtype: the detector mask and pixel type; csc: a pyFAI CSC matrix or None; cut, howmany,
    thresholds: as chunk2sparse; pipeline: passed through to it; exact: integer pixels, the powder
    is the int64 fixed-point sums (see chunk2sparseCSC), its scale in the sink's meta."""

    name = "cpu"

    def __init__(
        self,
        mask,
        dtype,
        csc=None,
        cut=0,
        howmany=None,
        thresholds=None,
        pipeline=None,
        exact=False,
    ):
        self.mask, self.csc, self.dtype, self.cut = mask, csc, np.dtype(dtype), cut
        self.exact = exact
        self.howmany, self.thresholds, self.pipeline = howmany, thresholds, pipeline

    def make_worker(self):
        """A decoder for one worker: fn(buffer, offsets, lengths) -> (npx, values, indices,
        powder)."""
        import bslz4_to_sparse as b

        kw = dict(howmany=self.howmany, thresholds=self.thresholds, pipeline=self.pipeline)
        if self.csc is None:
            c = b.chunk2sparse(self.mask, self.dtype, **kw)
        else:
            c = b.chunk2sparseCSC(self.mask, self.csc, self.dtype, exact=self.exact, **kw)
        cut = self.cut

        def run(buf, offsets, lengths):
            r = c.decode_offsets(buf, offsets, lengths, cut)
            return r[0], r[1][0], r[1][1], (r[2] if len(r) > 2 else None)

        run.powder_scale = getattr(c, "powder_scale", None)
        return run


class DecompressBackend(object):
    """The front half alone: bslz4_to_sparse.decompress into a preallocated array, no sparse step.
    unshuffle=False gives each block as stored (before the bit unshuffle); True the pixel values.
    Returns npx = 0 for every frame, values = the decompressed frames, no indices or powder.
    dtype, shape: pixel type and frame shape; batch: frames per call."""

    name = "decompress"

    def __init__(self, dtype, shape, unshuffle=True, batch=16):
        self.dtype, self.shape, self.unshuffle, self.batch = (
            np.dtype(dtype),
            tuple(shape),
            unshuffle,
            batch,
        )

    def make_worker(self):
        """A decompressor for one worker: fn(buffer, offsets, lengths) -> (npx, frames, None,
        None)."""
        import bslz4_to_sparse as b

        npix = int(np.prod(self.shape))
        if self.unshuffle:
            out = np.empty((self.batch, npix), self.dtype)
        else:
            out = np.empty((self.batch, npix * self.dtype.itemsize), np.uint8)
        npx = np.zeros(self.batch, np.int32)
        dtype, unshuffle = self.dtype, self.unshuffle

        def run(buf, offsets, lengths):
            n = len(offsets)
            b.decompress_offsets(buf, offsets, lengths, dtype, unshuffle=unshuffle, out=out[:n])
            return npx[:n], out[:n], None, None

        return run


class GpuBackend(object):
    """bslz4_to_sparse.cuda (cupy): one decoder per worker, each with its own CUDA stream.
    Build the workers after any fork: CUDA must be initialised in the process that uses it.
    mask, dtype, csc, cut: as CpuBackend; batch: frames per call; acc: accumulator type of the CSC
    sum ("f64"); howmany, thresholds: as chunk2sparse.
    """

    name = "gpu"

    def __init__(
        self, mask, dtype, csc=None, cut=0, batch=16, acc="f64", howmany=None, thresholds=None
    ):
        self.mask, self.csc, self.dtype, self.cut = mask, csc, np.dtype(dtype), cut
        self.batch, self.acc, self.howmany, self.thresholds = batch, acc, howmany, thresholds

    def make_worker(self):
        """A GPU decoder for one worker: fn(buffer, offsets, lengths) -> (npx, values, indices,
        powder)."""
        from bslz4_to_sparse import cuda

        kw = dict(batch=self.batch, howmany=self.howmany, thresholds=self.thresholds)
        if self.csc is None:
            c = cuda.chunk2sparse(self.mask, self.dtype, **kw)
        else:
            c = cuda.chunk2sparseCSC(self.mask, self.csc, self.dtype, acc=self.acc, **kw)
        cut = self.cut

        def run(buf, offsets, lengths):
            r = c.decode_offsets(buf, offsets, lengths, cut)
            return r[0], r[1][0], r[1][1], (r[2] if len(r) > 2 else None)

        return run
