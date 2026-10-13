"""The decoders behind one interface: make_worker() -> fn(buf, offsets, lengths) ->
(npx, values, indices, powder).  Every worker owns one decoder and is the only user of it."""

import numpy as np


def _prepared_matrix(mask, csc, dtype, exact, maxval, float_asked=False):
    """The matrix a backend gives the decoder: the demo's policy for exact sums.  exact=None:
    bslz4_to_sparse.fixed_point_matrix for 8 and 16-bit integer pixels (and wider ones when maxval
    is given), unless float_asked or the matrix cannot be fixed-point, then csc as it is;
    exact=True: fixed_point_matrix or an error; exact=False: csc as it is."""
    import bslz4_to_sparse as b

    dt = np.dtype(dtype)
    if exact is None:
        if float_asked or dt.kind not in "ui" or (dt.itemsize > 2 and maxval is None):
            return csc
        try:
            return b.fixed_point_matrix(csc, mask, dt, maxval)
        except ValueError:
            return csc
    return b.fixed_point_matrix(csc, mask, dt, maxval) if exact else csc


def _names_float_dot(pipeline):
    import bslz4_to_sparse as b

    name = b._pipeline.NAMES["dot"][b._pipeline.parse(pipeline)[b._pipeline.DOT]]
    return name != "auto" and name not in b._matrix.INT_DOTS


class CpuBackend(object):
    """bslz4_to_sparse.chunk2sparse, or chunk2sparseCSC with a matrix (the GIL is released).
    mask, dtype: the detector mask and pixel type; csc: a pyFAI CSC matrix or None; cut, howmany,
    thresholds: as chunk2sparse; pipeline: passed through to it; exact, maxval: as chunk2sparseCSC:
    by default the powder is int64 fixed-point sums for 16-bit pixels (the scale goes to the sink's
    meta as powder_scale) and float64 sums otherwise; exact=False for a matrix that is not a sum of
    counts (rings, q-weighted rows)."""

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
        exact=None,
        maxval=None,
    ):
        self.mask, self.csc, self.dtype, self.cut = mask, csc, np.dtype(dtype), cut
        self.exact, self.maxval = exact, maxval
        self.howmany, self.thresholds, self.pipeline = howmany, thresholds, pipeline
        self._fixed = None

    def _matrix(self):
        if self._fixed is None:  # once, shared by the workers
            self._fixed = _prepared_matrix(
                self.mask,
                self.csc,
                self.dtype,
                self.exact,
                self.maxval,
                float_asked=_names_float_dot(self.pipeline),
            )
        return self._fixed

    def make_worker(self):
        """A decoder for one worker: fn(buffer, offsets, lengths) -> (npx, values, indices,
        powder)."""
        import bslz4_to_sparse as b

        kw = dict(howmany=self.howmany, thresholds=self.thresholds, pipeline=self.pipeline)
        if self.csc is None:
            c = b.chunk2sparse(self.mask, self.dtype, **kw)
        else:
            c = b.chunk2sparseCSC(self.mask, self._matrix(), self.dtype, **kw)
        cut = self.cut

        def run(buf, offsets, lengths):
            r = c.from_buffer(buf, offsets, lengths, cut)
            return r[0], r[1][0], r[1][1], (r[2] if len(r) > 2 else None)

        run.powder_scale = getattr(c, "powder_scale", None)
        return run


class DecompressBackend(object):
    """The front half alone: bslz4_to_sparse.decompress into a preallocated array, no sparse step.
    unshuffle=False gives each block as stored (before the bit unshuffle); True the pixel values.
    Returns npx = 0 for every frame, values = the decompressed frames, no indices or powder.
    dtype, shape: pixel type and frame shape; batch: the most frames per call (run's batch must
    not be larger)."""

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
            b.decompress_from_buffer(buf, offsets, lengths, dtype, unshuffle=unshuffle, out=out[:n])
            return npx[:n], out[:n], None, None

        return run


class GpuBackend(object):
    """bslz4_to_sparse.cuda (cupy): one decoder per worker, each with its own CUDA stream.
    Build the workers after any fork: CUDA must be initialised in the process that uses it.
    mask, dtype, csc, cut: as CpuBackend; batch: frames per call; acc: accumulator type of the CSC
    sum (None: 'i64' where exact applies, else 'f64'); howmany, thresholds: as chunk2sparse; exact,
    maxval: as CpuBackend, with the same integers for the same matrix.
    """

    name = "gpu"

    def __init__(
        self,
        mask,
        dtype,
        csc=None,
        cut=0,
        batch=16,
        acc=None,
        howmany=None,
        thresholds=None,
        exact=None,
        maxval=None,
    ):
        self.mask, self.csc, self.dtype, self.cut = mask, csc, np.dtype(dtype), cut
        self.exact, self.maxval = exact, maxval
        self.batch, self.acc, self.howmany, self.thresholds = batch, acc, howmany, thresholds
        self._fixed = None

    def _matrix(self):
        if self._fixed is None:  # once, shared by the workers
            self._fixed = _prepared_matrix(
                self.mask,
                self.csc,
                self.dtype,
                self.exact,
                self.maxval,
                float_asked=self.acc not in (None, "i64"),
            )
        return self._fixed

    def make_worker(self):
        """A GPU decoder for one worker: fn(buffer, offsets, lengths) -> (npx, values, indices,
        powder)."""
        from bslz4_to_sparse import cuda

        kw = dict(batch=self.batch, howmany=self.howmany, thresholds=self.thresholds)
        if self.csc is None:
            c = cuda.chunk2sparse(self.mask, self.dtype, **kw)
        else:
            c = cuda.chunk2sparseCSC(
                self.mask, self._matrix(), self.dtype, acc=self.acc, **kw
            )
        cut = self.cut

        def run(buf, offsets, lengths):
            r = c.from_buffer(buf, offsets, lengths, cut)
            return r[0], r[1][0], r[1][1], (r[2] if len(r) > 2 else None)

        run.powder_scale = getattr(c, "powder_scale", None)
        return run
