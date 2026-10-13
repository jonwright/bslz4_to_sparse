"""ImageD11's sparse pixel file, one per input file, with the powder sums beside it.

ImageD11's lima_segmenter writes one sparse file per lima file, so workers never share a file and
a later step (assemble_label) gathers them.  This sink does the same, so ImageD11 reads the result
as it reads its own.  Layout, in the group named like the input dataset (the lima path):

    row, col    uint16 (npx,)        pixel coordinates (dimension 0, dimension 1), frame after frame
    intensity   pixel dtype (npx,)   the pixel values, always the input dtype
    nnz         uint32 (nframes,)    the number of pixels of each frame
    attributes  nframes, shape0, shape1, npx, itype ("uint16", the type of row and col);
                min_pixels_in_object when small objects were dropped; sparse_skipped (the reason)
                when the pixels were not sparse and are not written

and, when the run has a matrix, in the same group

    powder      (nframes, nbins)     the summed powder: int64 sums if the backend is exact (then the
                                     attribute scale is the factor to float), else float64;
                                     powder_dtype float32 or float64 stores it as float (the sums
                                     were accumulated in 64 bits, scale applied, no scale attribute)
    attributes  of powder: scale (int64 only), integrator (the path of an Integrator file, if given)

With a stack (IntegratorStack) there is a dataset powder_<name> for every block, the rows of the
block, with the same attributes.

Only the sparse pixels are ImageD11's; powder and its attributes are ours, and ImageD11 ignores them.
When the pixels of a file were not sparse (a GuardSink skipped them) the file holds no pixels and
the attribute sparse_skipped says why.  No cleaning is applied unless min_pixels is given: then the pixels of connected
objects smaller than that are dropped (ImageD11's pixels_in_spot, its labelling).  Datasets are compressed with the bitshuffle filter of hdf5plugin (compression
"bslz4", "bszstd", "lzf" or None).

A sink writes whole files: use make_work_units with max_unit_bytes large enough that a unit is a file.
"""

import os

import numpy as np

from ._sinks import Sink


def imaged11_name(path):
    """ImageD11's name for the sparse file of an input file: / become _, .h5 becomes _sparse.h5"""
    return path.strip("/").replace("/", "_").replace(".h5", "") + "_sparse.h5"


def _compression(name):
    import hdf5plugin

    if name == "bslz4":
        return hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
    if name == "bszstd":
        return hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=1)
    if name == "lzf":
        return dict(compression="lzf", shuffle=True)
    if name is None:
        return {}
    raise ValueError("compression: bslz4, bszstd, lzf or None")


def keep_large_objects(flat, values, shape, min_pixels):
    """Which pixels belong to a connected object of at least min_pixels pixels: a boolean array.
    Pixels (flat indices into a frame of `shape`, in ascending order, and their values) are
    labelled into connected objects by ImageD11 (ImageD11.sparseframe, the labelling of its
    lima_segmenter), and the pixels of the objects that are too small, single pixels above all, are
    False.  Needs ImageD11."""
    try:
        from ImageD11 import sparseframe
    except ImportError:
        raise ImportError("min_pixels needs ImageD11 (its connected-object labelling)")
    nj = shape[1]
    s = sparseframe.sparse_frame(
        (flat // nj).astype(np.uint16), (flat % nj).astype(np.uint16), tuple(shape)
    )
    s.set_pixels("f32", np.asarray(values, np.float32))
    nobj = sparseframe.sparse_connected_pixels(s, threshold=0, data_name="f32", label_name="cp")
    labels = s.pixels["cp"]
    return np.bincount(labels, minlength=nobj)[labels] >= min_pixels


class ImageD11SparseSink(Sink):
    """out_dir      where the files go (created);  dset  the dataset path of the input, which is the
    group written;  name  function input path -> file name (imaged11_name);  compression  see the
    module docstring;  powder_dtype  None (as the decoder gives it: int64 or float64), np.float32 or
    np.float64;  integrator  path of an Integrator file to record beside the powder;  stack  the
    IntegratorStack the run's matrix is: each block is written as its own dataset powder_<name>
    (raw sums of the block's rows, with the normalisation in the integrator file);  min_pixels  drop
    the pixels of connected objects (by ImageD11's labelling) smaller than this, 2 or 3 say: single
    pixels are mostly noise, and the file is much smaller (None: keep every pixel; needs ImageD11).
    """

    def __init__(
        self,
        out_dir,
        dset,
        name=imaged11_name,
        compression="bslz4",
        powder_dtype=None,
        integrator=None,
        stack=None,
        min_pixels=None,
    ):
        self.out_dir, self.dset, self.name = out_dir, dset.strip("/"), name
        self.comp, self.pdtype, self.integrator = compression, powder_dtype, integrator
        self.stack = stack
        self.min_pixels = min_pixels if min_pixels and min_pixels > 1 else None
        self.files, self.powder = [], []

    def begin(self, meta):
        self.meta = meta
        self.shape = tuple(meta["shape"])
        if max(self.shape) > 65536:
            raise ValueError(
                "row and col are uint16: a frame side of %d does not fit" % max(self.shape)
            )
        os.makedirs(self.out_dir, exist_ok=True)

    def begin_unit(self, unit):
        if unit.frame0 != 0 or unit.n != unit.nfile:
            raise ValueError(
                "%s: the sparse file is written per input file, but this unit is frames %d-%d of "
                "%d: make_work_units(max_unit_bytes=...) larger"
                % (unit.path, unit.frame0, unit.frame0 + unit.n - 1, unit.nfile)
            )
        self.held = []
        self.skipped_reason = None

    def sparse_skipped(self, unit, reason):
        """The pixels of this file are not sparse: the file is written without them, and says why."""
        self.skipped_reason = reason

    def write(self, file, frame0, npx, values, indices, powder):
        """Keep the batch (trimmed to its pixels) until the file is complete."""
        for j in range(len(npx)):
            k = int(npx[j])
            v, ix = values[j, :k].copy(), indices[j, :k].copy()
            if self.min_pixels and k:
                keep = keep_large_objects(ix, v, self.shape, self.min_pixels)
                v, ix = v[keep], ix[keep]
            self.held.append((len(v), v, ix))
        if powder is not None:
            if self.stack is not None and powder.dtype.kind in "iu":
                float_blocks = [
                    k
                    for k, b in self.stack.blocks.items()
                    if b.kind not in ("powder1d", "powder2d")
                ]
                if float_blocks:
                    raise ValueError(
                        "the blocks %s of the stack are not sums of counts: make the backend with "
                        "exact=False" % float_blocks
                    )
            self.powder.append(powder.copy())

    def end_unit(self, unit):
        """Write the file of the unit."""
        import h5py

        nj = self.shape[1]
        nnz = np.array([h[0] for h in self.held], np.uint32)
        flat = np.concatenate([h[2] for h in self.held]) if self.held else np.zeros(0, np.uint32)
        val = np.concatenate([h[1] for h in self.held])
        out = os.path.join(self.out_dir, self.name(unit.path))
        kw = _compression(self.comp)
        with h5py.File(out + ".tmp", "w") as h:
            g = h.require_group(self.dset)
            g.attrs.update(
                nframes=len(nnz),
                shape0=self.shape[0],
                shape1=self.shape[1],
                npx=len(flat),
                itype="uint16",
            )
            if self.min_pixels:
                g.attrs["min_pixels_in_object"] = self.min_pixels
            if self.skipped_reason:  # nnz is 0 everywhere: no pixels, not "no signal"
                g.attrs["sparse_skipped"] = self.skipped_reason
            chunks = (
                dict(chunks=(max(1, min(len(flat), 10000)),), maxshape=(None,)) if len(flat) else {}
            )
            ck = kw if len(flat) else {}
            opts = dict(chunks)
            opts.update(ck)
            g.create_dataset("row", data=(flat // nj).astype(np.uint16), **opts)
            g.create_dataset("col", data=(flat % nj).astype(np.uint16), **opts)
            g.create_dataset("intensity", data=val, **opts)
            g.create_dataset("nnz", data=nnz)
            pw = self.powder
            if pw:
                pw = np.concatenate(pw)
                scale = self.meta.get("powder_scale")
                if self.pdtype is not None:
                    if pw.dtype.kind == "i":
                        pw = pw * np.float64(scale)
                    pw = pw.astype(self.pdtype)
                parts = {"powder": pw}
                if self.stack is not None:  # one dataset per block of the stacked matrix
                    parts = {"powder_" + k: pw[:, sl] for k, sl in self.stack.slices.items()}
                for name, rows in parts.items():
                    d = g.create_dataset(
                        name,
                        data=np.ascontiguousarray(rows),
                        chunks=(min(len(rows), 64), rows.shape[1]),
                        **kw
                    )
                    if rows.dtype.kind == "i" and scale is not None:
                        d.attrs["scale"] = scale
                    if self.integrator:
                        d.attrs["integrator"] = self.integrator
        os.replace(out + ".tmp", out)
        self.files.append(out)
        self.held, self.powder = [], []

    def end(self):
        """The files written."""
        return dict(files=self.files)
