"""ImageD11's sparse pixel file, one per input file, with the powder sums beside it.

ImageD11's lima_segmenter writes one sparse file per lima file, so workers never share a file and
a later step (assemble_label) gathers them.  This sink does the same, so ImageD11 reads the result
as it reads its own.  Layout, in the group named like the input dataset (the lima path):

    row, col    uint16 (npx,)        pixel coordinates (dimension 0, dimension 1), frame after frame
    intensity   pixel dtype (npx,)   the pixel values, always the input dtype
    nnz         uint32 (nframes,)    the number of pixels of each frame
    attributes  nframes, shape0, shape1, npx, itype ("uint16", the type of row and col)

and, when the run has a matrix, in the same group

    powder      (nframes, nbins)     the summed powder: int64 sums if the backend is exact (then the
                                     attribute scale is the factor to float), else float64;
                                     powder_dtype float32 or float64 stores it as float (the sums
                                     were accumulated in 64 bits, scale applied, no scale attribute)
    attributes  of powder: scale (int64 only), integrator (the path of an Integrator file, if given)

Only the sparse pixels are ImageD11's; powder and its attributes are ours, and ImageD11 ignores them.
When the pixels of a file were not sparse (a GuardSink skipped them) the file holds no pixels and
the attribute sparse_skipped says why.  No cleaning is applied (ImageD11's pixels_in_spot is a
policy of its own).  Datasets are compressed with the bitshuffle filter of hdf5plugin (compression
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


class ImageD11SparseSink(Sink):
    """out_dir      where the files go (created);  dset  the dataset path of the input, which is the
    group written;  name  function input path -> file name (imaged11_name);  compression  see the
    module docstring;  powder_dtype  None (as the decoder gives it: int64 or float64), np.float32 or
    np.float64;  integrator  path of an Integrator file to record beside the powder;  stack  the
    IntegratorStack the run's matrix is: each block is written as its own dataset powder_<name>
    (raw sums of the block's rows, with the normalisation in the integrator file).
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
    ):
        self.out_dir, self.dset, self.name = out_dir, dset.strip("/"), name
        self.comp, self.pdtype, self.integrator = compression, powder_dtype, integrator
        self.stack = stack
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
            self.held.append((k, values[j, :k].copy(), indices[j, :k].copy()))
        if powder is not None:
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
