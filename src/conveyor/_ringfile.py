"""Ring windows per frame, one file per input file, in the order the frames came in.

For texture pipelines: for every Bragg ring and every azimuth bin, the sum over a window of radial
bins of the normalised 2D cake, with no background subtracted, and, either side of the ring, the
same sum over two background windows.  The reader decides what to do with them and how to sort the
frames into scan order.  Layout, in the group named like the input dataset:

    intensity     (nframes, nwindows, nazim)  float32 (or the dtype asked for): the window sums
    window_ranges (nwindows, 2)  lower and upper edge of each window, in `unit`
    ring_index    (nwindows,)    the ring each window belongs to
    role          (nwindows,)    int8: -1 background below the ring, 0 the ring, +1 background above
    window_nbins  (nwindows,)    how many radial bins each window holds (the sums are not means)
    azimuthal     (nazim,)       the azimuth of each bin, degrees
    attributes    unit, nframes, integrator (path of the Integrator file, if given)

Frame i of the file is frame i of the input file.  Written like ImageD11SparseSink, one whole input
file at a time.
"""

import os

import numpy as np

from ._sinks import Sink
from ._sparsefile import _compression


def rings_name(path):
    """The name of the ring file of an input file: / become _, .h5 becomes _rings.h5"""
    return path.strip("/").replace("/", "_").replace(".h5", "") + "_rings.h5"


def peak_windows(centres, width, background_width=None, gap=0.0):
    """(ranges, ring_index, role): for each ring centre a window of `width` around it with the
    background windows of background_width (default: width) either side, `gap` away from it, all
    in the unit of the centres.  Windows are half-open [lower, upper) like Integrator.rings.  Order:
    below, ring, above, for ring after ring."""
    bw = width if background_width is None else background_width
    ranges, ring, role = [], [], []
    for k, c in enumerate(np.asarray(centres, np.float64)):
        lo, hi = c - width / 2.0, c + width / 2.0
        for r, (a, b) in zip(
            (-1, 0, 1), ((lo - gap - bw, lo - gap), (lo, hi), (hi + gap, hi + gap + bw))
        ):
            ranges.append((a, b))
            ring.append(k)
            role.append(r)
    return np.array(ranges), np.array(ring), np.array(role, np.int8)


class RingSink(Sink):
    """integrator   the "rings" Integrator the run's backend uses (csc=integrator.csc, float dot)
    out_dir, dset, name, compression  as ImageD11SparseSink (the default name ends in _rings.h5)
    dtype        what intensity is stored as (np.float32);  integrator_path  recorded in attributes
    ring_index, role  from peak_windows: which ring each window is for, and below/ring/above
    rows         the rows of the powder that are this integrator's (a slice, from
    IntegratorStack.slices) when the run's matrix is a stack"""

    def __init__(
        self,
        integrator,
        out_dir,
        dset,
        ring_index=None,
        role=None,
        name=rings_name,
        compression="bslz4",
        dtype=np.float32,
        integrator_path=None,
        rows=None,
    ):
        if integrator.kind != "rings":
            raise ValueError("RingSink needs a rings Integrator, not %r" % integrator.kind)
        self.integ, self.out_dir, self.dset, self.name = integrator, out_dir, dset.strip("/"), name
        self.comp, self.dtype, self.ipath = compression, np.dtype(dtype), integrator_path
        nw = len(integrator.ring_ranges)
        self.ring_index = np.arange(nw) if ring_index is None else np.asarray(ring_index)
        self.role = np.zeros(nw, np.int8) if role is None else np.asarray(role, np.int8)
        r = integrator.radial
        self.nbins = np.array(
            [int(((r >= lo) & (r < hi)).sum()) for lo, hi in integrator.ring_ranges], np.int32
        )
        self.rows = rows
        self.held, self.files = [], []

    def begin(self, meta):
        os.makedirs(self.out_dir, exist_ok=True)

    def begin_unit(self, unit):
        if unit.frame0 != 0 or unit.n != unit.nfile:
            raise ValueError(
                "%s: ring files are written per input file, but this unit is frames %d-%d of %d: "
                "make_units(max_bytes=...) larger"
                % (unit.path, unit.frame0, unit.frame0 + unit.n - 1, unit.nfile)
            )
        self.held = []

    def write(self, file, frame0, npx, values, indices, powder):
        """Normalise the batch and keep it as (frames, windows, azimuth) until the file is done."""
        naz, nw = self.integ.bins_shape
        if self.rows is not None:
            powder = powder[:, self.rows]
        inten = self.integ.intensity(powder).reshape(len(powder), naz, nw)
        self.held.append(np.ascontiguousarray(inten.transpose(0, 2, 1), self.dtype))

    def end_unit(self, unit):
        """Write the file of the unit."""
        import h5py

        data = np.concatenate(self.held)
        out = os.path.join(self.out_dir, self.name(unit.path))
        kw = _compression(self.comp)
        with h5py.File(out + ".tmp", "w") as h:
            g = h.require_group(self.dset)
            g.create_dataset(
                "intensity", data=data, chunks=(min(len(data), 16),) + data.shape[1:], **kw
            )
            g.create_dataset("window_ranges", data=self.integ.ring_ranges)
            g.create_dataset("ring_index", data=self.ring_index)
            g.create_dataset("role", data=self.role)
            g.create_dataset("window_nbins", data=self.nbins)
            g.create_dataset("azimuthal", data=self.integ.azimuthal)
            g.attrs["unit"], g.attrs["nframes"] = self.integ.unit, len(data)
            if self.ipath:
                g.attrs["integrator"] = self.ipath
        os.replace(out + ".tmp", out)
        self.files.append(out)
        self.held = []

    def end(self):
        """The files written."""
        return dict(files=self.files)
