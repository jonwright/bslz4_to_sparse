"""The integrator state: what is needed to turn the summed powder of a run into intensities.

A conveyor run returns, per frame, the sum over the pixels of each bin of weight x pixel value
(pyFAI's sum_signal).  pyFAI's intensity is that divided by sum_normalization, the same sum for an
image of ones with the corrections (flat, solid angle, polarisation, ...) folded in.  An Integrator
keeps what that needs, in one file that results files point to:

    poni      the pyFAI geometry, as the text of a .poni file
    csc       the sparse matrix (bins x pixels), CSC: data, indices, indptr
    norm      sum_normalization of an image of ones, one per bin (float64)
    radial    the bin positions in `unit` (the azimuth too for a 2D matrix);  ring_ranges
              (nrings, 2) in `unit` when the bins are rings;  offset, per bin, added to the
              normalised intensity (the background of rings)

Rows of the matrix: "powder1d" the radial bins; "powder2d" radial-major, row = radial * nazim +
azimuth; "rings" azimuth-major, row = azimuth * nrings + ring (reshape to (nazim, nrings)).

File (HDF5; hdf5plugin supplies the bitshuffle filter): attributes bslz4_to_sparse_integrator
(format version), kind ("powder1d", "rings" or "powder2d"), unit, shape (the detector), nbins,
bins_shape (what the powder of one frame reshapes to); datasets poni, norm, radial,
azimuthal, ring_ranges and offset when present; group csc (attribute format = "CSC", shape = (nbins, npix))
with datasets data (float32), indices (int32), indptr (int32), as scipy and pyFAI name them.

Needs pyFAI only for from_pyfai and ai(), h5py only for save and load.
"""

import os
import tempfile

import numpy as np

VERSION = 1


class Integrator(object):
    """csc        scipy.sparse CSC matrix (nbins, npix), float32 weights
    norm       (nbins,) float64, sum_normalization of an image of ones
    shape      (ni, nj) of the detector;  unit  the radial unit (pyFAI name, e.g. "q_nm^-1")
    radial     bin positions in unit: (nrad,) for 1D or rings, with azimuthal (nazim,) for 2D
    kind       "powder1d", "rings" or "powder2d";  poni  text of a .poni file
    ring_ranges (nrings, 2) lower and upper edge of each ring in unit (kind "rings");
    offset     (nbins,) added to the intensity (kind "rings": minus the background of the ring)"""

    def __init__(
        self,
        csc,
        norm,
        shape,
        unit,
        radial=None,
        azimuthal=None,
        kind="powder1d",
        poni=None,
        ring_ranges=None,
        offset=None,
    ):
        self.csc = csc.tocsc()
        self.norm = np.asarray(norm, np.float64).ravel()
        if self.norm.size != self.csc.shape[0]:
            raise ValueError("norm has %d values for %d bins" % (self.norm.size, self.csc.shape[0]))
        if self.csc.shape[1] != int(np.prod(shape)):
            raise ValueError(
                "matrix columns %d are not the pixels of %r" % (self.csc.shape[1], shape)
            )
        self.shape, self.unit, self.kind, self.poni = tuple(shape), unit, kind, poni
        self.radial = None if radial is None else np.asarray(radial, np.float64)
        self.azimuthal = None if azimuthal is None else np.asarray(azimuthal, np.float64)
        self.ring_ranges = None if ring_ranges is None else np.asarray(ring_ranges, np.float64)
        self.offset = None if offset is None else np.asarray(offset, np.float64).ravel()

    @property
    def nbins(self):
        return self.csc.shape[0]

    @property
    def bins_shape(self):
        """The shape the powder of one frame reshapes to (see the rows of the matrix above)."""
        if self.kind == "powder2d":
            return (len(self.radial), len(self.azimuthal))
        if self.kind == "rings":
            return (len(self.azimuthal), len(self.ring_ranges))
        return (self.nbins,)

    @classmethod
    def from_pyfai(cls, ai, npt, unit="q_nm^-1", npt_azim=None, split="bbox", **integrate):
        """The matrix and normalisation of a pyFAI AzimuthalIntegrator: npt radial bins (and
        npt_azim azimuthal ones for a 2D result).  split is pyFAI's pixel splitting ("no", "bbox",
        "full").  integrate: keywords for pyFAI's integrate1d/2d on the image of ones, so that
        correctSolidAngle, polarization_factor, flat, normalization_factor ... are the ones the
        data will be corrected with.  The detector mask of ai is in the matrix."""
        import scipy.sparse as sp
        from pyFAI.method_registry import IntegrationMethod

        from bslz4_to_sparse import _matrix

        dim = 1 if npt_azim is None else 2
        method = IntegrationMethod.select_one_available((split, "csc", "cython"), dim=dim)
        ones = np.ones(ai.detector.shape, np.float32)
        if dim == 1:
            res = ai.integrate1d(ones, npt, unit=unit, method=method, **integrate)
            norm, azimuthal = np.asarray(res.sum_normalization).ravel(), None
        else:
            res = ai.integrate2d(ones, npt, npt_azim, unit=unit, method=method, **integrate)
            norm = np.asarray(res.sum_normalization).T.ravel()  # (radial, azimuth) like the rows
            azimuthal = res.azimuthal
        nm = _matrix.normalise_matrix(ai.engines[method].engine, npix=ones.size)
        csc = sp.csc_matrix((nm.data, nm.indices, nm.indptr), shape=nm.shape)
        return cls(
            csc,
            norm,
            ai.detector.shape,
            unit,
            radial=res.radial,
            azimuthal=azimuthal,
            kind="powder1d" if dim == 1 else "powder2d",
            poni=poni_text(ai),
        )

    def ai(self):
        """The pyFAI AzimuthalIntegrator the poni text describes."""
        from pyFAI import load

        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "geometry.poni")
            with open(fn, "w") as f:
                f.write(self.poni)
            return load(fn)

    def intensity(self, powder, scale=None):
        """Normalised intensity of summed powder: powder (nbins,) or (nframes, nbins), integers
        times scale (from the decoder: its power of two) or floats.  Empty bins (norm 0) are 0, as
        in pyFAI."""
        p = np.asarray(powder, np.float64)
        if scale is not None:
            p = p * scale
        out = np.zeros_like(p)
        np.divide(p, self.norm, out=out, where=self.norm > 0)
        if self.offset is not None:
            out = out + self.offset
        return out

    def rings(self, ranges, background=None):
        """A "rings" Integrator from this 2D one: for each (lower, upper) in ranges (in unit),
        every azimuth bin gets the sum over the radial bins whose centre lies in [lower, upper) of
        the normalised intensity (summed signal / norm of each bin), minus the sum over them of
        `background` (a value per radial bin, default none).  The division by each bin's norm is
        folded into the weights, so the result of the matrix product already is the intensity: use
        a float dot (not exact), norm is 1.  This is the sum of the normalised 2D cake over each
        peak window, per azimuth, that texture pipelines compute from pyFAI's integrate2d."""
        import scipy.sparse as sp

        if self.kind != "powder2d":
            raise ValueError("rings needs a powder2d integrator, this is %r" % self.kind)
        nrad, naz = len(self.radial), len(self.azimuthal)
        ranges = np.asarray(ranges, np.float64).reshape(-1, 2)
        inv = np.zeros(self.norm.size)
        np.divide(1.0, self.norm, out=inv, where=self.norm > 0)
        inv = inv.reshape(nrad, naz)
        rows, cols, vals = [], [], []
        offset = np.zeros((naz, len(ranges)))
        for k, (lo, hi) in enumerate(ranges):
            inside = np.flatnonzero((self.radial >= lo) & (self.radial < hi))
            if background is not None:
                offset[:, k] = -np.asarray(background, np.float64)[inside].sum()
            a = np.tile(np.arange(naz), len(inside))
            r = np.repeat(inside, naz)
            rows.append(a * len(ranges) + k)
            cols.append(r * naz + a)
            vals.append(inv[r, a])
        S = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(naz * len(ranges), nrad * naz),
        )
        R = (S @ self.csc.tocsr()).astype(np.float32).tocsc()
        return Integrator(
            R,
            np.ones(R.shape[0]),
            self.shape,
            self.unit,
            azimuthal=self.azimuthal,
            kind="rings",
            poni=self.poni,
            ring_ranges=ranges,
            offset=offset.ravel(),
        )

    def save(self, path, compression=True):
        """Write the integrator to an HDF5 file (see the module docstring)."""
        import h5py

        comp = {}
        if compression:
            import hdf5plugin

            comp = hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
        c = self.csc
        with h5py.File(path, "w") as h:
            h.attrs["bslz4_to_sparse_integrator"] = VERSION
            h.attrs["kind"], h.attrs["unit"] = self.kind, self.unit
            h.attrs["shape"], h.attrs["nbins"] = np.array(self.shape), self.nbins
            h.attrs["bins_shape"] = np.array(self.bins_shape)
            if self.poni is not None:
                h["poni"] = self.poni
            h.create_dataset("norm", data=self.norm)
            for name in ("radial", "azimuthal", "ring_ranges", "offset"):
                v = getattr(self, name)
                if v is not None:
                    h.create_dataset(name, data=v).attrs["unit"] = (
                        self.unit if name != "azimuthal" else "deg"
                    )
            g = h.create_group("csc")
            g.attrs["format"], g.attrs["shape"] = "CSC", np.array(c.shape)
            for name, a, dt in (
                ("data", c.data, np.float32),
                ("indices", c.indices, np.int32),
                ("indptr", c.indptr, np.int32),
            ):
                a = np.ascontiguousarray(a, dt)
                kw = (
                    dict(chunks=(min(len(a), 1 << 16),), **comp)
                    if compression and len(a) > 1
                    else {}
                )
                g.create_dataset(name, data=a, **kw)

    @classmethod
    def load(cls, path):
        """Read an integrator file written by save."""
        import h5py
        import scipy.sparse as sp

        with h5py.File(path, "r") as h:
            if h.attrs.get("bslz4_to_sparse_integrator") != VERSION:
                raise ValueError("%s is not a version %d integrator file" % (path, VERSION))
            g = h["csc"]
            if g.attrs["format"] != "CSC":
                raise ValueError("%s: the matrix is %r, not CSC" % (path, g.attrs["format"]))
            csc = sp.csc_matrix(
                (g["data"][()], g["indices"][()], g["indptr"][()]), shape=tuple(g.attrs["shape"])
            )
            opt = {n: h[n][()] for n in ("radial", "azimuthal", "ring_ranges", "offset") if n in h}
            poni = h["poni"][()] if "poni" in h else None
            if isinstance(poni, bytes):
                poni = poni.decode("utf-8")
            return cls(
                csc,
                h["norm"][()],
                tuple(h.attrs["shape"]),
                h.attrs["unit"],
                kind=h.attrs["kind"],
                poni=poni,
                **opt
            )


def poni_text(ai):
    """The text of the .poni file of a pyFAI AzimuthalIntegrator."""
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "geometry.poni")
        ai.save(fn)
        with open(fn) as f:
            return f.read()
