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

Kinds: "powder1d", "powder2d", "rings", and "moment" (a 1D matrix and, below it, the same with every
pixel's weights multiplied by its q: the sum of q x signal per bin; Integrator.first_moment).  An
IntegratorStack puts several of them in one matrix, so one pass gives all of them.

Rows of the matrix: "powder1d" the radial bins; "powder2d" radial-major, row = radial * nazim +
azimuth; "rings" azimuth-major, row = azimuth * nrings + ring (reshape to (nazim, nrings)).

File (HDF5; hdf5plugin supplies the bitshuffle filter): attributes bslz4_to_sparse_integrator
(format version), kind ("powder1d", "rings", "powder2d" or "moment"), unit, shape (the detector), nbins,
bins_shape (what the powder of one frame reshapes to); datasets poni, norm, radial,
azimuthal, ring_ranges and offset when present; group csc (attribute format = "CSC", shape = (nbins, npix))
with datasets data (float32), indices (int32), indptr (int32), as scipy and pyFAI name them.

Needs pyFAI only for from_pyfai and ai(), h5py only for save and load.
"""

from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

import numpy as np

from ._files import temporary_directory

VERSION = 1


class Integrator(object):
    """csc        scipy.sparse CSC matrix (nbins, npix); from_pyfai, rings, first_moment and the stack
               make float32 weights
    norm       (nbins,) float64, sum_normalization of an image of ones
    shape      (ni, nj) of the detector;  unit  the radial unit (pyFAI name, e.g. "q_nm^-1")
    radial     bin positions in unit: (nrad,) for 1D or rings, with azimuthal (nazim,) for 2D
    kind       "powder1d", "rings", "powder2d" or "moment";  poni  text of a .poni file
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
        """The number of rows of the matrix: the length of the powder of one frame."""
        return self.csc.shape[0]

    @property
    def bins_shape(self):
        """The shape the powder of one frame reshapes to (see the rows of the matrix in the module docstring; "moment": the first nbins // 2 rows, then the q-weighted ones)."""
        if self.kind == "powder2d":
            return (len(self.radial), len(self.azimuthal))
        if self.kind == "rings":
            return (len(self.azimuthal), len(self.ring_ranges))
        if self.kind == "moment":
            return (2, self.nbins // 2)
        return (self.nbins,)

    @classmethod
    def from_pyfai(cls, ai, npt, unit="q_nm^-1", npt_azim=None, split="bbox", **integrate):
        """The matrix and normalisation of a pyFAI AzimuthalIntegrator: npt radial bins (and
        npt_azim azimuthal ones for a 2D result).  split is pyFAI's pixel splitting ("no", "bbox",
        "full").  integrate: keywords for pyFAI's integrate1d/2d on the image of ones, so that
        correctSolidAngle, polarization_factor, flat, normalization_factor ... are the ones the
        data will be corrected with.  The detector mask of ai is in the matrix.

        pyFAI's CSC integrator is used, else its CSR one, converted (pyFAI 0.17 and 0.18 have no
        CSC); ValueError when it has neither for this splitting (0.18: none for "full" in 2D;
        0.17 raises NotImplementedError).  When pyFAI gives no sum_normalization (0.17, 0.18),
        it is built here from pyFAI's corrections (solid angle, polarisation, flat,
        normalization_factor), and checked against pyFAI's intensity of the image of ones.

        The keywords dummy and delta_dummy are refused (ValueError): they mask pixels by their
        value in each frame, and this matrix and norm are for one fixed set of pixels.  The
        dummy a pyFAI detector carries by default (Eiger, Pilatus: pixels at the type's maximum)
        is not applied either: those pixels stay in the sums unless the mask holds them."""
        for name in ("dummy", "delta_dummy"):
            if integrate.get(name) is not None:
                raise ValueError(
                    "%s is not supported: it masks pixels by their value in each frame, and the "
                    "matrix and normalisation here are for a fixed mask.  Put the pixels in the "
                    "detector mask." % name
                )
        csc, res = pyfai_matrix(ai, npt, unit, npt_azim, split, **integrate)
        norm = getattr(res, "sum_normalization", None)
        if norm is None:
            norm = _norm_from_corrections(ai, res, csc, integrate)
        else:  # (radial, azimuth) like the rows, for 2D
            norm = np.asarray(norm).T.ravel()
        return cls(
            csc,
            norm,
            ai.detector.shape,
            unit,
            radial=res.radial,
            azimuthal=None if npt_azim is None else res.azimuthal,
            kind="powder1d" if npt_azim is None else "powder2d",
            poni=poni_text(ai),
        )

    def ai(self):
        """The pyFAI AzimuthalIntegrator the poni text describes."""
        from pyFAI import load

        with temporary_directory() as d:
            fn = os.path.join(d, "geometry.poni")
            with open(fn, "w") as f:
                f.write(self.poni)
            return load(fn)

    def intensity(self, powder, scale=None):
        """Normalised intensity of summed powder: powder (nbins,) or (nframes, nbins), integers
        divided by scale (from the decoder: its power of two) or floats.  Empty bins (norm 0) are 0, as
        in pyFAI."""
        p = np.asarray(powder, np.float64)
        if scale is not None:
            p = p * (1.0 / scale)  # a power of two: exact
        if self.kind == "moment":  # the first half: the signal; mean_q gives the moment
            p = p[..., : self.nbins // 2]
        out = np.zeros_like(p)
        norm = self.norm[: p.shape[-1]]
        np.divide(p, norm, out=out, where=norm > 0)
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
        R = S.dot(self.csc.tocsr()).astype(np.float32).tocsc()
        return Integrator(
            R,
            np.ones(R.shape[0]),
            self.shape,
            self.unit,
            radial=self.radial,  # the bins the windows were taken from
            azimuthal=self.azimuthal,
            kind="rings",
            poni=self.poni,
            ring_ranges=ranges,
            offset=offset.ravel(),
        )

    def save(self, path, compression=True):
        """Write the integrator to an HDF5 file (see the module docstring)."""
        import h5py

        with h5py.File(path, "w") as h:
            self._write(h, compression)

    def _write(self, h, compression=True):
        """Write into the open HDF5 file or group h."""
        comp = {}
        if compression:
            import hdf5plugin  # noqa: F401  (registers the bitshuffle filter)

            from ._sparsefile import BSLZ4

            comp = dict(BSLZ4)
        c = self.csc
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
            kw = dict(chunks=(min(len(a), 1 << 16),), **comp) if compression and len(a) > 1 else {}
            g.create_dataset(name, data=a, **kw)

    @classmethod
    def load(cls, path):
        """Read an integrator file written by save."""
        import h5py

        with h5py.File(path, "r") as h:
            return cls._read(h, path)

    @classmethod
    def _read(cls, h, where=""):
        import scipy.sparse as sp

        if h.attrs.get("bslz4_to_sparse_integrator") != VERSION:
            raise ValueError("%s is not a version %d integrator file" % (where, VERSION))
        g = h["csc"]
        if g.attrs["format"] != "CSC":
            raise ValueError("%s: the matrix is %r, not CSC" % (where, g.attrs["format"]))
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

    def first_moment(self, q=None):
        """A "moment" Integrator from this 1D one: the same matrix with, below it, the same
        matrix with every pixel's weights multiplied by that pixel's q (in unit; default: pyFAI's
        value at the pixel centre), so one product gives, per bin, the sum of weight x signal and
        the sum of weight x q x signal.  Their ratio is the mean q of the signal in the bin.  The
        weights are no longer 1-like numbers and the second half cannot be normalised by an image
        of ones: use a float dot (not exact).  Rows: the first nbins, then the moment."""
        import scipy.sparse as sp

        if self.kind != "powder1d":
            raise ValueError("first_moment needs a powder1d integrator, this is %r" % self.kind)
        if q is None:
            q = self.ai().array_from_unit(shape=self.shape, typ="center", unit=self.unit)
        q = np.asarray(q, np.float64).ravel()
        if q.size != self.csc.shape[1]:
            raise ValueError("q has %d values for %d pixels" % (q.size, self.csc.shape[1]))
        both = sp.vstack([self.csc, self.csc.dot(sp.diags(q))]).astype(np.float32).tocsc()
        return Integrator(
            both,
            np.concatenate([self.norm, np.zeros(self.nbins)]),
            self.shape,
            self.unit,
            radial=self.radial,
            kind="moment",
            poni=self.poni,
        )

    def mean_q(self, powder):
        """Of a "moment" integrator's powder: the mean q of the signal of each bin, the second
        half of the rows over the first (0 where the bin has no signal)."""
        if self.kind != "moment":
            raise ValueError("mean_q needs a moment integrator")
        p = np.asarray(powder, np.float64)
        n = self.nbins // 2
        out = np.zeros_like(p[..., :n])
        np.divide(p[..., n:], p[..., :n], out=out, where=p[..., :n] != 0)
        return out


class IntegratorStack(object):
    """Several Integrators of one detector as one matrix: everything comes out of one pass over
    the pixels, whatever their number (a 1D powder, ring windows, the first moment ...), each pixel
    reaching all of their bins.  blocks: name -> Integrator, as (name, Integrator) pairs or a dict
    (the order is kept; a plain dict has an order only on Python 3.7+).

    csc     the blocks' matrices stacked, rows block after block (float: a block of rings or of
            moments makes the whole product float, so use the float dot, not exact)
    slices  name -> the rows of csc, and of every powder, that belong to the block
    split(powder, scale=None)  name -> the intensity of that block (powder: (nbins_total,) or (nframes, ...))
    raw(powder)    name -> the rows of the powder of that block, as the decoder gives them"""

    def __init__(self, blocks):
        import scipy.sparse as sp

        self.blocks = OrderedDict(blocks)
        shapes = set(b.shape for b in self.blocks.values())
        if len(shapes) != 1:
            raise ValueError("the blocks are for detectors of different shapes: %r" % shapes)
        self.shape = shapes.pop()
        self.slices, n = OrderedDict(), 0
        for name, b in self.blocks.items():
            self.slices[name] = slice(n, n + b.nbins)
            n += b.nbins
        self.csc = sp.vstack([b.csc for b in self.blocks.values()]).astype(np.float32).tocsc()
        self.nbins = n

    def raw(self, powder):
        """name -> the rows of the powder of each block."""
        p = np.asarray(powder)
        return {k: p[..., sl] for k, sl in self.slices.items()}

    def split(self, powder, scale=None):
        """name -> the intensity of each block (Integrator.intensity of its rows)."""
        return {k: self.blocks[k].intensity(v, scale) for k, v in self.raw(powder).items()}

    def save(self, path, compression=True):
        """Write the blocks to one HDF5 file, a group each (their matrices are not repeated as one)."""
        import h5py

        with h5py.File(path, "w") as h:
            h.attrs["bslz4_to_sparse_integrator_stack"] = VERSION
            h.attrs["names"] = np.array(list(self.blocks), dtype="S")
            for k, b in self.blocks.items():
                b._write(h.create_group(k), compression)

    @classmethod
    def load(cls, path):
        """Read a stack written by save."""
        import h5py

        with h5py.File(path, "r") as h:
            if h.attrs.get("bslz4_to_sparse_integrator_stack") != VERSION:
                raise ValueError("%s is not a version %d integrator stack" % (path, VERSION))
            names = [n.decode() if isinstance(n, bytes) else n for n in h.attrs["names"]]
            return cls([(n, Integrator._read(h[n], path)) for n in names])


def sparse_method(split, dim):
    """pyFAI's cython CSC integrator for split ("no", "bbox", "full") in dim (1 or 2) dimensions,
    else its CSR one; ValueError when it has neither.  Matched exactly here: pyFAI's own
    selection silently falls back to another splitting or algorithm."""
    import pyFAI
    from pyFAI.method_registry import IntegrationMethod

    for algo in ("csc", "csr"):
        for m in IntegrationMethod.select_method(dim, split, algo, "cython"):
            if (m.dimension, m.split_lower, m.algo_lower, m.impl_lower) == (dim, split, algo, "cython"):
                return m
    raise ValueError(
        "pyFAI %s has no CSC or CSR integrator for %r pixel splitting in %dD"
        % (pyFAI.version, split, dim)
    )


def pyfai_matrix(ai, npt, unit="q_nm^-1", npt_azim=None, split="bbox", **integrate):
    """pyFAI's matrix for npt radial (and npt_azim azimuthal) bins with split pixel splitting, as
    a scipy CSC (bins x pixels; its CSC as it is, its CSR converted), and the result of the
    integration of an image of ones that built it (integrate: keywords for integrate1d/2d).
    ValueError when pyFAI has no such matrix or built none."""
    import scipy.sparse as sp

    from bslz4_to_sparse import _matrix

    dim = 1 if npt_azim is None else 2
    method = sparse_method(split, dim)
    # pyFAI 0.17 and 0.18 keep one CSR engine under this name and reuse it whatever the pixel
    # splitting asked for: dropped, so that it is built for this one
    ai.engines.pop("csr_integrator", None)
    ones = np.ones(ai.detector.shape, np.float32)
    if dim == 1:
        res = ai.integrate1d(ones, npt, unit=unit, method=method, **integrate)
    else:
        res = ai.integrate2d(ones, npt, npt_azim, unit=unit, method=method, **integrate)
    nbins, npix = npt * (npt_azim or 1), ones.size
    holder = ai.engines.get(method) or ai.engines.get("csr_integrator")
    if holder is None or holder.engine is None:
        raise ValueError("pyFAI built no engine for %s" % (method,))
    e = holder.engine
    if method.algo_lower == "csr":  # rows are the bins: scipy checks len(indptr) == nbins + 1
        e = sp.csr_matrix(
            (np.asarray(e.data), np.asarray(e.indices), np.asarray(e.indptr)), shape=(nbins, npix)
        )
    nm = _matrix.normalise_matrix(e, npix=npix)
    if nm.shape != (nbins, npix):
        raise ValueError("pyFAI's matrix is %r, expected %r" % (nm.shape, (nbins, npix)))
    return sp.csc_matrix((nm.data, nm.indices, nm.indptr), shape=nm.shape), res


_CORRECTIONS = ("correctSolidAngle", "polarization_factor", "flat", "normalization_factor")


def _norm_from_corrections(ai, res, csc, integrate):
    """sum_normalization for a pyFAI that does not give it: per bin, the sum of weight x the
    correction of each pixel (solid angle x polarisation x flat x normalization_factor), as
    newer pyFAI computes it.  Checked against the intensity res of the image of ones, which such
    a pyFAI computes as the weighted mean of 1 / correction; ValueError when it differs or when
    integrate has a keyword besides the corrections, mask and the ranges."""
    other = set(integrate) - set(_CORRECTIONS) - {"mask", "radial_range", "azimuth_range"}
    if other:
        raise ValueError(
            "this pyFAI gives no sum_normalization; it is built here only from %s, not %s"
            % (", ".join(_CORRECTIONS), ", ".join(sorted(other)))
        )
    shape = ai.detector.shape
    c = np.full(shape, float(integrate.get("normalization_factor", 1.0)))
    if integrate.get("correctSolidAngle", True):
        c = c * ai.solidAngleArray(shape)
    if integrate.get("polarization_factor") is not None:
        c = c * ai.polarization(shape, integrate["polarization_factor"])
    flat = integrate.get("flat")
    flat = ai.detector.flatfield if flat is None else flat
    if flat is not None:
        c = c * flat
    c = c.ravel().astype(np.float64)
    count = csc.dot(np.ones(c.size))
    inv = np.zeros(c.size)
    np.divide(1.0, c, out=inv, where=c != 0)
    hit = count > 0
    mean_inv = csc.dot(inv)[hit] / count[hit]
    got = np.asarray(res.intensity, np.float64)
    got = (got.T if got.ndim == 2 else got).ravel()[hit]  # 2D: (azimuth, radial) to the rows
    # pyFAI's intensity is float32: measured 1.3e-7 at most (0.17.1, every correction and split)
    if not np.allclose(got, mean_inv, rtol=1e-6, atol=0):
        raise ValueError(
            "pyFAI's intensity of an image of ones differs from the corrections built here "
            "(max relative difference %.2g)" % np.max(np.abs(got - mean_inv) / np.abs(mean_inv))
        )
    return csc.dot(c)


def poni_text(ai):
    """The text of the .poni file of a pyFAI AzimuthalIntegrator."""
    with temporary_directory() as d:
        fn = os.path.join(d, "geometry.poni")
        ai.save(fn)
        with open(fn) as f:
            return f.read()
