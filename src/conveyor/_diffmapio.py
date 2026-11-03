"""Write a pyFAI-diffmap-compatible HDF5 from integrated frames.

The output is a single NeXus-layout file that pyFAI-diffmap-viewer,
diffmap-refine (reader "diffmap") and xrdmap can all open with defaults:

    /entry_0000                       (program_name="pyfai", title="diff_map")
      /measurement/images_0001        raw frames (external link -> master VDS)
      /pyFAI/                         NXprocess
        program/version/date/PONIfile/offset(/maskfile)
        dim0, dim1                    scalar ints             <- xrdmap
        /configuration/data           WorkerConfig JSON       <- both viewers
        /configuration/integrate_config  {"poni": path}       <- xrdmap (optional)
        /result/                      NXdata
          intensity    (dim0,dim1,n_rad) float32  interpretation="spectrum"
          sum_intensity(dim0,dim1,n_rad) int64
          errors       (opt)
          intensity_flat(dim0*dim1,n_rad) float32   (VDS of the cube) <- xrdmap
          2th          (n_rad,) float    unit="deg", interpretation="scalar"
          slow, fast   (dim0,),(dim1,)   long_name
          map_ptr      (dim0,dim1) int32

Deriving the map grid from a Bliss master is handled by
:func:`bliss_geometry`; a ``Mesh``/per-frame placement is described by a
``Geometry`` dict:
``map_shape`` (slow, fast), ``row``/``col`` (per frame), ``frame_index``,
``exposure``, ``wtmax`` = exposure*1e7, ``radial_label``/``radial_unit``,
``entry``, ``title``, ``raw_ds`` (absolute path for the external link),
``beam_ds`` (absolute path), ``dtype``/``frame_shape``, ``source`` (where the
chunks of each frame live, see _chunk_sources).
"""

import datetime
import json
import os
import re

import numpy as np

_MESH_TITLE = re.compile(
    r"^\s*(?P<kind>[ad])mesh\s+"
    r"(?P<fast>\S+)\s+(?P<fast_start>\S+)\s+(?P<fast_stop>\S+)\s+(?P<fast_intervals>\d+)\s+"
    r"(?P<slow>\S+)\s+(?P<slow_start>\S+)\s+(?P<slow_stop>\S+)\s+(?P<slow_intervals>\d+)"
)


def _axis(vals, npos=None):
    """Grid positions of the measured motor values: (centres, index of each value).

    The sorted values are split at their npos - 1 largest gaps (every gap above half the largest
    one when npos is None), so jitter in the measured positions stays within one position.  Refused
    unless each kept gap is more than twice every gap inside a position."""
    vals = np.asarray(vals, dtype=float)
    order = np.argsort(vals, kind="stable")
    s = vals[order]
    gaps = np.diff(s)
    if npos is None:
        split = gaps > 0.5 * gaps.max() if gaps.size and gaps.max() > 0 else np.zeros(gaps.size, bool)
    else:
        if not 1 <= npos <= len(vals):
            raise ValueError("%d positions for %d values" % (npos, len(vals)))
        split = np.zeros(gaps.size, bool)
        split[np.argsort(gaps, kind="stable")[gaps.size - (npos - 1):]] = True
    if split.any() and (~split).any() and gaps[split].min() <= 2 * gaps[~split].max():
        raise ValueError("the measured positions do not fall on %d grid lines (a gap between "
                         "lines %.3g, within a line %.3g)" % (int(split.sum()) + 1,
                                                              gaps[split].min(), gaps[~split].max()))
    group = np.concatenate([[0], np.cumsum(split)])
    idx = np.empty(len(vals), np.int64)
    idx[order] = group
    centres = np.bincount(group, weights=s) / np.bincount(group)
    return centres, idx


def _fscan_parameters(par, fast, slow):
    """(fast, slow, (slow positions, fast positions)) from an fscan's instrument/fscan_parameters:
    the numbers of points (as ImageD11's sinograms.dataset reads them, None where absent) and,
    unless given, the motor names."""
    def get(key):
        if key not in par:
            return None
        v = par[key][()]
        return v.decode() if isinstance(v, bytes) else v

    npts = tuple(None if get(k) is None else int(get(k)) for k in ("slow_npoints", "fast_npoints"))
    return fast or get("fast_motor"), slow or get("slow_motor"), npts


def _motor_values(e, motor):
    """The per-frame positions of motor in the scan entry e."""
    for path in ("instrument/%s/value" % motor, "measurement/%s" % motor):
        if path in e:
            return np.asarray(e[path][:], dtype=float)
    raise ValueError("%s: no positions of %r (instrument/%s/value or measurement/%s)"
                     % (e.name, motor, motor, motor))


def _chunk_sources(imgs, master, n):
    """Where the compressed chunks of frames 0..n-1 live: a list of (file, dataset, first frame,
    first frame in the file, number of frames), one per file of a VDS, else the dataset itself."""
    if not imgs.is_virtual:
        return [(os.path.abspath(master), imgs.name, 0, 0, n)]
    import h5py

    out = []
    for vs in imgs.virtual_sources():
        (v0,), (v1,) = [b[:1] for b in vs.vspace.get_select_bounds()]
        s0 = 0  # the whole source dataset: HDF5 hands back its space with no extent
        if vs.src_space.get_select_type() != h5py.h5s.SEL_ALL:
            (s0,), (s1,) = [b[:1] for b in vs.src_space.get_select_bounds()]
            if v1 - v0 != s1 - s0:
                raise ValueError("%s: a VDS mapping that is not one frame per frame" % vs.file_name)
        fname = vs.file_name
        if fname == ".":
            fname = master
        elif not os.path.isabs(fname):
            fname = os.path.join(os.path.dirname(os.path.abspath(master)), fname)
        out.append((fname, vs.dset_name, int(v0), int(s0), int(v1 - v0 + 1)))
    return sorted(out, key=lambda x: x[2])


def bliss_geometry(master, entry, frame_shape=None, fast=None, slow=None, map_shape=None):
    """Read the map geometry of a Bliss scan from ``master::<entry>``.

    Returns a dict (see the module docstring).  ``frame_shape``, if given,
    overrides the detected detector frame shape.

    ``fast``/``slow`` name the two scanning motors; they are taken from an
    ``amesh``/``dmesh`` title when available, otherwise they must be given (this
    is how ``fscan``/``fscan2d`` fly-scan entries are handled, unless their
    ``instrument/fscan_parameters`` name them).  ``map_shape`` (rows, cols) is the
    number of positions of the slow and fast motor; an ``amesh``/``dmesh`` title
    gives it (intervals + 1), as do the ``slow_npoints``/``fast_npoints`` of an
    fscan's ``instrument/fscan_parameters``; otherwise it is found from the gaps
    between the measured positions, and the frames must then fill the grid.

    The point -> (row, col) placement is computed from the *measured* per-frame
    motor arrays (``instrument/<motor>/value``): the values are grouped onto the
    grid lines (see _axis), which is safe for serpentine/fly-scan orderings and
    for jitter in the measured positions.  It must be 1:1 (one distinct cell
    per frame).
    """
    import h5py

    with h5py.File(master, "r") as f:
        e = f[entry]
        title = e["title"][()]
        if isinstance(title, bytes):
            title = title.decode()
        m = _MESH_TITLE.match(title)
        npos = (None, None)
        if m:
            fast = fast or m.group("fast")
            slow = slow or m.group("slow")
            if (fast, slow) == (m.group("fast"), m.group("slow")):
                npos = (int(m.group("slow_intervals")) + 1, int(m.group("fast_intervals")) + 1)
        elif "fscan" in title and "instrument/fscan_parameters" in e:
            fast, slow, npos = _fscan_parameters(e["instrument/fscan_parameters"], fast, slow)
        if not (fast and slow):
            raise ValueError(
                "%s: %r names no mesh motors; pass --fast/--slow so the two scanning motors are "
                "known" % (entry, title)
            )
        if map_shape is not None:
            map_shape = (int(map_shape[0]), int(map_shape[1]))
            if None not in npos and map_shape != npos:
                raise ValueError("%s: map_shape %r, but the scan %r gives %r"
                                 % (entry, map_shape, title, npos))
            npos = map_shape

        py = _motor_values(e, fast)
        pz = _motor_values(e, slow)
        n = py.size
        if pz.size != n:
            raise ValueError("%s: %d %s and %d %s positions" % (entry, n, fast, pz.size, slow))

        fast_u, col = _axis(py, npos[1])
        slow_u, row = _axis(pz, npos[0])
        rows, cols = len(slow_u), len(fast_u)
        if None in npos and rows * cols != n:
            raise ValueError("%s: %d frames on %d x %d grid lines, guessed from the measured "
                             "positions: give map_shape (--map-shape rows cols)" % (entry, n, rows, cols))
        if len(set(zip(row.tolist(), col.tolist()))) != n:
            raise ValueError("%s: %d frames do not map 1:1 onto a %dx%d grid" % (entry, n, rows, cols))

        exposure = float(e["instrument/eiger/acq_parameters/acq_expo_time"][()])
        frame_index = np.full((rows, cols), -1, dtype=np.int32)
        frame_index[row, col] = np.arange(n)

        imgs = e["instrument/eiger/image"]
        if imgs.shape[0] < n:
            raise ValueError("%s: %d motor positions but %d frames" % (entry, n, imgs.shape[0]))
        shape = frame_shape or imgs.shape[1:]
        raw_ds = "/%s/measurement/eiger" % entry
        beam_ds = "/%s/measurement/current" % entry if "measurement/current" in e else None

        return dict(
            master=master, entry=entry, title=title,
            map_shape=(rows, cols), row=row, col=col, frame_index=frame_index,
            n_points=n, exposure=exposure, wtmax=exposure * 1e7,
            slow_axis=slow, fast_axis=fast,
            slow_values=slow_u, fast_values=fast_u,
            raw_ds=raw_ds, beam_ds=beam_ds,
            dtype=imgs.dtype, frame_shape=tuple(shape),
            source=_chunk_sources(imgs, master, n),
        )


def worker_config_dict(ai, n_bins, shape, unit="2th_deg"):
    """A JSON-serializable pyFAI WorkerConfig dict from an AzimuthalIntegrator.

    Built directly: importing pyFAI.io (for WorkerConfig) drags in pyFAI's GUI model
    and silx.gui.qt/PySide6 (~1.3 s) which this tool never uses.  ai.get_config() is
    already a plain JSON-safe dict, so the WorkerConfig.as_dict() schema is assembled
    here instead.
    """
    cfg = dict(ai.get_config())
    return {
        "poni": cfg,
        "nbpt_rad": int(n_bins),
        "nbpt_azim": None,
        "unit": unit,
        "chi_discontinuity_at_0": False,
        "polarization_description": None,
        "normalization_factor": 1.0,
        "val_dummy": None,
        "delta_dummy": None,
        "correct_solid_angle": True,
        "dark_current": None,
        "flat_field": None,
        "mask_file": None,
        "error_model": "no",
        "method": None,
        "opencl_device": None,
        "azimuth_range": None,
        "radial_range": None,
        "integrator_class": "AzimuthalIntegrator",
        "integrator_method": None,
        "extra_options": None,
        "monitor_name": None,
        "shape": [int(v) for v in shape],
    }


def _cube_frame_vds(group, name, cube_ds, row, col, n_rad, dtype, fill=0.0):
    """``name`` = per-frame (file-order) rows of a (slow, fast, radial) cube.

    Frame ``i`` sits at ``(row[i], col[i])``; the dataset is a virtual
    permutation of ``cube_ds`` (no bytes are duplicated).
    """
    import h5py

    layout = h5py.VirtualLayout(shape=(len(row), n_rad), dtype=dtype)
    src = h5py.VirtualSource(cube_ds)
    i = 0
    while i < len(row):  # one mapping per run of frames along a row of the cube
        j = i + 1
        while j < len(row) and row[j] == row[i] and col[j] == col[j - 1] + 1:
            j += 1
        layout[i:j, :] = src[row[i], col[i]:col[i] + (j - i), :]
        i = j
    group.create_virtual_dataset(name, layout, fillvalue=fill)


def write_diffmap(
    path,
    geometry,
    worker_config,
    radial,
    intensity_cube,
    sum_cube,
    poni_path,
    frame_shape,
    errors=None,
    maskfile=None,
    clipped=None,
    unit="2th_deg",
    entry="entry_0000",
):
    """Write the diffmap H5 (module docstring layout) to ``path``.

    ``intensity_cube`` is (dim0, dim1, n_rad) float32; ``sum_cube`` the exact
    (dim0, dim1, n_rad) integer sums; ``worker_config`` a pyFAI WorkerConfig dict.
    ``clipped`` is an optional list of ``{frame, index, value}`` for saturated
    pixels (value > wtmax = exposure*1e7, clipped to wtmax), written under ``/pyFAI/clipped_pixels``.
    """
    import h5py

    rows, cols = geometry["map_shape"]
    n_rad = len(radial)
    dim = geometry["map_shape"]
    slow = geometry["slow_values"]
    fast = geometry["fast_values"]
    frame_index = geometry["frame_index"]
    master = geometry["master"]
    raw_ds = geometry["raw_ds"]

    # String metadata is stored as UTF-8 str (not bytes): pyFAI/diffmap-refine compare
    # attrs like `interpretation`/`NX_class` against str.
    # pyFAI names the radial dataset by its "space" (e.g. "2th" for 2th_deg).
    space = unit.split("_")[0] if "_" in unit else unit
    unit_attr = unit[len(space) + 1:] if "_" in unit else ""

    with h5py.File(path, "w") as f:
        f.attrs["default"] = entry
        ent = f.create_group(entry)
        ent.attrs["NX_class"] = "NXentry"
        ent.create_dataset("program_name", data="pyfai")
        ent.create_dataset("title", data="diff_map")
        # pyFAI Nexus.get_entries() needs a `start_time` (and `end_time`) dataset.
        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        ent.create_dataset("start_time", data=now)
        ent.create_dataset("end_time", data=now)

        meas = ent.create_group("measurement")
        meas.attrs["NX_class"] = "NXdata"
        # external link to the user-selected master scan/dataset (usually a VDS)
        meas["images_0001"] = h5py.ExternalLink(master, raw_ds)

        proc = ent.create_group("pyFAI")
        proc.attrs["NX_class"] = "NXprocess"
        proc.create_dataset("program", data="xrdmap-diffmap")
        proc.create_dataset("version", data="0.1")
        proc.create_dataset("date", data=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        proc.create_dataset("PONIfile", data=poni_path)
        proc.create_dataset("offset", data=np.int32(0))
        if maskfile is not None:
            proc.create_dataset("maskfile", data=maskfile)
        proc.create_dataset("dim0", data=np.int32(rows))
        proc.create_dataset("dim1", data=np.int32(cols))

        if clipped:
            clip = proc.create_group("clipped_pixels")
            cl_grp = clip.create_group("all")
            # frame is repeated per saturated pixel so the three arrays stay parallel.
            cl_grp.create_dataset("frame", data=np.concatenate([
                np.full(len(c["index"]), c["frame"], dtype=np.int32)
                for c in clipped]))
            cl_grp.create_dataset("pixel", data=np.concatenate([np.asarray(c["index"], dtype=np.int64) for c in clipped]))
            cl_grp.create_dataset("value", data=np.concatenate([np.asarray(c["value"], dtype=np.uint64) for c in clipped]))
            clip.attrs["wtmax"] = float(geometry["wtmax"])

        cfg = proc.create_group("configuration")
        cfg.attrs["NX_class"] = "NXnote"
        cfg.create_dataset("data", data=json.dumps(worker_config).encode())
        cfg.create_dataset("integrate_config",
                           data=json.dumps({"poni": poni_path}).encode())

        res = proc.create_group("result")
        res.attrs["NX_class"] = "NXdata"
        # intensity is stored (slow, fast, radial) with interpretation="spectrum",
        # so the axis datasets are listed in that dimension order.
        res.attrs["axes"] = ["slow", "fast", space]
        res.attrs["signal"] = "intensity"

        intg = res.create_dataset("intensity", data=np.asarray(intensity_cube, np.float32))
        intg.attrs["interpretation"] = "spectrum"
        intg.attrs["long_name"] = "Intensity"
        res.create_dataset("sum_intensity", data=np.asarray(sum_cube, np.int64)).attrs["interpretation"] = "spectrum"
        if errors is not None:
            res.create_dataset("errors", data=errors.astype(np.float32)).attrs["interpretation"] = "spectrum"

        # xrdmap convenience: flat (dim0*dim1, n_rad) view of the cube (VDS)
        try:
            layout = h5py.VirtualLayout(shape=(rows * cols, n_rad), dtype=np.float32)
            src = h5py.VirtualSource(res["intensity"])
            for i in range(rows):
                layout[i * cols:(i + 1) * cols, :] = src[i, :, :]
            res.create_virtual_dataset("intensity_flat", layout, fillvalue=0.0)
        except Exception:
            res.create_dataset("intensity_flat",
                               data=intensity_cube.reshape(rows * cols, n_rad).astype(np.float32))

        # Per-frame (acquisition / file order) data, so frame number -> motors + powder.
        row = np.asarray(geometry["row"], dtype=np.int64)
        col = np.asarray(geometry["col"], dtype=np.int64)
        n_frames = len(row)
        res.create_dataset("frame", data=np.arange(n_frames, dtype=np.int64))
        res.create_dataset("slow_value", data=np.asarray(slow, dtype=float)[row])
        res.create_dataset("fast_value", data=np.asarray(fast, dtype=float)[col])
        _cube_frame_vds(res, "intensity_frame", res["intensity"], row, col, n_rad,
                        np.float32, 0.0)
        _cube_frame_vds(res, "sum_intensity_frame", res["sum_intensity"], row, col, n_rad,
                        np.int64, 0)

        tth = res.create_dataset(space, data=np.asarray(radial, dtype=float))
        tth.attrs["unit"] = unit_attr or "deg"
        tth.attrs["long_name"] = {"2th": "2theta"}.get(space, space)
        tth.attrs["interpretation"] = "scalar"
        tth.attrs["axes"] = np.int32(0)

        for name, values, label in (("slow", slow, geometry["slow_axis"]), ("fast", fast, geometry["fast_axis"])):
            ds = res.create_dataset(name, data=np.asarray(values, dtype=float))
            ds.attrs["long_name"] = label
            ds.attrs["interpretation"] = "scalar"
            ds.attrs["axes"] = np.int32(0 if name == "slow" else 1)

        res.create_dataset("map_ptr", data=frame_index.astype(np.int32))
        res["map_ptr"].attrs["interpretation"] = "image"
