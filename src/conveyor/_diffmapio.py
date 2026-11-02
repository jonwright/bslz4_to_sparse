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
``beam_ds`` (absolute path), ``dtype``/``frame_shape``.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys

import numpy as np

_MESH_TITLE = re.compile(
    r"^\s*(?P<kind>[ad])mesh\s+"
    r"(?P<fast>\S+)\s+(?P<fast_start>\S+)\s+(?P<fast_stop>\S+)\s+(?P<fast_intervals>\d+)\s+"
    r"(?P<slow>\S+)\s+(?P<slow_start>\S+)\s+(?P<slow_stop>\S+)\s+(?P<slow_intervals>\d+)"
)


def bliss_geometry(master, entry, frame_shape=None, fast=None, slow=None, map_shape=None):
    """Read the map geometry of a Bliss scan from ``master::<entry>``.

    Returns a dict (see the module docstring).  ``frame_shape``, if given,
    overrides the detected detector frame shape.

    ``fast``/``slow`` name the two scanning motors; they are taken from an
    ``amesh``/``dmesh`` title when available, otherwise they must be given (this
    is how ``fscan``/``fscan2d`` fly-scan entries are handled).  ``map_shape``
    overrides the grid size; by default it is derived from the number of unique
    measured positions along each motor.

    The point -> (row, col) placement is computed from the *measured* per-frame
    motor arrays (``instrument/<motor>/value``): each value is ranked onto the
    sorted-unique positions, which is safe for serpentine/fly-scan orderings and
    does not depend on the scan title.  It must be 1:1 (one distinct cell per
    frame).
    """
    import h5py

    with h5py.File(master, "r") as f:
        e = f[entry]
        title = e["title"][()]
        if isinstance(title, bytes):
            title = title.decode()
        m = _MESH_TITLE.match(title)
        if m:
            fast = fast or m.group("fast")
            slow = slow or m.group("slow")
        if not (fast and slow):
            raise ValueError(
                f"{entry}: {title!r} names no mesh motors; pass --fast/--slow "
                "(or --fast/--slow for fscan2d) so the two scanning motors are known"
            )

        py = np.asarray(e[f"instrument/{fast}/value"][:], dtype=float)
        pz = np.asarray(e[f"instrument/{slow}/value"][:], dtype=float)
        n = py.size

        def axis(vals):
            u = np.unique(np.round(vals, 4))
            idx = np.searchsorted(u, np.round(vals, 4))
            return u, idx

        fast_u, col = axis(py)
        slow_u, row = axis(pz)
        if map_shape is not None:
            rows, cols = int(map_shape[0]), int(map_shape[1])
            slow_u = slow_u[::-1] if rows != len(slow_u) else slow_u
            fast_u = fast_u[::-1] if cols != len(fast_u) else fast_u
        else:
            rows, cols = len(slow_u), len(fast_u)
        if not ((row >= 0) & (row < rows) & (col >= 0) & (col < cols)).all():
            raise ValueError(f"{entry}: measured positions outside a {rows}x{cols} grid ({n} pts)")
        if len(set(zip(row.tolist(), col.tolist()))) != n:
            raise ValueError(f"{entry}: {n} frames do not map 1:1 onto a {rows}x{cols} grid")

        exposure = float(e["instrument/eiger/acq_parameters/acq_expo_time"][()])
        frame_index = np.full((rows, cols), -1, dtype=np.int32)
        frame_index[row, col] = np.arange(n)

        imgs = e["instrument/eiger/image"]
        shape = frame_shape or imgs.shape[1:]
        raw_ds = f"/{entry}/measurement/eiger"
        beam_ds = f"/{entry}/measurement/current" if "measurement/current" in e else None

        # Where the compressed chunks really live (a VDS -> the scanNNNN/eiger file).
        source = None
        try:
            vsrc = list(imgs.virtual_sources())
        except Exception:
            vsrc = []
        if vsrc:
            src = vsrc[0]
            src_file = src.file_name
            if not os.path.isabs(src_file):
                src_file = os.path.join(os.path.dirname(os.path.abspath(master)), src_file)
            source = (src_file, src.dset_name)

        return dict(
            master=master, entry=entry, title=title,
            map_shape=(rows, cols), row=row, col=col, frame_index=frame_index,
            n_points=n, exposure=exposure, wtmax=exposure * 1e7,
            slow_axis=slow, fast_axis=fast,
            slow_values=slow_u, fast_values=fast_u,
            raw_ds=raw_ds, beam_ds=beam_ds,
            dtype=imgs.dtype, frame_shape=tuple(shape),
            source=source,
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
    for i in range(len(row)):
        layout[i, :] = src[row[i], col[i], :]
    group.create_virtual_dataset(name, layout, fillvalue=fill)


def write_diffmap(
    path,
    *,
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
    pixels (value >= exposure*1e7), written under ``/pyFAI/clipped_pixels``.
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

    # Store string metadata as UTF-8 str (not bytes): pyFAI/diffmap-refine compare
    # attrs like `interpretation`/`NX_class` against str.
    ascii = lambda s: s  # noqa: E731
    # pyFAI names the radial dataset by its "space" (e.g. "2th" for 2th_deg).
    space = unit.split("_")[0] if "_" in unit else unit
    unit_attr = unit[len(space) + 1:] if "_" in unit else ""

    with h5py.File(path, "w") as f:
        f.attrs["default"] = ascii(entry)
        ent = f.create_group(entry)
        ent.attrs["NX_class"] = ascii("NXentry")
        ent.create_dataset("program_name", data=ascii("pyfai"))
        ent.create_dataset("title", data=ascii("diff_map"))
        # pyFAI Nexus.get_entries() needs a `start_time` (and `end_time`) dataset.
        now = ascii(datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        ent.create_dataset("start_time", data=now)
        ent.create_dataset("end_time", data=now)

        meas = ent.create_group("measurement")
        meas.attrs["NX_class"] = ascii("NXdata")
        # external link to the user-selected master scan/dataset (usually a VDS)
        meas["images_0001"] = h5py.ExternalLink(master, raw_ds)

        proc = ent.create_group("pyFAI")
        proc.attrs["NX_class"] = ascii("NXprocess")
        proc.create_dataset("program", data=ascii("xrdmap-diffmap"))
        proc.create_dataset("version", data=ascii("0.1"))
        proc.create_dataset("date", data=ascii(datetime.datetime.now().astimezone().isoformat(timespec="seconds")))
        proc.create_dataset("PONIfile", data=ascii(poni_path))
        proc.create_dataset("offset", data=np.int32(0))
        if maskfile is not None:
            proc.create_dataset("maskfile", data=ascii(maskfile))
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
        cfg.attrs["NX_class"] = ascii("NXnote")
        cfg.create_dataset("data", data=json.dumps(worker_config).encode())
        cfg.create_dataset("integrate_config",
                           data=json.dumps({"poni": poni_path}).encode())

        res = proc.create_group("result")
        res.attrs["NX_class"] = ascii("NXdata")
        # intensity is stored (slow, fast, radial) with interpretation="spectrum",
        # so the axis datasets are listed in that dimension order.
        res.attrs["axes"] = ["slow", "fast", space]
        res.attrs["signal"] = ascii("intensity")

        intg = res.create_dataset("intensity", data=intensity_cube.astype(np.float32))
        intg.attrs["interpretation"] = ascii("spectrum")
        intg.attrs["long_name"] = ascii("Intensity")
        res.create_dataset("sum_intensity", data=sum_cube.astype(np.int64)).attrs["interpretation"] = ascii("spectrum")
        if errors is not None:
            res.create_dataset("errors", data=errors.astype(np.float32)).attrs["interpretation"] = ascii("spectrum")

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
        tth.attrs["unit"] = ascii(unit_attr or "deg")
        tth.attrs["long_name"] = ascii("2theta")
        tth.attrs["interpretation"] = ascii("scalar")
        tth.attrs["axes"] = np.int32(0)

        for name, values, label in (("slow", slow, geometry["slow_axis"]), ("fast", fast, geometry["fast_axis"])):
            ds = res.create_dataset(name, data=np.asarray(values, dtype=float))
            ds.attrs["long_name"] = ascii(label)
            ds.attrs["interpretation"] = ascii("scalar")
            ds.attrs["axes"] = np.int32(0 if name == "slow" else 1)

        res.create_dataset("map_ptr", data=frame_index.astype(np.int32))
        res["map_ptr"].attrs["interpretation"] = ascii("image")
