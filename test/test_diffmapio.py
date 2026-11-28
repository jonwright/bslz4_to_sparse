"""The diffmap writer output is openable as pyFAI-diffmap layout.

A small, synthetic setup (no external data): build an Integrator, fill a tiny
map cube, and write it; then read it back the way xrdmap / diffmap-refine /
pyFAI-diffmap-view do, checking the axis/interpretation conventions.
"""

from __future__ import absolute_import, division, print_function

from conftest import bitshuffle
import json
import os
import sys

import numpy as np
import pytest

pyFAI = pytest.importorskip("pyFAI")
h5py = pytest.importorskip("h5py")

from pyFAI.detectors import Detector  # noqa: E402
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # pyFAI >= 2024.10
except ImportError:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # older pyFAI

import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import conveyor as bslz4_cv  # noqa: E402
from bslz4_to_sparse.conveyor import _diffmapio  # noqa: E402

SHAPE = (96, 112)
ROWS, COLS = 3, 4
N_BINS = 32


def geometry():
    """A fake Bliss mesh geometry for a 3x4 map, with a serpentine fast axis.

    Frame order (file order) is the acquisition order; the fast motor reverses
    on the odd row, so file order != grid row-major order. row/col are the map
    cell of each frame; frame_index is the file-order frame id per map cell.
    """
    row = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    col = np.asarray([0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3], dtype=np.int64)
    frame_index = np.asarray([[0, 1, 2, 3], [7, 6, 5, 4], [8, 9, 10, 11]], dtype=np.int32)
    return dict(
        master="phony-master.h5",
        raw_ds="/1.1/instrument/eiger/image",
        entry="1.1",
        title="amesh py 0 3 3 pz 0 2 2 0.5",
        map_shape=(ROWS, COLS),
        row=row,
        col=col,
        frame_index=frame_index,
        slow_values=np.asarray([23.0, 25.0, 27.0]),
        fast_values=np.asarray([15.0, 17.0, 19.0, 21.0]),
        slow_axis="pz",
        fast_axis="py",
        exposure=0.5,
        wtmax=5e6,
    )


def test_write_diffmap_layout(tmpdir):
    ai = AzimuthalIntegrator(
        dist=0.1, poni1=0.004, poni2=0.0045, wavelength=3e-11,
        detector=Detector(75e-6, 75e-6, max_shape=SHAPE),
    )
    integ = bslz4_cv.Integrator.from_pyfai(ai, N_BINS, unit="2th_deg", split="no")
    n = ROWS * COLS
    rng = np.random.RandomState(0)
    int_cube = rng.uniform(0, 100, (ROWS, COLS, N_BINS)).astype(np.float64)
    sum_cube = rng.randint(0, 10000, (ROWS, COLS, N_BINS)).astype(np.int64)
    geom = geometry()
    wc = _diffmapio.worker_config_dict(ai, N_BINS, SHAPE, unit="2th_deg")
    poni = os.path.join(str(tmpdir), "geom.poni")
    open(poni, "w").write(integ.poni)

    out = os.path.join(str(tmpdir), "map.h5")
    _diffmapio.write_diffmap(
        out, geometry=geom, worker_config=wc, radial=integ.radial,
        intensity_cube=int_cube, sum_cube=sum_cube, poni_path=poni,
        frame_shape=SHAPE, unit="2th_deg",
    )

    with h5py.File(out, "r") as f:
        proc = f["/entry_0000/pyFAI"]
        nd = proc["result"]
        dec = lambda v: v.decode() if isinstance(v, bytes) else v  # noqa: E731
        assert dec(f["/entry_0000/program_name"][()]) == "pyfai"
        assert dec(f["/entry_0000/title"][()]) == "diff_map"
        # pyFAI Nexus.get_entries() requires these (the pyFAI-diffmap-view auto-detect)
        assert "start_time" in f["/entry_0000"] and "end_time" in f["/entry_0000"]
        assert int(proc["dim0"][()]) == ROWS and int(proc["dim1"][()]) == COLS

        sig = nd["intensity"]
        assert sig.shape == (ROWS, COLS, N_BINS)
        assert sig.attrs["interpretation"] == "spectrum"
        assert [dec(x) for x in nd.attrs["axes"]] == ["slow", "fast", "2th"]

        # flat view is a virtual dataset of the cube (no duplicate bytes)
        flat = nd["intensity_flat"]
        assert flat.is_virtual and flat.shape == (n, N_BINS)
        assert np.allclose(flat[()], int_cube.reshape(n, N_BINS), atol=1e-3)

        assert np.array_equal(nd["map_ptr"][()], geom["frame_index"])
        assert nd["slow"].attrs["long_name"] == "pz"
        assert nd["fast"].attrs["long_name"] == "py"
        assert nd["2th"].attrs["unit"] == "deg"

        # per-frame (file-order) block: frame number -> motors + powder.
        # For the serpentine placement, file order is NOT grid row-major order.
        row, col = geom["row"], geom["col"]
        assert np.array_equal(nd["frame"][()], np.arange(n))
        assert np.allclose(nd["slow_value"][()], geom["slow_values"][row])
        assert np.allclose(nd["fast_value"][()], geom["fast_values"][col])
        assert np.allclose(nd["intensity_frame"][()], int_cube[row, col, :], atol=1e-3)
        assert np.array_equal(nd["sum_intensity_frame"][()], sum_cube[row, col, :])
        # frame number -> map cell (file order; serpentine row-1 col-3 is frame 4)
        assert int(nd["map_ptr"][()][1, 3]) == 4 and int(nd["map_ptr"][()][1, 0]) == 7

        # the two viewers' config: a WorkerConfig with nbpt_rad + poni (an older pyFAI, as on
        # Python 2, has neither the viewers nor WorkerConfig)
        try:
            from pyFAI.io.integration_config import WorkerConfig
        except ImportError:
            WorkerConfig = None
        if WorkerConfig is not None:
            wc_back = WorkerConfig.from_dict(json.loads(proc["configuration/data"][()]), inplace=True)
            assert wc_back.nbpt_rad == N_BINS
            assert list(wc_back.shape) == list(SHAPE)
            assert wc_back.poni is not None

        # raw frames exposed as an external link into the (master) VDS
        link = f.get("/entry_0000/measurement/images_0001", getlink=True)
        assert isinstance(link, h5py.ExternalLink)
        assert link.filename == "phony-master.h5"
        assert link.path == "/1.1/instrument/eiger/image"


def test_axis_groups_jittered_positions():
    """Measured positions with jitter fall on their grid lines; a scatter that is no grid is refused."""
    vals = np.array([0.0, 1.002, 2.0, 2.999, 3.0, 1.998, 0.997, 0.003])
    centres, idx = _diffmapio._axis(vals)
    assert len(centres) == 4 and list(idx) == [0, 1, 2, 3, 3, 2, 1, 0]
    assert list(_diffmapio._axis(vals, 4)[1]) == list(idx)
    with pytest.raises(ValueError):
        _diffmapio._axis(np.array([0.0, 0.4, 1.0, 1.5, 2.1]), 3)


EXPO = 1e-5  # wtmax = EXPO * 1e7 = 100 counts
FSHAPE = (16, 20)


def bliss_scan(tmpdir):
    """A Bliss-like master: 'amesh py 0 3 3 pz 0 2 2', serpentine, measured positions with jitter,
    its frames in two files behind a VDS, one pixel above wtmax."""
    hdf5plugin = pytest.importorskip("hdf5plugin")
    rng = np.random.RandomState(5)
    row = np.repeat(np.arange(ROWS), COLS)
    col = np.array([c if r % 2 == 0 else COLS - 1 - c for r in range(ROWS) for c in range(COLS)])
    n = len(row)
    frames = rng.poisson(3, (n,) + FSHAPE).astype(np.uint32)
    frames[5, 7, 9] = 1000
    os.makedirs(os.path.join(str(tmpdir), "scan0001"))
    parts = [(0, 7), (7, n)]
    for k, (a, z) in enumerate(parts):
        with h5py.File(os.path.join(str(tmpdir), "scan0001", "eiger_%04d.h5" % k), "w") as h:
            h.create_dataset("/entry_0000/data", data=frames[a:z], chunks=(1,) + FSHAPE,
                             **bitshuffle(nelems=0, cname="lz4"))
    master = os.path.join(str(tmpdir), "master.h5")
    with h5py.File(master, "w") as h:
        e = h.create_group("1.1")
        e["title"] = "amesh py 0 3 3 pz 0 2 2 0.1"
        e["instrument/py/value"] = col + rng.uniform(-0.003, 0.003, n)
        e["instrument/pz/value"] = row + rng.uniform(-0.003, 0.003, n)
        e["instrument/eiger/acq_parameters/acq_expo_time"] = EXPO
        layout = h5py.VirtualLayout((n,) + FSHAPE, np.uint32)
        for k, (a, z) in enumerate(parts):
            layout[a:z] = h5py.VirtualSource("scan0001/eiger_%04d.h5" % k, "/entry_0000/data",
                                             shape=(z - a,) + FSHAPE)
        e.create_virtual_dataset("instrument/eiger/image", layout)
    ai = AzimuthalIntegrator(dist=0.05, poni1=0.0006, poni2=0.0007, wavelength=3e-11,
                             detector=Detector(75e-6, 75e-6, max_shape=FSHAPE))
    poni = os.path.join(str(tmpdir), "geom.poni")
    ai.write(poni)
    return master, poni, frames, row, col


def run_cli(monkeypatch, *argv):
    from bslz4_to_sparse.conveyor import diffmap_integrate

    monkeypatch.setattr("sys.argv", ["bs_diffmap_integrate"] + [str(a) for a in argv])
    diffmap_integrate.main()


@pytest.mark.parametrize("mask_arg", ["auto", "file"])
def test_integrate_a_bliss_scan(tmpdir, monkeypatch, mask_arg):
    """Placement, exact clipped sums and the clipped-pixel list of a scan in two files; the
    recipe gives the same file again, and refuses other settings.  The mask: pyFAI's
    dynamic_mask of the first frame (it finds none here), or a file masking one pixel."""
    master, poni, frames, row, col = bliss_scan(tmpdir)
    if mask_arg == "file":
        fabio = pytest.importorskip("fabio")
        bad = np.zeros(FSHAPE, np.int8)
        bad[3, 4] = 1
        mask_arg = os.path.join(str(tmpdir), "mask.edf")
        fabio.edfimage.EdfImage(data=bad).write(mask_arg)
    elif not hasattr(Detector, "dynamic_mask"):
        with pytest.raises(SystemExit, match="dynamic_mask"):
            run_cli(monkeypatch, master, os.path.join(str(tmpdir), "x.h5"), "--entry", "1.1",
                    "--poni", poni, "--mask", "auto")
        return
    out = os.path.join(str(tmpdir), "out.h5")
    run_cli(monkeypatch, master, out, "--entry", "1.1", "--poni", poni, "--n-bins", N_BINS,
            "--split", "bbox", "--mask", mask_arg, "--threads", 2, "--no-pin")
    plan = os.path.join(str(tmpdir), "out_plan.h5")
    integ = bslz4_cv.Integrator.load(plan)
    with h5py.File(plan, "r") as h:
        mask = h["mask"][()]
    assert mask.sum() == mask.size - (mask_arg != "auto")  # the decoder's: 1 = use the pixel
    W = b.fixed_point_matrix(integ.csc, mask, np.uint32).to_scipy()
    wtmax = int(EXPO * 1e7)
    want = np.stack([W.dot(np.minimum(f, wtmax).ravel().astype(np.int64)) for f in frames])
    with h5py.File(out, "r") as f:
        res = f["/entry_0000/pyFAI/result"]
        assert np.array_equal(res["map_ptr"][()][row, col], np.arange(len(row)))
        assert np.array_equal(res["sum_intensity"][()][row, col], want)
        assert np.array_equal(res["sum_intensity_frame"][()], want)
        cl = f["/entry_0000/pyFAI/clipped_pixels/all"]
        assert list(cl["frame"][()]) == [5] and list(cl["value"][()]) == [1000]
        assert list(cl["pixel"][()]) == [7 * FSHAPE[1] + 9]
        first = {k: res[k][()] for k in ("intensity", "sum_intensity")}

    again = os.path.join(str(tmpdir), "again.h5")
    run_cli(monkeypatch, again, "--plan", plan, "--entry", "1.1", "--threads", 3)
    with h5py.File(again, "r") as f:
        for k, v in first.items():
            assert np.array_equal(f["/entry_0000/pyFAI/result"][k][()], v)

    with pytest.raises(SystemExit, match="n-bins"):
        run_cli(monkeypatch, again, "--plan", plan, "--n-bins", N_BINS + 1)


def test_the_chunk_routes_agree(tmpdir, monkeypatch):
    """diffmap_integrate finds each frame's chunk from the sidecars when present, else from HDF5's
    chunk index, else (h5py 2.10) reads it through HDF5: the same bytes every way."""
    from conftest import chunk_info_available
    from bslz4_to_sparse.conveyor import _diffmapio, diffmap_integrate as di

    master, poni, frames, row, col = bliss_scan(tmpdir)
    sources = _diffmapio.bliss_geometry(master, "1.1")["source"]
    assert len(sources) == 2
    want = []
    for f, ds, v0, s0, n in sources:
        with h5py.File(f, "r") as h:
            want += [bytes(b.read_direct_chunk(h[ds], (i, 0, 0))[1]) for i in range(n)]

    def chunks():
        chunk, close = di.open_frames(sources, len(row))
        try:
            return [bytes(chunk(i)) for i in range(len(row))]
        finally:
            close()

    if chunk_info_available():
        assert chunks() == want  # HDF5's chunk index
        for f, ds, v0, s0, n in sources:
            bslz4_cv.write_index(bslz4_cv.harvest(f, ds), f)

        def refuse(ds):
            raise AssertionError("the sidecar is there")
        monkeypatch.setattr(b, "locate_chunks", refuse)
        assert chunks() == want  # the sidecars

        def missing(ds):
            raise NotImplementedError("no get_chunk_info")
        for f, ds, v0, s0, n in sources:
            os.remove(bslz4_cv.sidecar_path(f))
        monkeypatch.setattr(b, "locate_chunks", missing)
    assert chunks() == want  # read through HDF5


@pytest.mark.parametrize("damage", ["truncated", "no nframes"])
def test_a_damaged_sidecar_falls_back_to_hdf5(tmpdir, damage):
    """A sidecar that cannot be read is passed over, whatever read_index raises for it."""
    import json
    import struct

    from bslz4_to_sparse.conveyor import _diffmapio, _index, diffmap_integrate as di

    master, poni, frames, row, col = bliss_scan(tmpdir)
    sources = _diffmapio.bliss_geometry(master, "1.1")["source"]
    for f, ds, v0, s0, n in sources:
        with open(bslz4_cv.sidecar_path(f), "wb") as out:
            if damage == "truncated":
                out.write(_index.MAGIC + b"\0\0")
            else:
                head = json.dumps({"version": _index.VERSION, "dataset": ds}).encode("utf-8")
                out.write(_index.MAGIC + struct.pack("<I", len(head)) + head)
    chunk, close = di.open_frames(sources, len(row))
    try:
        got = [bytes(chunk(i)) for i in range(len(row))]
    finally:
        close()
    want = []
    for f, ds, v0, s0, n in sources:
        with h5py.File(f, "r") as h:
            want += [bytes(b.read_direct_chunk(h[ds], (i, 0, 0))[1]) for i in range(n)]
    assert got == want


def test_frames_never_written_are_refused_before_the_run(tmpdir, monkeypatch):
    """Read through HDF5 (no sidecar, no chunk index): an acquisition that stopped early (HDF5
    declares 6 frames, 4 were written) is refused when the frames are opened, not in a worker."""
    from bslz4_to_sparse.conveyor import diffmap_integrate as di

    def missing(ds):
        raise NotImplementedError("no get_chunk_info")
    monkeypatch.setattr(b, "locate_chunks", missing)
    f = os.path.join(str(tmpdir), "partial.h5")
    frames = np.random.RandomState(3).poisson(3, (4,) + FSHAPE).astype(np.uint32)
    with h5py.File(f, "w") as h:
        ds = h.create_dataset("data", shape=(6,) + FSHAPE, dtype=np.uint32, chunks=(1,) + FSHAPE,
                              **bitshuffle(nelems=0, cname="lz4"))
        ds[:4] = frames
    chunk, close = di.open_frames([(f, "data", 0, 0, 4)], 4)  # the written frames: fine
    try:
        assert b.decompress([chunk(3)], np.uint32).reshape(FSHAPE).tolist() == frames[3].tolist()
    finally:
        close()
    with pytest.raises(ValueError, match="no chunk for frame 5"):
        di.open_frames([(f, "data", 0, 0, 6)], 6)


def test_timestamps_are_iso_8601_whatever_strftime_z_gives(monkeypatch):
    """%z is the zone's name on Windows and empty on some C libraries: not used."""
    import re
    import time

    from bslz4_to_sparse.conveyor import _diffmapio

    strftime = time.strftime
    monkeypatch.setattr(time, "strftime",
                        lambda fmt, *t: strftime(fmt.replace("%z", "W. Europe Standard Time"), *t))
    now = _diffmapio._now()
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$", now), now
    if hasattr(time.localtime(), "tm_gmtoff"):  # the offset is the local one
        off = time.localtime().tm_gmtoff
        assert now.endswith("%s%02d:%02d" % ("+" if off >= 0 else "-", abs(off) // 3600,
                                            abs(off) % 3600 // 60)), now


def fscan_master(path, row, col, params=True, offset=0.0, mode=None, spread=0.003):
    """An fscan2d entry: positions under measurement/, the reverse lines offset by offset."""
    rng = np.random.RandomState(6)
    n = len(row)
    back = np.array([r % 2 == 1 for r in row])
    with h5py.File(path, "w") as h:
        e = h.create_group("1.1")
        e["title"] = "fscan2d pz 0 1 %d py 0 1 %d 0.1" % (row.max() + 1, col.max() + 1)
        e["measurement/py"] = col + offset * back + rng.uniform(-spread, spread, n)
        e["measurement/pz"] = row + rng.uniform(-0.003, 0.003, n)
        e["instrument/eiger/acq_parameters/acq_expo_time"] = EXPO
        e.create_dataset("instrument/eiger/image", shape=(n,) + FSHAPE, dtype=np.uint32)
        if params:
            p = e.create_group("instrument/fscan_parameters")
            p["slow_motor"], p["fast_motor"] = "pz", "py"
            p["slow_npoints"], p["fast_npoints"] = row.max() + 1, col.max() + 1
            if mode is not None:
                p["fast_motor_mode"] = mode


def test_fscan2d_grid_from_fscan_parameters(tmpdir):
    """fscan2d: motors and numbers of points from instrument/fscan_parameters, zigzag offset
    lines on their grid line; a guessed grid with holes is refused unless map_shape is given."""
    row = np.repeat(np.arange(ROWS), COLS)
    col = np.array([c if r % 2 == 0 else COLS - 1 - c for r in range(ROWS) for c in range(COLS)])
    master = os.path.join(str(tmpdir), "fscan.h5")
    fscan_master(master, row, col, offset=0.3)
    g = _diffmapio.bliss_geometry(master, "1.1")
    assert g["map_shape"] == (ROWS, COLS) and (g["fast_axis"], g["slow_axis"]) == ("py", "pz")
    assert np.array_equal(g["row"], row) and np.array_equal(g["col"], col)

    # no parameters: the offset lines look like a grid twice as wide, half empty
    fscan_master(master, row, col, params=False, offset=0.5)
    with pytest.raises(ValueError, match="map_shape"):
        _diffmapio.bliss_geometry(master, "1.1", fast="py", slow="pz")
    with pytest.raises(ValueError, match="grid lines"):
        _diffmapio.bliss_geometry(master, "1.1", fast="py", slow="pz", map_shape=(ROWS, COLS))


def test_fscan2d_zigzag_by_frame_order(tmpdir):
    """A ZIGZAG fscan2d whose measured positions spread by close to a step (as id11's WAu_g1 1.1,
    step 0.5, spread 0.46) is placed by frame order; a frame order that is no zigzag is refused."""
    row = np.repeat(np.arange(ROWS), COLS)
    col = np.array([c if r % 2 == 0 else COLS - 1 - c for r in range(ROWS) for c in range(COLS)])
    master = os.path.join(str(tmpdir), "fscan.h5")
    fscan_master(master, row, col, offset=-0.3, mode="ZIGZAG", spread=0.2)
    with pytest.raises(ValueError, match="grid lines"):  # the measured positions are no grid
        _diffmapio._axis(h5py.File(master, "r")["1.1/measurement/py"][()], COLS)
    g = _diffmapio.bliss_geometry(master, "1.1")
    assert np.array_equal(g["row"], row) and np.array_equal(g["col"], col)
    assert np.allclose(g["fast_values"], np.arange(COLS) - 0.3 * (ROWS // 2) / ROWS, atol=0.2)

    fscan_master(master, row, np.tile(np.arange(COLS), ROWS), mode="ZIGZAG")
    with pytest.raises(ValueError, match="zigzag"):
        _diffmapio.bliss_geometry(master, "1.1")
