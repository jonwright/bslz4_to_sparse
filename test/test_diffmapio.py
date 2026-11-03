"""The diffmap writer output is openable as pyFAI-diffmap layout.

A small, synthetic setup (no external data): build an Integrator, fill a tiny
map cube, and write it; then read it back the way xrdmap / diffmap-refine /
pyFAI-diffmap-view do, checking the axis/interpretation conventions.
"""

import json
import os

import numpy as np
import pytest

pyFAI = pytest.importorskip("pyFAI")
h5py = pytest.importorskip("h5py")

from pyFAI.detectors import Detector  # noqa: E402
from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: E402

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


def test_write_diffmap_layout(tmp_path):
    ai = AzimuthalIntegrator(
        dist=0.1, poni1=0.004, poni2=0.0045, wavelength=3e-11,
        detector=Detector(75e-6, 75e-6, max_shape=SHAPE),
    )
    integ = bslz4_cv.Integrator.from_pyfai(ai, N_BINS, unit="2th_deg", split="no")
    n = ROWS * COLS
    rng = np.random.default_rng(0)
    int_cube = rng.uniform(0, 100, (ROWS, COLS, N_BINS)).astype(np.float64)
    sum_cube = rng.integers(0, 10000, (ROWS, COLS, N_BINS)).astype(np.int64)
    geom = geometry()
    wc = _diffmapio.worker_config_dict(ai, N_BINS, SHAPE, unit="2th_deg")
    poni = tmp_path / "geom.poni"
    poni.write_text(integ.poni)

    out = tmp_path / "map.h5"
    _diffmapio.write_diffmap(
        str(out), geometry=geom, worker_config=wc, radial=integ.radial,
        intensity_cube=int_cube, sum_cube=sum_cube, poni_path=str(poni),
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

        # the two viewers' config: a WorkerConfig with nbpt_rad + poni
        from pyFAI.io.integration_config import WorkerConfig
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


def bliss_scan(tmp_path):
    """A Bliss-like master: 'amesh py 0 3 3 pz 0 2 2', serpentine, measured positions with jitter,
    its frames in two files behind a VDS, one pixel above wtmax."""
    hdf5plugin = pytest.importorskip("hdf5plugin")
    rng = np.random.default_rng(5)
    row = np.repeat(np.arange(ROWS), COLS)
    col = np.array([c if r % 2 == 0 else COLS - 1 - c for r in range(ROWS) for c in range(COLS)])
    n = len(row)
    frames = rng.poisson(3, (n,) + FSHAPE).astype(np.uint32)
    frames[5, 7, 9] = 1000
    os.makedirs(tmp_path / "scan0001")
    parts = [(0, 7), (7, n)]
    for k, (a, z) in enumerate(parts):
        with h5py.File(tmp_path / "scan0001" / ("eiger_%04d.h5" % k), "w") as h:
            h.create_dataset("/entry_0000/data", data=frames[a:z], chunks=(1,) + FSHAPE,
                             **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    master = tmp_path / "master.h5"
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
    poni = tmp_path / "geom.poni"
    ai.write(str(poni))
    return master, poni, frames, row, col


def run_cli(monkeypatch, *argv):
    from bslz4_to_sparse.conveyor import diffmap_integrate

    monkeypatch.setattr("sys.argv", ["bs_diffmap_integrate"] + [str(a) for a in argv])
    diffmap_integrate.main()


def test_integrate_a_bliss_scan(tmp_path, monkeypatch):
    """Placement, exact clipped sums and the clipped-pixel list of a scan in two files; the
    recipe gives the same file again, and refuses other settings."""
    master, poni, frames, row, col = bliss_scan(tmp_path)
    out = tmp_path / "out.h5"
    run_cli(monkeypatch, master, out, "--entry", "1.1", "--poni", poni, "--n-bins", N_BINS,
            "--split", "bbox", "--mask", "auto", "--threads", 2, "--no-pin")
    plan = tmp_path / "out_plan.h5"
    integ = bslz4_cv.Integrator.load(str(plan))
    with h5py.File(plan, "r") as h:
        mask = h["mask"][()]
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

    again = tmp_path / "again.h5"
    run_cli(monkeypatch, again, "--plan", plan, "--entry", "1.1", "--threads", 3)
    with h5py.File(again, "r") as f:
        for k, v in first.items():
            assert np.array_equal(f["/entry_0000/pyFAI/result"][k][()], v)

    with pytest.raises(SystemExit, match="n-bins"):
        run_cli(monkeypatch, again, "--plan", plan, "--n-bins", N_BINS + 1)
