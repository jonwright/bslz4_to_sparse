"""RingSink: per input file, intensity[frame, window, azimuth] with a background window either side
of each ring, equal to pyFAI's cake summed over the windows."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
pytest.importorskip("pyFAI")

import test_integrator as ti  # noqa: E402
from test_conveyor import DSET  # noqa: E402

from bslz4_to_sparse import conveyor as cv  # noqa: E402


def make_files(tmp_path, nfiles, nframes):
    import hdf5plugin

    rng = np.random.default_rng(12)
    paths = []
    for k in range(nfiles):
        frames = (rng.random((nframes,) + ti.SHAPE) < 0.2) * rng.integers(
            0, 300, (nframes,) + ti.SHAPE
        )
        frames = frames.astype(np.uint16)
        p = str(tmp_path / ("r%d.h5" % k))
        with h5py.File(p, "w") as h:
            h.create_dataset(
                DSET,
                data=frames,
                chunks=(1,) + ti.SHAPE,
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
            )
        cv.write_index(cv.harvest(p, DSET), p)
        paths.append(p)
    return paths


def test_peak_windows_order_and_edges():
    r, ring, role = cv.peak_windows([1.0, 2.0], 0.2, background_width=0.1, gap=0.05)
    assert r.shape == (6, 2) and list(ring) == [0, 0, 0, 1, 1, 1] and list(role) == [-1, 0, 1] * 2
    assert np.allclose(r[:3], [(0.75, 0.85), (0.9, 1.1), (1.15, 1.25)])


def test_ring_files_follow_the_input_files(tmp_path):
    ai = ti.masked_ai()
    nrad, naz = 70, 12
    kw = dict(polarization_factor=0.95, radial_range=(0.3, 3.0))
    ranges, ring, role = cv.peak_windows([0.9, 1.4], 0.12, 0.1)
    integ = cv.Integrator.from_pyfai(ai, nrad, unit="2th_deg", npt_azim=naz, **kw).rings(ranges)
    paths = make_files(tmp_path, nfiles=2, nframes=9)
    # the frames are the files' own; the ring files must follow their order
    units, shape, dtype = cv.make_units(paths, DSET)
    mask = (1 - ai.detector.mask).astype(np.uint8)
    out = str(tmp_path / "rings")
    cv.run(
        units,
        cv.CpuBackend(mask, dtype, csc=integ.csc, cut=200, howmany=500),
        workers=2,
        sink=lambda k: cv.TeeSink(
            cv.RingSink(integ, out, DSET, ring, role, integrator_path="i.h5"),
            cv.ImageD11SparseSink(out, DSET),
        ),
        policy=cv.Policy(("sparse", "rings")),
    )
    for p in paths:
        with h5py.File(p, "r") as src:
            frames = src[DSET][:]
        with h5py.File("%s/%s" % (out, cv.rings_name(p)), "r") as h:
            g = h[DSET.strip("/")]
            inten = g["intensity"][:]
            assert inten.shape == (9, 6, naz) and inten.dtype == np.float32
            assert (
                list(g["role"][:]) == [-1, 0, 1] * 2
                and list(g["ring_index"][:]) == [0] * 3 + [1] * 3
            )
            assert g.attrs["integrator"] == "i.h5" and g.attrs["unit"] == "2th_deg"
            assert np.all(g["window_nbins"][:] > 0)
            assert np.allclose(g["window_ranges"][:], ranges)
        for i in (0, 4, 8):
            ref = ti.textom_process_one_frame(
                ai, frames[i].astype(np.uint16), nrad, naz, ranges, np.zeros(nrad), **kw
            ).T  # (window, azimuth)
            assert np.allclose(
                inten[i], ref, rtol=2e-7, atol=0
            )  # float32 storage + pyFAI float32 cake


def test_dense_frames_lose_their_pixels_but_not_their_rings(tmp_path):
    """cut 0 on frames that are all signal: the pixels are not sparse and are not written, the
    job goes on and the ring file is complete."""
    ai = ti.masked_ai()
    ranges, ring, role = cv.peak_windows([0.9], 0.12, 0.1)
    integ = cv.Integrator.from_pyfai(
        ai, 70, unit="2th_deg", npt_azim=12, radial_range=(0.3, 3.0)
    ).rings(ranges)
    rng = np.random.default_rng(3)
    import hdf5plugin

    p = str(tmp_path / "dense.h5")
    with h5py.File(p, "w") as h:
        h.create_dataset(
            DSET,
            data=rng.integers(1, 300, (6,) + ti.SHAPE).astype(np.uint16),
            chunks=(1,) + ti.SHAPE,
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
        )
    cv.write_index(cv.harvest(p, DSET), p)
    units, shape, dtype = cv.make_units([p], DSET)
    out = str(tmp_path / "o")
    with pytest.warns(UserWarning, match="not sparse"):
        r = cv.run(
            units,
            cv.CpuBackend(np.ones(ti.SHAPE, np.uint8), dtype, csc=integ.csc),
            workers=1,
            sink=lambda k: cv.TeeSink(
                cv.RingSink(integ, out, DSET, ring, role), cv.ImageD11SparseSink(out, DSET)
            ),
            policy=cv.Policy(("sparse", "rings")),
        )
    assert r["frames"] == 6 and len(r["sparse_skipped"]) == 1
    with h5py.File("%s/%s" % (out, cv.rings_name(p)), "r") as h:
        assert h[DSET.strip("/")]["intensity"].shape == (6, 3, 12)
        assert np.all(h[DSET.strip("/")]["intensity"][:] > 0)
    with h5py.File("%s/%s" % (out, cv.imaged11_name(p)), "r") as h:
        g = h[DSET.strip("/")]
        assert g["nnz"][:].sum() == 0 and "not sparse" in g.attrs["sparse_skipped"]


def test_everything_in_one_pass_through_the_conveyor(tmp_path):
    """Sparse pixels, a 1D powder, ring windows and the first moment from one pass over the pixels
    (one stacked matrix): each file and dataset equals what that output gives on its own."""
    ai = ti.masked_ai()
    kw = dict(polarization_factor=0.95, radial_range=(0.3, 3.0))
    one = cv.Integrator.from_pyfai(ai, 60, unit="2th_deg", **kw)
    cake = cv.Integrator.from_pyfai(ai, 70, unit="2th_deg", npt_azim=12, **kw)
    ranges, ring, role = cv.peak_windows([0.9, 1.4], 0.12, 0.1)
    rings = cake.rings(ranges)
    stack = cv.IntegratorStack({"powder1d": one, "rings": rings, "moment": one.first_moment()})
    paths = make_files(tmp_path, nfiles=2, nframes=7)
    units, shape, dtype = cv.make_units(paths, DSET)
    mask = (1 - ai.detector.mask).astype(np.uint8)
    out = str(tmp_path / "all")
    r = cv.run(
        units,
        cv.CpuBackend(mask, dtype, csc=stack.csc, cut=200, howmany=2000),
        workers=2,
        sink=lambda k: cv.TeeSink(
            cv.ImageD11SparseSink(out, DSET, stack=stack, integrator="stack.h5"),
            cv.RingSink(rings, out, DSET, ring, role, rows=stack.slices["rings"]),
        ),
        policy=cv.Policy(("sparse", "powder1d", "rings")),
    )
    assert r["frames"] == 14 and r["sparse_skipped"] == []
    method = ti.IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
    for p in paths:
        with h5py.File(p, "r") as src:
            frames = src[DSET][:]
        with h5py.File("%s/%s" % (out, cv.imaged11_name(p)), "r") as h:
            g = h[DSET.strip("/")]
            assert set(g) >= {"row", "col", "intensity", "nnz"}
            assert g["powder_powder1d"].shape == (7, 60) and g["powder_moment"].shape == (7, 120)
            assert (
                g["powder_rings"].shape == (7, 72)
                and g["powder_powder1d"].attrs["integrator"] == "stack.h5"
            )
            assert list(g["nnz"][:]) == [int(((f > 200) & (mask > 0)).sum()) for f in frames]
            i = 3
            ref = ai.integrate1d(
                frames[i],
                60,
                unit="2th_deg",
                method=method,
                polarization_factor=0.95,
                radial_range=(0.3, 3.0),
            )
            assert np.allclose(
                one.intensity(g["powder_powder1d"][i]), ref.intensity, rtol=1.2e-7, atol=0
            )
        with h5py.File("%s/%s" % (out, cv.rings_name(p)), "r") as h:
            inten = h[DSET.strip("/")]["intensity"][:]
        refr = ti.textom_process_one_frame(ai, frames[3], 70, 12, ranges, np.zeros(70), **kw).T
        assert np.allclose(inten[3], refr, rtol=2e-7, atol=0)
