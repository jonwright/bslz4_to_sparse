"""The conveyor: every frame of every unit reaches a sink exactly once and equals a direct decode,
with threads and with forked processes; a failing sink raises instead of hanging."""

import os
import sys
import threading

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import conveyor as cv  # noqa: E402

DSET = "/entry/data"
SHAPE = (64, 80)


def make_files(tmp_path, nfiles=3, nframes=37):
    rng = np.random.default_rng(3)
    paths = []
    for k in range(nfiles):
        frames = np.where(
            rng.random((nframes,) + SHAPE) < 0.05, rng.integers(1, 60, (nframes,) + SHAPE), 0
        )
        p = str(tmp_path / ("f%d.h5" % k))
        with h5py.File(p, "w") as h:
            h.create_dataset(
                DSET,
                data=frames.astype(np.uint16),
                chunks=(1,) + SHAPE,
                **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
            )
        paths.append(p)
        cv.write_index(cv.harvest(p, DSET), p)  # the sidecar: reading never needs HDF5 after this
    return paths


def direct(paths, mask):
    ref = {}
    c = b.chunk2sparse(mask, np.uint16, b.CODEC_LZ4)
    for fi, p in enumerate(paths):
        with h5py.File(p, "r") as h:
            ds = h[DSET]
            for i in range(ds.shape[0]):
                n, (v, ix) = c(ds.id.read_direct_chunk((i, 0, 0))[1], 0)
                ref[(fi, i)] = (
                    int(n),
                    int(v[:n].sum(dtype=np.int64)),
                    int(ix[:n].sum(dtype=np.int64)),
                )
    return ref


class Collect(cv.Sink):
    """Keeps (npixels, value sum, index sum) per frame, handed to a shared dict (threads only)."""

    lock = threading.Lock()
    out = {}

    def write(self, file, frame0, npx, values, indices, powder):
        with Collect.lock:
            for j, n in enumerate(npx):
                Collect.out[(file, frame0 + j)] = (
                    int(n),
                    int(values[j, :n].sum(dtype=np.int64)),
                    int(indices[j, :n].sum(dtype=np.int64)),
                )


@pytest.mark.parametrize("unit_bytes", [1 << 30, 3000])
def test_every_frame_once_and_equal_to_a_direct_decode(tmp_path, unit_bytes):
    paths = make_files(tmp_path)
    units, shape, dtype = cv.make_units(paths, DSET, max_bytes=unit_bytes)
    assert shape == SHAPE and dtype == np.uint16
    assert sum(u.n for u in units) == 3 * 37
    if unit_bytes < 1 << 20:
        assert len(units) > 3  # several units per file
    mask = np.ones(SHAPE, np.uint8)
    Collect.out = {}
    r = cv.run(
        units,
        cv.CpuBackend(mask, dtype),
        workers=3,
        sink=lambda k: Collect(),
        batch=8,
        io_bytes=1000,
    )  # reads in pieces too
    assert r["frames"] == 3 * 37 and r["mode"] == "threads"
    assert Collect.out == direct(paths, mask)


def test_processes_count_the_same_frames(tmp_path):
    paths = make_files(tmp_path)
    units, shape, dtype = cv.make_units(paths, DSET, max_bytes=4000)

    class Count(cv.Sink):
        def __init__(self):
            self.n = 0

        def write(self, file, frame0, npx, values, indices, powder):
            self.n += len(npx)

        def end(self):
            return dict(frames=self.n)

    r = cv.run(
        units,
        cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype),
        workers=2,
        sink=lambda k: Count(),
        mode="processes",
    )
    assert r["frames"] == 3 * 37 and r["mode"] == "processes"


def test_with_a_matrix_and_howmany(tmp_path):
    import scipy.sparse as sp

    paths = make_files(tmp_path, nfiles=1, nframes=20)
    units, shape, dtype = cv.make_units(paths, DSET)
    npix = SHAPE[0] * SHAPE[1]
    csc = sp.csc_matrix(
        (np.ones(npix, np.float32), (np.arange(npix) % 7, np.arange(npix))), shape=(7, npix)
    )
    seen = []

    class Check(cv.Sink):
        def write(self, file, frame0, npx, values, indices, powder):
            seen.append((powder.shape, int(npx.max())))

    cv.run(
        units,
        cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype, csc=csc, howmany=30),
        workers=1,
        sink=lambda k: Check(),
    )
    assert seen and all(s[0][1] == 7 and s[1] <= 30 for s in seen)


@pytest.mark.parametrize("mode", ["threads", "processes"])
def test_a_failing_sink_raises_and_does_not_hang(tmp_path, mode):
    paths = make_files(tmp_path)
    units, shape, dtype = cv.make_units(paths, DSET, max_bytes=4000)

    class Bad(cv.Sink):
        def write(self, file, frame0, npx, values, indices, powder):
            raise RuntimeError("sink broke")

    with pytest.raises(RuntimeError, match="sink broke"):
        cv.run(
            units,
            cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype),
            workers=2,
            sink=lambda k: Bad(),
            mode=mode,
        )


def test_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        cv.run([], cv.CpuBackend(np.ones(SHAPE, np.uint8), np.uint16), workers=1)
    paths = make_files(tmp_path, nfiles=1)
    units, shape, dtype = cv.make_units(paths, DSET)
    with pytest.raises(ValueError, match="mode"):
        cv.run(units, cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype), workers=1, mode="gpus")


def test_sidecar_equals_what_hdf5_reports(tmp_path):
    paths = make_files(tmp_path, nfiles=2, nframes=25)
    for p in paths:
        ix = cv.read_index(p)
        with h5py.File(p, "r") as h:
            ds = h[DSET]
            assert ix.shape == ds.shape and ix.dtype == ds.dtype and ix.codec == b.detect_codec(ds)
            for i in range(ds.shape[0]):
                info = ds.id.get_chunk_info(i)
                assert (ix.offsets[i], ix.lengths[i]) == (info.byte_offset, info.size)
        assert (
            os.path.getsize(cv.sidecar_path(p)) < 12 * 25 + 400
        )  # 12 bytes a frame and a small header


def test_sidecar_in_another_directory(tmp_path):
    paths = make_files(tmp_path, nfiles=1, nframes=5)
    ix = cv.harvest(paths[0], DSET)
    where = str(tmp_path / "idx")
    out = cv.write_index(ix, paths[0], where)
    assert out == cv.sidecar_path(paths[0], where) and os.path.dirname(out) == where
    assert np.array_equal(cv.read_index(paths[0], where).offsets, ix.offsets)


def test_a_changed_file_makes_the_sidecar_stale(tmp_path):
    paths = make_files(tmp_path, nfiles=1, nframes=5)
    with open(paths[0], "ab") as f:
        f.write(b"x")
    with pytest.raises(cv.StaleIndex):
        cv.read_index(paths[0])
    assert cv.read_index(paths[0], check=False).shape[0] == 5  # the caller may choose to trust it
    with pytest.raises(cv.StaleIndex):
        cv.make_units(paths, DSET)


def test_corrupt_and_missing_sidecars(tmp_path):
    paths = make_files(tmp_path, nfiles=1, nframes=5)
    sc = cv.sidecar_path(paths[0])
    raw = open(sc, "rb").read()
    open(sc, "wb").write(raw[:-5])
    with pytest.raises(ValueError, match="truncated"):
        cv.read_index(paths[0])
    open(sc, "wb").write(b"NOTANINDEX" + raw[10:])
    with pytest.raises(ValueError, match="not a bslz4"):
        cv.read_index(paths[0])
    os.remove(sc)
    with pytest.raises(FileNotFoundError, match="index_chunks"):
        cv.make_units(paths, DSET)
    units, shape, dtype = cv.make_units(paths, DSET, harvest=True)  # the explicit HDF5 path
    assert sum(u.n for u in units) == 5


def test_reading_and_decoding_need_no_hdf5(tmp_path, monkeypatch):
    paths = make_files(tmp_path)
    mask = np.ones(SHAPE, np.uint8)
    ref = direct(paths, mask)  # h5py only to make the reference
    monkeypatch.setitem(sys.modules, "h5py", None)  # from here "import h5py" raises ImportError
    with pytest.raises(ImportError):
        import h5py  # noqa: F401
    units, shape, dtype = cv.make_units(paths, DSET, max_bytes=3000)
    Collect.out = {}
    cv.run(units, cv.CpuBackend(mask, dtype), workers=2, sink=lambda k: Collect())
    assert Collect.out == ref


def test_an_acquisition_that_stopped_early(tmp_path):
    rng = np.random.default_rng(0)
    p = str(tmp_path / "partial.h5")
    with h5py.File(p, "w") as h:
        ds = h.create_dataset(
            DSET,
            shape=(10,) + SHAPE,
            dtype=np.uint16,
            chunks=(1,) + SHAPE,
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
        )
        ds[:4] = rng.integers(0, 50, (4,) + SHAPE).astype(
            np.uint16
        )  # frames 4..9 are never written
    ix = cv.harvest(p, DSET)
    assert ix.shape == (4,) + SHAPE and ix.meta["declared_nframes"] == 10 and len(ix.offsets) == 4
    cv.write_index(ix, p)
    units, shape, dtype = cv.make_units([p], DSET)
    assert sum(u.n for u in units) == 4
    q = str(tmp_path / "holes.h5")
    with h5py.File(q, "w") as h:
        ds = h.create_dataset(
            DSET,
            shape=(10,) + SHAPE,
            dtype=np.uint16,
            chunks=(1,) + SHAPE,
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
        )
        ds[0] = 1
        ds[5] = 1  # a hole in the middle
    with pytest.raises(ValueError, match="not all at the end"):
        cv.harvest(q, DSET)


def test_a_faster_worker_takes_more_units(tmp_path):
    """Pulling from the shared queue is the load balance: with one slow worker the others decode more."""
    import time

    paths = make_files(tmp_path, nfiles=4, nframes=60)
    units, shape, dtype = cv.make_units(paths, DSET, max_bytes=5000)
    assert len(units) > 40
    done = {}

    class Counting(cv.Sink):
        def __init__(self, k):
            self.k = k
            done[k] = 0

        def write(self, file, frame0, npx, values, indices, powder):
            done[self.k] += len(npx)
            if self.k == 0:
                time.sleep(0.02)  # worker 0 is slow

    cv.run(units, cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype), workers=3, sink=Counting, batch=4)
    assert sum(done.values()) == 4 * 60
    assert done[0] < min(done[1], done[2]) / 2, done


def test_report_and_the_guard_drops_only_pixels_that_are_not_sparse(tmp_path):
    pytest.importorskip("bitshuffle")
    import scipy.sparse as sp

    paths = make_files(tmp_path)
    units, shape, dtype = cv.make_units(paths, DSET, max_bytes=4000)
    mask = np.ones(SHAPE, np.uint8)
    npix = int(np.prod(SHAPE))
    m = sp.csc_matrix(
        (np.ones(npix, np.float32), (np.arange(npix) % 7, np.arange(npix))), (7, npix)
    )
    backend = cv.CpuBackend(mask, dtype, csc=m)
    want = direct(paths, mask)

    class Both(Collect):
        """pixels as Collect, and the powder sums per frame"""

        powder = {}

        def write(self, file, frame0, npx, values, indices, powder):
            Collect.write(self, file, frame0, npx, values, indices, powder)
            with Collect.lock:
                for j in range(len(npx)):
                    Both.powder[(file, frame0 + j)] = powder[j].copy()

    # within the limit: the inner sink gets exactly what a direct decode gives
    Collect.out, Both.powder = {}, {}
    r = cv.run(units, backend, workers=2, sink=lambda k: cv.GuardSink(Both(), limit=1e6), batch=8)
    assert Collect.out == want and r["sparse_skipped"] == []
    ref_powder = dict(Both.powder)
    assert r["in_bytes"] == sum(int(u.lengths.sum()) for u in units)
    assert r["out_bytes"] > 0 and r["sparse_stored_bytes"] > 0
    assert r["sparse_stored_over_in"] == pytest.approx(
        r["sparse_stored_bytes"] / r["in_bytes"], abs=1e-4
    )
    assert "sparse pixels" in cv.report(r) and "input" in cv.report(r)
    # over the limit: no pixels reach the sink, the job goes on, the powder is the same, and the
    # result and a warning say so
    Collect.out, Both.powder = {}, {}
    with pytest.warns(UserWarning, match="not sparse"):
        r = cv.run(
            units, backend, workers=2, sink=lambda k: cv.GuardSink(Both(), limit=1e-9), batch=8
        )
    assert r["frames"] == 3 * 37 and len(r["sparse_skipped"]) == len(units)
    assert all(v[0] == 0 for v in Collect.out.values())  # no pixels
    assert Both.powder.keys() == ref_powder.keys()
    assert all(np.array_equal(Both.powder[k], ref_powder[k]) for k in ref_powder)
    assert "NOT WRITTEN" in cv.report(r)
    zs = pytest.importorskip("zstandard") and cv.stored_size(
        [np.arange(5000, dtype=np.uint32)], "bszstd"
    )
    assert 0 < zs < 20000
    # without a guard the report has no sparse size
    r = cv.run(units, backend, workers=2, batch=8)
    assert r["sparse_stored_bytes"] is None and "sparse pixels" not in cv.report(r)


def test_policy_refuses_only_what_has_no_writer_or_too_many_bins(tmp_path):
    import scipy.sparse as sp

    paths = make_files(tmp_path)
    units, shape, dtype = cv.make_units(paths, DSET)
    mask = np.ones(SHAPE, np.uint8)
    npix = int(np.prod(SHAPE))
    for k in ("2d", "fazit", "decompress"):
        with pytest.raises(NotImplementedError, match="memory"):
            cv.Policy(k)
    with pytest.raises(ValueError):
        cv.Policy("nonsense")
    # sparse: nothing is refused beforehand, whatever howmany is: the guard sizes what comes
    cv.Policy("sparse").check(cv.CpuBackend(mask, dtype), units)
    cv.Policy("sparse").check(cv.CpuBackend(mask, dtype, howmany=100), units)
    # powder: needs a matrix, and its dense bins must be smaller than the compressed frame
    with pytest.raises(cv.OutputRefused, match="matrix"):
        cv.Policy("powder1d").check(cv.CpuBackend(mask, dtype), units)
    mean_in = sum(int(u.lengths.sum()) for u in units) / sum(u.n for u in units)
    for nbins, ok in ((4, True), (int(mean_in // 8) + 50, False)):
        rows = np.arange(npix) % nbins
        m = sp.csc_matrix((np.ones(npix, np.float32), (rows, np.arange(npix))), (nbins, npix))
        pol = cv.Policy(("powder1d", "sparse"))
        be = cv.CpuBackend(mask, dtype, csc=m, howmany=10)
        if ok:
            pol.check(be, units)
        else:
            with pytest.raises(cv.OutputRefused, match="not a reduction"):
                pol.check(be, units)


def test_iterate_gives_every_frame_and_stops_cleanly(tmp_path):
    paths = make_files(tmp_path)
    units, shape, dtype = cv.make_units(paths, DSET, max_bytes=3000)
    mask = np.ones(SHAPE, np.uint8)
    want = direct(paths, mask)
    got = {}
    for file, frame0, npx, values, indices, powder in cv.iterate(
        units, cv.CpuBackend(mask, dtype), workers=3, batch=4
    ):
        for j, n in enumerate(npx):
            got[(file, frame0 + j)] = (
                int(n),
                int(values[j, :n].sum(dtype=np.int64)),
                int(indices[j, :n].sum(dtype=np.int64)),
            )
    assert got == want
    it = cv.iterate(units, cv.CpuBackend(mask, dtype), workers=3, batch=2, depth=1)
    next(it)
    it.close()  # breaking out must not hang or raise
    with pytest.raises(ValueError, match="threads"):
        next(cv.iterate(units, cv.CpuBackend(mask, dtype), workers=2, mode="processes"))
