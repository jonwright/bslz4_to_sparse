"""The conveyor: every frame of every unit reaches a sink exactly once and equals a direct decode,
with threads and with forked processes; a failing sink raises instead of hanging."""

from __future__ import absolute_import, division, print_function

from conftest import bitshuffle, chunk_info_available, needs_chunk_info
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import conveyor as cv  # noqa: E402
from bslz4_to_sparse.conveyor._index import MTIME_SLACK_NS, mtime_ns  # noqa: E402

DSET = "/entry/data"
SHAPE = (64, 80)


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conveyor_files")
SETS = [(3, 37), (1, 20), (1, 37), (2, 25), (1, 5), (4, 60), (2, 11)]  # (nfiles, nframes) used


GENERATOR = 1  # bump when cached_files makes different files: the cache is keyed on it and SHAPE


def cached_files(nfiles=3, nframes=37):
    """The test files and their sidecars, made once in test/conveyor_files.  Making a sidecar needs
    h5py's chunk index (h5py >= 3), so Python 2.7 reads the ones made under Python 3
    (python3 test/make_testcases.py conveyor makes every set in SETS)."""
    d = os.path.join(CACHE, "g%d_%dx%d_%dx%d" % ((GENERATOR,) + SHAPE + (nfiles, nframes)))
    paths = [os.path.join(d, "f%d.h5" % k) for k in range(nfiles)]
    if os.path.isdir(d):
        return paths
    if not chunk_info_available():
        pytest.skip("no %s: make it under Python 3 with test/make_testcases.py conveyor" % d)
    if not os.path.isdir(CACHE):
        os.makedirs(CACHE)
    tmp = tempfile.mkdtemp(dir=CACHE)  # built aside and renamed: a set is there whole or not at all
    rng = np.random.RandomState(3)
    for k in range(nfiles):
        p = os.path.join(tmp, "f%d.h5" % k)
        frames = np.where(
            rng.random_sample((nframes,) + SHAPE) < 0.05, rng.randint(1, 60, (nframes,) + SHAPE), 0
        )
        with h5py.File(p, "w") as h:
            h.create_dataset(
                DSET,
                data=frames.astype(np.uint16),
                chunks=(1,) + SHAPE,
                **bitshuffle(nelems=0, cname="lz4")
            )
        # the sidecar: reading never needs HDF5 after this; its path is beside the file, so it
        # moves with the directory, and make_files writes fresh ones for the copies anyway
        cv.write_index(cv.harvest(p, DSET), p)
    try:
        os.rename(tmp, d)
    except OSError:  # another run made it meanwhile
        shutil.rmtree(tmp)
    return paths


def make_files(tmpdir, nfiles=3, nframes=37):
    """Copies of cached_files in tmpdir, each with its own sidecar, for the test to change."""
    out = []
    for p in cached_files(nfiles, nframes):
        q = os.path.join(str(tmpdir), os.path.basename(p))
        shutil.copyfile(p, q)
        ix = cv.read_index(p, check=False)
        st = os.stat(q)
        ix.meta.update(file_size=st.st_size, file_mtime_ns=mtime_ns(st))
        cv.write_index(ix, q)
        out.append(q)
    return out


def direct(paths, mask):
    ref = {}
    c = b.chunk2sparse(mask, np.uint16, b.CODEC_LZ4)
    for fi, p in enumerate(paths):
        with h5py.File(p, "r") as h:
            ds = h[DSET]
            for i in range(ds.shape[0]):
                n, (v, ix) = c(b.read_direct_chunk(ds, (i, 0, 0))[1], 0)
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
def test_every_frame_once_and_equal_to_a_direct_decode(tmpdir, unit_bytes):
    paths = make_files(tmpdir)
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=unit_bytes)
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


needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason='mode="processes" forks: no fork here')


@needs_fork
def test_processes_count_the_same_frames(tmpdir):
    paths = make_files(tmpdir)
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=4000)

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


def test_with_a_matrix_and_howmany(tmpdir):
    import scipy.sparse as sp

    paths = make_files(tmpdir, nfiles=1, nframes=20)
    units, shape, dtype = cv.make_work_units(paths, DSET)
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


@pytest.mark.parametrize("mode", ["threads", pytest.param("processes", marks=needs_fork)])
def test_a_failing_sink_raises_and_does_not_hang(tmpdir, mode):
    paths = make_files(tmpdir)
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=4000)

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


def test_a_failure_after_the_start_is_not_a_failure_to_start(tmpdir, monkeypatch):
    """Worker 1 is slow past the start; worker 0's sink fails meanwhile.  Worker 1 still ends its
    sink (it started), and the error raised is the sink's."""
    from bslz4_to_sparse.conveyor import _engine

    paths = make_files(tmpdir)
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=4000)
    slow, ended = set(), []

    class SlowEvent(threading._Event if hasattr(threading, "_Event") else threading.Event):
        def wait(self, timeout=None):
            r = super(SlowEvent, self).wait(timeout)
            if threading.current_thread() in slow:
                time.sleep(0.5)
            return r

    class Sink(cv.Sink):
        def __init__(self, k):
            self.k = k
            if k == 1:
                slow.add(threading.current_thread())

        def write(self, file, frame0, npx, values, indices, powder):
            if self.k == 0:
                raise RuntimeError("sink broke")

        def end(self):
            ended.append(self.k)

    monkeypatch.setattr(_engine._ThreadContext, "Event", staticmethod(SlowEvent))
    with pytest.raises(RuntimeError, match="sink broke"):
        cv.run(units, cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype), workers=2, sink=Sink)
    assert 1 in ended


def test_threads_need_nothing_linux_only(tmpdir, monkeypatch):
    """Off Linux (macOS, Windows) there is no per-thread getrusage: the run is the same, the page
    faults are None, and the cpu time comes from time.thread_time (Python 3).  On Linux both are
    counted.  Without fork, mode="processes" says so."""
    from bslz4_to_sparse.conveyor import _engine

    if sys.platform.startswith("linux"):
        assert _engine._RESOURCE is not None
    paths = make_files(tmpdir, nfiles=2)
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=4000)
    mask = np.ones(SHAPE, np.uint8)
    monkeypatch.setattr(_engine, "_RESOURCE", None)
    Collect.out = {}
    r = cv.run(units, cv.CpuBackend(mask, dtype), workers=2, sink=lambda k: Collect())
    assert r["frames"] == 2 * 37 and Collect.out == direct(paths, mask)
    assert r["page_faults"] == dict(workers=None, readers=None)
    assert (r["worker_cpu_cores"] is None) == (not hasattr(time, "thread_time"))
    monkeypatch.delattr(os, "fork", raising=False)
    with pytest.raises(ValueError, match="no fork"):
        cv.run(units, cv.CpuBackend(mask, dtype), workers=1, mode="processes")


def test_bad_arguments(tmpdir):
    with pytest.raises(ValueError):
        cv.run([], cv.CpuBackend(np.ones(SHAPE, np.uint8), np.uint16), workers=1)
    paths = make_files(tmpdir, nfiles=1)
    units, shape, dtype = cv.make_work_units(paths, DSET)
    with pytest.raises(ValueError, match="mode"):
        cv.run(units, cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype), workers=1, mode="gpus")


@needs_chunk_info
def test_sidecar_equals_what_hdf5_reports(tmpdir):
    paths = make_files(tmpdir, nfiles=2, nframes=25)
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


@needs_chunk_info
def test_sidecar_in_another_directory(tmpdir):
    paths = make_files(tmpdir, nfiles=1, nframes=5)
    ix = cv.harvest(paths[0], DSET)
    where = os.path.join(str(tmpdir), "idx")
    out = cv.write_index(ix, paths[0], where)
    assert out == cv.sidecar_path(paths[0], where) and os.path.dirname(out) == where
    assert np.array_equal(cv.read_index(paths[0], where).offsets, ix.offsets)


def test_a_changed_file_makes_the_sidecar_stale(tmpdir):
    paths = make_files(tmpdir, nfiles=1, nframes=5)
    with open(paths[0], "ab") as f:
        f.write(b"x")
    with pytest.raises(cv.StaleIndex):
        cv.read_index(paths[0])
    assert cv.read_index(paths[0], check=False).shape[0] == 5  # the caller may choose to trust it
    with pytest.raises(cv.StaleIndex):
        cv.make_work_units(paths, DSET)


def test_the_recorded_mtime_may_be_a_microsecond_off(tmpdir):
    """A sidecar written by a Python with only the float st_mtime (2.7) records the time rounded,
    a few hundred ns off; it is still fresh to either Python.  Beyond a microsecond it is stale."""
    paths = make_files(tmpdir, nfiles=1, nframes=5)
    ix = cv.read_index(paths[0])
    for off, stale in ((MTIME_SLACK_NS, False), (-MTIME_SLACK_NS, False), (MTIME_SLACK_NS + 1, True)):
        ix.meta.update(file_mtime_ns=mtime_ns(os.stat(paths[0])) + off)
        cv.write_index(ix, paths[0])
        if stale:
            with pytest.raises(cv.StaleIndex):
                cv.read_index(paths[0])
        else:
            cv.read_index(paths[0])


def test_corrupt_and_missing_sidecars(tmpdir):
    paths = make_files(tmpdir, nfiles=1, nframes=5)
    sc = cv.sidecar_path(paths[0])
    raw = open(sc, "rb").read()
    open(sc, "wb").write(raw[:-5])
    with pytest.raises(ValueError, match="truncated"):
        cv.read_index(paths[0])
    open(sc, "wb").write(b"NOTANINDEX" + raw[10:])
    with pytest.raises(ValueError, match="not a bslz4"):
        cv.read_index(paths[0])
    os.remove(sc)
    with pytest.raises(EnvironmentError, match="index_chunks"):
        cv.make_work_units(paths, DSET)


def test_index_chunks_keeps_fresh_sidecars_and_makes_missing_ones(tmpdir):
    """tools/index_chunks.py under this Python: a fresh sidecar is kept, a missing one is made
    (or, with an h5py that has no chunk index, the file is reported as failed)."""
    paths = make_files(tmpdir, nfiles=2, nframes=5)
    os.remove(cv.sidecar_path(paths[1]))
    script = os.path.join(os.path.dirname(CACHE), "..", "tools", "index_chunks.py")
    env = dict(os.environ, BSLZ4_TO_SPARSE_PATH=os.path.dirname(os.path.dirname(b.__file__)))
    p = subprocess.Popen(
        [sys.executable, script, "--files", os.path.join(str(tmpdir), "*.h5"), "--dset", DSET,
         "--procs", "2"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True)
    out, err = p.communicate()
    assert "'kept': 1" in out, out + err
    if chunk_info_available():
        assert p.returncode == 0 and "'written': 1" in out, out + err
        assert cv.read_index(paths[1]).shape[0] == 5
    else:
        assert p.returncode == 1 and "'failed': 1" in out and "get_chunk_info" in err, out + err


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="tells the worker by /proc")
@pytest.mark.parametrize("flag", ["--check", "--force"])
def test_index_chunks_counts_a_worker_that_died_as_failed(tmpdir, flag):
    """A worker that exits 1 before it reads its files (an import failed: Python's own exit code
    for an uncaught exception) did nothing: its files are failed, not checked or written,
    although their old sidecars are still fresh."""
    paths = make_files(tmpdir, nfiles=2, nframes=5)
    site = os.path.join(str(tmpdir), "site")
    os.mkdir(site)
    with open(os.path.join(site, "sitecustomize.py"), "w") as f:
        f.write("import os\n"
                "with open('/proc/%d/cmdline' % os.getppid(), 'rb') as f:\n"
                "    if b'index_chunks.py' in f.read():\n"
                "        os._exit(1)  # the worker, before it reads stdin\n")
    script = os.path.join(os.path.dirname(CACHE), "..", "tools", "index_chunks.py")
    env = dict(os.environ, BSLZ4_TO_SPARSE_PATH=os.path.dirname(os.path.dirname(b.__file__)),
               PYTHONPATH=os.pathsep.join([site] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    p = subprocess.Popen(
        [sys.executable, script, "--files", os.path.join(str(tmpdir), "*.h5"), "--dset", DSET,
         "--procs", "2", flag],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True)
    out, err = p.communicate()
    assert p.returncode == 1 and "'failed': 2" in out and "crashed" in out, out + err


@needs_chunk_info
def test_a_missing_sidecar_is_harvested_when_asked(tmpdir):
    paths = make_files(tmpdir, nfiles=1, nframes=5)
    os.remove(cv.sidecar_path(paths[0]))
    units, shape, dtype = cv.make_work_units(paths, DSET, harvest=True)  # the explicit HDF5 path
    assert sum(u.n for u in units) == 5


class FailingSink(cv.NullSink):
    """begin() raises for the workers in `fail`, and blocks (until `release` is set) for those in
    `hang`."""

    def __init__(self, k, fail=(), hang=(), release=None):
        cv.NullSink.__init__(self)
        self.k, self.fail, self.hang, self.release = k, fail, hang, release

    def begin(self, meta):
        if self.k in self.fail:
            raise ValueError("sink %d refused" % self.k)
        if self.k in self.hang:
            self.release.wait()


def test_each_worker_that_fails_to_start_is_reported_once(tmpdir):
    paths = make_files(tmpdir, nfiles=2, nframes=5)
    units, shape, dtype = cv.make_work_units(paths, DSET)
    with pytest.raises(RuntimeError) as e:
        cv.run(units, cv.CpuBackend(np.ones(shape, np.uint8), dtype), workers=3,
               sink=lambda k: FailingSink(k, fail=(1, 2)))
    msg = str(e.value)
    assert msg.count("sink 1 refused") == 1 and msg.count("sink 2 refused") == 1, msg


def test_a_worker_that_hangs_at_start_does_not_hang_the_run(tmpdir, monkeypatch):
    from bslz4_to_sparse.conveyor import _engine

    monkeypatch.setattr(_engine, "STOP_GRACE_S", 1.0)
    paths = make_files(tmpdir, nfiles=2, nframes=5)
    units, shape, dtype = cv.make_work_units(paths, DSET)
    release = threading.Event()
    out = []

    def go():
        try:
            cv.run(units, cv.CpuBackend(np.ones(shape, np.uint8), dtype), workers=2,
                   sink=lambda k: FailingSink(k, hang=(0, 1), release=release), start_timeout=0.5)
        except RuntimeError as err:
            out.append(str(err))

    t = threading.Thread(target=go)
    t.daemon = True
    t.start()
    t.join(20)
    release.set()  # let the stuck worker go, whatever happened
    assert not t.is_alive(), "run() did not return"
    assert out and "timed out after 0.5 s" in out[0], out


def test_reading_and_decoding_need_no_hdf5(tmpdir, monkeypatch):
    paths = make_files(tmpdir)
    mask = np.ones(SHAPE, np.uint8)
    ref = direct(paths, mask)  # h5py only to make the reference
    monkeypatch.setitem(sys.modules, "h5py", None)  # from here "import h5py" raises ImportError
    with pytest.raises(ImportError):
        import h5py  # noqa: F401
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=3000)
    Collect.out = {}
    cv.run(units, cv.CpuBackend(mask, dtype), workers=2, sink=lambda k: Collect())
    assert Collect.out == ref


@needs_chunk_info
def test_an_acquisition_that_stopped_early(tmpdir):
    rng = np.random.RandomState(0)
    p = os.path.join(str(tmpdir), "partial.h5")
    with h5py.File(p, "w") as h:
        ds = h.create_dataset(
            DSET,
            shape=(10,) + SHAPE,
            dtype=np.uint16,
            chunks=(1,) + SHAPE,
            **bitshuffle(nelems=0, cname="lz4")
        )
        ds[:4] = rng.randint(0, 50, (4,) + SHAPE).astype(
            np.uint16
        )  # frames 4..9 are never written
    ix = cv.harvest(p, DSET)
    assert ix.shape == (4,) + SHAPE and ix.meta["declared_nframes"] == 10 and len(ix.offsets) == 4
    cv.write_index(ix, p)
    units, shape, dtype = cv.make_work_units([p], DSET)
    assert sum(u.n for u in units) == 4
    q = os.path.join(str(tmpdir), "holes.h5")
    with h5py.File(q, "w") as h:
        ds = h.create_dataset(
            DSET,
            shape=(10,) + SHAPE,
            dtype=np.uint16,
            chunks=(1,) + SHAPE,
            **bitshuffle(nelems=0, cname="lz4")
        )
        ds[0] = 1
        ds[5] = 1  # a hole in the middle
    with pytest.raises(ValueError, match="not all at the end"):
        cv.harvest(q, DSET)


def test_a_faster_worker_takes_more_units(tmpdir):
    """Pulling from the shared queue is the load balance: with one slow worker the others decode more."""
    import time

    paths = make_files(tmpdir, nfiles=4, nframes=60)
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=5000)
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


def test_report(tmpdir):
    paths = make_files(tmpdir)
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=4000)
    r = cv.run(units, cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype), workers=2, batch=8)
    assert r["in_bytes"] == sum(int(u.lengths.sum()) for u in units) and r["out_bytes"] > 0
    assert "input" in cv.report(r) and "output" in cv.report(r)


def test_a_writer_refuses_a_powder_over_a_tenth_of_the_frame_before_the_run(tmpdir):
    """8 x nbins bytes against the uncompressed frame: refused in begin(), before any frame, and no
    file is written; a powder within the limit is written."""
    import scipy.sparse as sp
    from bslz4_to_sparse.conveyor._sinks import POWDER_MAX_FRACTION

    paths = make_files(tmpdir)
    units, shape, dtype = cv.make_work_units(paths, DSET)
    mask = np.ones(SHAPE, np.uint8)
    npix = int(np.prod(SHAPE))
    frame = npix * np.dtype(dtype).itemsize
    limit = int(POWDER_MAX_FRACTION * frame) // 8
    for nbins, ok in ((limit, True), (limit + 1, False)):
        rows = np.arange(npix) % nbins
        m = sp.csc_matrix((np.ones(npix, np.float32), (rows, np.arange(npix))), (nbins, npix))
        out = os.path.join(str(tmpdir), "o%d" % nbins)
        go = lambda: cv.run(  # noqa: E731
            units,
            cv.CpuBackend(mask, dtype, csc=m, howmany=10),
            workers=2,
            sink=lambda k: cv.ImageD11SparseSink(out, DSET),
        )
        if ok:
            assert go()["frames"] == 3 * 37
        else:
            with pytest.raises(RuntimeError, match="not a reduction"):
                go()
            assert not os.path.exists(out) or not os.listdir(out)
    cv.check_powder_size(dict(shape=SHAPE, dtype=str(dtype)))  # no matrix: nothing to check


def test_iterate_gives_every_frame_and_stops_cleanly(tmpdir):
    paths = make_files(tmpdir)
    units, shape, dtype = cv.make_work_units(paths, DSET, max_unit_bytes=3000)
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


def test_cpulist_and_nodes_without_cpus():
    from bslz4_to_sparse.conveyor._engine import _allowed_cpus, _cpulist

    assert _cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert _cpulist("\n") == [] and _cpulist("") == []  # a memory-only node (GPU memory on POWER9)
    assert sum(len(n) for n in cv.numa_nodes()) == len(set(_allowed_cpus()))


def test_numa_nodes_keeps_to_the_cpus_allowed():
    """Under Slurm a job may use some CPUs of the node: numa_nodes lists only those, on Python 2
    too (no sched_getaffinity there: /proc/self/status).  taskset stands in for Slurm."""
    import subprocess

    path = os.environ.get("PATH", "").split(os.pathsep)
    if not any(os.path.exists(os.path.join(d, "taskset")) for d in path):
        pytest.skip("needs taskset")
    if not os.path.exists("/proc/self/status"):
        pytest.skip("needs /proc")
    cpu = sorted(set(sum(cv.numa_nodes(), [])))[-1]
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(b.__file__)))
    out = subprocess.check_output(
        ["taskset", "-c", str(cpu), sys.executable, "-c",
         "from bslz4_to_sparse import conveyor; print(conveyor.numa_nodes())"],
        env=dict(os.environ, PYTHONPATH=pkg),
    )
    assert out.decode().strip() == "[[%d]]" % cpu


def fazit_matrix(mask, extra=5, seed=11):
    """A permutation (FAZIT): every unmasked pixel to its own bin, weight 1; some bins empty."""
    import scipy.sparse as sp

    rng = np.random.RandomState(seed)
    pix = np.flatnonzero(mask.ravel())
    bins = rng.permutation(pix.size + extra)[: pix.size]
    return sp.csc_matrix(
        (np.ones(pix.size, np.float32), (bins, pix)), shape=(pix.size + extra, mask.size)
    ), pix, bins



def check_fazit_keeps_the_pixel_dtype(tmpdir, backend, **kw):
    """With the default exact=None, a permutation matrix gives the pixels in bin order in the
    pixel dtype, with no powder_scale, whatever exact is; a float dot asked for gives float64 sums of
    the same values."""
    paths = make_files(tmpdir, nfiles=2, nframes=11)
    units, shape, dtype = cv.make_work_units(paths, DSET)
    mask = (np.random.RandomState(4).random_sample(SHAPE) > 0.1).astype(np.uint8)
    m, pix, bins = fazit_matrix(mask)
    frames = {}
    for fi, p in enumerate(paths):
        with h5py.File(p, "r") as h:
            frames[fi] = h[DSET][()].reshape(-1, mask.size)
    out, scales = {}, []

    class Keep(cv.Sink):
        lock = threading.Lock()

        def begin(self, meta):
            scales.append(meta["powder_scale"])

        def write(self, file, frame0, npx, values, indices, powder):
            with self.lock:
                for j in range(len(npx)):
                    out[(file, frame0 + j)] = powder[j].copy()

    cv.run(units, backend(mask, dtype, csc=m, **kw), workers=2, sink=lambda k: Keep(), batch=4)
    assert scales[0] is None and len(out) == 22
    for (fi, i), p in out.items():
        want = np.zeros(m.shape[0], dtype)
        want[bins] = frames[fi][i, pix]
        assert p.dtype == (np.float64 if "pipeline" in kw or "acc" in kw else dtype)
        assert np.array_equal(p, want)


@pytest.mark.parametrize("kw", [{}, {"exact": False}, {"exact": True}, {"pipeline": {"dot": "csc-nosplit"}}])
def test_fazit_keeps_the_pixel_dtype(tmpdir, kw):
    check_fazit_keeps_the_pixel_dtype(tmpdir, cv.CpuBackend, **kw)
