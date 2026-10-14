"""plan: the matrix of a chunk2sparseCSC prepared once and shared by the objects of many threads.

A plan owns read-only copies of the matrix and the mask; objects made from it (plan.context(),
chunk2sparseCSC(mask, plan)) share them and each layout, built once for all (also when 8 threads
ask at once).  chunk2sparseCSC(mask, csc) alone still owns a private copy.
  * an object on a plan gives the powder of an independent object and holds the very same Layout
    arrays, for a histogram, a split and a ring-like matrix and the integer dots;
  * a change of the caller's arrays after construction changes nothing; none of the arrays the
    objects pass to C is one of the caller's, and none can be written;
  * an object at another block size adds its own layout to the plan and leaves the first alone;
  * the conveyor backend builds once for all its workers; objects and backends still pickle.
"""
import collections
import os
import pickle
import sys
import threading

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")
sp = pytest.importorskip("scipy.sparse")

import bslz4_to_sparse as b
from bslz4_to_sparse import _matrix
from test_layouts import available_dots, make_nosplit, make_split_1d, reference

SHAPE = (60, 70)
NPIX = SHAPE[0] * SHAPE[1]


def encode(frames, nelems=0):
    with h5py.File("p.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=nelems, cname="lz4"))
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]


def make_rings(npix, nbins, seed):
    """Only about 10 % of the pixels reach a run of 1-3 bins (occupancy < 0.5)."""
    M = make_split_1d(npix, nbins, seed).tolil()
    keep = np.random.default_rng(seed).random(npix) < 0.1
    M[:, np.flatnonzero(~keep)] = 0
    return sp.csc_matrix(M)


MATRICES = {
    "histogram": (lambda: make_nosplit(NPIX, 50, 1), ("csc-nosplit-int", "bsb-csr-nosplit", "csc")),
    "split": (lambda: make_split_1d(NPIX, 50, 2), ("padded", "bsb-csr", "csc")),
    "rings": (lambda: make_rings(NPIX, 50, 3), ("bsb-csr", "padded", "csc")),
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    frames = rng.poisson(0.4, (3,) + SHAPE).astype(np.uint16)
    mask = (rng.random(SHAPE) > 0.1).astype(np.uint8)
    return frames, mask, encode(frames)


@pytest.fixture
def count_builds(monkeypatch):
    calls = collections.Counter()
    real = _matrix.build

    def counting(dot, *a, **k):
        calls[dot] += 1
        return real(dot, *a, **k)

    monkeypatch.setattr(_matrix, "build", counting)
    return calls


def powder(obj, chunks):
    return np.array(obj.multi(chunks, 0)[2])


def same_arrays(a, b_):
    return len(a) == len(b_) and all(x is y for x, y in zip(a, b_))


@pytest.mark.parametrize("kind", sorted(MATRICES))
def test_objects_on_a_plan_equal_an_independent_object_and_share_its_layout(kind, data):
    frames, mask, chunks = data
    make, dots = MATRICES[kind]
    M = make()
    ref = reference(M, frames, mask)
    ran = 0
    for dot in [None] + [d for d in dots if d in available_dots()]:
        pipe = None if dot is None else {"dot": dot}
        try:
            alone = b.chunk2sparseCSC(mask, make(), codec=b.CODEC_LZ4, pipeline=pipe)
        except ValueError:                      # this dot cannot represent this matrix
            with pytest.raises(ValueError):
                b.plan(mask, M, codec=b.CODEC_LZ4, pipeline=pipe)
            continue
        pl = b.plan(mask, M, np.uint16, b.CODEC_LZ4, pipe)
        c1, c2 = pl.context(), b.chunk2sparseCSC(mask, pl, codec=b.CODEC_LZ4, pipeline=pipe)
        assert c1._layout is c2._layout and c1._family is pl and c1.dot == alone.dot
        assert same_arrays(c1._layout.args(), c2._layout.args())
        assert c1._nm is c2._nm and c1.mask is c2.mask
        assert c1.pipeline is not c2.pipeline
        p = powder(alone, chunks)
        for c in (c1, c2):
            assert np.array_equal(powder(c, chunks), p)
        assert c1._workspace is not c2._workspace and c1._powder is not c2._powder
        assert np.abs(p - ref).max() < 1e-4 * max(1.0, np.abs(ref).max())
        ran += 1
    assert ran >= 2


def test_the_same_csc_made_twice_is_two_independent_objects(data, count_builds):
    frames, mask, chunks = data
    M = make_nosplit(NPIX, 50, 1)
    c1 = b.chunk2sparseCSC(mask, M, pipeline={"dot": "bsb-csr-nosplit"})
    c2 = b.chunk2sparseCSC(mask, M, pipeline={"dot": "bsb-csr-nosplit"})
    assert c1._layout is not c2._layout and count_builds["bsb-csr-nosplit"] == 2
    assert not [k for k in vars(M) if "bslz4" in k]          # nothing is put on the caller's object


@pytest.mark.parametrize("dot", ["csc-int", "csc-run-int"])
def test_a_plan_of_a_fixed_point_matrix(dot, data, count_builds):
    frames, mask, chunks = data
    M = make_nosplit(NPIX, 50, 1)
    fx = b.fixed_point_matrix(M, mask, np.uint16)
    alone = b.chunk2sparseCSC(mask, fx, dtype=np.uint16, pipeline={"dot": dot})
    pl = b.plan(mask, fx, np.uint16, pipeline={"dot": dot})
    n = count_builds[dot]
    cs = [pl.context(), pl.context()]
    assert count_builds[dot] == n and all(c._layout is cs[0]._layout for c in cs)
    assert pl.fixed is not fx and not np.shares_memory(pl.fixed.data, fx.data)
    assert not np.shares_memory(pl.nm.indices, fx.nm.indices)
    assert not pl.fixed.data.flags.writeable and fx.data.flags.writeable
    p = powder(alone, chunks)
    assert p.dtype == np.int64
    for c in cs:
        assert np.array_equal(powder(c, chunks), p) and c.powder_scale == alone.powder_scale


def user_arrays(M, mask):
    return [M.data, M.indices, M.indptr, mask]


def arrays_in_use(c):
    out = [c.mask, c._nm.data, c._nm.indices, c._nm.indptr]
    out += [a for a in c._layout.args() if isinstance(a, np.ndarray)]
    return out


@pytest.mark.parametrize("kind", sorted(MATRICES))
def test_the_object_owns_its_arrays_and_they_are_read_only(kind, data):
    frames, mask, chunks = data
    make, dots = MATRICES[kind]
    M = make()
    mk = mask.copy()
    for dot in dots:
        if dot not in available_dots():
            continue
        try:
            c = b.chunk2sparseCSC(mk, M, codec=b.CODEC_LZ4, pipeline={"dot": dot})
        except ValueError:
            continue
        powder(c, chunks)                       # C accepts the read-only arrays
        for mine in arrays_in_use(c) + arrays_in_use(b.plan(mk, M, np.uint16, b.CODEC_LZ4, {"dot": dot}).context()):
            assert not mine.flags.writeable
            for theirs in user_arrays(M, mk):
                assert not np.shares_memory(mine, theirs), dot
        with pytest.raises(ValueError):
            c._nm.data[0] = 1
        with pytest.raises(ValueError):
            c.mask[0] = 1


def test_changes_after_construction_change_nothing(data):
    frames, mask, chunks = data
    M = make_split_1d(NPIX, 50, 2)
    mk = mask.copy()
    pl = b.plan(mk, M, np.uint16, b.CODEC_LZ4)
    c = b.chunk2sparseCSC(mk, M, codec=b.CODEC_LZ4)
    p0 = powder(c, chunks).copy()
    M.data[:] = M.data * 2                                  # in place
    mk[:] = 0                                               # the mask too
    assert np.array_equal(powder(c, chunks), p0)
    M.data = (M.data * 0.25).astype(np.float32)            # replaced
    M.indices[:] = 0
    assert np.array_equal(powder(c, chunks), p0)
    assert np.array_equal(powder(pl.context(), chunks), p0)
    assert np.array_equal(c.mask, mask.ravel())
    # and an object made now sees the new matrix
    assert not np.array_equal(powder(b.chunk2sparseCSC(mask, M, codec=b.CODEC_LZ4), chunks), p0)


def test_a_plan_refuses_another_mask(data):
    frames, mask, chunks = data
    pl = b.plan(mask, make_nosplit(NPIX, 50, 1))
    other = mask.copy()
    other[0, 0] = 1 - other[0, 0]
    with pytest.raises(ValueError, match="mask"):
        b.chunk2sparseCSC(other, pl)
    with pytest.raises(ValueError, match="mask"):
        b.chunk2sparseCSC(mask[:, :-1], pl)
    assert b.chunk2sparseCSC(mask.copy(), pl) is not None


def test_objects_of_one_plan_may_differ_in_what_is_per_object(data, count_builds):
    frames, mask, chunks = data
    M = make_split_1d(NPIX, 50, 2)
    pl = b.plan(mask, M, np.uint16, b.CODEC_LZ4, {"dot": "csc"})
    assert pl.context(howmany=5).howmany == 5 and pl.context().howmany is None
    c = b.chunk2sparseCSC(mask, pl, codec=b.CODEC_LZ4, pipeline={"dot": "bsb-csr"})
    assert c.dot == "bsb-csr" and c._family is pl and count_builds["bsb-csr"] == 1
    assert pl.context()._layout.dot == "csc"
    assert np.allclose(powder(c, chunks), powder(pl.context(), chunks), rtol=1e-5)


def test_eight_threads_build_each_layout_once(data, count_builds):
    frames, mask, chunks = data
    M = make_split_1d(NPIX, 50, 2)
    pl = b.plan(mask, M, np.uint16, b.CODEC_LZ4, {"dot": "csc"})
    barrier = threading.Barrier(8)
    out, errs = [], []

    def go(k):
        try:
            barrier.wait()
            c = b.chunk2sparseCSC(mask, pl, codec=b.CODEC_LZ4,
                                  pipeline={"dot": ("bsb-csr", "padded")[k % 2]})
            out.append((c, powder(c, chunks)))
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=go, args=(k,)) for k in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(out) == 8
    assert count_builds["bsb-csr"] == 1 and count_builds["padded"] == 1 and count_builds["csc"] == 1
    for c, p in out:
        assert c._layout is pl._layouts[(c.dot, c._layout.block_elems, None)]
        assert np.allclose(p, out[0][1], rtol=1e-5)


def test_an_object_at_another_block_size_adds_its_own_layout(count_builds):
    big = (1100, 1200)                                      # 1 MB blocks: 524288 pixels
    rng = np.random.default_rng(7)
    fr = rng.poisson(0.3, (2,) + big).astype(np.uint16)
    mk = (rng.random(big) > 0.1).astype(np.uint8)
    Mb = make_split_1d(big[0] * big[1], 50, 4)
    ch = encode(fr, nelems=(1 << 20) // 2)
    r = reference(Mb, fr, mk)
    pl = b.plan(mk, Mb, np.uint16, b.CODEC_LZ4, {"dot": "padded"})
    c0 = pl.context()
    first, args = c0._layout, c0._layout.args()
    assert first.block_elems == 8192 // 2 and count_builds["padded"] == 1
    s1, s2 = pl.context(), pl.context()
    for c in (s1, s2):
        assert np.abs(powder(c, ch) - r).max() < 1e-4 * r.max()
    assert s1._layout is s2._layout and s1._layout is not first
    assert s1._layout.block_elems == (1 << 20) // 2 and count_builds["padded"] == 2
    assert c0._layout is first and same_arrays(first.args(), args) and len(pl._layouts) == 2
    # c0 itself, used on these chunks, takes the plan's layout for them
    assert np.abs(powder(c0, ch) - r).max() < 1e-4 * r.max()
    assert c0._layout is s1._layout and count_builds["padded"] == 2


def test_objects_and_backends_still_pickle(data):
    from bslz4_to_sparse import conveyor as cv

    frames, mask, chunks = data
    M = make_split_1d(NPIX, 50, 2)
    c = b.chunk2sparseCSC(mask, M, codec=b.CODEC_LZ4)
    c2 = pickle.loads(pickle.dumps(c))
    assert np.array_equal(powder(c2, chunks), powder(c, chunks))
    pl = pickle.loads(pickle.dumps(b.plan(mask, M, np.uint16, b.CODEC_LZ4)))
    assert np.array_equal(powder(pl.context(), chunks), powder(c, chunks))
    assert not pl.nm.data.flags.writeable
    be = cv.CpuBackend(mask, np.uint16, csc=M, exact=False)
    be.prepare()
    be2 = pickle.loads(pickle.dumps(be))
    assert be2._plan_obj is None and be2.make_worker() is not None


@pytest.mark.parametrize("exact", [None, False])
def test_the_conveyor_backend_builds_once_for_all_workers(tmp_path, count_builds, exact):
    from bslz4_to_sparse import conveyor as cv

    rng = np.random.default_rng(3)
    p = str(tmp_path / "f.h5")
    fr = np.where(rng.random((24,) + SHAPE) < 0.05, rng.integers(1, 60, (24,) + SHAPE), 0)
    with h5py.File(p, "w") as h:
        h.create_dataset("/entry/data", data=fr.astype(np.uint16), chunks=(1,) + SHAPE,
                         **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    cv.write_index(cv.harvest(p, "/entry/data"), p)
    units, shape, dtype = cv.make_work_units([p], "/entry/data")
    M = make_nosplit(NPIX, 7, 1)
    seen = []

    class Check(cv.Sink):
        def write(self, file, frame0, npx, values, indices, powder):
            seen.append(powder.shape)

    cv.run(units, cv.CpuBackend(np.ones(SHAPE, np.uint8), dtype, csc=M, exact=exact),
           workers=4, sink=lambda k: Check())
    assert seen and sum(count_builds.values()) == 1, dict(count_builds)
