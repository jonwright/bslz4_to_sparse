"""The GPU classes (bslz4_to_sparse.cuda), against numpy and against the CPU classes.  Skipped
without cupy and a CUDA GPU.

  * collect: every pixel above the cut, exactly (dense frames take the host fallback);
  * matrices: the powder against numpy for the layouts and accumulators;
  * howmany / thresholds: identical to the CPU classes' (top_pixels rules, test_top_pixels.py),
    including the frames the
    device lists and the frames it cannot (host fallback, counted in last_host);
  * the powder never depends on howmany.
Matrix makers as in test_layouts.py.
"""

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")
sp = pytest.importorskip("scipy.sparse")
cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device", allow_module_level=True)
except Exception:  # noqa: BLE001 (no driver: cupy raises various errors)
    pytest.skip("no CUDA device", allow_module_level=True)

import bslz4_to_sparse as b  # noqa: E402
from bslz4_to_sparse import cuda as g  # noqa: E402

LZ4 = b.CODEC_LZ4


def write_chunks(path, frames):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "d",
            data=frames,
            chunks=(1,) + frames.shape[1:],
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
        )
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))], LZ4


def make_split_1d(npix, nbins, seed):
    """A 1D-like matrix: each pixel splits its intensity over 1-3 consecutive
    bins with weights summing to 1 (padded can represent it)."""
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 4, npix)
    start = rng.integers(0, nbins - 3, npix)
    indptr = np.concatenate(([0], np.cumsum(n)))
    indices = np.concatenate([np.arange(start[i], start[i] + n[i]) for i in range(npix)])
    w = rng.random(int(n.sum())) + 0.1
    w /= np.add.reduceat(w, indptr[:-1]).repeat(n)
    return sp.csc_matrix(
        (w.astype(np.float32), indices.astype(np.int32), indptr.astype(np.int32)),
        shape=(nbins, npix),
    )


def make_nosplit(npix, nbins, seed):
    """A histogram (pyFAI split='no'): each pixel reaches at most one bin,
    weight exactly 1 (csc-nosplit can represent it)."""
    rng = np.random.default_rng(seed)
    n = (rng.random(npix) > 0.1).astype(np.int64)
    indptr = np.concatenate(([0], np.cumsum(n)))
    indices = rng.integers(0, nbins, int(n.sum()))
    return sp.csc_matrix(
        (np.ones(int(n.sum()), np.float32), indices.astype(np.int32), indptr.astype(np.int32)),
        shape=(nbins, npix),
    )


def make_nosplit_moment(npix, nbins, seed):
    """A histogram with a first moment: each pixel reaches no bin or the pair
    (2k, 2k+1) with weights 1 (sum I) and its q (sum qI)."""
    rng = np.random.default_rng(seed)
    has = rng.random(npix) > 0.1
    n = np.where(has, 2, 0)
    k = rng.integers(0, nbins // 2, npix)
    q = rng.uniform(1.0, 30.0, npix).astype(np.float32)
    indptr = np.concatenate(([0], np.cumsum(n)))
    indices = np.stack([2 * k, 2 * k + 1], axis=1)[has].ravel()
    data = np.stack([np.ones(npix, np.float32), q], axis=1)[has].ravel()
    return sp.csc_matrix(
        (data, indices.astype(np.int32), indptr.astype(np.int32)), shape=(nbins, npix)
    )


def make_general(npix, nbins, seed):
    """Pixels reach arbitrary bins (padded must refuse it)."""
    rng = np.random.default_rng(seed)
    n = rng.integers(0, 4, npix)
    indptr = np.concatenate(([0], np.cumsum(n)))
    indices = rng.integers(0, nbins, int(n.sum()))
    data = rng.random(int(n.sum())).astype(np.float32) + 0.25
    return sp.csc_matrix(
        (data, indices.astype(np.int32), indptr.astype(np.int32)), shape=(nbins, npix)
    )


def big_value(dt):
    dt = np.dtype(dt)
    return np.iinfo(dt).max if dt.kind in "ui" else 1e30


def reference(M, frames, mask):
    keep = (mask.ravel() > 0).astype(np.float64)
    return np.array([M @ (f.ravel().astype(np.float64) * keep) for f in frames])


TOL = {"f64": 1e-9, "i64": 1e-9, "f32": 5e-5}


@pytest.mark.parametrize(
    "dt", [np.uint8, np.uint16, np.uint32, np.int32, np.float32, np.int16, np.int8]
)
@pytest.mark.parametrize("shape", [(61, 83), (93, 217), (64, 128), (16, 8)])
def test_collect(tmp_path, dt, shape):
    rng = np.random.default_rng(1)
    n = (3,) + shape
    mask = (rng.random(shape) > 0.2).astype(np.uint8)
    for dens in (0.01, 0.5, 30):
        a = rng.poisson(dens, n)
        if np.dtype(dt).kind == "i":
            a = a - rng.poisson(dens, n)
        frames = a.astype(dt)
        frames[:, mask == 0] = big_value(dt)
        chunks, codec = write_chunks(tmp_path / "f.h5", frames)
        c = g.chunk2sparse(mask, dt, codec)
        for cut in (-1, 0, 1, 5, 40000.5):
            npx, (v, idx) = c.multi(chunks, cut)
            for i in range(len(frames)):
                f = frames[i].ravel()
                want = np.flatnonzero((f > cut) & (mask.ravel() > 0))
                assert c.last_nabove[i] == len(want) and c.last_cut[i] == cut
                assert npx[i] == len(want), (dt, shape, dens, cut, npx[i], len(want))
                assert (idx[i, : npx[i]] == want).all()
                assert (v[i, : npx[i]] == f[want]).all()
            assert c.last_host == sum(1 for i in range(len(frames)) if c.last_nabove[i] > c.cap)


@pytest.mark.parametrize("acc", ["f64", "i64", "f32"])
@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.int32, np.float32])
@pytest.mark.parametrize("kind", ["split1d", "general", "nosplit", "moment", "many_bins"])
def test_matrices(tmp_path, dt, kind, acc):
    if acc == "i64" and np.dtype(dt).kind == "f":
        pytest.skip("integer accumulators need integer pixels")
    shape = (61, 83)
    npix = shape[0] * shape[1]
    rng = np.random.default_rng(3)
    mask = (rng.random(npix) > 0.2).astype(np.uint8).reshape(shape)
    frames = rng.poisson(0.8, (3,) + shape).astype(dt)
    frames[:, mask == 0] = big_value(dt)
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    M = {
        "split1d": lambda: make_split_1d(npix, 200, 4),
        "general": lambda: make_general(npix, 300, 5),
        "nosplit": lambda: make_nosplit(npix, 250, 8),
        "moment": lambda: make_nosplit_moment(npix, 240, 9),
        "many_bins": lambda: make_general(npix, 100000, 6),
    }[
        kind
    ]()  # more bins than a uint16
    mv = int(frames[:, mask > 0].max()) if np.dtype(dt).kind != "f" else None
    c = g.chunk2sparseCSC(mask, M, dt, codec, acc=acc, maxval=mv, exact=False)
    npx, (v, idx), p = c.multi(chunks, 0)
    ref = reference(M, frames, mask)
    assert np.abs(p - ref).max() / max(np.abs(ref).max(), 1.0) < TOL[acc]


def mixed_frames(dt, shape, seed=7):
    """nearly empty to dense, with a few bright spots, and masked pixels at the maximum"""
    rng = np.random.default_rng(seed)
    mask = (rng.random(shape) > 0.1).astype(np.uint8)
    frames = np.stack(
        [
            rng.poisson(lam, shape) + (rng.random(shape) > 0.999) * 200
            for lam in (0.01, 0.3, 2.0, 8.0, 30.0)
        ]
    )
    if np.dtype(dt).kind == "i":
        frames = frames - rng.poisson(0.5, frames.shape)
    frames = np.minimum(frames, 250).astype(dt)
    frames[:, mask == 0] = (
        big_value(dt) if np.dtype(dt).kind != "u" or np.dtype(dt).itemsize > 1 else 255
    )
    return mask, frames


@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.int32, np.float32])
def test_howmany_matches_the_cpu_classes(tmp_path, dt):
    mask, frames = mixed_frames(dt, (93, 217))
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    M = make_split_1d(mask.size, 60, 3)
    cap = mask.size * np.dtype(dt).itemsize // (4 + np.dtype(dt).itemsize)
    ref = reference(M, frames, mask)
    cases = (
        (None, None),
        (50, None),
        (500, [2.5, 5, 10]),
        (3, [1, 2]),
        (1000, []),
        (0, [1]),
        (50, [0, 1, 3, 7]),
        (cap // 2, None),
        (cap, None),
        (cap + 10, None),
        (cap + 10, [2, 4]),
    )
    for howmany, thresholds in cases:
        gpu = g.chunk2sparseCSC(mask, M, dt, codec, howmany=howmany, thresholds=thresholds, exact=False)
        cpu = b.chunk2sparseCSC(mask, M, dt, codec, howmany=howmany, thresholds=thresholds, exact=False)
        for cut in (0, 2):  # the CPU classes take an integer cut
            n1, (v1, i1), p1 = gpu.multi(chunks, cut)
            n0, (v0, i0), p0 = cpu.multi(chunks, cut)
            assert np.abs(p1 - ref).max() / np.abs(ref).max() < 1e-9  # the powder uses every pixel
            assert (n1 == n0).all(), (dt, howmany, thresholds, cut, n1, n0)
            for f in range(len(frames)):
                assert (i1[f, : n1[f]] == i0[f, : n0[f]]).all() and (
                    v1[f, : n1[f]] == v0[f, : n0[f]]
                ).all()
            assert (gpu.last_nabove == cpu.last_nabove).all() and (
                gpu.last_cut == cpu.last_cut
            ).all()
    with pytest.raises(ValueError):
        g.chunk2sparse(mask, dt, codec, thresholds=(3, 1))
    with pytest.raises(ValueError):
        g.chunk2sparse(mask, dt, codec, howmany=-1)


def test_dense_frames_are_listed_on_the_host_and_counted(tmp_path):
    rng = np.random.default_rng(0)
    shape = (93, 217)
    mask = np.ones(shape, np.uint8)
    frames = np.stack([rng.poisson(0.02, shape), rng.poisson(6, shape)]).astype(
        np.uint16
    )  # sparse, then dense
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    c = g.chunk2sparse(mask, np.uint16, codec)
    npx, (v, i) = c.multi(chunks, 0)
    assert c.last_host == 1 and c.last_nabove[1] > c.cap >= c.last_nabove[0]
    for f in range(2):
        want = np.flatnonzero(frames[f].ravel() > 0)
        assert npx[f] == len(want) and (i[f, : npx[f]] == want).all()
    c2 = g.chunk2sparse(
        mask, np.uint16, codec, howmany=1000
    )  # asked to limit: the device does it, no fallback
    n2, _ = c2.multi(chunks, 0)
    assert c2.last_host == 0 and n2[1] <= 1000


def test_single_offsets_coo(tmp_path):
    rng = np.random.default_rng(2)
    shape = (93, 217)
    mask = np.ones(shape, np.uint8)
    frames = rng.poisson(0.3, (5,) + shape).astype(np.uint16)
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    c = g.chunk2sparseCSC(mask, make_split_1d(mask.size, 50, 1), np.uint16, codec, exact=False)
    n1, (v1, i1), p1 = c(chunks[1], 1)
    npx, (v, i), p = c.multi(chunks, 1)
    assert (
        n1 == npx[1]
        and (v1[:n1] == v[1, :n1]).all()
        and (i1[:n1] == i[1, :n1]).all()
        and (p1 == p[1]).all()
    )
    blob = b"".join(bytes(x) for x in chunks)
    lens = np.array([len(x) for x in chunks])
    offs = np.concatenate(([0], np.cumsum(lens)[:-1]))
    n2, _, p2 = c.from_buffer(np.frombuffer(blob, np.uint8), offs, lens, 1)
    assert (n2 == npx).all() and (p2 == p).all()
    nn, row, col, vals, pw = c.coo(chunks[0], 1)
    assert nn == npx[0] and (row.astype(np.int64) * shape[1] + col == i[0, :nn]).all()


def test_corrupt_is_reported(tmp_path):
    frames = np.random.default_rng(0).poisson(1.0, (1, 93, 217)).astype(np.uint16)
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    c = g.chunk2sparse(np.ones((93, 217), np.uint8), np.uint16, codec)
    bad = bytearray(chunks[0])
    bad[12:16] = (len(bad)).to_bytes(4, "big")
    with pytest.raises(Exception, match="corrupt|Error decoding"):
        c.multi([bytes(bad)], 0)
