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
    if acc == "i64":  # integer sums need the matrix prepared
        c = g.chunk2sparseCSC(mask, b.fixed_point_matrix(M, mask, dt, maxval=mv), dt, codec)
    else:
        c = g.chunk2sparseCSC(mask, M, dt, codec, acc=acc)
    npx, (v, idx), p = c.multi(chunks, 0)
    if c.powder_scale is not None:
        p = p * (1.0 / c.powder_scale)
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
        gpu = g.chunk2sparseCSC(mask, M, dt, codec, howmany=howmany, thresholds=thresholds)
        cpu = b.chunk2sparseCSC(mask, M, dt, codec, howmany=howmany, thresholds=thresholds)
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
    c = g.chunk2sparseCSC(mask, make_split_1d(mask.size, 50, 1), np.uint16, codec)
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


@pytest.mark.parametrize("dt", [np.uint16, np.int32, np.float32])
def test_batch_and_group_give_the_same_result(tmp_path, dt):
    mask, frames = mixed_frames(dt, (93, 217))
    frames = np.concatenate([frames, frames[::-1], frames[1:3]])  # 12 frames, dense ones among
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    M = make_split_1d(mask.size, 60, 3)
    for howmany in (None, 50):
        ref = g.chunk2sparseCSC(mask, M, dt, codec, batch=len(frames), howmany=howmany)
        n0, (v0, i0), p0 = ref.multi(chunks, 1)
        for batch, group in ((1, 1), (2, 5), (3, 7), (4, 4), (16, 3), (5, None), (4, 12)):
            c = g.chunk2sparseCSC(mask, M, dt, codec, batch=batch, group=group, howmany=howmany)
            n1, (v1, i1), p1 = c.multi(chunks, 1)
            assert (n1 == n0).all() and (p1 == p0).all(), (batch, group)
            for f in range(len(frames)):
                assert (i1[f, : n1[f]] == i0[f, : n0[f]]).all()
                assert (v1[f, : n1[f]] == v0[f, : n0[f]]).all()
            assert (c.last_nabove == ref.last_nabove).all() and (c.last_cut == ref.last_cut).all()
            assert c.last_host == ref.last_host
    bad = bytearray(chunks[9])
    bad[12:16] = (len(bad)).to_bytes(4, "big")
    c = g.chunk2sparse(mask, dt, codec, batch=2, group=4)
    with pytest.raises(Exception, match=r"chunks \[9\]"):
        c.multi(chunks[:9] + [bytes(bad)] + chunks[10:], 1)


@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.int32, np.float32])
def test_unblock_on_the_cpu_gives_the_same_result(tmp_path, dt):
    mask, frames = mixed_frames(dt, (93, 217))
    frames = np.concatenate([frames, frames[::-1]])
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    M = make_split_1d(mask.size, 60, 3)
    ref = g.chunk2sparseCSC(mask, M, dt, codec, batch=3, group=7)
    n0, (v0, i0), p0 = ref.multi(chunks, 1)
    c = g.chunk2sparseCSC(mask, M, dt, codec, batch=3, group=7, unblock="cpu")
    n1, (v1, i1), p1 = c.multi(chunks, 1)
    assert (n1 == n0).all() and (p1 == p0).all() and c.last_host == ref.last_host
    for f in range(len(frames)):
        assert (i1[f, : n1[f]] == i0[f, : n0[f]]).all()
        assert (v1[f, : n1[f]] == v0[f, : n0[f]]).all()
    bad = bytearray(chunks[8])
    bad[12:16] = (len(bad)).to_bytes(4, "big")
    with pytest.raises(Exception, match=r"chunks \[8\]"):
        c.multi(chunks[:8] + [bytes(bad)] + chunks[9:], 1)
    with pytest.raises(ValueError):
        g.chunk2sparse(mask, dt, codec, unblock="disk")


@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.int32, np.float32])
@pytest.mark.parametrize("shape", [(61, 83), (64, 128), (16, 8)])  # 61 x 83: rows padded to 16 B
def test_decompress_and_integrate_stay_on_the_device(tmp_path, dt, shape):
    rng = np.random.default_rng(4)
    mask = (rng.random(shape) > 0.2).astype(np.uint8)
    frames = rng.poisson(2.0, (7,) + shape).astype(dt)
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    M = make_split_1d(mask.size, 40, 2)
    c = g.chunk2sparseCSC(mask, M, dt, codec, batch=2, group=3, device=0)
    d = c.decompress(chunks)
    assert d.shape == frames.shape and d.dtype == frames.dtype
    assert (cp.asarray(d).get() == frames).all()  # __cuda_array_interface__ (or a cupy array)
    assert (cp.from_dlpack(d).get() == frames).all()
    assert int(cp.asarray(d[2:5]).sum()) == int(frames[2:5].astype(np.int64).sum())
    p = c.integrate(chunks)
    _, _, p0 = c.multi(chunks, 0)
    assert p.shape == (len(frames), M.shape[0]) and (cp.asarray(p).get() == p0).all()
    bad = bytearray(chunks[4])
    bad[12:16] = (len(bad)).to_bytes(4, "big")
    with pytest.raises(Exception, match=r"chunks \[4\]"):
        c.decompress(chunks[:4] + [bytes(bad)] + chunks[5:])


def write_chunks_codec(path, frames, cname):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "d", data=frames, chunks=(1,) + frames.shape[1:],
            **hdf5plugin.Bitshuffle(nelems=0, cname=cname)
        )
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]


@pytest.mark.parametrize("dt", [np.uint16, np.int32])
def test_chunks_already_on_the_device(tmp_path, dt):
    mask, frames = mixed_frames(dt, (61, 83))
    frames = np.concatenate([frames, frames[::-1]])
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    lens = np.array([len(c) for c in chunks], np.int64)
    offs = 100 + np.concatenate(([0], np.cumsum(lens)[:-1]))  # the chunks start 100 bytes in
    d_buf = cp.asarray(np.frombuffer(b"x" * 100 + b"".join(chunks), np.uint8))
    M = make_split_1d(mask.size, 50, 5)
    c = g.chunk2sparseCSC(mask, M, dt, codec, batch=3, group=4)
    n0, (v0, i0), p0 = _flat(c.multi(chunks, 1))  # copies: the next call reuses the buffers
    n1, (v1, i1), p1 = c.from_device(d_buf, offs, lens, 1)
    assert (n1 == n0).all() and (p1 == p0).all()
    for f in range(len(frames)):
        assert (i1[f, : n1[f]] == i0[f, : n0[f]]).all() and (v1[f, : n1[f]] == v0[f, : n0[f]]).all()
    assert (cp.asarray(c.decompress_from_device(d_buf, offs, lens)).get() == frames).all()
    want = cp.asarray(c.integrate(chunks)).get()
    assert (cp.asarray(c.integrate_from_device(d_buf, offs, lens)).get() == want).all()
    # the header checks are on the GPU: a frame of another size, a zstd chunk, offsets outside
    other = write_chunks_codec(tmp_path / "o.h5", frames[:2, :60], "lz4")[1]
    zst = write_chunks_codec(tmp_path / "z.h5", frames[:2], "zstd")[1]
    for bad, match in ((other, "-106"), (zst, "zstd")):
        blob = b"".join(chunks[:5]) + bad + b"".join(chunks[6:])
        ln = np.array([len(x) for x in chunks[:5]] + [len(bad)] + [len(x) for x in chunks[6:]])
        of = np.concatenate(([0], np.cumsum(ln)[:-1]))
        with pytest.raises(Exception, match=match + r".*chunks \[5\]"):
            c.decompress_from_device(cp.asarray(np.frombuffer(blob, np.uint8)), of, ln)
    with pytest.raises(Exception, match="-108"):
        c.from_device(d_buf, offs + 10**6, lens, 1)
    bad = bytearray(chunks[4])
    bad[12:16] = (len(bad)).to_bytes(4, "big")
    blob = b"".join(chunks[:4]) + bytes(bad) + b"".join(chunks[5:])
    with pytest.raises(Exception, match=r"corrupt.*chunks \[4\]"):
        c.integrate_from_device(cp.asarray(np.frombuffer(blob, np.uint8)), offs - 100, lens)


def _flat(r):
    npx, (v, i), p = r
    return npx.copy(), (v.copy(), i.copy()), p.copy()


def test_driver_backend_arrays_are_shared_without_a_copy():
    """The driver backend's exports, whichever backend the GPU classes use in this run."""
    import gc

    from bslz4_to_sparse.cuda import _backend_drv as drv

    host = np.arange(3000, dtype=np.uint16).reshape(30, 100)
    a = drv.asarray(host)
    x = cp.asarray(a)  # __cuda_array_interface__
    y = cp.from_dlpack(a)  # DLPack
    assert x.data.ptr == y.data.ptr == a.ptr and (x.get() == host).all() and (y.get() == host).all()
    assert (cp.asarray(a[10:20]).get() == host[10:20]).all()
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        t = torch.from_dlpack(a)
        assert t.data_ptr() == a.ptr and (t.cpu().numpy() == host).all()
        del t
    del x, y
    gc.collect()
    assert not drv._dl_alive  # every exported tensor was released by its consumer
    src = cp.arange(64, dtype=cp.uint8)
    b = drv.device_bytes(src, cp.cuda.Device().id)  # borrowing another library's memory
    assert b.size == 64 and b.ptr == src.data.ptr and (b.get() == np.arange(64)).all()
    with pytest.raises(BufferError):
        a.__dlpack__(dl_device=(2, 99))


def _device_of(a):
    ptr = a.data.ptr if hasattr(a, "data") and hasattr(a.data, "ptr") else getattr(a, "ptr", None)
    return None if not ptr else cp.cuda.runtime.pointerGetAttributes(ptr).device


def test_each_object_keeps_its_buffers_on_its_own_gpu(tmp_path):
    if g.device_count() < 2:
        pytest.skip("needs two GPUs")
    mask, frames = mixed_frames(np.uint16, (93, 217))
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    M = make_split_1d(mask.size, 60, 3)
    objs = [g.chunk2sparseCSC(mask, M, np.uint16, codec, device=d) for d in (0, 1)]
    results = []
    for d, o in enumerate(objs):
        n, (v, i), p = o.multi(chunks, 1)
        out = [o.decompress(chunks), o.integrate(chunks)]
        results.append((n.copy(), v.copy(), i.copy(), p.copy(), [cp.asarray(x).get() for x in out]))
        held = [(k, a) for k, a in vars(o).items() if k.startswith("d_")] + [("out", x) for x in out]
        assert len(held) > 10
        for k, a in held:
            assert _device_of(a) in (None, d), (d, k, _device_of(a))
    (n0, v0, i0, p0, o0), (n1, v1, i1, p1, o1) = results
    assert (n0 == n1).all() and (p0 == p1).all()
    for f in range(len(frames)):  # the lists: their first npixels entries
        assert (v0[f, : n0[f]] == v1[f, : n1[f]]).all() and (i0[f, : n0[f]] == i1[f, : n1[f]]).all()
    assert all((a == b).all() for a, b in zip(o0, o1))
