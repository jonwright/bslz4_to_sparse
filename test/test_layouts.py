"""Matrix-dot layouts (csc, padded*, bsb-csr, ...) against an
independent reference.

The per-layout tests (test_padded.py, test_bsb_csr.py) compare each dot with
pipeline={"dot": "csc"}, which goes through the same mask-folded matrix, so a mask or tail
bug common to both would cancel.  Here every available dot is checked
against numpy `M @ (img * mask)` instead, on both routes, plus the edge cases
of the frame/layout geometry:

  * masked pixels holding huge finite values must not reach the powder;
  * a frame that is a whole number of decode blocks (empty tail);
  * a frame with fewer pixels than the mask is an error (-110);
  * intensity is preserved for a matrix whose pixel weights sum to 1;
  * malformed layout arrays are refused (c2py checks / -109), not read.

Self-contained (no pyFAI).
"""

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")
sp = pytest.importorskip("scipy.sparse")

import bslz4_to_sparse as b
from bslz4_to_sparse import _pipeline

ROUTES = ("dense", "sparse")


def available_dots():
    """The dot names this CPU/build can run."""
    return [n for i, n in enumerate(_pipeline.NAMES["dot"])
            if i and _pipeline.available("dot", i) == 1]


def write_chunks(path, frames):
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + frames.shape[1:],
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        codec = b.detect_codec(ds)
        chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]
    return chunks, codec


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
    return sp.csc_matrix((w.astype(np.float32), indices.astype(np.int32),
                          indptr.astype(np.int32)), shape=(nbins, npix))


def make_nosplit(npix, nbins, seed):
    """A histogram (pyFAI split='no'): each pixel reaches at most one bin,
    weight exactly 1 (csc-nosplit can represent it)."""
    rng = np.random.default_rng(seed)
    n = (rng.random(npix) > 0.1).astype(np.int64)
    indptr = np.concatenate(([0], np.cumsum(n)))
    indices = rng.integers(0, nbins, int(n.sum()))
    return sp.csc_matrix((np.ones(int(n.sum()), np.float32), indices.astype(np.int32),
                          indptr.astype(np.int32)), shape=(nbins, npix))


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
    return sp.csc_matrix((data, indices.astype(np.int32), indptr.astype(np.int32)),
                         shape=(nbins, npix))


def make_general(npix, nbins, seed):
    """Pixels reach arbitrary bins (padded must refuse it)."""
    rng = np.random.default_rng(seed)
    n = rng.integers(0, 4, npix)
    indptr = np.concatenate(([0], np.cumsum(n)))
    indices = rng.integers(0, nbins, int(n.sum()))
    data = rng.random(int(n.sum())).astype(np.float32) + 0.25
    return sp.csc_matrix((data, indices.astype(np.int32), indptr.astype(np.int32)),
                         shape=(nbins, npix))


def big_value(dt):
    dt = np.dtype(dt)
    return np.iinfo(dt).max if dt.kind in "ui" else 1e30


def reference(M, frames, mask):
    keep = (mask.ravel() > 0).astype(np.float64)
    return np.array([M @ (f.ravel().astype(np.float64) * keep) for f in frames])


def expected(integ, frames, ref):
    """(reference powder, relative tolerance) for this integrator's dot.  The
    fixed-point dots sum exact integers, so they are checked against numpy
    with the same rounded weights (the layout's data, in the order of the
    mask-folded matrix)."""
    lay = integ._layout
    if lay.scale is not None:
        nm = integ._nm
        Mq = sp.csc_matrix((lay.args()[0].astype(np.float64) * lay.scale, nm.indices, nm.indptr),
                           shape=nm.shape)
        return np.array([Mq @ f.ravel().astype(np.float64) for f in frames]), 1e-12
    return ref, 1e-9


def run_all_dots(mask, M, frames, chunks, codec, dt):
    """Yield (dot, route, powder, npx, adr, integ) for every dot that accepts M
    (and the dtype: the integer dots refuse float pixels)."""
    for dot in available_dots():
        for route in ROUTES:
            try:
                integ = b.chunk2sparseCSC(mask, M, dtype=dt, codec=codec,
                                          pipeline={"dot": dot, "route": route})
            except ValueError:      # padded refusing a general matrix
                continue
            npx, (_v, adr), p = integ.multi(chunks, 0)
            yield dot, route, p.copy(), npx.copy(), adr.copy(), integ


@pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.int32, np.float32])
@pytest.mark.parametrize("kind", ["split1d", "general", "nosplit", "moment"])
def test_every_dot_matches_numpy_with_masked_big_values(tmp_path, dt, kind):
    shape = (61, 83)                              # 5063 px: a raw tail for every dtype
    npix = shape[0] * shape[1]
    rng = np.random.default_rng(3)
    mask = (rng.random(npix) > 0.2).astype(np.uint8).reshape(shape)
    frames = rng.poisson(0.8, (3,) + shape).astype(dt)
    frames[:, mask == 0] = big_value(dt)          # must never reach the powder
    chunks, codec = write_chunks(tmp_path / "f.h5", frames)
    M = {"split1d": lambda: make_split_1d(npix, 200, 4),
         "general": lambda: make_general(npix, 300, 5),
         "nosplit": lambda: make_nosplit(npix, 250, 8),
         "moment": lambda: make_nosplit_moment(npix, 240, 9)}[kind]()
    ref = reference(M, frames, mask)
    seen = set()
    keep = (mask > 0)
    for dot, route, p, _npx, _adr, integ in run_all_dots(mask, M, frames, chunks, codec, dt):
        seen.add(dot)
        r, tol = expected(integ, frames * keep, ref)
        err = np.abs(p - r).max() / max(np.abs(r).max(), 1.0)
        assert err < tol, (dot, route, err)
    assert "bsb-csr" in seen and "csc" in seen
    if kind == "split1d":
        assert "padded" in seen and "csc-run" in seen
        if "padded-avx2" in available_dots():
            assert "padded-avx2" in seen
        assert "csc-nosplit" not in seen
    if kind == "general":
        assert "csc-run" not in seen and "csc-nosplit" not in seen
    if kind == "nosplit":
        assert "csc-run" in seen and "csc-nosplit" in seen and "bsb-csr-nosplit" in seen
    else:
        assert "bsb-csr-nosplit" not in seen
    assert ("csc-nosplit-moment" in seen) == (kind == "moment")
    if kind == "moment":
        assert "csc-run" in seen and "csc-nosplit" not in seen


def test_whole_number_of_blocks(tmp_path):
    """8192 u16 pixels = exactly two decode blocks: the driver's tail call is
    empty and must not index past the per-block pointer arrays."""
    shape = (128, 64)
    npix = shape[0] * shape[1]
    rng = np.random.default_rng(6)
    frames = rng.poisson(0.8, (2,) + shape).astype(np.uint16)
    chunks, codec = write_chunks(tmp_path / "e.h5", frames)
    mask = np.ones(shape, np.uint8)
    M = make_split_1d(npix, 150, 7)
    ref = reference(M, frames, mask)
    for dot, route, p, _npx, _adr, integ in run_all_dots(mask, M, frames, chunks, codec, np.uint16):
        r, tol = expected(integ, frames, ref)
        assert np.abs(p - r).max() < tol * max(np.abs(r).max(), 1.0), (dot, route)
    integ = b.chunk2sparseCSC(mask, M, dtype=np.uint16, pipeline={"dot": "bsb-csr"})
    blk_ptr = integ._layout.args()[0]
    assert len(blk_ptr) == npix // integ._layout.block_elems + 1


def test_frame_smaller_than_mask_is_an_error(tmp_path):
    shape = (61, 83)
    frames = np.ones((1,) + shape, np.uint16)
    chunks, codec = write_chunks(tmp_path / "s.h5", frames)
    npix = shape[0] * shape[1] + 3000
    mask = np.ones((1, npix), np.uint8)
    M = make_split_1d(npix, 150, 8)
    for dot in available_dots():
        try:
            integ = b.chunk2sparseCSC(mask, M, dtype=np.uint16, codec=codec,
                                      pipeline={"dot": dot})
        except ValueError:      # a dot that cannot represent M (csc-nosplit)
            continue
        with pytest.raises(Exception, match="-110"):
            integ.multi(chunks, 0)
    with pytest.raises(Exception, match="-110"):
        b.chunk2sparse(mask, dtype=np.uint16, codec=codec).multi(chunks, 0)


@pytest.mark.parametrize("masked", [False, True])
def test_intensity_preserved(tmp_path, masked):
    """Pixel weights sum to 1, so sum(powder) == sum of the (unmasked) image."""
    shape = (61, 83)
    npix = shape[0] * shape[1]
    rng = np.random.default_rng(9)
    mask = (rng.random(npix) > (0.1 if masked else -1)).astype(np.uint8).reshape(shape)
    frames = rng.poisson(3.0, (2,) + shape).astype(np.uint16)
    chunks, codec = write_chunks(tmp_path / "i.h5", frames)
    M = make_split_1d(npix, 200, 10)
    total = np.array([f[mask > 0].sum() for f in frames], np.float64)
    for dot, route, p, _npx, _adr, _integ in run_all_dots(mask, M, frames, chunks, codec, np.uint16):
        rel = np.abs(p.sum(axis=1) / total - 1).max()
        assert rel < 1e-6, (dot, route, rel)      # float32 weights sum to 1 +- 6e-8


def test_malformed_layout_arrays_are_refused(tmp_path):
    shape = (61, 83)
    npix = shape[0] * shape[1]
    frames = np.ones((1,) + shape, np.uint16)
    chunks, codec = write_chunks(tmp_path / "m.h5", frames)
    mask = np.ones(shape, np.uint8)
    M = make_split_1d(npix, 150, 11)

    names = {"padded": ("base", "weights", "pixels", "rowmap", "row_ptr", "width",
                        "listed", "block_elems"),
             "bsb-csr": ("blk_ptr", "bins", "bin_ptr", "idx", "data", "csc_data",
                         "csc_indices", "csc_indptr", "block_elems")}

    def make(dot):
        integ = b.chunk2sparseCSC(mask, M, dtype=np.uint16, codec=codec, pipeline={"dot": dot})
        integ.multi(chunks, 0)                    # allocate buffers
        return integ

    def args_of(integ):
        lay = integ._layout
        return dict(zip(names[lay.entry], lay.args()))

    def call_with(integ, **override):
        lay = integ._layout
        args = args_of(integ)
        args.update(override)
        lay._args = tuple(args[n] for n in names[lay.entry])
        return integ.multi(chunks, 0)

    p = args_of(make("padded"))
    with pytest.raises(ValueError):               # weights.n != base.n * width
        call_with(make("padded"), weights=p["weights"][:-1])
    with pytest.raises(ValueError):
        call_with(make("padded"), width=65)
    with pytest.raises(ValueError):               # float base passes itemsize, not format
        call_with(make("padded"), base=p["base"].astype(np.float32))
    with pytest.raises(Exception, match="-109"):  # row_ptr does not cover every block
        call_with(make("padded"), row_ptr=p["row_ptr"][:-1])
    q = args_of(make("bsb-csr"))
    with pytest.raises(ValueError):               # bin_ptr.n != bins.n + 1
        call_with(make("bsb-csr"), bin_ptr=q["bin_ptr"][:-1])
    with pytest.raises(ValueError):               # idx.n != data.n
        call_with(make("bsb-csr"), idx=q["idx"][:-1])
    with pytest.raises(ValueError):               # csc_indptr.n != mask.n + 1
        call_with(make("bsb-csr"), csc_indptr=q["csc_indptr"][:-1])
    with pytest.raises(Exception, match="-109"):  # blk_ptr does not cover every block
        call_with(make("bsb-csr"), blk_ptr=q["blk_ptr"][:-1])
