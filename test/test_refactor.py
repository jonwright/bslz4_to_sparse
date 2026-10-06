"""
Gate for the pipeline refactor:

  * the Python and C step/value tables agree;
  * pipeline validation rejects unknown values, steps and lengths;
  * an explicit pipeline actually forces the selected collect value;
  * the per-block counters show the selected value ran in every route and in
    the tail, and that the scalar collect was not picked for a SIMD dtype;
  * resolution / matrix errors surface as the documented codes.

The chunk fixture is grown by hand (literal-only LZ4 blocks) so no h5py /
pyFAI is needed: one full 8192-byte block plus a 200-byte tail (192 bytes =
one 8*2-aligned tail block, 8 bytes of raw remainder), which exercises both
the full-block and the tail path.
"""
from __future__ import print_function

import os
import struct
import sys

import pytest
import numpy as np

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import bslz4_to_sparse as b
from bslz4_to_sparse import _pipeline, _testing, _matrix

BS = 8192
NB = 2                 # uint16
TAIL = 200             # bytes of tail
NPIX = (BS + TAIL) // NB            # 4196
NBINS = 10


def _lz4_zeros(n):
    """LZ4 block holding n zero bytes as one literal-only sequence."""
    ext = n - 15
    return bytes(bytearray([0xF0])) + b"\xff" * (ext // 255) + bytes(bytearray([ext % 255])) + b"\0" * n


def _chunk():
    tail_block = 16 * (TAIL // 16)            # 8*NB-aligned bytes of the tail
    rem = TAIL - tail_block                   # raw (literal) remainder bytes
    full = _lz4_zeros(BS)
    tail = _lz4_zeros(tail_block)
    c = bytearray(struct.pack(">QI", BS + TAIL, BS))
    c += struct.pack(">I", len(full)) + full
    c += struct.pack(">I", len(tail)) + tail
    # The raw remainder holds the final NB/2 = 4 uint16 pixels directly (no
    # bitshuffle/transpose), so give them a value above cut so tail data is
    # actually sparsified.  The aligned full+tail blocks stay zeros.
    c += bytes([5, 0]) * (rem // NB)
    return bytes(c)


def _csc():
    indptr = np.arange(NPIX + 1, dtype=np.uint32)
    indices = (np.arange(NPIX) % NBINS).astype(np.uint32)
    data = np.ones(NPIX, np.float32)
    return type("_CSC", (), {"indptr": indptr, "indices": indices, "data": data,
                             "shape": (NBINS, NPIX)})()


def _usable(step):
    return [n for i, n in enumerate(_pipeline.NAMES[step])
            if i and _pipeline.available(step, i) == 1]


# ---- Python/C value-table agreement -----------------------------------------

def test_ids_and_c_agree():
    # names and ids are defined in both Python (src/_pipeline.py) and C
    # (src/pipeline/pipeline.h); every (step, value) must be known to C and
    # no unknown value may be reported as usable.
    for step in _pipeline.STEPS:
        n = len(_pipeline.NAMES[step])
        for i in range(1, n):
            assert _pipeline.available(step, i) != -1, (step, i)
        assert _pipeline.available(step, n) == -1, step
        assert _pipeline.available(step, 99) == -1, step
        assert _pipeline.available(step, -1) == -1, step
    # the scalar fallbacks are always usable
    assert "scalar" in _usable("collect")
    assert "scalar" in _usable("untranspose")
    assert "csc" in _usable("dot")


# ---- pipeline validation ------------------------------------------------------

def test_pipeline_defaults_and_rejects():
    mask = np.ones((1, NPIX), np.uint8)
    c = b.chunk2sparse(mask, dtype=np.uint16)
    p = c.pipeline
    assert isinstance(p, np.ndarray) and p.dtype == np.uint16 and p.shape == (6,)
    # every sparsify step resolved, route and dot unused without a matrix
    assert all(p[s] != 0 for s in range(_pipeline.ROUTE))
    assert p[_pipeline.ROUTE] == 0 and p[_pipeline.DOT] == 0
    d = b.describe(p)
    assert d["route"] is None and d["dot"] is None
    assert d["collect"] in _usable("collect")

    for step in ("decode", "mask", "untranspose", "collect"):
        with pytest.raises(ValueError, match="pipeline\\[%r\\]=99 is unknown" % step):
            b.chunk2sparse(mask, pipeline={step: 99})
        with pytest.raises(ValueError, match="pipeline\\[%r\\]='bogus' is unknown" % step):
            b.chunk2sparse(mask, pipeline={step: "bogus"})
    for step in ("route", "dot"):
        with pytest.raises(ValueError, match="is unknown"):
            b.chunk2sparseCSC(mask, _csc(), pipeline={step: 99})
        # route and dot need a matrix
        with pytest.raises(ValueError, match="this object has no matrix"):
            b.chunk2sparse(mask, pipeline={step: 1})
    with pytest.raises(ValueError, match="a pipeline has 6 values"):
        b.chunk2sparse(mask, pipeline=[0, 0, 0, 0, 0])    # the old 5-entry stages array
    with pytest.raises(ValueError, match="unknown steps 'options'"):
        b.chunk2sparse(mask, pipeline={"options": 64})


# ---- an explicit pipeline forces the selected collect value -------------------

def test_pipeline_forces_collect_value():
    simd = [n for n in _usable("collect") if n != "scalar"]

    def _run(name):
        c = b.chunk2sparse(np.ones((1, NPIX), np.uint8), dtype=np.uint16,
                           pipeline={"collect": name})
        assert b.describe(c.pipeline)["collect"] == name
        _testing.reset_counters()
        c.multi([_chunk()], 0)
        return _testing.counters()["collect"]

    # scalar, always available
    col = _run("scalar")
    assert col.get("scalar", 0) > 0
    assert set(col) == {"scalar"}, col

    for name in simd:
        col = _run(name)
        # the SIMD value ran, and nothing else did
        assert col.get(name, 0) > 0, (name, col)
        assert set(col) == {name}, (name, col)


# ---- per-block counters: every route and the tail, scalar not used -----------

@pytest.mark.parametrize("route", ["sparse", "dense"])
def test_counters_every_route_and_tail(route):
    c = b.chunk2sparseCSC(np.ones((1, NPIX), np.uint8), _csc(), dtype=np.uint16,
                          pipeline={"route": route})
    names = b.describe(c.pipeline)
    if names["collect"] == "scalar":
        pytest.skip("no SIMD collect value on this CPU/build")
    _testing.reset_counters()
    npx, (outpx, outadr), powder = c.multi([_chunk()], 1)
    cnt = _testing.counters()
    # the selected values ran in the full block AND the tail (the single
    # frame decodes 1 full block + 1 tail block), the dot ran, and scalar
    # collect was not used for a SIMD dtype.
    assert cnt["dot"].get(c.dot, 0) > 0, (route, cnt)
    assert sum(cnt["decode"].values()) == 2, (route, "1 full block + 1 tail block decoded", cnt)
    assert sum(cnt["untranspose"].values()) == 2, (route, "untranspose ran once per block", cnt)
    assert cnt["route"] == {route: 2}, (route, cnt)
    assert cnt["collect"].get(names["collect"], 0) > 0, (route, "SIMD collect did not run", cnt)
    assert "scalar" not in cnt["collect"], (route, "scalar collect should not run", cnt)
    # and the tail pixels are in the powder
    want = np.zeros(NBINS)
    np.add.at(want, np.arange(NPIX - 4, NPIX) % NBINS, 5.0)
    np.testing.assert_array_equal(powder[0], want)


# ---- tail data reaches the sparse output -------------------------------------
# _chunk() has a non-zero raw literal remainder (4 uint16 pixels, value 5) and
# zeros everywhere else, so cut=1 must sparsify exactly those 4 tail pixels.

def test_tail_remainder_pixels_are_sparsified():
    c = b.chunk2sparse(np.ones((1, NPIX), np.uint8), dtype=np.uint16)
    npx, (vals, adr) = c.multi([_chunk()], 1)
    assert int(npx[0]) == 4, npx
    np.testing.assert_array_equal(vals[0][:4], np.full(4, 5, dtype=np.uint16))
    np.testing.assert_array_equal(
        adr[0][:4], np.arange(NPIX - 4, NPIX, dtype=np.uint32))


# ---- resolution / matrix errors ----------------------------------------------

def test_normalise_matrix_accepts_scipy_csr_and_bsr():
    sp = pytest.importorskip("scipy.sparse")
    data = np.array([1, 2, 3, 4, 5], np.float32)
    indices = np.array([0, 2, 1, 0, 2], np.int32)
    indptr = np.array([0, 1, 2, 3, 4, 5], np.int32)
    shape = (3, 5)  # (nbins, npix)
    csc = sp.csc_matrix((data, indices, indptr), shape=shape)
    csr = sp.csr_matrix(csc)
    bsr = sp.bsr_matrix(csc)
    ref = _matrix.normalise_matrix(csc, npix=5)
    for name, m in (("csr", csr), ("bsr", bsr)):
        nm = _matrix.normalise_matrix(m, npix=5)
        np.testing.assert_array_equal(nm.indptr, ref.indptr)
        np.testing.assert_array_equal(nm.indices, ref.indices)
        np.testing.assert_array_equal(nm.data, ref.data)


def test_resolution_errors():
    _ext = b._ext
    ptr = np.empty(1, np.int64)
    ptr[0] = 0
    lens = np.array([len(_chunk())], np.int32)
    mask = np.zeros(NPIX, np.uint8)
    pipe = b.chunk2sparse(np.ones((1, NPIX), np.uint8)).pipeline
    cpipe = b.chunk2sparseCSC(np.ones((1, NPIX), np.uint8), _csc(),
                              pipeline={"dot": "csc"}).pipeline

    # dtype out of range -> ERR_DTYPE (-113)
    assert _ext.sparsify(ptr, lens, mask, np.zeros(NPIX, np.uint16),
                         np.empty(NPIX, np.uint32), np.empty(1, np.int32), 0,
                         np.zeros(3 * BS, np.uint8), np.empty(1, np.int64),
                         11, pipe) == -113

    # an unresolved (auto) pipeline is refused by C -> -111
    assert _ext.sparsify(ptr, lens, mask, np.zeros(NPIX, np.uint16),
                         np.empty(NPIX, np.uint32), np.empty(1, np.int32), 0,
                         np.zeros(3 * BS, np.uint8), np.empty(1, np.int64),
                         1, np.zeros(6, np.uint16)) == -111

    # int64 indices / indptr arrays are rejected at the boundary
    ws = np.zeros(3 * BS + (BS // NB) * (4 + NB), np.uint8)
    data = np.ones(NPIX, np.float32)
    ind64 = np.arange(NPIX, dtype=np.int64)
    indptr = np.arange(NPIX + 1, dtype=np.int32)
    with pytest.raises(Exception):
        _ext.sparsify_and_dot(ptr, lens, mask, np.zeros(NPIX, np.uint16),
                              np.empty(NPIX, np.uint32), np.empty(1, np.int32), 0,
                              np.zeros(NBINS, np.float64), data, ind64, indptr,
                              ws, np.empty(1, np.int64), NBINS, 1, cpipe)

    # wrong matrix size (indptr not npix+1) is rejected
    with pytest.raises(Exception):
        _ext.sparsify_and_dot(ptr, lens, mask, np.zeros(NPIX, np.uint16),
                              np.empty(NPIX, np.uint32), np.empty(1, np.int32), 0,
                              np.zeros(NBINS, np.float64), data,
                              np.arange(NPIX, dtype=np.uint32),
                              np.arange(NPIX, dtype=np.uint32),
                              ws, np.empty(1, np.int64), NBINS, 1, cpipe)


# ---- dot values: the per-dtype/kernel mechanism ------------------------------
# Every dot that can represent a one-bin-per-pixel, weight-1 matrix must give
# bit-identical powder and sparse output to csc on both routes, and the
# counters must show the *selected* dot ran (and no other).

def _variant_chunks(h5py, hdf5plugin, tmp_path, nf=2, h=8, w=8):
    """A tiny real bitshuffle-lz4 dataset with a mix of above/below-cut pixels
    so every dot has real work to do."""
    arr = (np.arange(nf * h * w, dtype=np.uint32).reshape(nf, h, w) % 8).astype(np.uint16)
    fn = str(tmp_path / "_bslz4_variant.h5")
    with h5py.File(fn, "w") as f:
        f.create_dataset("d", data=arr, chunks=(1, h, w),
                         **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    with h5py.File(fn, "r") as f:
        return arr, [f["d"].id.read_direct_chunk((i, 0, 0))[1] for i in range(nf)]


DOTS = ("csc", "padded", "padded-avx2", "csc-run", "csc-nosplit", "bsb-csr",
        "bsb-csr-nosplit", "csc-int", "csc-run-int", "csc-run-int16")


def test_dot_values_agree(tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf5plugin = pytest.importorskip("hdf5plugin")

    arr, chunks = _variant_chunks(h5py, hdf5plugin, tmp_path)
    mask = np.ones((1, 8 * 8), np.uint8)
    npix = mask.size
    nbins = 4
    indptr = np.arange(npix + 1, dtype=np.uint32)
    indices = (np.arange(npix) % nbins).astype(np.uint32)
    data = np.ones(npix, np.float32)
    csc = type("_CSC", (), {"indptr": indptr, "indices": indices, "data": data,
                            "shape": (nbins, npix)})()
    ref = np.zeros((len(arr), nbins))
    for f in range(len(arr)):
        np.add.at(ref[f], indices, arr[f].ravel().astype(np.float64))

    def _run(dot, route):
        c = b.chunk2sparseCSC(mask, csc, dtype=np.uint16,
                              pipeline={"dot": dot, "route": route})
        assert c.dot == dot
        _testing.reset_counters()
        npx, (vals, adr), powder = c.multi(chunks, 1)
        cnt = _testing.counters()["dot"]
        assert cnt.get(dot, 0) > 0, (dot, "selected dot did not run", cnt)
        assert set(cnt) == {dot}, (dot, "another dot ran", cnt)
        return npx.copy(), vals.copy(), adr.copy(), np.asarray(powder, np.float64).copy()

    usable = [d for d in DOTS if d in _usable("dot")]
    for route in ("dense", "sparse"):
        npx0, v0, a0, p0 = _run("csc", route)
        np.testing.assert_array_equal(p0, ref)
        for dot in usable:
            npx1, v1, a1, p1 = _run(dot, route)
            np.testing.assert_array_equal(npx0, npx1)
            for i in range(len(npx0)):
                n = int(npx0[i])
                np.testing.assert_array_equal(v0[i][:n], v1[i][:n])
                np.testing.assert_array_equal(a0[i][:n], a1[i][:n])
            np.testing.assert_array_equal(p0, p1, err_msg="%s %s" % (dot, route))
