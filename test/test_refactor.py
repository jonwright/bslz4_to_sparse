"""
Phase 2 gate for the refactor (see plan.md sections 5, 11, 12):

  * the Python and C stage/implementation id tables agree;
  * pack_pipeline rejects unknown ids / option bits;
  * an explicit pipeline actually forces the selected collect tier;
  * the per-block counters show the selected impl ran in every route and in
    the tail, and that the scalar collect tier was not picked for a SIMD dtype;
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


# ---- Python/C id-table agreement (plan.md section 5) -----------------------

def test_ids_and_c_agree():
    # ids, names and dtype masks are defined in both Python (src/__init__.py)
    # and C (bslz4_registry.h); every (stage, id) must be known to C and no
    # unknown id may be reported as usable.
    known = {
        b._STAGE_DECOMPRESS: (b.CODEC_LZ4, b.CODEC_ZSTD),
        b._STAGE_UNTRANSPOSE: tuple(b._BACKEND_TO_ID.values()),
        b._STAGE_COLLECT: tuple(b._COLLECT_ID_TO_NAME),
        b._STAGE_DOT: (0,),
    }
    for stage, ids in known.items():
        for i in ids:
            assert b.impl_available(stage, i) != -1, (stage, i)
        assert b.impl_available(stage, 99) == -1, stage
        assert b.impl_available(stage, -1) == -1, stage


# ---- pack_pipeline validation ------------------------------------------------

def test_pack_pipeline_defaults_and_rejects():
    p = b.pack_pipeline()
    # the default stages array must be accepted by the decoder
    assert isinstance(p, np.ndarray) and p.dtype == np.uint16 and p.shape == (5,)
    np.testing.assert_array_equal(
        p,
        b._stages(b._default_codec(), b._BACKEND_TO_ID[b._default_backend],
                  b._active_collect_tier(), 0, 0),
    )

    with pytest.raises(NotImplementedError):
        b.pack_pipeline(collect=99)
    with pytest.raises(NotImplementedError):
        b.pack_pipeline(untranspose=99)
    with pytest.raises(NotImplementedError):
        b.pack_pipeline(decompress=99)
    with pytest.raises(NotImplementedError):
        b.pack_pipeline(dot=99)
    with pytest.raises(ValueError):
        b.pack_pipeline(options=64)     # bits 0-5 are in use (see bslz4_common.h BSLZ4_OPT_*)


# ---- an explicit pipeline forces the selected collect tier -------------------

def test_pipeline_forces_collect_tier():
    simd = next((i for i in (1, 2, 3, 4, 5) if b.impl_available(b._STAGE_COLLECT, i) == 1), None)

    def _run(pipe):
        c = b.chunk2sparseMulti(np.ones((1, NPIX), np.uint8), dtype=np.uint16, pipeline=pipe)
        b.reset_counters()
        c([_chunk()], 0)
        out = np.empty(4 * b._COUNTER_SLOTS, np.uint64)
        b.read_counters(out)
        return out

    # scalar tier, always available
    col0 = _run(b.pack_pipeline(collect=0))
    assert col0[b._STAGE_COLLECT * b._COUNTER_SLOTS + 0] > 0
    assert col0[b._STAGE_COLLECT * b._COUNTER_SLOTS + simd] == 0 if simd is not None else True

    if simd is not None:
        cols = _run(b.pack_pipeline(collect=simd))
        # the SIMD tier ran, and scalar did not
        assert cols[b._STAGE_COLLECT * b._COUNTER_SLOTS + simd] > 0
        assert cols[b._STAGE_COLLECT * b._COUNTER_SLOTS + 0] == 0
        # and it picked the non-scalar collect tier for a u16 dtype


# ---- per-block counters: every route and the tail, scalar not used -----------

def test_counters_every_route_and_tail():
    simd = b._active_collect_tier()     # the default tier the integrator uses
    if simd == 0:
        pytest.skip("no SIMD collect tier in this build")

    saved = b.get_dense_sparse_threshold()
    try:
        for label, thr in (("sparse-route", 0.0), ("dense-route", 1e9)):
            b.set_dense_sparse_threshold(thr)
            c = b.chunk2sparseCSCmulti(np.ones((1, NPIX), np.uint8), _csc(), dtype=np.uint16)
            b.reset_counters()
            npx, (outpx, outadr), powder = c([_chunk()], 1)
            out = np.empty(4 * b._COUNTER_SLOTS, np.uint64)
            b.read_counters(out)
            C = b._STAGE_COLLECT * b._COUNTER_SLOTS
            D = b._STAGE_DOT * b._COUNTER_SLOTS
            # the selected impl ran in the full block AND the tail (the single
            # frame decodes 1 full block + 1 tail block), the dot impl ran once
            # per block, and scalar collect was not used for a SIMD dtype.
            assert out[D + 0] > 0, label
            assert out[0:b._COUNTER_SLOTS].sum() == 2, (label, "1 full block + 1 tail block decompressed")
            assert out[b._COUNTER_SLOTS:2 * b._COUNTER_SLOTS].sum() == 2, (label, "untranspose ran once per block")
            assert out[C + simd] > 0, (label, "SIMD collect did not run")
            assert out[C + 0] == 0, (label, "scalar collect should not run")
    finally:
        b.set_dense_sparse_threshold(saved)


# ---- tail data reaches the sparse output -------------------------------------
# _chunk() has a non-zero raw literal remainder (4 uint16 pixels, value 5) and
# zeros everywhere else, so cut=1 must sparsify exactly those 4 tail pixels.

def test_tail_remainder_pixels_are_sparsified():
    c = b.chunk2sparseMulti(np.ones((1, NPIX), np.uint8), dtype=np.uint16)
    npx, (vals, adr) = c([_chunk()], 1)
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
    ref = b.normalise_matrix(csc, npix=5)
    for name, m in (("csr", csr), ("bsr", bsr)):
        nm = b.normalise_matrix(m, npix=5)
        np.testing.assert_array_equal(nm.indptr, ref.indptr)
        np.testing.assert_array_equal(nm.indices, ref.indices)
        np.testing.assert_array_equal(nm.data, ref.data)

def test_resolution_errors():
    _ext = b._ext
    ptr = np.empty(1, np.int64)
    ptr[0] = 0
    lens = np.array([len(_chunk())], np.int32)
    mask = np.zeros(NPIX, np.uint8)

    # dtype out of range -> ERR_DTYPE (-113)
    assert _ext.sparsify(ptr, lens, mask, np.zeros(NPIX, np.uint16),
                         np.empty(NPIX, np.uint32), np.empty(1, np.int32), 0,
                         np.zeros(3 * BS, np.uint8), np.empty(1, np.int64),
                         11, b.pack_pipeline()) == -113

    # int64 indices / indptr arrays are rejected at the boundary
    ws = np.zeros(3 * BS + (BS // NB) * (4 + NB), np.uint8)
    data = np.ones(NPIX, np.float32)
    ind64 = np.arange(NPIX, dtype=np.int64)
    indptr = np.arange(NPIX + 1, dtype=np.int32)
    with pytest.raises(Exception):
        _ext.sparsify_and_dot(ptr, lens, mask, np.zeros(NPIX, np.uint16),
                              np.empty(NPIX, np.uint32), np.empty(1, np.int32), 0,
                              np.zeros(NBINS, np.float64), data, ind64, indptr,
                              ws, np.empty(1, np.int64), NBINS, 1e9, 1, b.pack_pipeline())

    # wrong matrix size (indptr not npix+1) is rejected
    with pytest.raises(Exception):
        _ext.sparsify_and_dot(ptr, lens, mask, np.zeros(NPIX, np.uint16),
                              np.empty(NPIX, np.uint32), np.empty(1, np.int32), 0,
                              np.zeros(NBINS, np.float64), data,
                              np.arange(NPIX, dtype=np.uint32),
                              np.arange(NPIX, dtype=np.uint32),
                              ws, np.empty(1, np.int64), NBINS, 1e9, 1, b.pack_pipeline())


# ---- dot variants: the per-dtype/kernel mechanism ---------------------------
# dot id 0 = csc "dot then threshold" (a dot.dense pass, then a separate >cut
# collect); dot id 1 = a new variant that fuses the dense CSC loop with the
# >cut sparsify into a single pass.  Both must give bit-identical powder and
# sparse output, and the counters must show the *selected* dot impl ran.

def _variant_chunks(h5py, hdf5plugin, nf=2, h=8, w=8):
    """A tiny real bitshuffle-lz4 dataset with a mix of above/below-cut pixels
    so both dot variants have real work to do."""
    arr = (np.arange(nf * h * w, dtype=np.uint32).reshape(nf, h, w) % 8).astype(np.uint16)
    fn = "/tmp/_bslz4_variant.h5"
    with h5py.File(fn, "w") as f:
        f.create_dataset("d", data=arr, chunks=(1, h, w),
                         **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
    with h5py.File(fn, "r") as f:
        return [f["d"].id.read_direct_chunk((i, 0, 0))[1] for i in range(nf)]


def test_dot_variants_fused_and_plain_agree():
    h5py = pytest.importorskip("h5py")
    hdf5plugin = pytest.importorskip("hdf5plugin")

    # the fused variant must be a known, available dot implementation
    assert b.impl_available(b._STAGE_DOT, 1) == 1
    assert b.impl_available(b._STAGE_DOT, 0) == 1
    assert b.impl_available(b._STAGE_DOT, 99) == -1
    b.pack_pipeline(dot=1)   # must not raise

    chunks = _variant_chunks(h5py, hdf5plugin)
    mask = np.ones((1, 8 * 8), np.uint8)
    npix = mask.size
    nbins = 4
    indptr = np.arange(npix + 1, dtype=np.uint32)
    indices = (np.arange(npix) % nbins).astype(np.uint32)
    data = np.ones(npix, np.float32)
    csc = type("_CSC", (), {"indptr": indptr, "indices": indices, "data": data,
                            "shape": (nbins, npix)})()

    def _run(dot, thr):
        b.set_dense_sparse_threshold(thr)
        c = b.chunk2sparseCSCmulti(mask, csc, dtype=np.uint16, dot=dot)
        b.reset_counters()
        npx, (vals, adr), powder = c(chunks, 1)
        out = np.empty(4 * b._COUNTER_SLOTS, np.uint64)
        b.read_counters(out)
        D = b._STAGE_DOT * b._COUNTER_SLOTS
        assert out[D + dot] > 0, (dot, "selected dot impl did not run")
        assert out[D + 1 - dot] == 0, (dot, "the other dot impl should not run")
        return npx, vals[0:], adr[0:], powder

    def _check_equal(npx_a, v_a, a_a, p_a, npx_b, v_b, a_b, p_b):
        np.testing.assert_array_equal(npx_a, npx_b)
        for i in range(len(npx_a)):
            n = int(npx_a[i])
            np.testing.assert_array_equal(v_a[i][:n], v_b[i][:n])
            np.testing.assert_array_equal(a_a[i][:n], a_b[i][:n])
        np.testing.assert_array_equal(p_a, p_b)

    saved = b.get_dense_sparse_threshold()
    try:
        # dense route: dot=0 (dot then threshold) vs dot=1 (fused) -- identical
        npx0, v0, a0, p0 = _run(0, 1e9)
        npx1, v1, a1, p1 = _run(1, 1e9)
        _check_equal(npx0, v0, a0, p0, npx1, v1, a1, p1)
        # sparse route: both dot ids run the same sparse kernel -- identical too
        npxs0, vs0, as0, ps0 = _run(0, 0.0)
        npxs1, vs1, as1, ps1 = _run(1, 0.0)
        _check_equal(npxs0, vs0, as0, ps0, npxs1, vs1, as1, ps1)
    finally:
        b.set_dense_sparse_threshold(saved)
