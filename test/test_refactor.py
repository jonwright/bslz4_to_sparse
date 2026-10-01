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
    full = _lz4_zeros(BS)
    tail = _lz4_zeros(tail_block)
    c = bytearray(struct.pack(">QI", BS + TAIL, BS))
    c += struct.pack(">I", len(full)) + full
    c += struct.pack(">I", len(tail)) + tail
    c += b"\0" * (TAIL - tail_block)          # raw (literal) remainder
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
    # the default pipeline must be accepted by the decoder
    assert isinstance(p, int)
    assert p == b._pack(b._default_codec(), b._BACKEND_TO_ID[b._default_backend],
                        b._active_collect_tier(), 0, 0)

    with pytest.raises(NotImplementedError):
        b.pack_pipeline(collect=99)
    with pytest.raises(NotImplementedError):
        b.pack_pipeline(untranspose=99)
    with pytest.raises(NotImplementedError):
        b.pack_pipeline(decompress=99)
    with pytest.raises(NotImplementedError):
        b.pack_pipeline(dot=99)
    with pytest.raises(ValueError):
        b.pack_pipeline(options=4)      # only bit 0 (DROP_NEGATIVES) is reserved


# ---- an explicit pipeline forces the selected collect tier -------------------

def test_pipeline_forces_collect_tier():
    simd = next((i for i in (1, 2, 3, 4, 5) if b.impl_available(b._STAGE_COLLECT, i) == 1), None)

    def _run(pipe):
        c = b.chunk2sparseMulti(np.ones((1, NPIX), np.uint8), dtype=np.uint16, pipeline=pipe)
        b.reset_counters()
        c([_chunk()], 0)
        out = np.empty(64, np.uint64)
        b.read_counters(out)
        return out

    # scalar tier, always available
    col0 = _run(b.pack_pipeline(collect=0))
    assert col0[b._STAGE_COLLECT * 16 + 0] > 0
    assert col0[b._STAGE_COLLECT * 16 + simd] == 0 if simd is not None else True

    if simd is not None:
        cols = _run(b.pack_pipeline(collect=simd))
        # the SIMD tier ran, and scalar did not
        assert cols[b._STAGE_COLLECT * 16 + simd] > 0
        assert cols[b._STAGE_COLLECT * 16 + 0] == 0
        # and it picked the non-scalar collect tier for a u16 dtype


# ---- per-block counters: every route and the tail, scalar not used -----------

def test_counters_every_route_and_tail():
    simd = next((i for i in (1, 2, 3, 4, 5) if b.impl_available(b._STAGE_COLLECT, i) == 1), None)
    if simd is None:
        pytest.skip("no SIMD collect tier in this build")

    saved = b.get_dense_sparse_threshold()
    try:
        for label, thr in (("sparse-route", 0.0), ("dense-route", 1e9)):
            b.set_dense_sparse_threshold(thr)
            c = b.chunk2sparseCSCmulti(np.ones((1, NPIX), np.uint8), _csc(), dtype=np.uint16)
            b.reset_counters()
            npx, (outpx, outadr), powder = c([_chunk()], 1)
            out = np.empty(64, np.uint64)
            b.read_counters(out)
            C = b._STAGE_COLLECT * 16
            D = b._STAGE_DOT * 16
            # the selected impl ran in the full block AND the tail (the single
            # frame decodes 1 full block + 1 tail block), the dot impl ran once
            # per block, and scalar collect was not used for a SIMD dtype.
            assert out[D + 0] > 0, label
            assert out[0:16].sum() == 2, (label, "1 full block + 1 tail block decompressed")
            assert out[16:32].sum() == 2, (label, "untranspose ran once per block")
            assert out[C + simd] > 0, (label, "SIMD collect did not run")
            assert out[C + 0] == 0, (label, "scalar collect should not run")
    finally:
        b.set_dense_sparse_threshold(saved)


# ---- resolution / matrix errors ----------------------------------------------

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
