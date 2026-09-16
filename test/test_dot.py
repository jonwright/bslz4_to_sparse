
import os
import sys

path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if path:
    sys.path.insert(0, path)

import bslz4_to_sparse
from bslz4_to_sparse import chunk2sparseCSC

print("Running from", bslz4_to_sparse.__file__)

try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # pyFAI >= 2024.10
except ImportError:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # older pyFAI
import pyFAI
import numpy as np
import h5py
import hdf5plugin
import timeit

NFR = 5

dtypes = np.uint8, np.uint16,  np.uint32

if not os.path.exists( 'sparsetest.h5' ):
    s = (NFR, 2162, 2068)
    testdata = np.random.poisson( 0.01, s )
    with h5py.File( 'sparsetest.h5', 'a' ) as h5f:
        for dt in dtypes:
            data = testdata.astype( dt )
            name = f'frm{data.itemsize}'
            dset = h5f.create_dataset( name, data = data,
                                       chunks = (1, s[1], s[2]),
                                       compression = 32008,
                                       compression_opts = (0, 2), )


with h5py.File('sparsetest.h5','r') as h5f:
    chunks = {}
    for name in list(h5f):
        dset = h5f[name]
        chunks[name] = ( dset.dtype, dset.shape,
                         [ dset.id.read_direct_chunk( (i,0,0) )
                           for i in range(dset.shape[0]) ] )
    testdata = h5f['frm2'][:]
    
                                             
npt = 1500


ai = AzimuthalIntegrator(
    dist = 0.25,
    poni1 = 0.07,
    poni2 = 0.08,
    rot1 = 0.01,
    rot2 = 0.02,
    rot3 = 0.03,
    pixel1 = 75e-6,
    pixel2 = 75e-6,
    wavelength = 12.3984/43.57,
    detector = pyFAI.detector_factory("Eiger2CdTe_4M"),
    )


method = ('bbox','CSC','python')
result = ai.integrate1d( testdata[0],
                1500, method = method )
reference_results = [ ai.integrate1d( frm, npt, method=method )
                      for frm in testdata ]
method = [ e for e in ai.engines if ( e.algorithm == 'CSC' ) ][0]


def R( y1, y2 ):
    e = abs(y1 - y2)
    j = np.argmax(e)
    return e[j], e[j]/y2[j], j


def testfun( ):
    for name in chunks:
        dt, shp, chunklist = chunks[name]
        c2s = chunk2sparseCSC( 1-ai.mask,
                               ai.engines[method].engine,
                               dtype=np.dtype(dt) )
        tims = []
        for i,(filt, chunk) in enumerate(chunklist):
            t0 = timeit.default_timer()
            npx, (val, idx), powder = c2s( chunk, 1 )
            t1  = timeit.default_timer()
            tims.append( t1 - t0 )
            e = R(powder, reference_results[i].sum_signal )
            assert e[0] < 1e-4, e
        tavg = np.average(tims)
        j = np.argmax(e)
        print(f'{name} {dt} {tavg*1e3:.3f} ms {1/tavg:.1f} fps/core, max error abs, rel',e)
        
testfun()


# ---------------------------------------------------------------------------
# Multi-frame batching, dense/sparse routing, and u64/i64 dtype coverage.
#
# "single is multi with n==1" (bslz4_core.hpp): chunk2sparse/chunk2sparseCSC
# are thin wrappers over the same templates with nframes==1, so these check
# that the batched and single-frame paths agree with each other and with the
# pyFAI reference.
# ---------------------------------------------------------------------------

from bslz4_to_sparse import chunk2sparse, chunk2sparseCSCmulti, chunk2sparseMulti
from bslz4_to_sparse import set_dense_sparse_threshold, get_dense_sparse_threshold


def test_csc_multi_matches_single_and_reference():
    for name in chunks:
        dt, shp, chunklist = chunks[name]
        bufs = [c for (filt, c) in chunklist]
        c2s = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt))
        c2sm = chunk2sparseCSCmulti(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt))

        npx_multi, (val_multi, adr_multi), powder_multi = c2sm(bufs, 1)

        for i, buf in enumerate(bufs):
            npx1, (val1, adr1), powder1 = c2s(buf, 1)
            assert npx1 == npx_multi[i], (name, i, npx1, npx_multi[i])
            s1 = set(zip(adr1[:npx1].tolist(), val1[:npx1].tolist()))
            s2 = set(zip(adr_multi[i, :npx_multi[i]].tolist(), val_multi[i, :npx_multi[i]].tolist()))
            assert s1 == s2, (name, i, "single vs multi sparse set differs")
            e = R(powder_multi[i], reference_results[i].sum_signal)
            assert e[0] < 1e-4, (name, i, "multi powder vs pyFAI", e)


def test_csc_dense_and_sparse_routes_both_match_reference():
    saved = get_dense_sparse_threshold()
    try:
        # bslz4_csc_dense_sparse_threshold() is compared against each
        # block's own compression RATIO (blocksize/compressed_bytes):
        # ratio > threshold picks the sparse (compaction) route, so a
        # huge threshold like 1e9 is virtually never exceeded -- that
        # forces DENSE -- and 0.0 is virtually always exceeded -- that
        # forces SPARSE. (Backwards from what the numbers might suggest
        # at a glance: this bit a benchmark script once already.)
        for threshold, label in ((1e9, "dense"), (0.0, "sparse")):
            set_dense_sparse_threshold(threshold)
            for name in chunks:
                dt, shp, chunklist = chunks[name]
                bufs = [c for (filt, c) in chunklist]
                c2sm = chunk2sparseCSCmulti(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt))
                npx, (outpx, adr), powder = c2sm(bufs, 1)
                for i in range(len(bufs)):
                    e = R(powder[i], reference_results[i].sum_signal)
                    assert e[0] < 1e-4, (name, label, i, e)
    finally:
        set_dense_sparse_threshold(saved)


def test_plain_multi_matches_single():
    for name in chunks:
        dt, shp, chunklist = chunks[name]
        bufs = [c for (filt, c) in chunklist]
        c2s = chunk2sparse(1 - ai.mask, dtype=np.dtype(dt))
        c2sm = chunk2sparseMulti(1 - ai.mask, dtype=np.dtype(dt))

        npx_multi, (val_multi, adr_multi) = c2sm(bufs, 0)
        for i, buf in enumerate(bufs):
            npx1, (val1, adr1) = c2s(buf, 0)
            assert npx1 == npx_multi[i], (name, i)
            s1 = set(zip(adr1[:npx1].tolist(), val1[:npx1].tolist()))
            s2 = set(zip(adr_multi[i, :npx_multi[i]].tolist(), val_multi[i, :npx_multi[i]].tolist()))
            assert s1 == s2, (name, i, "single vs multi sparse set differs")


def test_every_available_backend_matches():
    """Each untranspose backend available_backends() advertises must decode
    identically -- they differ only in the bit/byte de-shuffle kernel. Also
    checks a backend it does not advertise is refused rather than silently
    running upstream bitshuffle's absent-ISA stub."""
    from bslz4_to_sparse import available_backends, set_backend

    usable = available_backends()
    assert "kcb" in usable and "scal" in usable, usable
    reference = {}
    try:
        for name in usable:
            set_backend(name)
            for dsname in chunks:
                dt, shp, chunklist = chunks[dsname]
                bufs = [c for (filt, c) in chunklist]
                npx, (val, adr) = chunk2sparseMulti(1 - ai.mask, dtype=np.dtype(dt))(bufs, 0)
                # Only the first npx[i] entries of each row are written; the
                # rest of the worst-case-sized buffer is uninitialised.
                got = (npx.copy(), [(val[i, :npx[i]].copy(), adr[i, :npx[i]].copy())
                                    for i in range(len(bufs))])
                ref = reference.setdefault(dsname, got)
                assert np.array_equal(got[0], ref[0]), (name, dsname, "npx differs")
                for i, ((v, a), (rv, ra)) in enumerate(zip(got[1], ref[1])):
                    assert np.array_equal(v, rv), (name, dsname, i, "values differ")
                    assert np.array_equal(a, ra), (name, dsname, i, "indices differ")
    finally:
        set_backend(None)

    for name in ("kcb", "sse", "neon", "scal"):
        if name not in usable:
            try:
                set_backend(name)
            except ValueError:
                pass
            else:
                set_backend(None)
                raise AssertionError("set_backend(%r) should have been refused" % name)


def _make_tiny_csc(npix, nbins, seed=1):
    """1 entry/pixel, weight 1.0 -- sum(csc) must equal sum(masked image)
    exactly, including negative pixel values, independent of pyFAI."""
    class CSC:
        pass
    rng = np.random.default_rng(seed)
    csc = CSC()
    csc.indptr = np.arange(npix + 1, dtype=np.uint32)
    csc.indices = rng.integers(0, nbins, size=npix).astype(np.uint32)
    csc.data = np.ones(npix, dtype=np.float32)
    csc.shape = (nbins,)
    return csc


def test_u64_i64_and_signed_negative_values():
    """u64/i64 dtype coverage, and the signed/unsigned zero-compare fix:
    the CSC compaction gate must be val != 0 in T's own domain, not a cast
    through an unsigned type (which would silently drop negative pixels).
    Self-contained (no pyFAI dependency): checks sum(csc) == sum(masked
    image) and the sparse set against a plain numpy reference, for both
    routes, for uint64 and for int64 with genuinely negative values."""
    NI, NJ = 48, 40
    npix = NI * NJ
    mask2d = np.ones((NI, NJ), np.uint8)
    csc = _make_tiny_csc(npix, nbins=8)
    rng = np.random.default_rng(2)

    fname = "u64i64_test.h5"
    try:
        with h5py.File(fname, "w") as f:
            u64 = rng.poisson(1.0, (3, NI, NJ)).astype(np.uint64)
            i64 = (rng.poisson(1.0, (3, NI, NJ)).astype(np.int64) - 1)  # forces negatives
            f.create_dataset("u64", data=u64, chunks=(1, NI, NJ),
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4", clevel=0))
            f.create_dataset("i64", data=i64, chunks=(1, NI, NJ),
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4", clevel=0))

        saved = get_dense_sparse_threshold()
        try:
            for dsname, npdt in (("u64", np.uint64), ("i64", np.int64)):
                with h5py.File(fname, "r") as f:
                    ds = f[dsname]
                    dense = [ds[i].copy() for i in range(3)]
                    bufs = [np.frombuffer(ds.id.read_direct_chunk((i, 0, 0))[1], np.uint8)
                            for i in range(3)]
                assert dsname != "i64" or (dense[0] < 0).any(), "test setup should include negatives"

                for threshold in (1e9, 0.0):  # sparse route, dense route
                    set_dense_sparse_threshold(threshold)
                    c2sm = chunk2sparseCSCmulti(mask2d, csc, dtype=npdt)
                    npx, (outpx, adr), powder = c2sm(bufs, 0)
                    for i in range(3):
                        img = dense[i].ravel().astype(np.int64)
                        ref_sum = float(img.sum())
                        ref_sparse = set(zip(np.nonzero(img > 0)[0].tolist(), img[img > 0].tolist()))
                        got_sparse = set(zip(adr[i, :npx[i]].tolist(), outpx[i, :npx[i]].tolist()))
                        assert abs(powder[i].sum() - ref_sum) < 1e-6, (dsname, threshold, i, powder[i].sum(), ref_sum)
                        assert got_sparse == ref_sparse, (dsname, threshold, i, "sparse set differs")
        finally:
            set_dense_sparse_threshold(saved)
    finally:
        if os.path.exists(fname):
            os.remove(fname)


# ---------------------------------------------------------------------------
# Ctypes-free chunk feeding (issue #16) and the base+offsets fast path,
# for callers who already hold the whole HDF5 file as one buffer.
#
# note_chunk()/_gather_chunks() are exercised by every test above.
# harvest_chunk_offsets()/pack_offsets_lengths()/bslz4_csc_multi_base_*
# have no other coverage, so they are checked here against
# chunk2sparseCSCmulti reading the very same chunks.
# ---------------------------------------------------------------------------

from bslz4_to_sparse import harvest_chunk_offsets, pack_offsets_lengths
from bslz4_to_sparse import _BSLZ4_CSC_MULTI_BASE, _suffix_for_dtype, _workspace_bytes_csc


def test_csc_multi_base_matches_multi():
    with open("sparsetest.h5", "rb") as fh:
        base = fh.read()

    for name in chunks:
        dt, shp, chunklist = chunks[name]
        nframes = len(chunklist)
        frames = list(range(nframes))

        with h5py.File("sparsetest.h5", "r") as h5f:
            frame_offsets = harvest_chunk_offsets(h5f[name])
        offsets, lengths = pack_offsets_lengths(frame_offsets, frames)

        csc = ai.engines[method].engine
        mask = (1 - ai.mask).ravel()
        npix = mask.size
        nbins = csc.shape[0] if hasattr(csc, "shape") else csc.bins

        outpx = np.empty((nframes, npix), dt)
        output_adr = np.empty((nframes, npix), np.uint32)
        npx_out = np.empty(nframes, np.int32)
        powder = np.empty((nframes, nbins), np.float64)
        cursors = np.empty(nframes, np.int64)
        workspace = np.empty(_workspace_bytes_csc(chunklist[0][1], np.dtype(dt).itemsize), np.uint8)

        fn = _BSLZ4_CSC_MULTI_BASE[_suffix_for_dtype(dt)]
        ret = fn(base, offsets, lengths, mask, outpx.ravel(), output_adr.ravel(),
                  npx_out, 1, powder.ravel(), csc.data, csc.indices, csc.indptr,
                  workspace, cursors, nbins, 2)
        assert ret >= 0, (name, ret)

        c2sm = chunk2sparseCSCmulti(1 - ai.mask, csc, dtype=np.dtype(dt))
        bufs = [c for (filt, c) in chunklist]
        npx_ref, (val_ref, adr_ref), powder_ref = c2sm(bufs, 1)

        for i in range(nframes):
            assert npx_out[i] == npx_ref[i], (name, i)
            s1 = set(zip(output_adr[i, :npx_out[i]].tolist(), outpx[i, :npx_out[i]].tolist()))
            s2 = set(zip(adr_ref[i, :npx_ref[i]].tolist(), val_ref[i, :npx_ref[i]].tolist()))
            assert s1 == s2, (name, i, "multi_base vs multi sparse set differs")
            np.testing.assert_allclose(powder[i], powder_ref[i])


# ---------------------------------------------------------------------------
# SIMD mask+threshold collect tiers (u16/u32 only). A tier this machine
# cannot run is skipped, not passed. These run under pytest only, and are
# absent from the call list at the end of this file: pytest.skip() raised
# outside a pytest run aborts collection of the whole file.
# ---------------------------------------------------------------------------

from bslz4_to_sparse import (
    avx512_collect_available, get_avx512_collect, set_avx512_collect,
    avx2_collect_available, get_avx2_collect, set_avx2_collect,
    sse2_collect_available, get_sse2_collect, set_sse2_collect,
    vsx_collect_available, get_vsx_collect, set_vsx_collect,
)


def _check_collect_tier_matches_scalar(tier, available, getter, setter):
    """Shared body for test_{avx512,avx2,sse2,vsx}_collect_matches_scalar:
    a SIMD collect tier must give byte-identical results to the scalar
    path it replaces -- same values, same indices, same order (both
    visit pixels in ascending index order): bslz4_multi_u16/u32's plain
    threshold-collect, and bslz4_csc_multi_u16/u32's sparse-route
    compaction (the dense route never calls the collect kernel at all,
    so only checked here for its own pyFAI-vs-reference agreement)."""
    if not available():
        import pytest  # here, not at module top: a direct `python3
        # test_dot.py` run must not need pytest installed
        pytest.skip(f"{tier} not available on this build/CPU")
    # Best available tier is on by default now (decided in
    # bslz4_collect_simd.hpp, not here) -- don't assume either state,
    # just restore whatever it was when this test started.
    original_state = getter()

    saved_threshold = get_dense_sparse_threshold()
    try:
        for name in ("frm2", "frm4"):
            dt, shp, chunklist = chunks[name]
            bufs = [c for (filt, c) in chunklist]

            setter(False)
            c2sm = chunk2sparseMulti(1 - ai.mask, dtype=np.dtype(dt))
            npx_s, (val_s, adr_s) = c2sm(bufs, 0)
            npx_s, val_s, adr_s = npx_s.copy(), val_s.copy(), adr_s.copy()

            references = {}
            for threshold in (1e9, 0.0):  # dense route, sparse route (see the
                                           # comment in the dense/sparse test above)
                set_dense_sparse_threshold(threshold)
                c2scm = chunk2sparseCSCmulti(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt))
                npx_c, (outpx_c, outadr_c), powder_c = c2scm(bufs, 1)
                references[threshold] = (npx_c.copy(), outpx_c.copy(), outadr_c.copy(), powder_c.copy())

            setter(True)
            try:
                c2sm2 = chunk2sparseMulti(1 - ai.mask, dtype=np.dtype(dt))
                npx_a, (val_a, adr_a) = c2sm2(bufs, 0)
                for i in range(len(bufs)):
                    n = npx_s[i]
                    assert npx_a[i] == n, (tier, name, i, "plain npx differs")
                    assert np.array_equal(val_a[i, :n], val_s[i, :n]), (tier, name, i, "plain values differ")
                    assert np.array_equal(adr_a[i, :n], adr_s[i, :n]), (tier, name, i, "plain indices differ")

                for threshold, label in ((1e9, "dense"), (0.0, "sparse")):
                    set_dense_sparse_threshold(threshold)
                    c2scm2 = chunk2sparseCSCmulti(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt))
                    npx_c2, (outpx_c2, outadr_c2), powder_c2 = c2scm2(bufs, 1)
                    npx_ref, outpx_ref, outadr_ref, powder_ref = references[threshold]
                    for i in range(len(bufs)):
                        e = R(powder_c2[i], reference_results[i].sum_signal)
                        assert e[0] < 1e-4, (tier, name, label, i, "powder vs pyFAI", e)
                        n = npx_c2[i]
                        assert n == npx_ref[i], (tier, name, label, i, "csc npx differs")
                        assert np.array_equal(outpx_c2[i, :n], outpx_ref[i, :n]), (tier, name, label, i, "csc values differ")
                        assert np.array_equal(outadr_c2[i, :n], outadr_ref[i, :n]), (tier, name, label, i, "csc indices differ")
            finally:
                setter(original_state)
    finally:
        set_dense_sparse_threshold(saved_threshold)


def test_avx512_collect_matches_scalar():
    _check_collect_tier_matches_scalar("avx512", avx512_collect_available, get_avx512_collect, set_avx512_collect)


def test_avx2_collect_matches_scalar():
    _check_collect_tier_matches_scalar("avx2", avx2_collect_available, get_avx2_collect, set_avx2_collect)


def test_sse2_collect_matches_scalar():
    _check_collect_tier_matches_scalar("sse2", sse2_collect_available, get_sse2_collect, set_sse2_collect)


def test_vsx_collect_matches_scalar():
    _check_collect_tier_matches_scalar("vsx", vsx_collect_available, get_vsx_collect, set_vsx_collect)


test_csc_multi_matches_single_and_reference()
test_csc_dense_and_sparse_routes_both_match_reference()
test_plain_multi_matches_single()
test_u64_i64_and_signed_negative_values()
test_csc_multi_base_matches_multi()
test_every_available_backend_matches()
print("all multi-frame / dense-sparse-route / u64-i64 / multi_base tests passed")
print("(SIMD collect tier tests only run under pytest -- see test_*_collect_matches_scalar)")
