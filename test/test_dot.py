
import os
import sys

path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if path:
    sys.path.insert(0, path)

import bslz4_to_sparse
from bslz4_to_sparse import chunk2sparseCSC, _pipeline, _testing

print("Running from", bslz4_to_sparse.__file__)
print("Usable here:", {s: [n for (i, n, ok, need) in v if ok]
                       for s, v in bslz4_to_sparse.describe().items()})

import pytest
pyFAI = pytest.importorskip("pyFAI")
try:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # pyFAI >= 2024.10
except ImportError:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # older pyFAI
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
# "single is multi with n==1": c(chunk) is a thin wrapper over c.multi with
# nframes==1, so these check that the batched and single-frame paths agree
# with each other and with the pyFAI reference.
# ---------------------------------------------------------------------------

from bslz4_to_sparse import chunk2sparse


def test_csc_multi_matches_single_and_reference():
    for name in chunks:
        dt, shp, chunklist = chunks[name]
        bufs = [c for (filt, c) in chunklist]
        c2s = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt))
        c2sm = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt))

        npx_multi, (val_multi, adr_multi), powder_multi = c2sm.multi(bufs, 1)

        for i, buf in enumerate(bufs):
            npx1, (val1, adr1), powder1 = c2s(buf, 1)
            assert npx1 == npx_multi[i], (name, i, npx1, npx_multi[i])
            s1 = set(zip(adr1[:npx1].tolist(), val1[:npx1].tolist()))
            s2 = set(zip(adr_multi[i, :npx_multi[i]].tolist(), val_multi[i, :npx_multi[i]].tolist()))
            assert s1 == s2, (name, i, "single vs multi sparse set differs")
            e = R(powder_multi[i], reference_results[i].sum_signal)
            assert e[0] < 1e-4, (name, i, "multi powder vs pyFAI", e)


def test_csc_dense_and_sparse_routes_both_match_reference():
    # the route step forces every block down one route; the counters
    # confirm only that route ran
    for label in ("dense", "sparse"):
        for name in chunks:
            dt, shp, chunklist = chunks[name]
            bufs = [c for (filt, c) in chunklist]
            c2sm = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt),
                                   pipeline={"route": label})
            _testing.reset_counters()
            npx, (outpx, adr), powder = c2sm.multi(bufs, 1)
            assert set(_testing.counters()["route"]) == {label}, (name, label, _testing.counters())
            for i in range(len(bufs)):
                e = R(powder[i], reference_results[i].sum_signal)
                assert e[0] < 1e-4, (name, label, i, e)


def test_cut_above_dtype_max_produces_empty_sparse_output():
    """A >cut threshold above the dtype's maximum is common when harvesting
    (texture tomography): no pixel can exceed it, so the sparse output must be
    empty and the powder identical.  This guards the wrapped `(T)threshold`
    cast from emitting garbage against a small wrapped cut."""
    for name in chunks:
        dt, shp, chunklist = chunks[name]
        if np.dtype(dt).kind not in ("u", "i"):
            continue
        # the >cut threshold is a C int, so only dtypes whose max it can
        # exceed are meaningful for this skip
        if int(np.iinfo(dt).max) >= (1 << 31) - 1:
            continue
        cut = int(np.iinfo(dt).max) + 1  # just above the dtype's max
        bufs = [c for (filt, c) in chunklist]
        for dot in ("csc", "bsb-csr"):
            # dense route: the >cut collect runs
            c2sm = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine,
                                   dtype=np.dtype(dt),
                                   pipeline={"route": "dense", "dot": dot})
            npx0, (v0, a0), p0 = c2sm.multi(bufs, 0)      # any pixels
            p0 = p0.copy()
            nxph, (vh, ah), ph = c2sm.multi(bufs, cut)    # > dtype max
            for i in range(len(bufs)):
                assert int(nxph[i]) == 0, (name, dot, i, "expected empty sparse output")
                assert np.array_equal(p0[i], ph[i]), (name, dot, i, "powder changed")


def test_plain_multi_matches_single():
    for name in chunks:
        dt, shp, chunklist = chunks[name]
        bufs = [c for (filt, c) in chunklist]
        c2s = chunk2sparse(1 - ai.mask, dtype=np.dtype(dt))
        c2sm = chunk2sparse(1 - ai.mask, dtype=np.dtype(dt))

        npx_multi, (val_multi, adr_multi) = c2sm.multi(bufs, 0)
        for i, buf in enumerate(bufs):
            npx1, (val1, adr1) = c2s(buf, 0)
            assert npx1 == npx_multi[i], (name, i)
            s1 = set(zip(adr1[:npx1].tolist(), val1[:npx1].tolist()))
            s2 = set(zip(adr_multi[i, :npx_multi[i]].tolist(), val_multi[i, :npx_multi[i]].tolist()))
            assert s1 == s2, (name, i, "single vs multi sparse set differs")


def _usable(step):
    return [n for (i, n, ok, need) in bslz4_to_sparse.describe()[step] if ok]


def test_every_available_untranspose_matches():
    """Each untranspose value usable here must decode identically -- they
    differ only in the bit/byte de-shuffle kernel (the lowplanes-* values are
    u16 only). Also checks a value this CPU lacks is refused rather than
    silently running upstream bitshuffle's absent-ISA stub, and that the
    removed "sse" backend is an unknown name."""
    usable = _usable("untranspose")
    assert "kcb" in usable and "scalar" in usable, usable
    reference = {}
    for uname in usable:
        for dsname in chunks:
            dt, shp, chunklist = chunks[dsname]
            if uname.startswith("lowplanes") and np.dtype(dt) != np.uint16:
                with pytest.raises(ValueError, match="is for u16 pixels"):
                    chunk2sparse(1 - ai.mask, dtype=np.dtype(dt), pipeline={"untranspose": uname})
                continue
            bufs = [c for (filt, c) in chunklist]
            npx, (val, adr) = chunk2sparse(1 - ai.mask, dtype=np.dtype(dt),
                                           pipeline={"untranspose": uname}).multi(bufs, 0)
            # Only the first npx[i] entries of each row are written; the
            # rest of the worst-case-sized buffer is uninitialised.
            got = (npx.copy(), [(val[i, :npx[i]].copy(), adr[i, :npx[i]].copy())
                                for i in range(len(bufs))])
            ref = reference.setdefault(dsname, got)
            assert np.array_equal(got[0], ref[0]), (uname, dsname, "npx differs")
            for i, ((v, a), (rv, ra)) in enumerate(zip(got[1], ref[1])):
                assert np.array_equal(v, rv), (uname, dsname, i, "values differ")
                assert np.array_equal(a, ra), (uname, dsname, i, "indices differ")

    for uname in _pipeline.NAMES["untranspose"][1:]:
        if uname not in usable:
            with pytest.raises(ValueError, match="lacks"):
                chunk2sparse(1 - ai.mask, dtype=np.uint16, pipeline={"untranspose": uname})
    with pytest.raises(ValueError, match="unknown"):
        chunk2sparse(1 - ai.mask, dtype=np.uint16, pipeline={"untranspose": "sse"})


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

        for dsname, npdt in (("u64", np.uint64), ("i64", np.int64)):
            with h5py.File(fname, "r") as f:
                ds = f[dsname]
                dense = [ds[i].copy() for i in range(3)]
                bufs = [np.frombuffer(ds.id.read_direct_chunk((i, 0, 0))[1], np.uint8)
                        for i in range(3)]
            assert dsname != "i64" or (dense[0] < 0).any(), "test setup should include negatives"

            for route in ("dense", "sparse"):
                c2sm = chunk2sparseCSC(mask2d, csc, dtype=npdt, pipeline={"route": route})
                npx, (outpx, adr), powder = c2sm.multi(bufs, 0)
                for i in range(3):
                    img = dense[i].ravel().astype(np.int64)
                    ref_sum = float(img.sum())
                    ref_sparse = set(zip(np.nonzero(img > 0)[0].tolist(), img[img > 0].tolist()))
                    got_sparse = set(zip(adr[i, :npx[i]].tolist(), outpx[i, :npx[i]].tolist()))
                    assert abs(powder[i].sum() - ref_sum) < 1e-6, (dsname, route, i, powder[i].sum(), ref_sum)
                    assert got_sparse == ref_sparse, (dsname, route, i, "sparse set differs")
    finally:
        if os.path.exists(fname):
            os.remove(fname)


# ---------------------------------------------------------------------------
# Ctypes-free chunk feeding (issue #16) and the base+offsets fast path,
# for callers who already hold the whole HDF5 file as one buffer.
#
# _gather_chunks() is exercised by every test above.
# harvest_chunk_offsets()/pack_offsets_lengths()/decode_offsets have no
# other coverage, so they are checked here against .multi reading the very
# same chunks.
# ---------------------------------------------------------------------------

from bslz4_to_sparse import harvest_chunk_offsets, pack_offsets_lengths


def test_decode_offsets_matches_call():
    # the public face of the base+offsets path, fed from an mmap of the file
    import mmap
    with open("sparsetest.h5", "rb") as fh, h5py.File("sparsetest.h5", "r") as h5f:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for name in chunks:
                dt, shp, chunklist = chunks[name]
                frames = list(range(len(chunklist)))
                offsets, lengths = pack_offsets_lengths(harvest_chunk_offsets(h5f[name]), frames)
                saved = offsets.copy()
                c2sm = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine,
                                       dtype=np.dtype(dt))
                # copies: the next call reuses the same output arrays
                npx, (val, adr), powder = c2sm.decode_offsets(mm, offsets, lengths, 1)
                npx, val, adr, powder = npx.copy(), val.copy(), adr.copy(), powder.copy()
                assert np.array_equal(offsets, saved), "caller's offsets were modified"
                npx_ref, (val_ref, adr_ref), powder_ref = c2sm.multi([c for (filt, c) in chunklist], 1)
                assert np.array_equal(npx, npx_ref), name
                for i in frames:
                    n = npx[i]
                    assert np.array_equal(adr[i, :n], adr_ref[i, :n]), (name, i)
                    assert np.array_equal(val[i, :n], val_ref[i, :n]), (name, i)
                assert np.array_equal(powder, powder_ref), name
                for i in frames:
                    e = R(powder[i], reference_results[i].sum_signal)
                    assert e[0] < 1e-4, (name, i, "decode_offsets powder vs pyFAI", e)

                bad = offsets.copy()
                bad[-1] = len(mm) - lengths[-1] + 1        # runs one byte past the end
                try:
                    c2sm.decode_offsets(mm, bad, lengths, 1)
                except Exception as e:
                    assert "-108" in str(e), e
                else:
                    raise AssertionError("an out-of-bounds chunk was not refused")
        finally:
            mm.close()


# ---------------------------------------------------------------------------
# Sparse-output ordering: the >cut list must be strictly increasing in raster
# order with no repeats (np.diff(adr) > 0). The existing suite only checks
# ordering indirectly (array_equal against an ascending reference); this pins
# it directly, and it is the property the bsb-csr bin-major layout must
# deliberately preserve with a second pixel-major pass.
# ---------------------------------------------------------------------------

def test_sparse_output_order_is_strictly_increasing():
    for name in chunks:
        dt, shp, chunklist = chunks[name]
        bufs = [c for (filt, c) in chunklist]
        for route in ("dense", "sparse"):
            c2sm = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine, dtype=np.dtype(dt),
                                   pipeline={"route": route})
            npx, (outpx, adr), powder = c2sm.multi(bufs, 1)
            for f in range(len(bufs)):
                n = int(npx[f])
                if n > 1:
                    assert np.all(np.diff(adr[f, :n].astype(np.int64)) > 0), \
                        (name, "csc", route, f, "order/repeat")
        # the plain sparse path (no dot) must be ascending too
        p2sm = chunk2sparse(1 - ai.mask, dtype=np.dtype(dt))
        npx, (outpx, adr) = p2sm.multi(bufs, 1)
        for f in range(len(bufs)):
            n = int(npx[f])
            if n > 1:
                assert np.all(np.diff(adr[f, :n].astype(np.int64)) > 0), \
                    (name, "plain", f, "order/repeat")


# ---------------------------------------------------------------------------
# SIMD mask+threshold collect values (u16/u32 only). A value this machine
# cannot run is skipped, not passed. These run under pytest only, and are
# absent from the call list at the end of this file: pytest.skip() raised
# outside a pytest run aborts collection of the whole file.
# ---------------------------------------------------------------------------

SIMD_COLLECTS = [n for n in _pipeline.NAMES["collect"][1:] if n != "scalar"]


@pytest.mark.parametrize("collect", SIMD_COLLECTS)
@pytest.mark.parametrize("untranspose", [None, "kcb"])
def test_simd_collect_matches_scalar(collect, untranspose):
    """A SIMD collect value must give byte-identical results to the scalar
    collect it replaces -- same values, same indices, same order (both visit
    pixels in ascending index order): the plain threshold-collect, and the
    CSC dense and sparse routes, with the powder also checked against pyFAI.
    untranspose None (auto: lowplanes for u16, collect fused) and kcb (the
    collect kernel runs on its own)."""
    if _pipeline.available("collect", _pipeline.NAMES["collect"].index(collect)) != 1:
        pytest.skip("%s not available on this build/CPU" % collect)

    def P(c, **kw):
        p = {"collect": c, "untranspose": untranspose}
        p.update(kw)
        return p

    for name in ("frm2", "frm4"):
        dt, shp, chunklist = chunks[name]
        bufs = [c for (filt, c) in chunklist]

        # scalar reference
        c2sm = chunk2sparse(1 - ai.mask, dtype=np.dtype(dt), pipeline=P("scalar"))
        npx_s, (val_s, adr_s) = c2sm.multi(bufs, 0)

        references = {}
        for route in ("dense", "sparse"):
            c2scm = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine,
                                    dtype=np.dtype(dt), pipeline=P("scalar", route=route))
            npx_c, (outpx_c, outadr_c), powder_c = c2scm.multi(bufs, 1)
            references[route] = (npx_c.copy(), outpx_c.copy(), outadr_c.copy(), powder_c.copy())

        # the value under test
        c2sm2 = chunk2sparse(1 - ai.mask, dtype=np.dtype(dt), pipeline=P(collect))
        _testing.reset_counters()
        npx_a, (val_a, adr_a) = c2sm2.multi(bufs, 0)
        assert set(_testing.counters()["collect"]) == {collect}, _testing.counters()
        for i in range(len(bufs)):
            n = npx_s[i]
            assert npx_a[i] == n, (collect, name, i, "plain npx differs")
            assert np.array_equal(val_a[i, :n], val_s[i, :n]), (collect, name, i, "plain values differ")
            assert np.array_equal(adr_a[i, :n], adr_s[i, :n]), (collect, name, i, "plain indices differ")

        for route in ("dense", "sparse"):
            c2scm2 = chunk2sparseCSC(1 - ai.mask, ai.engines[method].engine,
                                     dtype=np.dtype(dt), pipeline=P(collect, route=route))
            npx_c2, (outpx_c2, outadr_c2), powder_c2 = c2scm2.multi(bufs, 1)
            npx_ref, outpx_ref, outadr_ref, powder_ref = references[route]
            for i in range(len(bufs)):
                e = R(powder_c2[i], reference_results[i].sum_signal)
                assert e[0] < 1e-4, (collect, name, route, i, "powder vs pyFAI", e)
                n = npx_c2[i]
                assert n == npx_ref[i], (collect, name, route, i, "csc npx differs")
                assert np.array_equal(outpx_c2[i, :n], outpx_ref[i, :n]), (collect, name, route, i, "csc values differ")
                assert np.array_equal(outadr_c2[i, :n], outadr_ref[i, :n]), (collect, name, route, i, "csc indices differ")


def test_bad_collect_values_refused():
    # the removed tiers ("avx512", "avx2") are unknown names; a SIMD collect
    # needs u16/u32 pixels; route/dot need a matrix
    for bad in ("avx512", "avx2", 7):
        with pytest.raises(ValueError, match="unknown"):
            chunk2sparse(1 - ai.mask, dtype=np.uint16, pipeline={"collect": bad})
    with pytest.raises(ValueError, match="u16/u32"):
        chunk2sparse(1 - ai.mask, dtype=np.uint8, pipeline={"collect": "sse2"})
    with pytest.raises(ValueError, match="no matrix"):
        chunk2sparse(1 - ai.mask, dtype=np.uint16, pipeline={"route": "dense"})


test_csc_multi_matches_single_and_reference()
test_csc_dense_and_sparse_routes_both_match_reference()
test_plain_multi_matches_single()
test_u64_i64_and_signed_negative_values()
test_decode_offsets_matches_call()
test_every_available_untranspose_matches()
print("all multi-frame / dense-sparse-route / u64-i64 / decode_offsets tests passed")
print("(SIMD collect tests only run under pytest -- see test_simd_collect_matches_scalar)")
