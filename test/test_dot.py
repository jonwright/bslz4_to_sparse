

from bslz4_to_sparse import chunk2sparseCSC

import pyFAI.integrator.azimuthal
import numpy as np
import h5py
import hdf5plugin
import os
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


ai = pyFAI.integrator.azimuthal.AzimuthalIntegrator(
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
method = [ e for e in ai.engines if ( e.algo == 'CSC' ) ][0]


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
# are thin wrappers over the same bslz4_decode_multi/bslz4_csc_decode_multi
# templates chunk2sparseMulti/chunk2sparseCSCmulti call with nframes>1, so
# these tests check the batched and single-frame paths agree with each
# other and with the pyFAI reference, not two independently-implemented
# decoders.
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
        for threshold, label in ((1e9, "sparse"), (0.0, "dense")):
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


test_csc_multi_matches_single_and_reference()
test_csc_dense_and_sparse_routes_both_match_reference()
test_plain_multi_matches_single()
test_u64_i64_and_signed_negative_values()
print("all multi-frame / dense-sparse-route / u64-i64 tests passed")
