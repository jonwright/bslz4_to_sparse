

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
