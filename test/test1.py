
import h5py, hdf5plugin
import numpy as np
import sys
sys.path.insert(0,'../build/lib.linux-x86_64-cpython-38')
import bslz4_to_sparse

CASES = [( "/data/id11/jon/hdftest/eiger4m_u32.h5", "/entry_0000/ESRF-ID11/eiger/data"),
         ( "/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5",
           "/entry_0000/ESRF-ID11/eiger/data"),
         ("/data/id11/jon/hdftest/kevlar.h5", "/entry/data/data" ) ]


indices = np.zeros(2)

def pysparse( ds, num, cut, mask = None ):
    frame = ds[num]
    if mask is not None:
        frame *= mask
        assert frame.dtype == ds.dtype
    pixels = frame > cut
    values = frame[pixels]
    global indices
    if indices.size != frame.size:
        indices = np.arange( frame.size )
    return values, indices[pixels.ravel()]

def Csparse( ds, num, cut, mask = None):
    filtinfo, buffer = ds.id.read_direct_chunk( (num, 0, 0) )
    npbuffer = np.frombuffer( buffer, np.uint8 )
    if mask is None:
        mask = np.ones( (ds.shape[1], ds.shape[2]), np.uint8 )
    indices = np.empty( (ds.shape[1], ds.shape[2]), np.uint32 ).ravel()
    values  = np.empty( (ds.shape[1], ds.shape[2]), ds.dtype  ).ravel()
    if ds.dtype == np.uint32:
        fun = bslz4_to_sparse.bslz4_u32
    elif ds.dtype == np.uint16:
        fun = bslz4_to_sparse.bslz4_u16
    else:
        raise Exception("no decoder for your type")
    npixels = fun(npbuffer, mask, values, indices, cut)
    return values[:npixels], indices[:npixels]
        
def testok():
    for hname, dset in CASES:
        with h5py.File(hname, 'r') as hin:
            dataset = hin[dset]
            print(dataset.shape, dataset.dtype, hname)
            if dataset.dtype == np.uint16:
                mask = dataset[0] < pow(2,16)-1
            elif dataset.dtype == np.uint32:
                mask = dataset[0] < pow(2,16)-1 ## Careful !!
            for frame in np.arange(0,len(dataset),len(dataset)//10):
                for cut in (0,10,100,1000):
                    pv, pi = pysparse( dataset, frame, cut, mask )
                    cv, ci = Csparse(  dataset, frame, cut, mask )
                    if cv.shape != pv.shape:
                        print(cv.shape, cv[:10],ci[:10])
                        print(pv.shape, pv[:10],pi[:10])
                        raise
                    assert (cv == pv).all()
                    assert (ci == pi).all()
    #                    print('...', frame, cut, len(pv))
    print('No errors found')

if __name__=='__main__':
    testok()
                
        # py-spy record -n -r 200 -f speedscope python3 test1.py
