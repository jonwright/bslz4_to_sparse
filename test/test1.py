
import h5py, hdf5plugin
import numpy as np
import sys, os
# sys.path.insert(0,'../build/lib.linux-x86_64-cpython-38')
import bslz4_to_sparse

print("Running from", bslz4_to_sparse.__file__)

TESTCASES = [( "/data/id11/jon/hdftest/eiger4m_u32.h5", "/entry_0000/ESRF-ID11/eiger/data"),
         ( "/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5",
           "/entry_0000/ESRF-ID11/eiger/data"),
         ("/data/id11/jon/hdftest/kevlar.h5", "/entry/data/data" ) ]

CASES = []
for f,d in TESTCASES:
    if os.path.exists(f):
        CASES.append((f,d))
    else:
        f = os.path.split(f)[-1]
        if os.path.exists(f):
            CASES.append((f,d))
        else:
            print("Missing",f)
        

if not os.path.exists('bslz4testcases.h5'):
    print('Making more testcases')
    ret = os.system(sys.executable + ' make_testcases.py')
    assert ret == 0

with h5py.File('bslz4testcases.h5','r') as hin:
    for dataset in list(hin):
        CASES.append(( 'bslz4testcases.h5', dataset ) )
        
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

        
def testok():
    for hname, dset in CASES:
        with h5py.File(hname, 'r') as hin:
            dataset = hin[dset]
            print(dataset.shape, dataset.dtype, hname, dset)
            mbool = dataset[0] == pow(2,16)-1
            if dataset.dtype == np.uint32:
                mbool |= (dataset[0] == pow(2,32)-1) 
            mask = (1-mbool.astype(np.uint8))
            step = max(1, len(dataset)//10)
            for frame in np.arange(0,len(dataset),step):
                for cut in (0,10,100,1000):
                    if cut > np.iinfo( dataset.dtype ).max:
                        continue
                    pv, pi = pysparse( dataset, frame, cut, mask )
                    npx, (cv, ci) = bslz4_to_sparse.bslz4_to_sparse( dataset, 
                                                                    frame, cut, mask )
                    if len(pv) != npx:
                        print('cut',cut)
                        print(npx, cv[:10],ci[:10])
                        print(npx, cv[:npx][-10:],ci[:npx][-10:])
                        print(pv.shape[0], pv[:10],pi[:10])
                        print(pv.shape[0], pv[-10:],pi[-10:])
                        raise
                    assert (cv[:npx] == pv).all()
                    assert (ci[:npx] == pi).all()
    print('No errors found')
    
def testcaller():
    for hname, dset in CASES:
        with h5py.File(hname, 'r') as hin:
            dataset = hin[dset]
            print(dataset.shape, dataset.dtype, hname, dset)
            mbool = dataset[0] == pow(2,16)-1
            if dataset.dtype == np.uint32:
                mbool |= (dataset[0] == pow(2,32)-1) 
            mask = (1-mbool.astype(np.uint8))
            step = max(1, len(dataset)//10)
            funcobj = bslz4_to_sparse.chunk2sparse( mask, dataset.dtype )
            for frame in np.arange(0,len(dataset),step):
                for cut in (0,10,100,1000):
                    if cut > np.iinfo( dataset.dtype ).max:
                        continue
                    pv, pi = pysparse( dataset, frame, cut, mask )
                    filters, buffer = dataset.id.read_direct_chunk( (frame, 0, 0 ) )
                    npx, (cv, ci) = funcobj( buffer, cut )
                    if len(pv) != npx:
                        print('cut',cut)
                        print(npx, cv[:10],ci[:10])
                        print(npx, cv[:npx][-10:],ci[:npx][-10:])
                        print(pv.shape[0], pv[:10],pi[:10])
                        print(pv.shape[0], pv[-10:],pi[-10:])
                        raise
                    assert (cv[:npx] == pv).all()
                    assert (ci[:npx] == pi).all()
    print('No errors found') 

if __name__=='__main__':
    testcaller()
    testok()
                
    # py-spy record -n -r 200 -f speedscope python3 test1.py
