
import h5py, hdf5plugin
import numpy as np
import sys, os
# sys.path.insert(0,'../build/lib.linux-x86_64-cpython-38')
import bslz4_to_sparse

print("Running from", bslz4_to_sparse.__file__)

CASES = [ ('bslz4testcases.h5','Primes_u16_Range_1'),
    ( "/data/id11/jon/hdftest/eiger4m_u32.h5", "/entry_0000/ESRF-ID11/eiger/data"),
         ( "/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5",
           "/entry_0000/ESRF-ID11/eiger/data"),
         ("/data/id11/jon/hdftest/kevlar.h5", "/entry/data/data" ) ]

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
        frame *= mask.reshape( frame.shape )
        assert frame.dtype == ds.dtype
    pixels = frame > cut
    values = frame[pixels]
    global ij
    if ij[0].size != frame.size:
        ij = np.mgrid[ 0:frame.shape[0] , 0:frame.shape[1] ]
    return values, ij[0].flat[pixels.ravel()], ij[1].flat[pixels.ravel()]

        
def testok():
    for hname, dset in CASES:
        with h5py.File(hname, 'r') as hin:
            dataset = hin[dset]
            print(dataset.shape, dataset.dtype, hname, dset)
            mbool = dataset[0] == pow(2,16)-1
            if dataset.dtype == np.uint32:
                mbool |= (dataset[0] == pow(2,32)-1) 
            mask = (1-mbool.astype(np.uint8)).ravel()
            step = max(1, len(dataset)//10)
            for frame in np.arange(0,len(dataset),step):
                for cut in (0,10,100,1000):
                    if cut > np.iinfo( dataset.dtype ).max:
                        continue
                    pv, pi = pysparse( dataset, frame, cut, mask )
                    npx, (cv, ci) = bslz4_to_sparse.bslz4_to_sparse( dataset, 
                                                                    frame, cut, mask )
                    def pd():
                        print('npx',npx,'len(pv)',len(pv),'cut',cut,'masksize',mask.size)
                        print('data')
                        dsf = dataset[frame]
                        print(dsf)
                        for i in range(len(pv)):
                            if (ci[i] == pi[i]) and (cj[i] == pj[i]) and (cv[i] == pv[i]):
                                continue
                            else:
                                print('i %d ci %d cj %d cv %d pi %d pj %d pv %d'%(
                                    i, ci[i],cj[i],cv[i],pi[i],pj[i],pv[i]
                                ))
                                return
                        
                    if len(pv) != npx:
                        print('cut',cut)
                        print(npx, cv[:10],ci[:10])
                        print(npx, cv[:npx][-10:],ci[:npx][-10:])
                        print(pv.shape[0], pv[:10],pi[:10])
                        print(pv.shape[0], pv[-10:],pi[-10:])
                        raise
                    assert (cv[:npx] == pv).all(), pd()
                    assert (ci[:npx] == pi).all(), pd()
                    assert (cj[:npx] == pj).all(), pd()
                

    print('No errors found')

if __name__=='__main__':
    for c in CASES:
        print(c)
    testok()
                
    # py-spy record -n -r 200 -f speedscope python3 test1.py
