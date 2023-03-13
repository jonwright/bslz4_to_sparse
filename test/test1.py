
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

ij = (np.empty(2), np.empty(2))

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
            r,c = np.mgrid[ 0: dataset.shape[1], 0: dataset.shape[2] ]
            r = r.ravel().astype(np.uint16)
            c = c.ravel().astype(np.uint16)
            mbool = dataset[0] == pow(2,16)-1
            if dataset.dtype == np.uint32:
                mbool |= (dataset[0] == pow(2,32)-1) 
            mask = 1-mbool.astype(np.uint8).ravel()
            step = max( len(dataset)//10, 1 )
            print(dataset.shape, dataset.dtype, hname, dset, mask.sum()/mask.size)
            for frame in np.arange(0,len(dataset),step):
                for cut in (0,10,100,1000,100000):
                    if cut > np.iinfo( dataset.dtype ).max:
                        continue
                    pv, pi, pj = pysparse( dataset, frame, cut, mask )
                    npx, (cv, ci, cj) = bslz4_to_sparse.bslz4_to_sparse( dataset, 
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
                        print('Wrong number of pixels found:')
                        pd()
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
