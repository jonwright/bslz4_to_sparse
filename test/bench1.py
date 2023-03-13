
import h5py, hdf5plugin
import numpy as np
import sys
import timeit
# sys.path.insert(0,'../build/lib.linux-x86_64-cpython-38')
import bslz4_to_sparse

print("Running from", bslz4_to_sparse.__file__)

CASES = [( "/data/id11/jon/hdftest/eiger4m_u32.h5", "/entry_0000/ESRF-ID11/eiger/data"),
         ( "/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5",
           "/entry_0000/ESRF-ID11/eiger/data"),
         ("/data/id11/jon/hdftest/kevlar.h5", "/entry/data/data" ) ]

def bench(hname, dsname, mask, cut):
    start = timeit.default_timer()
    npixels = 0
    with h5py.File(hname, 'r') as hin:
        dataset = hin[dsname]
        npt = dataset.size
        nframes = min(len(dataset), 300)
        buf = None
        for i in range(nframes):
            npx, buf = bslz4_to_sparse.bslz4_to_sparse( dataset, i, cut, mask, pixelbuffer = buf )
            npixels += npx
    end = timeit.default_timer()
    dt = end - start
    print('%d Sparsity %.3f %%, %d frames in %.4f /s,  %.3f fps'%( npixels,
        100*(npixels / npt),nframes, dt, nframes/dt))
        
def testok():
    for hname, dset in CASES:
        cut = 0
        with h5py.File(hname, 'r') as hin:
            dataset = hin[dset]
            print(dataset.shape, dataset.dtype, hname, cut)
            mbool = dataset[0] == pow(2,16)-1
            if dataset.dtype == np.uint32:
                mbool |= (dataset[0] == pow(2,32)-1) 
            mask = (1-mbool.astype(np.uint8)).ravel()
        cut = 0
        bench(hname, dset, mask, cut)

if __name__=='__main__':
    testok()
                
    # py-spy record -n -r 200 -f speedscope python3 test1.py
