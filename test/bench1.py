
import h5py, hdf5plugin
import numpy as np
import sys, os
import timeit

path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if path:
    sys.path.insert(0, path)
import bslz4_to_sparse

print("Running from", bslz4_to_sparse.__file__)


def example_cases():
    """The example data (tools/example_data.py), 8 kB bitshuffle-lz4, masked pixels at the maximum."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import example_data as ed

    return [(ed.path(k, "lz4mask"), ed.DSET) for k in ed.FILES]


TESTCASES = example_cases()

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


def bench1(hname, dsname, mask, cut):
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
      
def bench2(hname, dsname, mask, cut):
    start = timeit.default_timer()
    npixels = 0
    with h5py.File(hname, 'r') as hin:
        dataset = hin[dsname]
        npt = dataset.size
        nframes = min(len(dataset), 300)
        buf = None
        fun = bslz4_to_sparse.chunk2sparse( mask, dataset.dtype )
        for i in range(nframes):
            filters, buffer = dataset.id.read_direct_chunk( (i,0,0,) )
            npx, buf = fun( buffer, cut )
            npixels += npx
    end = timeit.default_timer()
    dt = end - start
    print('%d Sparsity %.3f %%, %d frames in %.4f /s,  %.3f fps'%( npixels,
        100*(npixels / npt),nframes, dt, nframes/dt))        
        
def testok():
    for hname, dset in CASES:
        with h5py.File(hname, 'r') as hin:
            dataset = hin[dset]
            print(dataset.shape, dataset.dtype, hname)
            mbool = dataset[0] == pow(2,16)-1
            if dataset.dtype == np.uint32:
                mbool |= (dataset[0] == pow(2,32)-1) 
            mask = (1-mbool.astype(np.uint8))
        cut = 0
        for i in range(3):
            bench1(hname, dset, mask, cut)
        for i in range(3):
            bench2(hname, dset, mask, cut)

if __name__=='__main__':
    testok()
                
    # py-spy record -n -r 200 -f speedscope python3 test1.py
