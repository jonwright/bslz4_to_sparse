

import hdf5plugin, h5py, numpy as np, os, sys, json

def write_array( h5name, dsetname, ary ):
    assert len(ary.shape) == 3
    with h5py.File( h5name, "a" ) as h5f:
        # Hard wire the bitshuffle numbers
        dset = h5f.create_dataset( dsetname, data = ary,
                                   chunks = (1, ary.shape[1], ary.shape[2]),
                                   compression = 32008,
                                   compression_opts = (0,2), )
        print("Wrote",h5name,dsetname)


METHODS = {
    # use less disk space:
    'uniform15' : lambda nelem, dt : np.random.randint( 0, 15, size=nelem, dtype = dt ),
    'Poisson_1'  : lambda nelem, dt : np.random.poisson( lam=1.0, size=nelem ).astype( dt ),
    # Range for debugging. Gives the array indices in output in small blocks
    'Range_31' :  lambda nelem, dt : ( (1/31)*np.arange( 0, nelem, dtype=np.float32 )).astype( dt ),
    # Random numbers. Hard to look at but hopefully a stringent test as they should not compress.
    'uniform_random' : lambda nelem, dt : np.random.randint( 0, pow(2,8*dt(0).itemsize),
                                                            size=nelem, dtype = dt ),
    # Poisson random numbers. For timing:
    'Poisson_0.01' : lambda nelem, dt : np.random.poisson( lam=0.01, size=nelem).astype( dt ),
    'Poisson_100'  : lambda nelem, dt : np.random.poisson( lam=100.0, size=nelem).astype( dt ),
    # Range for debugging. Gives the array indices in output
    '1_Range_1' :  lambda nelem, dt : np.arange( nelem, dtype = dt ),
    # Range for debugging. Gives the array indices in output in blocks
    'Range_1024' : lambda nelem, dt : ( (1./1024)*np.arange( 0, nelem, dtype=np.float32 )).astype( dt ),
}

N = 2
BIG = [ ('Frelon2K', (N, 2048, 2048) ),
        ('Eiger4M' , (N, 2162, 2068) ),
        ('Primes'  , (N, 521, 523) ) ]

SMALL = [ ('256 square', (N, 256, 256) ),
          ('Small Primes', (N, 149, 211) ),]

def make_testcases( hname, cases ) :
    np.random.seed(10007*10009)
    for name, shp in cases:
        for dtyp, label in ( ( np.uint8 , 'u8' ),
                             ( np.uint16, 'u16' ),
                             ( np.uint32, 'u32' ) ):
            ary = np.empty( shp, dtype = dtyp )
            for i, method in enumerate(METHODS):
                ary = METHODS[method]( ary.size, dtyp ).reshape( shp )
                print(method, ary.mean(dtype=float), ary.std())
                dsname = "_".join((name, label, method))
                write_array( hname, dsname , ary )






# --- decode matrix fixture -------------------------------------------------
# A small, exhaustive decode test data generator.  Every cell is a single
# (1, rows, cols) frame, so the whole matrix stays small.  The paired index
# file records each chunk's (byte offset, length) plus dtype/codec/cut and the
# expected >cut pixels, because the Python 2.7 test can only read chunks by
# raw offset (h5py 2.10 has no get_chunk_info and read_direct_chunk is broken).
# Run with:  python3 test/make_testcases.py matrix
#
# Image dims are primes (83 x 101): the frame is not a multiple of 8 elements,
# so every block size leaves a non-SIMD-aligned tail (the decoder's raw-copy
# path).  Block sizes are in ELEMENTS and MUST be a multiple of 8
# (BSHUF_BLOCKED_MULT) -- bitshuffle rejects anything else, so the awkward
# "prime-ish" sizes are rounded to the nearest multiple of 8.  0 == default
# (8192 bytes).  8384 is >= the 8383-element frame (1-block, tail-only); 0 and
# 1224 are smaller (multi-block).
MATRIX_ROWS = 83
MATRIX_COLS = 101
MATRIX_CUT = 0
MATRIX_DTYPES = [
    ("u8", np.uint8), ("u16", np.uint16), ("u32", np.uint32), ("u64", np.uint64),
    ("i8", np.int8), ("i16", np.int16), ("i32", np.int32), ("i64", np.int64),
    ("f32", np.float32), ("f64", np.float64),
]
MATRIX_BLOCKS = [0, 1224, 8384]           # elements; 0 == pack/8192 default
MATRIX_CODECS = [(2, "lz4"), (3, "zstd")]
MATRIX_SPARSITY = [0.01, 0.3, 0.9]


def _matrix_frame(dtype, density, rng):
    """A (1, rows, cols) frame with ~density of elements > CUT (0)."""
    elem = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype)
    flat = elem.reshape(-1)
    hot = rng.random(flat.shape) < density
    n = int(hot.sum())
    if n:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            vals = (rng.random(n) * int(info.max) + 1).astype(dtype)
        else:
            vals = (rng.random(n).astype(dtype) * 1000 + 1).astype(dtype)
        flat[hot] = vals
    if np.issubdtype(dtype, np.floating):
        nonzero = [[int(i), float(v)] for i, v in enumerate(flat) if v > MATRIX_CUT]
    else:
        nonzero = [[int(i), int(v)] for i, v in enumerate(flat) if v > MATRIX_CUT]
    return elem.reshape(1, MATRIX_ROWS, MATRIX_COLS), nonzero


def make_decode_matrix(h5name, indexname):
    rng = np.random.default_rng(20240617)
    index = {"rows": MATRIX_ROWS, "cols": MATRIX_COLS, "cut": MATRIX_CUT,
             "chunks": []}
    if os.path.exists(h5name):
        os.remove(h5name)
    with h5py.File(h5name, "w") as h5f:
        for dsuffix, dtype in MATRIX_DTYPES:
            for codec, clabel in MATRIX_CODECS:
                for block in MATRIX_BLOCKS:
                    frames = []
                    nonzeros = []
                    for density in MATRIX_SPARSITY:
                        frame, nonzero = _matrix_frame(dtype, density, rng)
                        frames.append(frame)
                        nonzeros.append(nonzero)
                    data = np.concatenate(frames, axis=0)
                    name = "_".join([dsuffix, clabel, str(block)])
                    dset = h5f.create_dataset(
                        name, data=data, chunks=(1, MATRIX_ROWS, MATRIX_COLS),
                        compression=32008, compression_opts=(block, codec))
                    for i, density in enumerate(MATRIX_SPARSITY):
                        info = dset.id.get_chunk_info(i)
                        index["chunks"].append({
                            "name": "%s#%s" % (name, density),
                            "offset": int(info.byte_offset),
                            "length": int(info.size),
                            "dtype": dsuffix,
                            "codec": codec,
                            "density": density,
                            "rows": MATRIX_ROWS,
                            "cols": MATRIX_COLS,
                            "cut": MATRIX_CUT,
                            "nonzero": nonzeros[i],
                        })
    with open(indexname, "w") as fp:
        json.dump(index, fp, indent=1)
    print("wrote", h5name, "(%d bytes)" % os.path.getsize(h5name))
    print("wrote", indexname, "(%d chunks)" % len(index["chunks"]))


if __name__=="__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "matrix":
        here = os.path.dirname(os.path.abspath(__file__))
        make_decode_matrix(os.path.join(here, "matrix.h5"),
                           os.path.join(here, "matrix.json"))
    else:
        hname = "bslz4testcases.h5"
        if os.path.exists(hname):
            print("Removed", hname)
            os.remove(hname)
        if len(sys.argv)>1 and sys.argv[1]=='all':
            make_testcases(hname, SMALL+BIG )
        else:
            make_testcases(hname, SMALL )




