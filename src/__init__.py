
import numpy as np
import ctypes
from .bslz4_to_sparse import bslz4_uint32_t, bslz4_uint16_t, bslz4_uint8_t

version = '0.0.5'

# We cast away the 'read-only' nature of python bytes.
# Not needed for the latest numpy.
buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
buffer_from_memory.restype = ctypes.py_object
buffer_from_memory.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)

class chunk2sparse:
    def __init__(self, mask, dtype = np.uint16):
        self.mask = mask.ravel()
        self.indices = np.empty(mask.shape, np.uint32)
        self.values = np.empty(mask.shape, dtype)
        self.fun = (None, 
                    bslz4_uint8_t, 
                    bslz4_uint16_t, 
                    None, 
                    bslz4_uint32_t)[ dtype.itemsize ]
        
    def __call__(self, buffer, cut):
        npixels = self.fun(  np.frombuffer( 
            buffer_from_memory( buffer, len(buffer), 0x200), np.uint8 ),
            self.mask, self.values, self.indices, cut)
        return npixels, (self.values, self.indices)
    
    
def indices2ij(indices, nfast = 2068):
    row = np.zeros( len(indices), np.uint16 )
    col = np.zeros( len(indices), np.uint16 )
    np.divmod( indices, nfast, out = (row, col) )
    return row, col
              
        
def bslz4_to_sparse( ds, num, cut, mask = None, pixelbuffer = None):
    """
    Reads a bitshuffle compressed hdf5 dataset and converts this 
    directly into a sparse format (indices, values) when decoding
    the data.
    
    ds = hdf5 dataset containing [nframes, ni, nj] pixels
    num = frame number to read
    cut = threshold, pixels below this value are ignored
    mask = detector mask. Active pixels > 0.
    pixelbuffer = None or (values, indices) storage space
    
    returns (number_of_pixels, (values, indices))
    """
    if mask is None:
        mask = np.ones( (ds.shape[1], ds.shape[2]), np.uint8 ).ravel()
    if pixelbuffer is None:
        indices = np.empty( (ds.shape[1], ds.shape[2]), np.uint32 ).ravel()
        values  = np.empty( (ds.shape[1], ds.shape[2]), ds.dtype  ).ravel()
    else:
        values, indices = pixelbuffer
    # todo : h5py malloc free version coming? see https://github.com/h5py/h5py/pull/2232
    filtinfo, buffer = ds.id.read_direct_chunk( (num, 0, 0) )
    #
    # h5py returns a bytes object that is read only. Older versions of numpy insist
    # to make a copy. We work around that using ctypes to set a writeable flag.
    #                                                      PyBUF_WRITE 0x200
    if ds.dtype == np.uint16:
        npixels = bslz4_uint16_t(  np.frombuffer( 
            buffer_from_memory( buffer, len(buffer), 0x200), np.uint8 ),
            mask, values, indices, cut)
    elif ds.dtype == np.uint32:
        npixels = bslz4_uint32_t(np.frombuffer( 
            buffer_from_memory( buffer, len(buffer), 0x200), np.uint8 ),
            mask, values, indices, cut)
    elif ds.dtype == np.uint8:
        npixels = bslz4_uint8_t(  np.frombuffer( 
            buffer_from_memory( buffer, len(buffer), 0x200), np.uint8 ),
            mask, values, indices, cut)
    else:
        raise Exception("no decoder for your type")
    if npixels < 0:
        raise Exception("Error decoding: %d"%(npixels))
    return npixels, (values, indices)
