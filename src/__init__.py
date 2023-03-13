
import numpy as np
import ctypes
from .bslz4_to_sparse import bslz4_uint32_t, bslz4_uint16_t, bslz4_uint8_t

version = '0.0.4'

# We cast away the 'read-only' nature of python bytes.
# Not needed for the latest numpy.
buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
buffer_from_memory.restype = ctypes.py_object
buffer_from_memory.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)

    
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
        irow = np.empty( (ds.shape[1], ds.shape[2]), np.uint16 ).ravel()
        jcol = np.empty( (ds.shape[1], ds.shape[2]), np.uint16 ).ravel()
        values  = np.empty( (ds.shape[1], ds.shape[2]), ds.dtype ).ravel()
    else:
        values, irow, jcol = pixelbuffer
    # todo : h5py malloc free version coming? see https://github.com/h5py/h5py/pull/2232
    filtinfo, buffer = ds.id.read_direct_chunk( (num, 0, 0) )
    #
    # h5py returns a bytes object that is read only. Older versions of numpy insist
    # to make a copy. We work around that using ctypes to set a writeable flag.
    #                                                      PyBUF_WRITE 0x200
    if ds.dtype == np.uint16:
        npixels = bslz4_uint16_t(  np.frombuffer( 
            buffer_from_memory( buffer, len(buffer), 0x200), np.uint8 ),
            mask, values, irow, jcol, cut, ds.shape[2])
    elif ds.dtype == np.uint32:
        npixels = bslz4_uint32_t(np.frombuffer( 
            buffer_from_memory( buffer, len(buffer), 0x200), np.uint8 ),
            mask, values, irow, jcol, cut, ds.shape[2])
    elif ds.dtype == np.uint8:
        npixels = bslz4_uint8_t(np.frombuffer( 
            buffer_from_memory( buffer, len(buffer), 0x200), np.uint8 ),
            mask, values, irow, jcol, cut, ds.shape[2])
    else:
        raise Exception("no decoder for your type")
    if npixels < 0:
        raise Exception("Error decoding: %d"%(npixels))
    # indices = icol.astype(np.uint32) * ds.shape[1] + jrow
    return npixels, (values, irow, jcol)
