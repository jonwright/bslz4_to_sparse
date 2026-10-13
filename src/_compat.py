"""The names of the released versions (0.0.15, 0.0.16) that are not part of the current design:
npbuf, buffer_from_memory and the raw functions bslz4_uint8_t / 16 / 32 and bslz4_csc_uint8_t / 16 /
32.  They are thin wrappers on the current kernels and behave as the released ones did:

    npixels = bslz4_uint16_t(cmp, mask, values, indices, cut)
    npixels = bslz4_csc_uint16_t(cmp, mask, values, indices, cut, powder, data, csc_indices, indptr)

cmp is the compressed chunk as a uint8 array, mask uint8 (active > 0), values (the pixel dtype),
indices (uint32), powder float64 (nbins), data float32, csc_indices and indptr uint32.  The return
is the number of pixels above cut, or negative for an error.  The outputs are written in place.
"""

import ctypes

import numpy as np

from . import _pipeline, _toppix

if hasattr(ctypes.pythonapi, "PyMemoryView_FromMemory"):
    buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
    buffer_from_memory.restype = ctypes.py_object
    buffer_from_memory.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
else:

    def buffer_from_memory(buf, l, f):
        if type(buf) == str and buf[:6] == "array(":
            from array import array  # noqa: F401  (python 2.7 and old h5py)

            return eval(buf)
        raise Exception("Unknown code path for buffer decoding " + str(type(buf)))


def npbuf(buf):
    """A uint8 numpy array on the bytes of buf (no copy where numpy allows)."""
    if isinstance(buf, np.ndarray):
        return buf
    elif isinstance(buf, memoryview):
        return np.frombuffer(buf, np.uint8)
    else:
        return np.frombuffer(buffer_from_memory(buf, len(buf), 0x200), np.uint8)


def _call(ext, dtype, cmp, msk, outpx, outP, thresh, dot=None):
    """One chunk through ext.sparsify (dot None) or ext.sparsify_and_dot (dot = (powder, d, ind, ip))."""
    from . import _blocksize_bytes, _codec_of_chunk, _cut_array

    dtype = np.dtype(dtype)
    codec = _codec_of_chunk(cmp)
    pointers = np.empty(1, np.int64)
    lengths = np.empty(1, np.int32)
    ext.note_chunk(cmp, 0, pointers, lengths)
    npx = np.empty(1, np.int32)
    cursors = np.empty(1, np.int64)
    be = _blocksize_bytes(cmp) // dtype.itemsize
    workspace = np.empty(3 * _blocksize_bytes(cmp) + be * (4 + dtype.itemsize), np.uint8)
    di = _pipeline.DTYPE_INDEX[_pipeline.dtype_suffix(dtype)]
    cut = _cut_array(thresh, dtype)
    head = (pointers, lengths, msk, outpx, outP, npx, cut)
    tail = (workspace, cursors, _toppix.EMPTY_TOP, _toppix.EMPTY_OUT)
    if dot is None:
        pipe = _pipeline.resolve(None, dtype, codec, msk)
        ret = ext.sparsify(*(head + tail + (di, pipe)))
    else:
        powder, d, ind, ip = dot
        pipe = _pipeline.resolve({"dot": "csc"}, dtype, codec, msk, matrix=True)
        ret = ext.sparsify_and_dot(*(head + (powder, d, ind, ip) + tail + (len(powder), di, pipe)))
    return int(ret) if ret < 0 else int(npx[0])


def _make(dtype, csc):
    from . import _ext

    if csc:

        def f(cmp, msk, outpx, outP, thresh, out, d, ind, ip):
            return _call(_ext, dtype, cmp, msk, outpx, outP, thresh, (out, d, ind, ip))

        f.__name__ = "bslz4_csc_%s_t" % np.dtype(dtype).name
    else:

        def f(cmp, msk, outpx, outP, thresh):
            return _call(_ext, dtype, cmp, msk, outpx, outP, thresh)

        f.__name__ = "bslz4_%s_t" % np.dtype(dtype).name
    return f


bslz4_uint8_t = _make(np.uint8, False)
bslz4_uint16_t = _make(np.uint16, False)
bslz4_uint32_t = _make(np.uint32, False)
bslz4_csc_uint8_t = _make(np.uint8, True)
bslz4_csc_uint16_t = _make(np.uint16, True)
bslz4_csc_uint32_t = _make(np.uint32, True)
