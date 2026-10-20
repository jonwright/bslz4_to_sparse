"""The CUDA calls the GPU classes need (the interface of _backend.py), on the CUDA driver API alone:
libcuda is opened at run time (ctypes), the kernels come precompiled by nvcc in one fatbin
(tools/build_cuda_fatbin.py).  No cupy, NVRTC or CUDA runtime.

Uses the device's primary context (shared with cupy, torch, jax).  A kernel runs on the calling
thread's current stream (Stream.use()), as in cupy.  Every device-to-host copy (.get) waits for its
stream before returning; .set queues on the stream given; zeros/asarray wait for their copy.
"""

import ctypes
import json
import os
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FATBIN = os.environ.get("BSLZ4_CUDA_FATBIN", os.path.join(HERE, "bslz4_kernels.fatbin"))


def _load():
    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise ImportError("no CUDA driver (libcuda) found")


_cu = _load()
_P = ctypes.c_void_p
_SZ = ctypes.c_size_t


def _sig(name, *args):
    f = getattr(_cu, name)
    f.argtypes, f.restype = list(args), ctypes.c_int
    return f


cuInit = _sig("cuInit", ctypes.c_uint)
cuDeviceGetCount = _sig("cuDeviceGetCount", ctypes.POINTER(ctypes.c_int))
cuDeviceGetName = _sig("cuDeviceGetName", ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
cuDeviceGet = _sig("cuDeviceGet", ctypes.POINTER(ctypes.c_int), ctypes.c_int)
cuDeviceGetAttribute = _sig(
    "cuDeviceGetAttribute", ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
)
cuDeviceTotalMem = _sig("cuDeviceTotalMem_v2", ctypes.POINTER(_SZ), ctypes.c_int)
try:
    cuDevicePrimaryCtxRetain = _sig("cuDevicePrimaryCtxRetain", ctypes.POINTER(_P), ctypes.c_int)
except AttributeError:  # drivers before CUDA 7.0: no primary context, make one
    cuDevicePrimaryCtxRetain = None
cuCtxCreate = _sig("cuCtxCreate_v2", ctypes.POINTER(_P), ctypes.c_uint, ctypes.c_int)
cuCtxSetCurrent = _sig("cuCtxSetCurrent", _P)
cuCtxGetCurrent = _sig("cuCtxGetCurrent", ctypes.POINTER(_P))
cuCtxGetDevice = _sig("cuCtxGetDevice", ctypes.POINTER(ctypes.c_int))
cuCtxSynchronize = _sig("cuCtxSynchronize")
cuCtxPushCurrent = _sig("cuCtxPushCurrent_v2", _P)
cuCtxPopCurrent = _sig("cuCtxPopCurrent_v2", ctypes.POINTER(_P))
cuModuleLoadData = _sig("cuModuleLoadData", ctypes.POINTER(_P), ctypes.c_char_p)
cuModuleGetFunction = _sig("cuModuleGetFunction", ctypes.POINTER(_P), _P, ctypes.c_char_p)
cuLaunchKernel = _sig(
    "cuLaunchKernel", _P, *([ctypes.c_uint] * 6), ctypes.c_uint, _P, ctypes.POINTER(_P), _P
)
cuMemAlloc = _sig("cuMemAlloc_v2", ctypes.POINTER(ctypes.c_uint64), _SZ)
cuMemFree = _sig("cuMemFree_v2", ctypes.c_uint64)
cuMemHostAlloc = _sig("cuMemHostAlloc", ctypes.POINTER(_P), _SZ, ctypes.c_uint)
cuMemFreeHost = _sig("cuMemFreeHost", _P)
cuMemcpyHtoDAsync = _sig("cuMemcpyHtoDAsync_v2", ctypes.c_uint64, _P, _SZ, _P)
cuMemcpyDtoHAsync = _sig("cuMemcpyDtoHAsync_v2", _P, ctypes.c_uint64, _SZ, _P)
cuMemcpy2DAsync = _sig("cuMemcpy2DAsync_v2", _P, _P)
cuPointerGetAttribute = _sig("cuPointerGetAttribute", _P, ctypes.c_int, ctypes.c_uint64)
cuMemsetD8Async = _sig("cuMemsetD8Async", ctypes.c_uint64, ctypes.c_ubyte, _SZ, _P)
cuStreamCreate = _sig("cuStreamCreate", ctypes.POINTER(_P), ctypes.c_uint)
cuStreamDestroy = _sig("cuStreamDestroy_v2", _P)
cuStreamSynchronize = _sig("cuStreamSynchronize", _P)
cuGetErrorString = _sig("cuGetErrorString", ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))

CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
CU_STREAM_NON_BLOCKING = 1
CU_MEMHOSTALLOC_PORTABLE = 1


class CudaError(RuntimeError):
    pass


def _check(rc):
    if rc:
        s = ctypes.c_char_p()
        cuGetErrorString(rc, ctypes.byref(s))
        raise CudaError("CUDA driver error %d: %s" % (rc, s.value.decode() if s.value else "?"))


_check(cuInit(0))
_tls = threading.local()  # .stream: the current stream of this thread
_ctx = {}  # device -> primary context
_ctx_lock = threading.Lock()


def _context(dev):
    with _ctx_lock:
        if dev not in _ctx:
            d, c = ctypes.c_int(), _P()
            _check(cuDeviceGet(ctypes.byref(d), dev))
            if cuDevicePrimaryCtxRetain is not None:
                _check(cuDevicePrimaryCtxRetain(ctypes.byref(c), d))
            else:
                _check(cuCtxCreate(ctypes.byref(c), 0, d))  # pushed current: pop it
                _check(cuCtxPopCurrent(ctypes.byref(_P())))
            _ctx[dev] = c
        return _ctx[dev]


def device_id():
    """The current device; with no current context, device 0's is made current (as the CUDA
    runtime does)."""
    c = _P()
    _check(cuCtxGetCurrent(ctypes.byref(c)))
    if not c.value:
        use_device(0)
        return 0
    d = ctypes.c_int()
    _check(cuCtxGetDevice(ctypes.byref(d)))
    return d.value


def device_count():
    n = ctypes.c_int()
    _check(cuDeviceGetCount(ctypes.byref(n)))
    return n.value


def device_name(dev):
    d, s = ctypes.c_int(), ctypes.create_string_buffer(256)
    _check(cuDeviceGet(ctypes.byref(d), dev))
    _check(cuDeviceGetName(s, 256, d))
    return s.value.decode()


def use_device(dev):
    """Make `dev` (its primary context) current in this thread."""
    _check(cuCtxSetCurrent(_context(dev)))


class device(object):
    """Context manager: `dev` current inside the block."""

    def __init__(self, dev):
        self.dev = dev

    def __enter__(self):
        self.prev = _P()
        _check(cuCtxGetCurrent(ctypes.byref(self.prev)))
        use_device(self.dev)
        return self

    def __exit__(self, *exc):
        _check(cuCtxSetCurrent(self.prev))


def l2_bytes(dev):
    v = ctypes.c_int()
    _check(cuDeviceGetAttribute(ctypes.byref(v), CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev))
    return v.value


def max_grid_x(dev):
    v = ctypes.c_int()
    _check(cuDeviceGetAttribute(ctypes.byref(v), CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev))
    return v.value


def memory_bytes(dev):
    v = _SZ()
    _check(cuDeviceTotalMem(ctypes.byref(v), dev))
    return v.value


def synchronize(dev):
    with device(dev):
        _check(cuCtxSynchronize())


def _in_context(ctx, f, *args):
    """f(*args) with `ctx` current (a free may run in any thread, by the garbage collector)."""
    try:
        push, pop = cuCtxPushCurrent, cuCtxPopCurrent
        if _P is None or push is None or pop is None or f is None:  # interpreter shutdown
            return
        if push(ctx) == 0:
            try:
                f(*args)
            finally:
                pop(ctypes.byref(_P()))
    except Exception:
        pass  # interpreter shutdown tears the bindings and the driver down in any order


def _current_stream():
    s = getattr(_tls, "stream", None)
    return None if s is None else s.handle


class Stream(object):
    """A non-blocking stream; .use() makes it this thread's current stream."""

    def __init__(self):
        self._ctx, h = _context(device_id()), _P()
        _check(cuStreamCreate(ctypes.byref(h), CU_STREAM_NON_BLOCKING))
        self.handle = h

    def use(self):
        _tls.stream = self
        return self

    def synchronize(self):
        _check(cuStreamSynchronize(self.handle))

    def __del__(self):
        if getattr(self, "handle", None):
            _in_context(self._ctx, cuStreamDestroy, self.handle)


class _Alloc(object):
    """One cuMemAlloc, freed with its last view."""

    def __init__(self, nbytes):
        self.dev = device_id()
        self.ctx, p = _context(self.dev), ctypes.c_uint64()
        _check(cuMemAlloc(ctypes.byref(p), max(int(nbytes), 1)))
        self.ptr = p.value

    def __del__(self):
        if getattr(self, "ptr", None) and _in_context is not None:  # None: interpreter shutdown
            _in_context(self.ctx, cuMemFree, self.ptr)


class _Borrowed(object):
    """Another library's device memory: kept alive by holding its owner, never freed here."""

    def __init__(self, ptr, dev, owner):
        self.ptr, self.dev, self.owner = ptr, dev, owner


class ndarray(object):
    """A C-contiguous device array: slicing along the first axis, .set, .get, .view, .size.  Other
    libraries take it without a copy through __cuda_array_interface__ (cupy, numba, torch, pycuda)
    or DLPack (torch, jax, tensorflow, cupy)."""

    def __init__(self, shape, dtype, alloc=None, offset=0):
        self.shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.nbytes = self.size * self.dtype.itemsize
        self._alloc = alloc if alloc is not None else _Alloc(self.nbytes)
        self.ptr = self._alloc.ptr + offset

    @property
    def __cuda_array_interface__(self):
        """Version 3.  The data is complete when the GPU classes hand an array out: no stream."""
        return {"shape": self.shape, "typestr": self.dtype.str, "data": (self.ptr, False),
                "strides": None, "version": 3, "stream": None}

    def __dlpack_device__(self):
        return (_KDLCUDA, self._alloc.dev)

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        """A DLPack (0.8, unversioned) capsule viewing this memory; it keeps the memory alive until
        the consumer calls its deleter.  `stream` is not needed: the data is complete."""
        if copy:
            raise BufferError("a device array is exported without a copy")
        return _dlpack_capsule(self)

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise IndexError("device arrays take a slice of the first axis")
        start, stop, _ = key.indices(self.shape[0])
        stop = max(stop, start)
        row = self.nbytes // max(self.shape[0], 1)
        return ndarray(
            (stop - start,) + self.shape[1:], self.dtype, self._alloc,
            self.ptr - self._alloc.ptr + start * row,
        )

    def view(self, dtype):
        dtype = np.dtype(dtype)
        if len(self.shape) != 1 or self.nbytes % dtype.itemsize:
            raise ValueError("view of a 1D array whose bytes divide by the new itemsize")
        return ndarray(self.nbytes // dtype.itemsize, dtype, self._alloc, self.ptr - self._alloc.ptr)

    def set(self, host, stream=None):
        host = np.ascontiguousarray(host)
        if host.nbytes > self.nbytes:
            raise ValueError("%d bytes into a %d-byte device array" % (host.nbytes, self.nbytes))
        h = stream.handle if stream is not None else _current_stream()
        _check(cuMemcpyHtoDAsync(self.ptr, host.ctypes.data, host.nbytes, h))

    def get(self, out=None, stream=None):
        """A host copy; returns once the copy has landed."""
        if out is None:
            out = np.empty(self.shape, self.dtype)
        if out.nbytes != self.nbytes or not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous with %d bytes" % self.nbytes)
        h = stream.handle if stream is not None else _current_stream()
        _check(cuMemcpyDtoHAsync(out.ctypes.data, self.ptr, self.nbytes, h))
        _check(cuStreamSynchronize(h))
        return out

    def fill(self, byte, stream=None):
        """Every byte set to `byte`; returns once done."""
        h = stream.handle if stream is not None else _current_stream()
        _check(cuMemsetD8Async(self.ptr, byte, self.nbytes, h))
        _check(cuStreamSynchronize(h))


def empty(shape, dtype):
    return ndarray(shape, dtype)


class _CUDA_MEMCPY2D(ctypes.Structure):
    _fields_ = [("srcXInBytes", _SZ), ("srcY", _SZ), ("srcMemoryType", ctypes.c_int),
                ("srcHost", _P), ("srcDevice", ctypes.c_uint64), ("srcArray", _P),
                ("srcPitch", _SZ), ("dstXInBytes", _SZ), ("dstY", _SZ),
                ("dstMemoryType", ctypes.c_int), ("dstHost", _P), ("dstDevice", ctypes.c_uint64),
                ("dstArray", _P), ("dstPitch", _SZ), ("WidthInBytes", _SZ), ("Height", _SZ)]


def device_bytes(obj, dev):
    """The GPU memory of obj (__cuda_array_interface__, C-contiguous, on device dev) as a uint8
    array viewing it; waits for obj's stream first (version 3)."""
    cai = obj.__cuda_array_interface__
    shape, dt = tuple(cai["shape"]), np.dtype(cai["typestr"])
    st = cai.get("strides")
    if st is not None and tuple(st) != tuple(
        int(np.prod(shape[k + 1 :])) * dt.itemsize for k in range(len(shape))
    ):
        raise ValueError("the chunks must be in one C-contiguous GPU buffer")
    ptr = int(cai["data"][0])
    if cai.get("stream"):
        _check(cuStreamSynchronize(_P(cai["stream"])))
    d = ctypes.c_int()
    if cuPointerGetAttribute(ctypes.byref(d), CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr) == 0:
        if d.value != dev:  # (drivers before CUDA 9.2 cannot tell: not checked there)
            raise ValueError("the buffer is on GPU %d, this object on GPU %d" % (d.value, dev))
    return ndarray(int(np.prod(shape)) * dt.itemsize, np.uint8, _Borrowed(ptr, dev, obj))


def copy_rows(dst, dst_pitch, src, src_pitch, width, height):
    """height rows of width bytes, device to device, on the current stream"""
    c = _CUDA_MEMCPY2D()
    c.srcMemoryType = c.dstMemoryType = 2  # CU_MEMORYTYPE_DEVICE
    c.srcDevice, c.srcPitch, c.dstDevice, c.dstPitch = src.ptr, src_pitch, dst.ptr, dst_pitch
    c.WidthInBytes, c.Height = width, height
    _check(cuMemcpy2DAsync(ctypes.byref(c), _current_stream()))


# DLPack 0.8 (dlpack.h): the C structs a capsule named "dltensor" points to
_KDLCUDA = 2


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", _P), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", _P),
                             ("deleter", _DLDELETER)]
_DLCODE = {"i": 0, "u": 1, "f": 2}
_dl_alive = {}  # address of an exported DLManagedTensor -> (it, its shape/strides, the array)


@_DLDELETER
def _dl_delete(p):  # called by the consumer when it lets go of the memory
    try:
        _dl_alive.pop(ctypes.addressof(p.contents), None)
    except Exception:  # interpreter shutdown
        pass


_DLNAME = ctypes.create_string_buffer(b"dltensor")  # the capsule keeps a pointer to its name
# our own handle: ctypes.pythonapi's functions are shared, and torch sets their argtypes too
_pyapi = ctypes.PyDLL(ctypes.pythonapi._name, handle=ctypes.pythonapi._handle)
_capsule_new = _pyapi.PyCapsule_New
_capsule_new.restype, _capsule_new.argtypes = ctypes.py_object, [_P, _P, _P]
_capsule_valid = _pyapi.PyCapsule_IsValid
_capsule_valid.restype, _capsule_valid.argtypes = ctypes.c_int, [_P, _P]
_capsule_ptr = _pyapi.PyCapsule_GetPointer
_capsule_ptr.restype, _capsule_ptr.argtypes = _P, [_P, _P]


@ctypes.CFUNCTYPE(None, _P)
def _dl_capsule_free(cap):  # a capsule never consumed (still named "dltensor") frees its tensor
    try:
        if _capsule_valid(cap, ctypes.addressof(_DLNAME)):
            _dl_alive.pop(_capsule_ptr(cap, ctypes.addressof(_DLNAME)), None)
    except Exception:  # interpreter shutdown
        pass


def _dlpack_capsule(a):
    if a.dtype.kind not in _DLCODE:
        raise BufferError("dtype %s has no DLPack code here" % a.dtype)
    nd = len(a.shape)
    shape = (ctypes.c_int64 * max(nd, 1))(*a.shape)
    strides = (ctypes.c_int64 * max(nd, 1))(*[int(np.prod(a.shape[k + 1 :])) for k in range(nd)])
    m = _DLManagedTensor()
    t = m.dl_tensor
    t.data, t.ndim, t.byte_offset = a.ptr, nd, 0
    t.device = _DLDevice(_KDLCUDA, a._alloc.dev)
    t.dtype = _DLDataType(_DLCODE[a.dtype.kind], a.dtype.itemsize * 8, 1)
    t.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
    t.strides = ctypes.cast(strides, ctypes.POINTER(ctypes.c_int64))  # C order, in elements
    m.deleter = _dl_delete
    _dl_alive[ctypes.addressof(m)] = (m, shape, strides, a)
    return _capsule_new(ctypes.addressof(m), ctypes.addressof(_DLNAME),
                        ctypes.cast(_dl_capsule_free, _P))


def zeros(shape, dtype):
    """Zeroed, done on the default stream before returning (ready for any stream)."""
    a = ndarray(shape, dtype)
    _check(cuMemsetD8Async(a.ptr, 0, a.nbytes, None))
    _check(cuStreamSynchronize(None))
    return a


def asarray(host):
    """A device copy, landed before returning (ready for any stream)."""
    host = np.ascontiguousarray(host)
    a = ndarray(host.shape, host.dtype)
    _check(cuMemcpyHtoDAsync(a.ptr, host.ctypes.data, host.nbytes, None))
    _check(cuStreamSynchronize(None))
    return a


def pinned(nbytes):
    """Page-locked host memory (a ctypes buffer, freed with its last reference)."""
    p = _P()
    _check(cuMemHostAlloc(ctypes.byref(p), max(int(nbytes), 1), CU_MEMHOSTALLOC_PORTABLE))
    buf = (ctypes.c_char * int(nbytes)).from_address(p.value)
    buf._owner = _Pinned(p)
    return buf


class _Pinned(object):
    def __init__(self, p):
        self.p = p

    def __del__(self):
        if cuMemFreeHost is not None:
            cuMemFreeHost(self.p)


_ARG = {
    np.dtype(np.int8): ctypes.c_int8, np.dtype(np.uint8): ctypes.c_uint8,
    np.dtype(np.int16): ctypes.c_int16, np.dtype(np.uint16): ctypes.c_uint16,
    np.dtype(np.int32): ctypes.c_int32, np.dtype(np.uint32): ctypes.c_uint32,
    np.dtype(np.int64): ctypes.c_int64, np.dtype(np.uint64): ctypes.c_uint64,
    np.dtype(np.float32): ctypes.c_float, np.dtype(np.float64): ctypes.c_double,
}


class _Kernel(object):
    def __init__(self, module, symbol):
        self.module, self.symbol, self._fn = module, symbol, {}

    def __call__(self, grid, block, args, shared_mem=0):
        dev = device_id()
        fn = self._fn.get(dev)
        if fn is None:
            fn = _P()
            _check(cuModuleGetFunction(ctypes.byref(fn), self.module.handle(dev), self.symbol))
            self._fn[dev] = fn
        vals = []
        for a in args:
            if isinstance(a, ndarray):
                vals.append(ctypes.c_uint64(a.ptr))
            elif isinstance(a, np.generic):
                vals.append(_ARG[a.dtype](a))
            else:
                raise TypeError("kernel argument %r: a device array or a numpy scalar" % (a,))
        params = (_P * len(vals))(*[ctypes.cast(ctypes.byref(v), _P) for v in vals])
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        _check(cuLaunchKernel(fn, *g, *b, shared_mem, _current_stream(), params, None))


class _Module(object):
    """The fatbin, loaded once per device; get_function by the name _gpu.py uses."""

    _data = None

    def __init__(self):
        if _Module._data is None:
            with open(FATBIN, "rb") as fh:
                _Module._data = fh.read()
            with open(os.path.splitext(FATBIN)[0] + ".json") as fh:
                _Module._names = json.load(fh)
        self._mods = {}
        self._lock = threading.Lock()

    def handle(self, dev):
        with self._lock:
            if dev not in self._mods:
                m = _P()
                _check(cuModuleLoadData(ctypes.byref(m), _Module._data))
                self._mods[dev] = m
            return self._mods[dev]

    def get_function(self, name):
        sym = _Module._names.get(name)
        if sym is None:
            raise KeyError("%s is not in %s (tools/build_cuda_fatbin.py)" % (name, FATBIN))
        self.handle(device_id())  # load now: a missing architecture fails here, not at a launch
        return _Kernel(self, sym.encode())


_module = None


def build_module(source, names):
    """The precompiled kernels (`source` and `names` are for NVRTC backends: ignored)."""
    global _module
    if _module is None:
        _module = _Module()
    return _module
