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
cuMemsetD8Async = _sig("cuMemsetD8Async", ctypes.c_uint64, ctypes.c_ubyte, _SZ, _P)
cuStreamCreate = _sig("cuStreamCreate", ctypes.POINTER(_P), ctypes.c_uint)
cuStreamDestroy = _sig("cuStreamDestroy_v2", _P)
cuStreamSynchronize = _sig("cuStreamSynchronize", _P)
cuGetErrorString = _sig("cuGetErrorString", ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))

CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
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
        self.ctx, p = _context(device_id()), ctypes.c_uint64()
        _check(cuMemAlloc(ctypes.byref(p), max(int(nbytes), 1)))
        self.ptr = p.value

    def __del__(self):
        if getattr(self, "ptr", None):
            _in_context(self.ctx, cuMemFree, self.ptr)


class ndarray(object):
    """A C-contiguous device array: slicing along the first axis, .set, .get, .view, .size."""

    def __init__(self, shape, dtype, alloc=None, offset=0):
        self.shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.nbytes = self.size * self.dtype.itemsize
        self._alloc = alloc if alloc is not None else _Alloc(self.nbytes)
        self.ptr = self._alloc.ptr + offset

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
