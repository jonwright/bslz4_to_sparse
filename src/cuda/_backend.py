"""The only module that imports cupy: the CUDA calls the GPU classes need, so another binding (a ctypes layer on the driver
API, cuda-core) can replace it without touching the kernels or _gpu.py.

Rules for any backend: never create a CUDA context of our own (use the device's primary / current context, which cupy, torch
and jax share); a device array supports slicing, `.set(host_array, stream)`, `.get(out=host_array)` and `.view(dtype)`, and
can be passed to a kernel; pinned host buffers support the buffer protocol.
"""
import cupy as cp


def device_id():
    """The current device."""
    return cp.cuda.Device().id


def use_device(dev):
    """Make `dev` current; another library may have made a different device or context current."""
    cp.cuda.Device(dev).use()


def device(dev):
    """Context manager: `dev` current inside the block."""
    return cp.cuda.Device(dev)


def synchronize(dev):
    cp.cuda.Device(dev).synchronize()


def Stream():
    """A new non-blocking stream; .use() makes it current, .synchronize() waits for it."""
    return cp.cuda.Stream(non_blocking=True)


def empty(shape, dtype):
    return cp.empty(shape, dtype)


def zeros(shape, dtype):
    return cp.zeros(shape, dtype)


def asarray(host):
    """A device copy of a host array."""
    return cp.asarray(host)


def pinned(nbytes):
    """Page-locked host memory (buffer protocol)."""
    return cp.cuda.alloc_pinned_memory(nbytes)


def build_module(source, names):
    """Compile CUDA C++ `source` with NVRTC; `names` are the template instantiations to export.  The module has
    .get_function(name) returning a callable kernel(grid, block, args).  Older NVRTC versions know fewer C++ standards."""
    err = None
    for opts in (("--std=c++17",), ("--std=c++14",), ()):
        try:
            m = cp.RawModule(code=source, options=opts, name_expressions=names)
            m.get_function(names[0])                    # compiles now
            return m
        except Exception as e:                          # noqa: BLE001 (cupy CompileException)
            err = e
    raise err
