"""The only module that imports cupy: the CUDA calls the GPU classes need, so another binding (a
ctypes layer on the driver API, cuda-core) can replace it without touching the kernels or _gpu.py.

Rules for any backend: use the device's primary context, which cupy, torch and jax share (the
driver backend makes a context of its own only on drivers older than CUDA 7.0, which have none); a device array supports slicing, `.set(host_array,
stream)`, `.get(out=host_array)` and `.view(dtype)`, and
can be passed to a kernel; pinned host buffers support the buffer protocol.
"""

import cupy as cp


def device_id():
    """The current device."""
    return cp.cuda.Device().id


def device_count():
    """The number of CUDA devices."""
    return cp.cuda.runtime.getDeviceCount()


def device_name(dev):
    """The name of device `dev`."""
    name = cp.cuda.runtime.getDeviceProperties(dev)["name"]
    return name.decode() if isinstance(name, bytes) else name


def use_device(dev):
    """Make `dev` current; another library may have made a different device or context current."""
    cp.cuda.Device(dev).use()


def device(dev):
    """Context manager: `dev` current inside the block."""
    return cp.cuda.Device(dev)


def l2_bytes(dev):
    """The L2 cache size of device `dev` in bytes."""
    return int(cp.cuda.Device(dev).attributes["L2CacheSize"])


def max_grid_x(dev):
    """The most blocks along x of a grid on device `dev` (65535 before compute capability 3.0)."""
    return int(cp.cuda.Device(dev).attributes["MaxGridDimX"])


def memory_bytes(dev):
    """The memory of device `dev` in bytes."""
    return int(cp.cuda.runtime.getDeviceProperties(dev)["totalGlobalMem"])


def synchronize(dev):
    """Wait for everything queued on device `dev`."""
    cp.cuda.Device(dev).synchronize()


def Stream():
    """A new non-blocking stream; .use() makes it current, .synchronize() waits for it."""
    return cp.cuda.Stream(non_blocking=True)


def empty(shape, dtype):
    """An uninitialised device array."""
    return cp.empty(shape, dtype)


def device_bytes(obj, dev):
    """The GPU memory of obj (__cuda_array_interface__, C-contiguous, on device dev) as a uint8
    array viewing it; cupy waits for obj's stream."""
    a = cp.asarray(obj)
    if a.device.id != dev:
        raise ValueError("the buffer is on GPU %d, this object on GPU %d" % (a.device.id, dev))
    if not a.flags.c_contiguous:
        raise ValueError("the chunks must be in one C-contiguous GPU buffer")
    return a.reshape(-1).view(cp.uint8)


def copy_rows(dst, dst_pitch, src, src_pitch, width, height):
    """height rows of width bytes, device to device, on the current stream"""
    cp.cuda.runtime.memcpy2DAsync(
        dst.data.ptr, dst_pitch, src.data.ptr, src_pitch, width, height,
        cp.cuda.runtime.memcpyDeviceToDevice, cp.cuda.get_current_stream().ptr,
    )


def zeros(shape, dtype):
    """A zeroed device array."""
    return cp.zeros(shape, dtype)


def asarray(host):
    """A device copy of a host array."""
    return cp.asarray(host)


def pinned(nbytes):
    """Page-locked host memory (buffer protocol)."""
    return cp.cuda.alloc_pinned_memory(nbytes)


def build_module(source, names):
    """Compile CUDA C++ `source` with NVRTC; `names` are the template instantiations to export.  The
    module has .get_function(name) returning a callable kernel(grid, block, args).  Older NVRTC
    versions know fewer C++ standards."""
    err = None
    for opts in (("--std=c++17",), ("--std=c++14",), ()):
        try:
            m = cp.RawModule(code=source, options=opts, name_expressions=names)
            m.get_function(names[0])  # compiles now
            return m
        except Exception as e:  # noqa: BLE001 (cupy CompileException)
            err = e
    raise err
