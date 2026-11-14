# Example 14: Frames and powder patterns on the GPU, handed to torch, jax, cupy, numba, pycuda and tensorflow

bslz4_to_sparse.cuda decodes on an NVIDIA GPU.  With the driver backend (BSLZ4_CUDA_BACKEND=driver)
it needs no CUDA toolkit, compiler or cupy: the kernels come compiled in the package (a fatbin with
code for every GPU generation from Fermi on) and are launched through the CUDA driver, libcuda, with
ctypes.  Two calls keep their results on the GPU:

    decompress(chunks, out=None)   the dense frames: (nframes, ni, nj) of the pixel dtype
    integrate(chunks, out=None)    each frame's powder, the sums of pyFAI's sparse matrix:
                                   (nframes, nbins)

The arrays they return follow the two conventions GPU libraries use to share memory,
__cuda_array_interface__ (cupy, numba, pycuda, torch) and DLPack (torch, jax, tensorflow, cupy), so
a library takes them without a copy: it works on the same GPU memory.  The data is complete when
the call returns.  This example gives the frames, a 1D powder pattern and a 2D (q x azimuth) cake
to each of these libraries that is installed, computes something there, and checks it against the
CPU (h5py's decompression and pyFAI's integration).  A library uses the device's primary context,
as the driver backend does (pycuda needs to be told: retain_primary_context).

Whose memory: without `out`, each call returns a new array whose memory goes back to a pool (the
driver backend's, or cupy's) as soon as nothing refers to it.  With `out`, the results go into an
array you own and reuse.  The last part of this example shows both, and how to use a result
safely on a stream of your library's.

Source: [`examples/ex14_gpu_libraries.py`](https://github.com/jonwright/bslz4_to_sparse/blob/main/examples/ex14_gpu_libraries.py), run as `python examples/ex14_gpu_libraries.py`, or download the [notebook](ex14_gpu_libraries.ipynb) (figures and output as run when this page was built).

Output, from running it on real data when this page was made: `silicon_u16_zstd2_256k.h5` of the example data ([https://doi.org/10.5281/zenodo.23361659](https://doi.org/10.5281/zenodo.23361659)) repacked as lz4 by `tools/example_data.py`, with its `mask.h5` and `tools/calib.poni`.  This page is committed, made by `tools/real_data_pages.py`; CI does not run it on this data.

```python
import os
```

```python
os.environ["BSLZ4_CUDA_BACKEND"] = "driver"  # set before bslz4_to_sparse.cuda is imported
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # jax: allocate as needed
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # tensorflow: errors only
```

```text
'3'
```

```python
import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import matplotlib.pyplot as plt
import numpy as np
import pyFAI
from pyFAI.method_registry import IntegrationMethod

from bslz4_to_sparse import cuda
```

```python
h5filename = "eiger_0000.h5"  # your file: bitshuffle-lz4 (8 kB blocks), one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
mask_filename = (
    "mask.h5"  # your detector mask: a dataset "mask", pyFAI's convention (non-zero = masked)
)
poni_filename = "calib.poni"  # your pyFAI geometry
nframes = 16
```

```python
with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    dtype = ds.dtype
    n = min(nframes, ds.shape[0])
    chunks = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(n)]
    frames = ds[:n]  # h5py's own decompression, on the CPU: the reference
with h5py.File(mask_filename, "r") as h5:
    bad = h5["mask"][()] != 0  # the masked pixels
mask = (~bad).astype(np.uint8)
```

pyFAI's sparse matrices (CSC), 1D and 2D, and their normalisation (an image of ones)

```python
ai = pyFAI.load(poni_filename)
ones = np.ones(frames.shape[1:], np.float32)
m1 = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=1)
m2 = IntegrationMethod.select_one_available(("bbox", "csc", "cython"), dim=2)
nq, nazim = 500, 36
norm1 = np.asarray(ai.integrate1d(ones, nq, unit="q_nm^-1", method=m1, mask=bad).sum_normalization)
ai.integrate2d(ones, nq, nazim, unit="q_nm^-1", method=m2, mask=bad)
engine1, engine2 = ai.engines[m1].engine, ai.engines[m2].engine
```

On the GPU: the frames, the 1D powder and the 2D cake of every frame

```python
gpu1 = cuda.chunk2sparseCSC(mask, engine1, dtype)
gpu2 = cuda.chunk2sparseCSC(mask, engine2, dtype)
d_frames = gpu1.decompress(chunks)  # (n, ni, nj) uint16
d_powder = gpu1.integrate(chunks)  # (n, nq) float64: sum of weight x pixel per bin
d_cake = gpu2.integrate(chunks)  # (n, nq * nazim) float64, q-major as pyFAI's matrix
print("GPU:", cuda.device_name(0))
print("GPU arrays:", d_frames.shape, d_frames.dtype, d_powder.shape, d_cake.shape)
ptr = d_frames.__cuda_array_interface__["data"][0]
assert (d_frames.get() == frames).all()  # .get(): a host copy, without any library
ref1 = ai.integrate1d(frames[0], nq, unit="q_nm^-1", method=m1, mask=bad)
ref2 = ai.integrate2d(frames[0], nq, nazim, unit="q_nm^-1", method=m2, mask=bad)
assert np.allclose(d_powder[:1].get()[0], ref1.sum_signal, rtol=1e-9, atol=0)
assert np.allclose(d_cake[:1].get()[0].reshape(nq, nazim).T, ref2.sum_signal, rtol=1e-9, atol=0)
print("frame 0 equals h5py's frame; its powder and cake equal pyFAI's sum_signal")
```

```text
GPU: NVIDIA L40S
GPU arrays: (16, 2162, 2068) uint16 (16, 500) (16, 18000)
frame 0 equals h5py's frame; its powder and cake equal pyFAI's sum_signal
```

```python
counts = np.count_nonzero((frames > 3) & ~bad, axis=(1, 2))  # pixels above 3, for numba below
host_cake = d_cake.get().reshape(n, nq, nazim)
results = {}  # library -> what it computed, for the summary
```

```python
def with_cupy():
    import cupy as cp

    x = cp.asarray(d_frames)  # __cuda_array_interface__: no copy
    assert x.data.ptr == ptr
    mean = x.mean(axis=0, dtype=cp.float64)  # the mean frame, on the GPU
    assert np.allclose(mean.get(), frames.mean(axis=0, dtype=np.float64))
    return "mean frame of %d, same memory" % n
```

```python
def with_torch():
    import torch

    p = torch.from_dlpack(d_powder)  # DLPack: no copy
    intensity = p / torch.as_tensor(norm1, device=p.device).clamp(min=1e-30)  # pyFAI's intensity
    assert np.allclose(intensity[0].cpu().numpy(), ref1.intensity, rtol=1e-6, atol=0)
    x = torch.as_tensor(d_frames, device="cuda")  # or __cuda_array_interface__
    assert x.data_ptr() == ptr
    return "normalised 1D patterns, (%d, %d), same memory" % tuple(intensity.shape)
```

```python
def with_jax():
    """jax computes in float32 unless told otherwise (x64), so three ways to take the cake.  Not
    the int64 sums of a fixed_point_matrix: without x64 jax turns them into int32, wrapping each
    sum above 2**31 without a word; take them under x64 only."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import enable_x64

    sums = host_cake.reshape(n, -1)  # the float64 sums, as the GPU gave them
    cai_ptr = lambda a: a.__cuda_array_interface__["data"][0]

    # 1. exact: our float64 with jax's x64, here for one block (JAX_ENABLE_X64=1: the whole
    #    program).  No copy.
    with enable_x64():
        x = jax.dlpack.from_dlpack(d_cake)
        assert x.dtype == np.float64 and x.unsafe_buffer_pointer() == cai_ptr(d_cake)
        assert (np.asarray(x) == sums).all()
        c = x.reshape(n, nq, nazim)
        spread = jnp.std(c, axis=2) / jnp.maximum(jnp.mean(c, axis=2), 1e-30)  # azimuthal
        spread = np.asarray(spread)  # variation of each q bin
    want = host_cake.std(axis=2) / np.maximum(host_cake.mean(axis=2), 1e-30)
    assert np.allclose(spread, want, rtol=1e-12, atol=0)

    # 2. fast: float32 sums on the GPU (acc="f32", relative error about 1e-6), jax's default
    #    float32.  No copy.
    d_cake32 = cuda.chunk2sparseCSC(mask, engine2, dtype, acc="f32").integrate(chunks)
    x = jax.dlpack.from_dlpack(d_cake32)
    assert x.dtype == np.float32 and x.unsafe_buffer_pointer() == cai_ptr(d_cake32)
    nz = sums != 0
    err32 = (np.abs(np.asarray(x, np.float64) - sums)[nz] / np.abs(sums[nz])).max()
    assert np.allclose(np.asarray(x), sums, rtol=1e-6, atol=0)

    # 3. our float64 into jax's default float32: from_dlpack silently makes a float32 copy, each
    #    sum rounded.  The copy is jax's own: keep ours until it is done.
    x = jax.dlpack.from_dlpack(d_cake)
    x.block_until_ready()
    assert x.dtype == np.float32 and x.unsafe_buffer_pointer() != cai_ptr(d_cake)
    assert (np.asarray(x) == sums.astype(np.float32)).all()
    return ("cake 3 ways: float64 under x64, same memory, exact; acc='f32', same memory "
            "(largest relative error %.1e); float64 to float32, copied and rounded" % err32)
```

```python
def with_numba():
    from numba import cuda as ncuda

    @ncuda.jit
    def count_above(fr, msk, cut, out):  # one thread per (frame, row): pixels above the cut
        f, i = ncuda.grid(2)
        if f < fr.shape[0] and i < fr.shape[1]:
            k = 0
            for j in range(fr.shape[2]):
                if fr[f, i, j] > cut and msk[i, j]:
                    k += 1
            ncuda.atomic.add(out, f, k)

    x = ncuda.as_cuda_array(d_frames)  # __cuda_array_interface__: no copy
    assert x.device_ctypes_pointer.value == ptr
    out = ncuda.to_device(np.zeros(n, np.int64))
    count_above[(n, (x.shape[1] + 127) // 128), (1, 128)](x, ncuda.to_device(mask), 3, out)
    assert (out.copy_to_host() == counts).all()
    return "a kernel of its own: pixels above 3 per frame"
```

```python
def with_pycuda():
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray

    drv.init()
    ctx = drv.Device(0).retain_primary_context()  # the context the arrays live in
    ctx.push()
    try:
        x = gpuarray.GPUArray(d_frames.shape, d_frames.dtype, gpudata=ptr)  # no copy
        top = [int(gpuarray.max(x[i]).get()) for i in range(n)]
    finally:
        ctx.pop()
    assert top == [int(f.max()) for f in frames]
    return "brightest pixel of each frame"
```

```python
def with_tensorflow():
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    x = tf.experimental.dlpack.from_dlpack(d_frames.__dlpack__())  # DLPack: no copy
    total = tf.reduce_sum(tf.cast(x, tf.float64), axis=(1, 2)).numpy()
    assert np.allclose(total, frames.sum(axis=(1, 2), dtype=np.float64))
    return "sum of each frame"
```

```python
for name, fn in [("cupy", with_cupy), ("torch", with_torch), ("jax", with_jax),
                 ("numba", with_numba), ("pycuda", with_pycuda), ("tensorflow", with_tensorflow)]:
    try:
        __import__(name)
    except ImportError:
        results[name] = "not installed"
        continue
    results[name] = fn()
for name, what in results.items():
    print("%-10s %s" % (name, what))
```

```text
cupy       mean frame of 16, same memory
torch      normalised 1D patterns, (16, 500), same memory
jax        cake 3 ways: float64 under x64, same memory, exact; acc='f32', same memory (largest relative error 1.5e-07); float64 to float32, copied and rounded
numba      a kernel of its own: pixels above 3 per frame
pycuda     brightest pixel of each frame
tensorflow sum of each frame
```

Whose memory.  Every array above is ours: a block of the pool, given back as soon as the last
reference to it is dropped, a library's view (torch.from_dlpack, cp.asarray, ...) included.  The
next call may then get the same block and write it while work your library queued earlier
still reads it: on any stream of its own (torch.cuda.Stream, jax's), and with the cupy backend
on the default stream too.  Three ways to use a result safely:

  1. out=: the results go into an array your library owns, made once and reused, and nothing
     is allocated per call.  It must be on this GPU, C-contiguous and of the result's dtype
     (decompress: the pixels'; integrate: float64 here).  The call writes it as soon as it
     starts, on a stream of its own: whatever your library still has queued on that array must
     be finished first.  cupy tells us its stream and we wait for it; torch does not, so
     synchronize torch before each call.
  2. a copy your library owns, then drop ours.  The copy is queued on your library's stream:
     keep our array until the copy is done (synchronize, block_until_ready), or the pool may
     give its block to the next call while the copy is still reading it.
  3. ours, used in place on a stream of your library's: name that stream with
     d.record_stream(stream) (the driver backend's arrays, as torch's Tensor.record_stream).
     The block then goes back to the pool only once the work queued on that stream before
     the drop is done, so ours may be dropped at once.  Not for jax, whose streams are not
     public: use 2 there.

```python
kept = {}
if results.get("cupy") != "not installed":
    import cupy as cp

    c_frames = cp.empty(frames.shape, dtype)  # 1. cupy's memory, ours to reuse
    assert gpu1.decompress(chunks, out=c_frames) is c_frames
    assert (c_frames.get() == frames).all()
    kept["cupy"] = "decompress(chunks, out=cupy array)"
if results.get("torch") != "not installed":
    import torch

    t_powder = torch.empty((n, nq), dtype=torch.float64, device="cuda")  # 1. torch's memory
    total = torch.zeros(nq, dtype=torch.float64, device="cuda")
    for _ in range(3):  # a loop over the same frames, as over the batches of a scan
        torch.cuda.current_stream().synchronize()  # torch is done with t_powder: it may be written
        gpu1.integrate(chunks, out=t_powder)
        total += t_powder.sum(dim=0)  # queued on torch's stream, reading t_powder
    assert np.allclose(total.cpu().numpy(), 3 * d_powder.get().sum(axis=0), rtol=1e-12, atol=0)

    d = gpu1.integrate(chunks)  # 2. a new array of ours ...
    mine = torch.from_dlpack(d).clone()  # ... copied into torch's memory, on torch's stream
    torch.cuda.current_stream().synchronize()  # the copy is done ...
    del d  # ... so our block may go back to the pool
    assert (mine.cpu().numpy() == d_powder.get()).all()

    s = torch.cuda.Stream()  # 3. a stream of torch's own
    d = gpu1.integrate(chunks)
    with torch.cuda.stream(s):  # queued on s, reading ours in place
        intensity = torch.from_dlpack(d) / torch.as_tensor(norm1, device="cuda").clamp(min=1e-30)
    d.record_stream(s)  # s reads d ...
    del d  # ... so the block waits for s's work before it goes back to the pool
    s.synchronize()
    assert (intensity.cpu().numpy() == d_powder.get() / np.maximum(norm1, 1e-30)).all()
    kept["torch"] = ("integrate(chunks, out=torch tensor) in a loop; a clone of ours; ours on a "
                     "torch stream, recorded")
if results.get("jax") != "not installed":
    import jax
    import jax.numpy as jnp

    from jax.experimental import enable_x64

    d = gpu2.integrate(chunks)  # 2. jax arrays cannot be written: jax keeps a copy of ours
    with enable_x64():  # float64 kept as it is (without x64 the copy is float32, see above)
        mine = jnp.array(jax.dlpack.from_dlpack(d), copy=True)
        mine.block_until_ready()  # the copy is done ...
        assert mine.unsafe_buffer_pointer() != d.__cuda_array_interface__["data"][0]
        del d  # ... so our block may go back to the pool
        assert mine.dtype == np.float64 and (np.asarray(mine) == host_cake.reshape(n, -1)).all()
    kept["jax"] = "a copy of ours"
for name, what in kept.items():
    print("%-10s kept: %s" % (name, what))
```

```text
cupy       kept: decompress(chunks, out=cupy array)
torch      kept: integrate(chunks, out=torch tensor) in a loop; a clone of ours; ours on a torch stream, recorded
jax        kept: a copy of ours
```

The other direction: chunks already in GPU memory, as a GPUDirect Storage read (kvikio, cuFile)
would leave them, one buffer with the chunks at known offsets.  Here a copy made by one of the
libraries stands in for that read.  *_from_device read the chunks in place: nothing goes back to
the host, and the chunk headers are checked on the GPU.

```python
lengths = np.array([len(c) for c in chunks], np.int64)
offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
blob = np.frombuffer(b"".join(chunks), np.uint8)
d_chunks = None
if results.get("cupy") != "not installed":
    import cupy as cp

    d_chunks = cp.asarray(blob)
elif results.get("torch") != "not installed":
    import torch

    d_chunks = torch.from_numpy(blob.copy()).cuda()
if d_chunks is not None:
    d_powder2 = gpu1.integrate_from_device(d_chunks, offsets, lengths)
    assert (d_powder2.get() == d_powder.get()).all()
    assert (gpu1.decompress_from_device(d_chunks, offsets, lengths).get() == frames).all()
    print("chunks in a %s GPU buffer: integrate_from_device and decompress_from_device give the "
          "same results" % type(d_chunks).__module__.split(".")[0])
```

```text
chunks in a cupy GPU buffer: integrate_from_device and decompress_from_device give the same results
```

The 1D pattern and the cake of all the frames, as the GPU gave them

```python
q, chi = ref2.radial, ref2.azimuthal
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(ref1.radial, d_powder.get().sum(axis=0) / np.maximum(norm1, 1e-30))
axes[0].set_xlabel("q (nm$^{-1}$)")
axes[0].set_title("%d frames: integrate() / sum_normalization" % n)
axes[1].imshow(np.log1p(host_cake.sum(axis=0).T), aspect="auto", origin="lower",
               extent=(q[0], q[-1], chi[0], chi[-1]))
axes[1].set_xlabel("q (nm$^{-1}$)")
axes[1].set_ylabel("azimuth (deg)")
axes[1].set_title("%d frames: the 2D cake, log(1 + sum)" % n)
plt.tight_layout()
plt.show()
```

![figure 1](ex14_gpu_libraries_files/fig1.png)

<small>Generated by `tools/real_data_pages.py` from bslz4_to_sparse 0.0.22a3 (git `v0.0.22.a3-30-g958fc04`, sources sha256 `99238cbd72a8`).</small>
