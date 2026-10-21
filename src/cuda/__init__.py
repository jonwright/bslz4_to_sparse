"""GPU (CUDA) chunk2sparse and chunk2sparseCSC: the CPU classes' call API, decoding on the GPU.
Optional: importing this needs a CUDA GPU.  Two backends (`backend` says which is in use): cupy,
which compiles the kernels with NVRTC, and the driver backend, which needs only the NVIDIA driver
(the kernels are precompiled in bslz4_kernels.fatbin).  cupy is used when it imports, else the
driver backend; BSLZ4_CUDA_BACKEND=cupy or =driver chooses.  See _gpu.py for the differences.

Porting to other devices (design note): the lz4 decoder is the only part tied to NVIDIA's 32-lane
warps.  A future port, or zstd on the GPU, is expected to decompress on the CPU (without the bit
shuffle) and then run the shuffle, the matrix product and the collect on the device; those kernels
are small and need only a subgroup shuffle/ballot, shared memory and a barrier.  The CPU classes are
the test oracle for any such port."""

from ._gpu import backend, chunk2sparse, chunk2sparseCSC, device_count, device_name, release_shared

__all__ = [
    "backend", "chunk2sparse", "chunk2sparseCSC", "device_count", "device_name", "release_shared"
]
