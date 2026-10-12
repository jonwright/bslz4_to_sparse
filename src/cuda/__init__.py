"""GPU (CUDA) chunk2sparse and chunk2sparseCSC: the CPU classes' call API, decoding on the GPU.
Optional: importing this needs cupy and a CUDA GPU.  See _gpu.py for the differences.

Porting to other devices (design note): the lz4 decoder is the only part tied to NVIDIA's 32-lane
warps.  A future port, or zstd on the GPU, is expected to decompress on the CPU (without the bit
shuffle) and then run the shuffle, the matrix product and the collect on the device; those kernels
are small and need only a subgroup shuffle/ballot, shared memory and a barrier.  The CPU classes are
the test oracle for any such port."""

from ._gpu import chunk2sparse, chunk2sparseCSC, release_shared

__all__ = ["chunk2sparse", "chunk2sparseCSC", "release_shared"]
