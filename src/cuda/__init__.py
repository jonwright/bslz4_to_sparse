"""GPU (CUDA) chunk2sparse and chunk2sparseCSC: the CPU classes' call API, decoding on the GPU.
Optional: importing this needs cupy and a CUDA GPU.  See _gpu.py for the differences."""
from ._gpu import chunk2sparse, chunk2sparseCSC, release_shared

__all__ = ["chunk2sparse", "chunk2sparseCSC", "release_shared"]
