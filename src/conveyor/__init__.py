"""The conveyor: storage (or network) -> memory -> decoders -> sinks, in pure Python.

Not to be confused with the *pipeline* (decode, mask, untranspose, collect, route, dot), the chain
of steps one decoder runs on every block.  The conveyor moves work to the decoders:

    units     consecutive frames of a file, one large contiguous read each (make_units), seconds of
              work, so workers synchronise with the rest about once a second
    workers   each owns two buffers it first touches itself and a reader; decodes all the frames of
              a unit straight from a buffer, gives the sink the result
    readers   one per worker: when a buffer is free, take the next unit from the shared queue and
              read it with large preadv calls, in file order, while the worker decodes the previous
              one (a worker that decodes faster takes more units: the load balances itself)
    sinks     where results go (a file format is the sink's business: see Sink)

Workers are threads or forked processes (run(mode=...)) around the same code; the decoders release
the GIL, so threads are the default.  Pure Python 3, numpy only: the chunk index of each file is
read from a sidecar file (tools/index_chunks.py builds them, with h5py, once), so reading touches
no HDF5; cupy is needed only for the GPU backend.
"""

from ._index import read_index, write_index, harvest, sidecar_path, StaleIndex
from ._source import make_units
from ._sinks import Sink, NullSink
from ._backends import CpuBackend, DecompressBackend, GpuBackend
from ._engine import run, numa_nodes

__all__ = [
    "make_units",
    "read_index",
    "write_index",
    "harvest",
    "sidecar_path",
    "StaleIndex",
    "Sink",
    "NullSink",
    "CpuBackend",
    "DecompressBackend",
    "GpuBackend",
    "run",
    "numa_nodes",
]
