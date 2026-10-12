"""The conveyor: storage (or network) -> memory -> decoders -> sinks, in pure Python.

Not to be confused with the *pipeline* (decode, mask, untranspose, collect, route, dot), the chain
of steps one decoder runs on every block.  The conveyor moves work to the decoders:

    units     consecutive frames of a file, one large contiguous read each (make_units), seconds of
              work, so workers synchronise with the rest about once a second
    slots     preallocated buffers, one set per worker, first touched by the worker that owns them
    readers   take a unit and a free slot, read the unit into the slot in file order, hand it over
    workers   decode all the frames of a unit straight from the slot, give the sink the result
    sinks     where results go (a file format is the sink's business: see Sink)

Workers are threads or forked processes (run(mode=...)) around the same code; the decoders release
the GIL, so threads are the default.  Pure Python 3, numpy only: the chunk index of each file is read from a sidecar
(tools/index_chunks.py builds them, with h5py, once), so reading touches no HDF5; cupy for the GPU backend.
"""

from ._index import read_index, write_index, harvest, sidecar_path, StaleIndex
from ._source import make_units
from ._sinks import Sink, NullSink
from ._backends import CpuBackend, GpuBackend
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
    "GpuBackend",
    "run",
    "numa_nodes",
]
