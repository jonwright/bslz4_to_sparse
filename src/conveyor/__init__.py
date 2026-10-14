"""The conveyor: storage (or network) -> memory -> decoders -> sinks, in pure Python.

A worked example of the kernels at scale, and a test of them: nothing in the kernels needs it, and
its file formats are the example's choice.

Not to be confused with the *pipeline* (decode, mask, untranspose, collect, route, dot), the chain
of steps one decoder runs on every block.  The conveyor moves work to the decoders:

    units     consecutive frames of a file, one large contiguous read each (make_work_units), seconds of
              work, so workers synchronise with the rest about once a second
    workers   each owns two buffers it first touches itself and a reader; decodes all the frames of
              a unit straight from a buffer, gives the sink the result
    readers   one per worker: when a buffer is free, take the next unit from the shared queue and
              read it with large preadv calls, in file order, while the worker decodes the previous
              one (a worker that decodes faster takes more units: the load balances itself)
    sinks     where results go (a file format is the sink's business: see Sink)

Workers are threads or forked processes (run(mode=...)) around the same code; the decoders release
the GIL, so threads are the default.  Pure Python 3: reading and decoding need numpy only, the chunk index of each file is read from a
sidecar file (tools/index_chunks.py builds them, with h5py, once), so reading touches no HDF5.  The
parts that name them need scipy and pyFAI (Integrator), h5py and hdf5plugin (the writers),
ImageD11 (min_pixels) or cupy (the GPU backend).  The writers refuse, before the run starts, a powder
of more than a tenth of the uncompressed frame (check_powder_size): the point is to reduce data.
"""

from ._index import read_index, write_index, harvest, sidecar_path, StaleIndex, Index
from ._source import make_work_units, WorkUnit
from ._sinks import Sink, NullSink, TeeSink, check_powder_size
from ._backends import CpuBackend, DecompressBackend, GpuBackend
from ._iter import iterate
from ._integrator import Integrator, IntegratorStack
from ._clip import fixed_weights, clip_exact_powder
from ._ringfile import RingSink, peak_windows, rings_name
from ._sparsefile import ImageD11SparseSink, imaged11_name
from ._engine import run, report, numa_nodes

__all__ = [
    "make_work_units",
    "WorkUnit",
    "Index",
    "read_index",
    "write_index",
    "harvest",
    "sidecar_path",
    "StaleIndex",
    "Sink",
    "NullSink",
    "TeeSink",
    "check_powder_size",
    "CpuBackend",
    "DecompressBackend",
    "GpuBackend",
    "run",
    "iterate",
    "Integrator",
    "IntegratorStack",
    "ImageD11SparseSink",
    "RingSink",
    "peak_windows",
    "rings_name",
    "fixed_weights",
    "clip_exact_powder",
    "imaged11_name",
    "report",
    "numa_nodes",
]
