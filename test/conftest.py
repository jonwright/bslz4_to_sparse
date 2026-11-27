from __future__ import absolute_import, division, print_function

import os
import sys
import threading
import time

import pytest

_p = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _p:
    sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def _counters_off_after_each_test():
    """A test that counted blocks (_testing.reset_counters) must not leave counting on for the
    tests after it: decoding threads would contend for the counters' cache lines."""
    yield
    try:
        from bslz4_to_sparse import _testing
    except Exception:  # the package is not importable in this environment: nothing to switch off
        return
    _testing.stop_counters()


def _zstd_writer_available():
    """hdf5plugin can write bitshuffle-zstd: a (1,64,64) uint16 chunk written with codec 32008
    (options (0,3)) carries the zstd magic b\"\\x28\\xb5\\x2f\\xfd\" at bytes 16:20.  hdf5plugin
    2.3.2 accepts codec 3 but silently writes lz4, so the capability is measured, not assumed."""
    import h5py
    import hdf5plugin  # registers filter 32008 (bshuf) with HDF5
    import tempfile
    import numpy as np
    import bslz4_to_sparse as b

    d = tempfile.mkdtemp()
    try:
        fn = os.path.join(d, "zstd.h5")
        with h5py.File(fn, "w") as h:
            h.create_dataset("d", data=np.zeros((1, 64, 64), np.uint16), chunks=(1, 64, 64),
                             compression=32008, compression_opts=(0, 3))
        with h5py.File(fn, "r") as h:
            chunk = b.read_direct_chunk(h["d"], (0, 0, 0))[1]
        return len(chunk) > 20 and chunk[16:20] == b"\x28\xb5\x2f\xfd"
    except Exception:
        return False
    finally:
        os.remove(fn)
        os.rmdir(d)


def _chunk_info_available():
    """h5py's DatasetID.get_chunk_info (locate_chunks and the conveyor sidecar index need it)."""
    import h5py
    import tempfile
    import numpy as np

    d = tempfile.mkdtemp()
    try:
        fn = os.path.join(d, "ci.h5")
        with h5py.File(fn, "w") as h:
            h.create_dataset("d", data=np.zeros((2, 8, 8), np.uint8), chunks=(1, 8, 8))
        with h5py.File(fn, "r") as h:
            return hasattr(h["d"].id, "get_chunk_info")
    except Exception:
        return False
    finally:
        os.remove(fn)
        os.rmdir(d)


needs_zstd_writer = pytest.mark.skipif(
    not _zstd_writer_available(), reason="hdf5plugin cannot write bitshuffle-zstd here"
)
needs_chunk_info = pytest.mark.skipif(
    not _chunk_info_available(), reason="h5py lacks DatasetID.get_chunk_info"
)


class Barrier(object):
    """threading.Barrier's wait(), for the threading tests, as the barrier was added to
    threading only in Python 3.2 and the tests must run on Python 2.7 too."""

    def __init__(self, parties):
        self._parties = parties
        self._cond = threading.Condition()
        self._count = 0
        self._gen = 0

    def wait(self, timeout=None):
        with self._cond:
            gen = self._gen
            self._count += 1
            if self._count == self._parties:
                self._count = 0
                self._gen += 1
                self._cond.notify_all()
                return True
            if timeout is None:
                while gen == self._gen:
                    self._cond.wait()
                return True
            deadline = time.time() + timeout
            while gen == self._gen:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RuntimeError("the barrier timed out")
                self._cond.wait(remaining)


def bitshuffle(nelems=0, cname="lz4", clevel=3):
    """The kwargs h5py.create_dataset takes to write a bitshuffle chunk, from the raw HDF5 filter
    parameters rather than hdf5plugin.Bitshuffle, whose constructor differs between hdf5plugin 2.3.2
    (Python 2.7: Bitshuffle(nelems, lz4), no cname/clevel) and modern hdf5plugin.  The LZ4 chunk is
    byte-identical to Bitshuffle(nelems, cname=\"lz4\"); the zstd chunk to
    Bitshuffle(nelems, cname=\"zstd\", clevel=clevel).  Only the LZ4 path is usable in both
    environments: zstd writing needs the real zstd filter (needs_zstd_writer)."""
    codec = {"none": 0, "lz4": 2, "zstd": 3}[cname]
    if codec == 3:
        return dict(compression=32008, compression_opts=(nelems, codec, clevel))
    return dict(compression=32008, compression_opts=(nelems, codec))
