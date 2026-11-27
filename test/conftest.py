from __future__ import absolute_import, division, print_function

import os
import sys
import threading
import time

import pytest

_p = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _p:
    sys.path.insert(0, _p)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

# The kwargs h5py.create_dataset takes for a bitshuffle chunk, from the raw HDF5 filter parameters
# (hdf5plugin 2.3.2, Python 2.7, has no cname/clevel; given zstd it would write lz4 under a zstd
# label, so zstd raises there instead: needs_zstd_writer).  One copy, in tools/standin.py.
from standin import bitshuffle  # noqa: E402,F401


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
    """hdf5plugin writes bitshuffle-zstd: its Bitshuffle takes cname="zstd".  hdf5plugin 2.3.2
    (Python 2.7) has no cname, and given the raw options (nelems, 3) it writes lz4 under a zstd
    label, so bitshuffle() below asks Bitshuffle for zstd and this asks the same."""
    try:
        import hdf5plugin
    except ImportError:
        return False
    try:
        hdf5plugin.Bitshuffle(nelems=0, cname="zstd")
    except TypeError:
        return False
    return True


def _chunk_info_available():
    """h5py's DatasetID.get_chunk_info (locate_chunks and the conveyor sidecar index need it)."""
    try:
        import h5py
    except ImportError:
        return False
    return hasattr(h5py.h5d.DatasetID, "get_chunk_info")


_ZSTD_WRITER = _zstd_writer_available()


def zstd_writer_available():
    return _ZSTD_WRITER


needs_zstd_writer = pytest.mark.skipif(not _ZSTD_WRITER, reason="hdf5plugin cannot write bitshuffle-zstd here")
_CHUNK_INFO = _chunk_info_available()


def chunk_info_available():
    return _CHUNK_INFO


needs_chunk_info = pytest.mark.skipif(not _CHUNK_INFO, reason="h5py lacks DatasetID.get_chunk_info")


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
