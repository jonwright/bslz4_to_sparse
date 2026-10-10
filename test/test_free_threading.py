"""Free-threaded Python (3.14t, 3.15t): importing the extension must not
re-enable the GIL, and one processor object per thread decodes concurrently
with the same answers as a serial run.  Uses only the decode matrix fixture
(no h5py), so it runs wherever test_decode_matrix.py does."""

import os
import subprocess
import sys
import sysconfig
import threading

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

from bslz4_to_sparse import chunk2sparse

from test_decode_matrix import _DTYPES, _INDEX, H5_PATH

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded (t) Python")
def test_import_keeps_gil_disabled():
    env = dict(os.environ)
    env.pop("PYTHON_GIL", None)          # PYTHON_GIL=0 would hide a missing declaration
    code = ("import sys; import bslz4_to_sparse; "
            "sys.exit(1 if sys._is_gil_enabled() else 0)")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0, "the GIL was re-enabled on import:\n" + r.stdout.decode()


@pytest.mark.skipif(_INDEX is None, reason="decode matrix fixture absent")
def test_concurrent_decode_matrix():
    entries = _INDEX["chunks"]
    with open(H5_PATH, "rb") as fh:
        chunks = []
        for e in entries:
            fh.seek(e["offset"])
            chunks.append(fh.read(e["length"]))
    expected = [dict((int(i), v) for i, v in e["nonzero"]) for e in entries]
    nthreads = 8
    start = threading.Barrier(nthreads)
    errors = []

    def work(k):
        try:
            # one processor object per (thread, entry): no sharing
            procs = [chunk2sparse(np.ones((e["rows"], e["cols"]), np.uint8),
                                  dtype=_DTYPES[e["dtype"]], codec=e["codec"])
                     for e in entries]
            start.wait()
            for rep in range(10):
                for j in range(len(entries)):
                    j = (j + k) % len(entries)            # threads out of step
                    npx, (vals, adrs) = procs[j](chunks[j], entries[j]["cut"])
                    got = dict((int(adrs[i]), vals[i].item()) for i in range(npx))
                    if got != expected[j]:
                        errors.append((k, rep, entries[j]["name"]))
                        return
        except Exception as e:                            # pragma: no cover
            errors.append((k, repr(e)))

    ts = [threading.Thread(target=work, args=(k,)) for k in range(nthreads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors
