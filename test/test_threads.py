"""The compute entry points release the GIL: one processor object per thread
(own workspace and output buffers, no locks) decodes concurrently and gives
the same answer as a serial run, and a pure-Python thread keeps running while
a decode is in flight."""

import os
import sys
import threading
import time

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

import bslz4_to_sparse as b

SHAPE = (1024, 1024)


def _chunks(nframes, seed):
    rng = np.random.default_rng(seed)
    frames = rng.poisson(0.05, (nframes,) + SHAPE).astype(np.uint16)
    out = []
    with h5py.File("t.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + SHAPE,
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"))
        for i in range(nframes):
            out.append(ds.id.read_direct_chunk((i, 0, 0))[1])
    return frames, out


def test_concurrent_processors_match_serial():
    mask = np.ones(SHAPE, np.uint8)
    nthreads = 4
    data = [_chunks(8, s) for s in range(nthreads)]
    results = [None] * nthreads
    errors = []

    def work(k):
        try:
            c = b.chunk2sparse(mask)           # one processor object per thread
            for _ in range(20):
                n, (v, idx) = c.multi(data[k][1], 0)
            results[k] = (n.copy(), v.copy(), idx.copy())
        except Exception as e:                 # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(nthreads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    for k in range(nthreads):
        n, v, idx = results[k]
        frames = data[k][0]
        for i in range(frames.shape[0]):
            ref = np.flatnonzero(frames[i].ravel() > 0)
            assert n[i] == ref.size
            assert (idx[i, :n[i]] == ref).all()
            assert (v[i, :n[i]] == frames[i].ravel()[ref]).all()


def test_python_thread_runs_during_decode():
    mask = np.ones(SHAPE, np.uint8)
    frames, chunks = _chunks(8, 0)
    c = b.chunk2sparse(mask)
    ticks = [0]
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            ticks[0] += 1
            time.sleep(0)

    t = threading.Thread(target=spin)
    t.start()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        c.multi(chunks, 0)
    stop.set()
    t.join()
    # with the GIL held throughout the decode the spinner would barely tick
    assert ticks[0] > 1000
