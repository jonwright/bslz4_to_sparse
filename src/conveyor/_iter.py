"""iterate: take the results in memory, from a plain for loop, while the conveyor runs behind it."""

import queue
import threading

from ._engine import run
from ._sinks import Sink

_END = object()


class _Closed(Exception):
    """the consumer stopped iterating"""


class _QueueSink(Sink):
    """Copies each batch, trimmed to the pixels it holds, onto the consumer's queue."""

    def __init__(self, q, closed):
        self.q, self.closed = q, closed

    def write(self, file, frame0, npx, values, indices, powder):
        """Hand one batch to the consumer (blocks while the consumer is behind)."""
        m = int(npx.max()) if len(npx) else 0
        item = (
            file,
            frame0,
            npx.copy(),
            values[:, :m].copy() if indices is not None else values.copy(),
            None if indices is None else indices[:, :m].copy(),
            None if powder is None else powder.copy(),
        )
        while not self.closed.is_set():
            try:
                self.q.put(item, timeout=0.5)
                return
            except queue.Full:
                pass
        raise _Closed()


def iterate(units, backend, workers, depth=4, **kw):
    """A generator of (file, frame0, npx, values, indices, powder), one tuple per batch of frames,
    the arguments of Sink.write but copies the consumer may keep.  Batches come in the order the
    workers finish them, not file order: file and frame0 say where each belongs.

        for file, frame0, npx, values, indices, powder in iterate(units, backend, workers=8):
            ...

    The results wait in a queue of `depth` batches per worker: a consumer that is slower than the
    decoders slows them down instead of filling memory.  Threads only.  Stopping the loop early
    (break, or an exception) stops the run.  Other keywords go to run (batch, policy, ...)."""
    if kw.get("mode", "threads") != "threads":
        raise ValueError("iterate works with threads")
    q = queue.Queue(maxsize=depth * workers)
    closed = threading.Event()
    box = {}

    def go():
        try:
            box["result"] = run(units, backend, workers, sink=lambda k: _QueueSink(q, closed), **kw)
        except BaseException as e:  # noqa: BLE001
            box["error"] = e
        while True:  # the end marker, even when the queue is full of what nobody will read
            try:
                q.put(_END, timeout=0.5)
                return
            except queue.Full:
                if closed.is_set():
                    return

    t = threading.Thread(target=go, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                break
            yield item
    finally:
        closed.set()
        t.join()
    err = box.get("error")
    if err is not None and "_Closed" not in repr(err):
        raise err
