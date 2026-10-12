"""The conveyor engine: a queue of units, and workers that each own two buffers and a reader.

    todo       the units (whole files by default), in file order, all queued before the run
    reader k   whenever worker k has a free buffer, takes the next unit from todo and reads it into
               that buffer with large preadv calls in file order, while the worker decodes the
               previous one
    worker k   decodes the unit its reader delivered straight from the buffer and gives the sink
               the result

Pulling is the load balance: a worker that decodes faster frees its buffers sooner and so takes more
units, and reading is demand driven, each worker reading only as fast as it consumes.  Workers touch
the shared queue once per unit (a unit is seconds of work), and each worker and its reader meet once
per unit through two blocking queues; nobody polls or spins.  The buffers are one anonymous mmap,
cut per worker and first touched by that worker after it is pinned (near its CPU); not a file, not
SharedMemory.  Workers are threads or forked processes around the same code.

Memory: workers x slots x the largest unit.
"""

import mmap
import multiprocessing as mp
import os
import queue
import resource
import threading
import time

import numpy as np

from ._sinks import NullSink

_DONE = None


class _ThreadContext(object):
    Queue = staticmethod(queue.Queue)
    Event = staticmethod(threading.Event)
    Barrier = staticmethod(threading.Barrier)

    @staticmethod
    def Worker(target, args):
        return threading.Thread(target=target, args=args, daemon=True)


class _ForkContext(object):
    def __init__(self):
        self.ctx = mp.get_context("fork")
        self.Queue, self.Event, self.Barrier = self.ctx.Queue, self.ctx.Event, self.ctx.Barrier

    def Worker(self, target, args):
        return self.ctx.Process(target=target, args=args, daemon=True)


def numa_nodes(allowed=None):
    """[[cpu, ...], ...]: the CPUs this process may use, grouped by NUMA node."""
    allowed = set(os.sched_getaffinity(0)) if allowed is None else set(allowed)
    out = []
    d = "/sys/devices/system/node"
    if os.path.isdir(d):
        for n in sorted(x for x in os.listdir(d) if x.startswith("node") and x[4:].isdigit()):
            with open(os.path.join(d, n, "cpulist")) as f:
                cl = f.read().strip()
            cpus = []
            for part in cl.split(","):
                a, _, b = part.partition("-")
                cpus += list(range(int(a), int(b or a) + 1))
            cpus = [c for c in cpus if c in allowed]
            if cpus:
                out.append(cpus)
    return out or [sorted(allowed)]


def _faults():
    """minor page faults of the calling thread so far"""
    return resource.getrusage(resource.RUSAGE_THREAD).ru_minflt


def _reader(k, todo, ready_q, free_q, buf, slot_bytes, cpus, io_bytes, go, stop, res_q):
    st = dict(role="reader", k=k, bytes=0, read_s=0.0, wait_s=0.0, error=None, minflt=0)
    try:
        if cpus:
            os.sched_setaffinity(0, cpus)
        go.wait()
        flt0 = _faults()
        view = memoryview(buf)
        while True:
            t = time.perf_counter()
            slot = free_q.get()  # blocks until the worker is done with a buffer
            st["wait_s"] += time.perf_counter() - t
            if stop.is_set():
                break
            u = todo.get()  # the next unit, whoever asks first; _DONE when there are no more
            if u is _DONE:
                break
            base = slot * slot_bytes
            t = time.perf_counter()
            fd = os.open(u.path, os.O_RDONLY)
            try:
                got = 0
                while got < u.span:
                    want = min(io_bytes or u.span, u.span - got)
                    r = os.preadv(fd, [view[base + got : base + got + want]], u.start + got)
                    if r <= 0:
                        raise IOError("short read of %s" % u.path)
                    got += r
            finally:
                os.close(fd)
            st["read_s"] += time.perf_counter() - t
            st["bytes"] += u.span
            ready_q.put((slot, u))
        st["minflt"] = _faults() - flt0
        ready_q.put(_DONE)
    except BaseException as e:  # noqa: BLE001 (report it, and unblock the worker)
        st["error"] = repr(e)
        stop.set()
        ready_q.put(_DONE)
    res_q.put(st)


def _worker(
    k,
    backend,
    sink_factory,
    meta,
    ready_q,
    free_q,
    buf,
    slot_bytes,
    S,
    batch,
    cpu,
    barrier,
    go,
    stop,
    res_q,
):
    st = dict(
        role="worker", k=k, frames=0, decode_s=0.0, write_s=0.0, idle_s=0.0, error=None, minflt=0
    )
    t_first = t_last = None
    try:
        if cpu is not None:
            os.sched_setaffinity(0, [cpu])
        for s in range(k * S, (k + 1) * S):  # first touch: the pages go near this CPU
            buf[s * slot_bytes : (s + 1) * slot_bytes : 4096] = 0
        fn = backend.make_worker()
        sink = sink_factory(k)
        sink.begin(meta)
        for s in range(k * S, (k + 1) * S):
            free_q.put(s)
        barrier.wait()
        go.wait()
        c0, t0, flt0 = time.thread_time(), time.perf_counter(), _faults()
        while True:
            t = time.perf_counter()
            item = ready_q.get()  # blocks until the reader has a whole unit in
            st["idle_s"] += time.perf_counter() - t
            if item is _DONE or stop.is_set():
                break
            slot, u = item
            if t_first is None:
                t_first = time.perf_counter()
            base = slot * slot_bytes
            data = buf[base : base + u.span]
            for f0 in range(0, u.n, batch):
                n = min(batch, u.n - f0)
                t = time.perf_counter()
                npx, values, indices, powder = fn(
                    data, u.offsets[f0 : f0 + n], u.lengths[f0 : f0 + n]
                )
                t1 = time.perf_counter()
                sink.write(u.file, u.frame0 + f0, npx, values, indices, powder)
                st["decode_s"] += t1 - t
                st["write_s"] += time.perf_counter() - t1
                st["frames"] += n
            t_last = time.perf_counter()
            free_q.put(slot)
        # Whatever made us stop (the units are done, or another worker failed), our reader may be
        # blocked waiting for a free buffer: give it some, or it waits for ever.
        for _ in range(S + 1):
            free_q.put(0)
        st["minflt"] = _faults() - flt0
        st["sink"] = sink.end()
        st["wall_s"], st["cpu_s"] = time.perf_counter() - t0, time.thread_time() - c0
        st["active_s"] = (t_last - t_first) if t_first is not None and t_last is not None else 0.0
    except BaseException as e:  # noqa: BLE001 (report it, and unblock the reader)
        st["error"] = repr(e)
        stop.set()
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001
            pass
        for _ in range(S + 1):  # the reader may be waiting for a free buffer
            free_q.put(0)
    res_q.put(st)


def run(
    units,
    backend,
    workers,
    sink=None,
    mode="threads",
    batch=16,
    slots_per_worker=2,
    worker_cpus=None,
    reader_cpus=None,
    io_bytes=0,
    meta=None,
    start_timeout=900,
):
    """Run the units through the conveyor; returns a dict of what it took.

    units         from make_units; handed out in order to whichever worker is ready for one
    backend       CpuBackend, DecompressBackend or GpuBackend (anything with make_worker())
    workers       decoders; each has its own reader thread (or process, in "processes" mode)
    sink          factory k -> Sink, called in each worker (default: NullSink)
    mode          "threads" (default) or "processes" (forked): the same code, to compare
    batch         frames per decoder call
    slots_per_worker  buffers per worker (2: one decoding while the next is read)
    worker_cpus   list: worker k is pinned to worker_cpus[k % len]; None leaves it to the OS
    reader_cpus   list: every reader may run on any of these; None leaves it to the OS
    io_bytes      size of each read call (0: the whole unit in one call)
    """
    if not units:
        raise ValueError("no units")
    ctx = {"threads": _ThreadContext, "processes": None}.get(mode, 0)
    if ctx == 0:
        raise ValueError("mode must be 'threads' or 'processes'")
    ctx = ctx or _ForkContext()
    sink = sink or (lambda k: NullSink())
    S = slots_per_worker
    slot_bytes = (max(u.span for u in units) + 4095) // 4096 * 4096
    mm = mmap.mmap(-1, workers * S * slot_bytes, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
    buf = np.frombuffer(mm, np.uint8)
    res_q, todo = ctx.Queue(), ctx.Queue()
    ready_qs = [ctx.Queue() for _ in range(workers)]
    free_qs = [ctx.Queue() for _ in range(workers)]
    barrier, go, stop = ctx.Barrier(workers + 1), ctx.Event(), ctx.Event()
    ws = [
        ctx.Worker(
            _worker,
            (
                k,
                backend,
                sink,
                meta or {},
                ready_qs[k],
                free_qs[k],
                buf,
                slot_bytes,
                S,
                batch,
                None if not worker_cpus else worker_cpus[k % len(worker_cpus)],
                barrier,
                go,
                stop,
                res_q,
            ),
        )
        for k in range(workers)
    ]
    rs = [
        ctx.Worker(
            _reader,
            (
                k,
                todo,
                ready_qs[k],
                free_qs[k],
                buf,
                slot_bytes,
                reader_cpus,
                io_bytes,
                go,
                stop,
                res_q,
            ),
        )
        for k in range(workers)
    ]
    for p in ws + rs:
        p.start()
    try:
        barrier.wait(
            timeout=start_timeout
        )  # every worker pinned, its buffers touched, decoder built
    except Exception:  # noqa: BLE001
        stop.set()
        go.set()
        errs = [s["error"] for s in [res_q.get() for _ in range(workers)] if s.get("error")]
        raise RuntimeError("a worker failed to start: %r" % (errs,))
    for u in units:
        todo.put(u)
    for _ in range(workers):
        todo.put(_DONE)
    t0 = time.perf_counter()
    go.set()
    got = [res_q.get() for _ in range(2 * workers)]
    wall = time.perf_counter() - t0
    for p in ws + rs:
        p.join()
    errors = [s["error"] for s in got if s.get("error")]
    if errors:
        raise RuntimeError(errors[0])
    wk = sorted((s for s in got if s["role"] == "worker"), key=lambda s: s["k"])
    rd = [s for s in got if s["role"] == "reader"]
    frames = sum(s["frames"] for s in wk)
    busy = [s for s in wk if s["active_s"] > 0]
    return dict(
        mode=mode,
        workers=workers,
        frames=frames,
        wall_s=round(wall, 3),
        fps=round(frames / wall, 1),
        # steady state: every worker over the time between its first and last unit (the start,
        # while the first files arrive, and the end, while the slowest worker finishes, left out)
        steady_fps=round(sum(s["frames"] / s["active_s"] for s in busy), 1),
        read_gbs=round(sum(u.span for u in units) / wall / 1e9, 3),
        slot_mb=round(slot_bytes / 1e6),
        memory_gb=round(workers * S * slot_bytes / 1e9, 2),
        worker_cpu_cores=round(sum(s["cpu_s"] for s in wk) / wall, 2),
        fps_per_worker_mean=round(float(np.mean([s["frames"] / s["wall_s"] for s in wk])), 1),
        busy_frac=dict(
            decode=round(sum(s["decode_s"] for s in wk) / wall / workers, 3),
            write=round(sum(s["write_s"] for s in wk) / wall / workers, 3),
            idle=round(sum(s["idle_s"] for s in wk) / wall / workers, 3),
            read=round(sum(s["read_s"] for s in rd) / wall / workers, 3),
            read_wait_for_buffer=round(sum(s["wait_s"] for s in rd) / wall / workers, 3),
        ),
        # minor page faults taken by the workers and readers after the buffers were touched: with
        # no allocation per file this stays small and does not grow with the number of files
        page_faults=dict(
            workers=sum(s["minflt"] for s in wk), readers=sum(s["minflt"] for s in rd)
        ),
        files=len(units),
        sink=[s.get("sink") for s in wk][:1],
    )
