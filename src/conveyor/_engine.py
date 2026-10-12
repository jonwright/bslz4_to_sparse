"""The conveyor engine: readers -> slots -> workers -> sinks.

    main      builds the units, the slots and the queues, starts the readers and the workers
    readers   take a unit and a free slot, read the unit into the slot with large preadv calls in
              file order, and give (slot, unit) to the worker that owns the slot
    workers   each owns S slots in one anonymous shared mmap (first touched by the worker, after it
              is pinned, so the pages sit near its CPU) and one decoder; decode every frame of a
              unit straight from the slot (no copy) and give the sink the result

Queues carry one small message per unit, never per frame.  A worker that finishes sooner frees its
slots sooner and so receives more units: the load balances itself.  The slots are one anonymous
mmap, not SharedMemory and not a file, so the same code runs with threads or forked processes.
"""

import mmap
import multiprocessing as mp
import os
import queue
import threading
import time

import numpy as np

from ._sinks import NullSink

_STOP = None


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


def _reader(k, todo, free_q, ready_qs, buf, slot_bytes, S, cpus, io_bytes, go, stop, res_q):
    st = dict(role="reader", k=k, bytes=0, read_s=0.0, wait_s=0.0, error=None)
    try:
        if cpus:
            os.sched_setaffinity(0, cpus)
        go.wait()
        view = memoryview(buf)
        while not stop.is_set():
            u = todo.get()
            if u is _STOP:
                break
            t = time.perf_counter()
            slot = None
            while slot is None and not stop.is_set():
                try:
                    slot = free_q.get(timeout=0.5)
                except queue.Empty:
                    pass
            if slot is None:
                break
            st["wait_s"] += time.perf_counter() - t
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
            ready_qs[slot // S].put((slot, u))
    except BaseException as e:  # noqa: BLE001 (report it, and stop the others)
        st["error"] = repr(e)
        stop.set()
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
    st = dict(role="worker", k=k, frames=0, decode_s=0.0, write_s=0.0, idle_s=0.0, error=None)
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
        c0, t0 = time.thread_time(), time.perf_counter()
        while not stop.is_set():
            t = time.perf_counter()
            try:
                item = ready_q.get(timeout=0.5)
            except queue.Empty:
                st["idle_s"] += time.perf_counter() - t
                continue
            st["idle_s"] += time.perf_counter() - t
            if item is _STOP:
                break
            slot, u = item
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
            free_q.put(slot)
        st["sink"] = sink.end()
        st["wall_s"], st["cpu_s"] = time.perf_counter() - t0, time.thread_time() - c0
    except BaseException as e:  # noqa: BLE001 (report it, and stop the others)
        st["error"] = repr(e)
        stop.set()
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001
            pass
    res_q.put(st)


def run(
    units,
    backend,
    workers,
    readers=2,
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

    units         from make_units
    backend       CpuBackend or GpuBackend (anything with make_worker())
    workers       decoders, each in its own thread or process
    readers       reader threads or processes
    sink          factory k -> Sink, called in each worker (default: NullSink)
    mode          "threads" (default) or "processes" (forked): the same code, to compare
    batch         frames per decoder call
    slots_per_worker  buffers per worker: more lets the readers run further ahead
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
    todo, free_q, res_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    ready_qs = [ctx.Queue() for _ in range(workers)]
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
                free_q,
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
            (k, todo, free_q, ready_qs, buf, slot_bytes, S, reader_cpus, io_bytes, go, stop, res_q),
        )
        for k in range(readers)
    ]
    for p in ws + rs:
        p.start()
    errors = []

    def collect(n):
        out = []
        for _ in range(n):
            out.append(res_q.get())
        return out

    try:
        barrier.wait(timeout=start_timeout)  # every worker pinned, its slots touched, decoder built
    except Exception:  # noqa: BLE001
        stop.set()
        go.set()
        errors = [s["error"] for s in collect(workers) if s.get("error")]
        raise RuntimeError("a worker failed to start: %r" % (errors,))
    for u in units:
        todo.put(u)
    for _ in range(readers):
        todo.put(_STOP)
    t0 = time.perf_counter()
    go.set()
    got = []
    while len(got) < readers:  # readers finish when the units are all read
        got.append(res_q.get())
    for q in ready_qs:
        q.put(_STOP)
    got += collect(workers)
    wall = time.perf_counter() - t0
    for p in ws + rs:
        p.join()
    errors = [s["error"] for s in got if s.get("error")]
    if errors:
        raise RuntimeError(errors[0])
    wk = sorted((s for s in got if s["role"] == "worker"), key=lambda s: s["k"])
    rd = [s for s in got if s["role"] == "reader"]
    frames = sum(s["frames"] for s in wk)
    return dict(
        mode=mode,
        workers=workers,
        readers=readers,
        frames=frames,
        wall_s=round(wall, 3),
        fps=round(frames / wall, 1),
        read_gbs=round(sum(u.span for u in units) / wall / 1e9, 3),
        slot_mb=round(slot_bytes / 1e6),
        worker_cpu_cores=round(sum(s["cpu_s"] for s in wk) / wall, 2),
        fps_per_worker_mean=round(float(np.mean([s["frames"] / s["wall_s"] for s in wk])), 1),
        fps_per_worker_min=round(min(s["frames"] / s["wall_s"] for s in wk), 1),
        busy_frac=dict(
            decode=round(sum(s["decode_s"] for s in wk) / wall / workers, 3),
            write=round(sum(s["write_s"] for s in wk) / wall / workers, 3),
            idle=round(sum(s["idle_s"] for s in wk) / wall / workers, 3),
            read=round(sum(s["read_s"] for s in rd) / wall / readers, 3),
            read_wait_for_slot=round(sum(s["wait_s"] for s in rd) / wall / readers, 3),
        ),
        sink=[s.get("sink") for s in wk][:1],
    )
