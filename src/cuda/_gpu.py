"""GPU (CUDA, cupy) versions of chunk2sparse and chunk2sparseCSC, with the same call API as the CPU
classes:

    c = chunk2sparse(mask, dtype, codec)           c(chunk, cut) -> npixels, (values, indices)
    c = chunk2sparseCSC(mask, csc, dtype, codec)   c(chunk, cut) -> npixels, (values, indices),
                                                   powder
    c.multi(chunks, cut), c.from_buffer(base, offsets, lengths, cut), c.coo(chunk, cut)

Differences from the CPU classes:
  * lz4 only, 8 kB blocks (the bitshuffle default), pixels u8/u16/u32/i8/i16/i32/f32; `pipeline` is
    accepted and ignored;
  * `acc` chooses the accumulator of the matrix product: 'f64' (default: f32 weights, f64 sums as
    pyFAI), 'f32' (fast; relative error about 1e-6); a matrix from fixed_point_matrix is summed as
    'i64' (exact integer sums, returned as int64: divide by `powder_scale` for float);
  * `batch` = frames decoded together (the dense frames in GPU memory: batch x frame bytes).
    None: as many as fit in 3/4 of the device's L2 cache when that is at least 8 (the collect
    re-reads them from the cache), else 64; at most 64, and the frames and lists of a batch take
    at most 1/16 of the device memory (a guess, for small cards).  Measured on Eiger 4M u16
    frames, kernel time per frame: L40S (96 MB L2) best at 8; H200 (60 MB) and A100 (40 MB) best
    at 64, and 3 to 16 slower there.
    `group` = frames uploaded and unblocked together; None: every frame of the call, at most 256.
    The block headers of a frame are walked by one GPU thread, so the walk costs about the same
    for 1 or 256 frames.
    `unblock` = 'gpu' (that walk, a kernel) or 'cpu' (bslz4_to_sparse.block_starts on the host,
    uploaded: for a small group, or a small GPU).
    The result arrays are numpy (as the CPU classes) and are
    overwritten by the next call.
  * the matrix is the one chunk2sparseCSC takes (scipy csc / pyFAI CSC engine, or
    fixed_point_matrix); normalise_matrix and the mask folding are bslz4_to_sparse._matrix's.  Per
    batch: decompress into dense frames in GPU memory, then the matrix product (CSR, one thread
    block per bin, no atomics) and the collect of the pixels above the cut, as separate kernels
    (bslz4_split.cu).

howmany / thresholds are the CPU classes' (see _toppix.py).  On the GPU, a frame's list is written
by the device when it holds at most `cap` = nbytes / (4 + bpp) pixels (indices + values never take
more than the raw frame) and, with a howmany, by top_pixels on the device.  A frame with more than
cap pixels above the cut, when all of them were asked for (howmany None or above cap), is not
sparse: its dense frame is copied back and listed on the host (slow, and counted in `last_host`).
GPU memory per object: the dense frames (batch x frame bytes) and batch x min(howmany, cap) listed
pixels; the matrix and the mask are shared by every object of the process.
"""

import hashlib
import os
import threading
import numpy as np

from . import _backend as cu
from .. import _matrix, _toppix
from .._pipeline import CODEC_LZ4

BLOCK = 8192
HERE = os.path.dirname(os.path.abspath(__file__))
CTYPE = {
    "u1": "unsigned char",
    "i1": "signed char",
    "u2": "unsigned short",
    "i2": "short",
    "u4": "unsigned int",
    "i4": "int",
    "f4": "float",
}
ACC = {
    "f32": ("float", "float", np.float32),
    "f64": ("float", "double", np.float64),  # weight, accumulator
    "i64": ("unsigned int", "long long", np.int64),
}
FRAMES_PER_READ = (
    2  # frames each (weight, index) read of csr1d is applied to (2 was best on a GTX 1060)
)
MAXTHR = _toppix.MAXTHR  # top_pixels thresholds (MAXTHR in bslz4_split.cu)
_mods = {}
_shared_dev = (
    {}
)  # (device, kind, digest) -> device arrays (read only), shared by all objects of the process
_shared_lock = threading.Lock()


def _digest(arrays):
    h = hashlib.blake2b(digest_size=20)
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _shared(dev, kind, arrays):
    """Device copies of read-only host arrays (matrix, mask), uploaded once per process and
    device."""
    key = (dev, kind, _digest(arrays))
    with _shared_lock:
        if key not in _shared_dev:
            with cu.device(dev):
                _shared_dev[key] = tuple(cu.asarray(np.ascontiguousarray(a)) for a in arrays)
                cu.synchronize(dev)
        return _shared_dev[key]


def release_shared():
    """Drop the shared matrices and masks (objects still alive keep their references)."""
    with _shared_lock:
        _shared_dev.clear()


def _ctype(dt):
    return CTYPE[dt.kind + str(dt.itemsize)]


def _csr_name(dt, acc, F):
    w, a, _ = ACC[acc]
    return "csr1d<%s, %s, %s, %d>" % (_ctype(dt), w, a, F)


def _module(dtype, acc=None):
    """The kernels for one pixel dtype (and accumulator, for the matrix product), compiled by
    NVRTC."""
    key = (dtype.str, acc)
    if key not in _mods:
        src = "".join(
            open(os.path.join(HERE, f)).read()
            for f in ("nvcomp_lz4.cuh", "shuffle_warp.cuh", "bslz4_split.cu")
        )
        names = ["count_above<%s>" % _ctype(dtype), "collect_write<%s>" % _ctype(dtype)]
        if acc is not None:
            names += [_csr_name(dtype, acc, F) for F in sorted({1, FRAMES_PER_READ})]
        _mods[key] = cu.build_module(src, names)
    return _mods[key]


def _fixed_np(dt):
    return ACC[dt][2]


def _chunk_header(cmp):
    """(total bytes, block bytes) from the 12-byte bitshuffle header."""
    h = bytes(memoryview(cmp)[:12])
    return int.from_bytes(h[:8], "big"), int.from_bytes(h[8:12], "big")


class chunk2sparse(object):
    """Decode bitshuffle-lz4 chunks of (ni, nj) frames into the pixels above a cut (and not
    masked).  lz4 only (a zstd chunk is refused), and no from_frames; last_host counts the frames
    that fell back to the host (see the module docstring)."""

    def __init__(
        self,
        mask,
        dtype=np.uint16,
        codec=None,
        pipeline=None,
        batch=None,
        howmany=None,
        thresholds=None,
        group=None,
        unblock="gpu",
    ):
        self._dev = cu.device_id()
        cu.use_device(self._dev)  # another library may have made its own context current
        if codec is not None and codec != CODEC_LZ4:
            raise ValueError("the GPU decoder does lz4 only")
        self._codec_pending = codec is None
        self.dtype = np.dtype(dtype)
        if (
            self.dtype.itemsize not in (1, 2, 4)
            or self.dtype.kind not in "uif"
            or (self.dtype.kind == "f" and self.dtype.itemsize != 4)
        ):
            raise ValueError(
                "GPU pixel dtype %s not supported (u8,u16,u32,i8,i16,i32,f32)" % self.dtype
            )
        self.nfast = mask.shape[1]
        self.mask = np.ascontiguousarray(mask.ravel(), np.uint8)
        self.npix = mask.size
        self.bpp = self.dtype.itemsize
        self.kind = {"u": 0, "i": 1, "f": 2}[self.dtype.kind]
        self.nbytes = self.npix * self.bpp
        self.stride = (
            (self.nbytes + BLOCK - 1) // BLOCK * BLOCK
        )  # bytes per dense frame (16-byte aligned stores)
        self.stride_el = self.stride // self.bpp
        nfull = self.nbytes // BLOCK
        part = self.nbytes - nfull * BLOCK
        self.tocopy = part % (self.bpp * 8)
        self.nstored = nfull + ((part - self.tocopy) > 0)  # blocks with an lz4 stream
        self.ntot = nfull + (part > 0)  # blocks of pixels
        if batch is None:
            fit = int(0.75 * cu.l2_bytes(self._dev)) // self.stride
            room = cu.memory_bytes(self._dev) // (16 * 2 * self.stride)  # frames + lists
            batch = max(1, min(fit if fit >= 8 else 64, 64, room))
        self.batch = int(batch)
        self.group = None if group is None else int(group)
        if self.batch < 1 or (self.group is not None and self.group < 1):
            raise ValueError("batch and group must be at least 1")
        if unblock not in ("gpu", "cpu"):
            raise ValueError("unblock must be 'gpu' or 'cpu'")
        self.unblock = unblock
        self.stream = cu.Stream()
        self.mod = _module(self.dtype, self._acc())
        self.k_unblock = self.mod.get_function("unblock")
        self.k_decompress = self.mod.get_function("decompress")
        self.k_count = self.mod.get_function("count_above<%s>" % _ctype(self.dtype))
        self.k_write = self.mod.get_function("collect_write<%s>" % _ctype(self.dtype))
        self.k_scan = self.mod.get_function("scan_counts")
        self.k_choose = self.mod.get_function("choose_level")
        self.k_images = self.mod.get_function("scan_images")
        (self.d_mask,) = _shared(self._dev, "mask", [self.mask])
        self.cap = self.nbytes // (
            4 + self.bpp
        )  # pixels a frame's list may hold: indices + values <= frame bytes
        self._cap = 0
        self._gcap = 0
        self._slab_cap = 0
        self._nframes = 0
        self.d_thr_i = cu.zeros(MAXTHR, np.int64)
        self.d_thr_f = cu.zeros(MAXTHR, np.float32)
        self.last_host = 0
        self.set_top_pixels(howmany, thresholds)

    def set_top_pixels(self, howmany=None, thresholds=None):
        """As the constructor's howmany and thresholds (_toppix): None for every pixel above the
        cut."""
        self.howmany, self.thresholds = _toppix.check(howmany, thresholds)
        # the device lists at most hm pixels per frame; thresholds only matter when it can honour
        # howmany
        self._fallback = self.howmany is None or self.howmany > self.cap
        self._hm = self.cap if self._fallback else self.howmany
        self._ndev_thr = (
            0 if self._fallback else (MAXTHR if self.thresholds is None else len(self.thresholds))
        )
        self._cap = 0  # the next call sizes the lists and the per-level counts again

    # -- the matrix subclass overrides these
    def _acc(self):
        return None

    # -- buffers
    def _ensure(self, nframes, slab_bytes, G):
        G = min(G, nframes)
        B = min(self.batch, G)
        if self._gcap < G or self._slab_cap < slab_bytes:  # one group: chunks and block starts
            self._gcap = max(self._gcap, G)
            self._slab_cap = max(self._slab_cap, int(slab_bytes * 1.25) + 64)
            g = self._gcap
            self.h_slab = cu.pinned(self._slab_cap)
            self.d_slab = cu.empty(self._slab_cap, np.uint8)
            self.d_off = cu.empty(g, np.int64)
            self.d_end = cu.empty(g, np.int64)
            self.d_err = cu.empty(g, np.int32)
            self.d_starts = cu.empty((g, max(self.nstored, 1)), np.uint32)
            if self.unblock == "cpu":
                ns = g * max(self.nstored, 1)
                self.h_starts = np.frombuffer(cu.pinned(ns * 4), np.uint32, ns)
                self.h_err = np.frombuffer(cu.pinned(g * 4), np.int32, g)
        if self._cap < B:  # one batch: dense frames, counts, lists
            self._cap = max(self._cap, B)
            n = self._cap
            if getattr(self, "d_dense", None) is None or self.d_dense.size < n * self.stride:
                self.d_dense = cu.empty(n * self.stride, np.uint8)
            lv = self._ndev_thr + 1  # levels: the cut, then each threshold
            self.d_counts = cu.empty((lv * n, self.ntot), np.uint32)
            self.d_offs = cu.empty((lv * n, self.ntot), np.uint32)
            self.d_npxl = cu.empty(lv * n, np.int32)
            self.d_level = cu.empty(n, np.int32)
            self.d_npx = cu.empty(n, np.int32)
            self.d_imgoff = cu.empty(n, np.uint32)
            nl = max(n * self._hm, 1)  # lists of n frames: at most n * nbytes bytes
            self.d_outv = cu.empty(nl, self.dtype)
            self.d_outi = cu.empty(nl, np.uint32)
            self.h_outv = np.frombuffer(cu.pinned(nl * self.bpp), self.dtype, nl)
            self.h_outi = np.frombuffer(cu.pinned(nl * 4), np.uint32, nl)
            self._on_resize(n)
        if self._nframes < nframes:  # host results keep the largest size seen; views are returned
            self._values_buf = np.empty((nframes, self.npix), self.dtype)
            self._indices_buf = np.empty((nframes, self.npix), np.uint32)
            self._nframes = nframes
        self._values = self._values_buf[:nframes]
        self._indices = self._indices_buf[:nframes]

    def _on_resize(self, n):
        pass

    def _cut(self, cut):
        if self.kind == 2:
            return np.int64(0), np.float32(cut)
        c = np.floor(cut) if np.isfinite(cut) else (2.0**62 if cut > 0 else -(2.0**62))
        return np.int64(int(max(min(c, 2.0**62), -(2.0**62)))), np.float32(0)

    # -- one group of up to `group` chunks: upload, walk the block headers
    def _upload(self, chunk_views):
        n = len(chunk_views)
        lens = np.array([len(v) for v in chunk_views], np.int64)
        ends = np.cumsum(lens)
        offs = ends - lens
        slab = np.frombuffer(self.h_slab, np.uint8, int(ends[-1]))
        for v, o, l in zip(chunk_views, offs, lens):
            slab[o : o + l] = v
        s = self.stream
        self.d_slab[: slab.size].set(slab, s)
        self.d_off[:n].set(offs, s)
        self.d_end[:n].set(ends, s)
        s.use()
        if self.unblock == "cpu":
            self._cpu_unblock(slab, offs, lens, n)
            return n
        self.k_unblock(
            ((n + 63) // 64,),
            (64,),
            (
                self.d_slab,
                self.d_off,
                self.d_end,
                self.d_starts,
                np.uint32(self.nstored),
                np.uint32(self.tocopy),
                np.uint32(n),
                self.d_err,
            ),
        )
        return n

    def _cpu_unblock(self, slab, offs, lens, n):
        from .. import _block_starts, _decode_error, _offset_pointers

        ptrs, lens32, _ = _offset_pointers(slab, offs, lens)
        starts = self.h_starts[: n * self.nstored]
        err = self.h_err[:n]
        rc, st, status = _block_starts(ptrs, lens32, self.dtype, starts, err)
        if rc < 0 and rc != -107:  # -107: the bad chunks are flagged and skipped, as by the kernel
            raise _decode_error(rc)
        if st.shape[1] != self.nstored:
            raise Exception(
                "Error decoding batch: %d blocks per chunk, expected %d"
                % (st.shape[1], self.nstored)
            )
        np.not_equal(status, 0, out=err)  # -107 -> 1
        s = self.stream
        self.d_starts[:n].set(st, s)
        self.d_err[:n].set(err, s)

    # -- frames s0 .. s0 + n of the group (n <= batch): decompress, collect, matrix product
    def _launch(self, s0, n, cut):
        s = self.stream
        self._s0 = s0
        d_off, d_end, d_err = self.d_off[s0:], self.d_end[s0:], self.d_err[s0:]
        grid = (n, (self.ntot + 1) // 2)
        self.k_decompress(
            grid,
            (32, 2),
            (
                self.d_slab,
                d_off,
                d_end,
                self.d_starts[s0:],
                d_err,
                np.uint32(self.nstored),
                np.uint32(self.ntot),
                np.uint32(self.nbytes),
                np.uint32(self.bpp),
                self.d_dense,
                np.uint64(self.stride),
            ),
        )
        cut_i, cut_f = self._cut(cut)
        nthr = self._ndev_thr
        self._thr_call = (
            _toppix.default_thresholds(cut) if self.thresholds is None else self.thresholds
        )
        if nthr:
            ti, tf = zip(*[self._cut(t) for t in self._thr_call])
            self.d_thr_i[:nthr].set(np.array(ti, np.int64), s)
            self.d_thr_f[:nthr].set(np.array(tf, np.float32), s)
        common = (
            self.d_dense,
            np.uint64(self.stride_el),
            np.uint32(self.npix),
            np.uint32(self.ntot),
            d_err,
            self.d_mask,
            cut_i,
            cut_f,
            self.d_thr_i,
            self.d_thr_f,
        )
        self.k_count(grid, (32, 2), common + (np.int32(nthr), np.uint32(n), self.d_counts))
        rows = (nthr + 1) * n  # (level, image) rows; level k of image i is row k * n + i
        self.k_scan(
            (rows,), (256,), (self.d_counts, self.d_offs, self.d_npxl, np.uint32(self.ntot))
        )
        self.k_choose(
            ((n + 63) // 64,),
            (64,),
            (
                self.d_npxl,
                np.uint32(n),
                np.int32(nthr),
                np.uint32(self._hm),
                np.uint32(self._fallback),
                self.d_level,
                self.d_npx,
            ),
        )
        self.k_images((1,), (32,), (self.d_npx, self.d_imgoff, np.uint32(n)))
        self.k_write(
            grid,
            (32, 2),
            common
            + (
                np.uint32(n),
                self.d_counts,
                self.d_offs,
                self.d_level,
                self.d_imgoff,
                np.uint32(self._hm),
                self.d_outv,
                self.d_outi,
            ),
        )
        self._dot(n)  # last: the collect re-reads the dense frames while they may still be in L2
        return n

    def _dot(self, n):
        pass

    def _fetch(self, f0, n, npx_all):
        """results of the n frames just launched -> host arrays rows f0.."""
        self.stream.synchronize()
        err = self.d_err[self._s0 : self._s0 + n].get()
        if err.any():
            raise Exception(
                "Error decoding batch: a chunk is corrupt or truncated (its block lengths do not "
                "match its size); chunks %s" % (np.flatnonzero(err) + f0).tolist()
            )
        npx = self.d_npx[:n].get()
        level = self.d_level[:n].get()
        npxl = self.d_npxl[: (self._ndev_thr + 1) * n].get().reshape(self._ndev_thr + 1, n)
        self.last_nabove[f0 : f0 + n] = npxl[
            np.clip(level, 0, None), np.arange(n)
        ]  # pixels above the cut each frame ended at
        thr = np.array(self._thr_call if self._ndev_thr else [self._call_cut], np.float64)
        self.last_cut[f0 : f0 + n] = np.where(
            level > 0, thr[np.clip(level - 1, 0, len(thr) - 1)], self._call_cut
        )
        total = int(npx.sum())
        if total:
            self.d_outv[:total].get(out=self.h_outv[:total])
            self.d_outi[:total].get(out=self.h_outi[:total])
        pos = 0
        for j in range(n):
            k = int(npx[j])
            if k:
                self._values[f0 + j, :k] = self.h_outv[pos : pos + k]
                self._indices[f0 + j, :k] = self.h_outi[pos : pos + k]
                pos += k
        for j in np.flatnonzero(level < 0):  # not sparse: listed on the host from the dense frame
            self._host_list(f0, int(j))
            npx[j] = self._npx_host
        npx_all[f0 : f0 + n] = npx
        self._fetch_powder(f0, n)

    def _host_list(self, f0, j):
        dense = self.d_dense[j * self.stride : j * self.stride + self.nbytes].view(self.dtype).get()
        sel = np.flatnonzero((dense > np.float64(self._call_cut)) & (self.mask > 0)).astype(
            np.uint32
        )
        k = len(sel)
        self._values[f0 + j, :k] = dense[sel]
        self._indices[f0 + j, :k] = sel
        used = self._call_cut
        if self.howmany is not None:
            thr = (
                _toppix.default_thresholds(self._call_cut)
                if self.thresholds is None
                else self.thresholds
            )
            k, t, na = _toppix.top_pixels(
                self._values[f0 + j], self._indices[f0 + j], k, self.howmany, thr
            )
            used = used if t is None else t
        self.last_cut[f0 + j] = used
        self.last_nabove[f0 + j] = na if self.howmany is not None else k
        self.last_host += 1
        self._npx_host = k

    def _fetch_powder(self, f0, n):
        pass

    def _check(self, views):
        if self._codec_pending:  # the first chunk decides, as in the CPU classes
            from .. import _codec_of_chunk

            if _codec_of_chunk(views[0]) != CODEC_LZ4:
                raise ValueError("the GPU decoder does lz4 only")
            self._codec_pending = False
        for v in views:
            tot, bs = _chunk_header(v)
            if tot != self.nbytes:
                raise Exception(
                    "Error decoding chunk: -106: the chunk is for %d bytes, expected %d"
                    % (tot, self.nbytes)
                )
            if bs not in (0, BLOCK):  # 0: the default, 8192 bytes
                raise Exception(
                    "Error decoding chunk: block size %d bytes; the GPU decoder takes %d"
                    % (bs, BLOCK)
                )

    def _run(self, views, cut):
        if len(views) == 0:
            raise Exception("Error decoding batch: -105: no chunks were given")
        cu.use_device(self._dev)  # another library may have made its own context current
        self._check(views)
        nf = len(views)
        G = max(self.batch, min(nf, 256)) if self.group is None else self.group
        self._ensure(nf, max(sum(len(v) for v in views[i : i + G]) for i in range(0, nf, G)), G)
        npx = np.zeros(nf, np.int32)
        self.last_cut = np.empty(nf, np.float64)  # the cut each frame used (top_pixels)
        self.last_nabove = np.empty(nf, np.int64)  # pixels above last_cut, before truncation
        self._call_cut = float(cut)
        self.last_host = 0  # frames listed on the host (not sparse)
        self._prepare_outputs(nf)
        for g0 in range(0, nf, G):
            ng = self._upload(views[g0 : g0 + G])
            for s0 in range(0, ng, self.batch):
                n = self._launch(s0, min(self.batch, ng - s0), cut)
                self._fetch(g0 + s0, n, npx)
        return (npx, (self._values, self._indices)) + self._extra()

    def _prepare_outputs(self, nf):
        pass

    def _extra(self):
        return ()

    def multi(self, chunks, cut):
        """Decode a batch of chunks (one frame each): npixels[nframes], (values, indices) of shape
        (nframes, ni*nj)."""
        return self._run([memoryview(c).cast("B") for c in chunks], cut)

    def from_buffer(self, base, offsets, lengths, cut):
        """As multi, for the chunks at byte offsets[i] (lengths[i]) inside the one buffer base."""
        b = (
            memoryview(base).cast("B")
            if not isinstance(base, np.ndarray)
            else base.reshape(-1).view(np.uint8)
        )
        views = []
        for o, l in zip(offsets, lengths):
            o, l = int(o), int(l)
            if o < 0 or l < 12 or o + l > len(b):
                raise Exception(
                    "Error decoding batch: -108: a chunk offset or size lies outside the buffer"
                )
            views.append(b[o : o + l])
        return self._run(views, cut)

    def __call__(self, buffer, cut):
        """Decode one chunk: npixels, (values, indices), and for an object with a matrix
        (chunk2sparseCSC) also the powder.  values and indices are full-frame-size views of this
        object's buffers, overwritten by the next call."""
        r = self.multi([buffer], cut)
        npx, (values, indices) = r[0], r[1]
        return (int(npx[0]), (values[0], indices[0])) + tuple(x[0] for x in r[2:])

    def coo(self, buffer, cut):
        """Decode one chunk to (npixels, row, col, values) as arrays of their own (safe to keep),
        and for an object with a matrix also the powder."""
        r = self.__call__(buffer, cut)
        npixels, (values, indices) = r[0], r[1]
        row, col = np.divmod(indices[:npixels].astype(np.int64), self.nfast)
        return (
            npixels,
            row.astype(np.uint16),
            col.astype(np.uint16),
            values[:npixels].copy(),
        ) + tuple(x.copy() for x in r[2:])


class chunk2sparseCSC(chunk2sparse):
    """As chunk2sparse, and every frame also goes through the sparse matrix product powder = csc @
    frame (mask folded into the matrix: masked pixels contribute nothing).

    csc: a scipy csc / pyFAI CSC engine, used as it is (float32 weights; acc 'f64' sums by default,
    or 'f32'), or the result of bslz4_to_sparse.fixed_point_matrix: the powder is then the int64
    sums of its integer weights (acc 'i64'), not converted to float: float powder = powder /
    powder_scale.  They are the CPU class's integers for the same matrix.  The matrix is not
    changed here."""

    def __init__(
        self,
        mask,
        csc,
        dtype=np.uint16,
        codec=None,
        pipeline=None,
        batch=None,
        acc=None,
        howmany=None,
        thresholds=None,
        group=None,
        unblock="gpu",
    ):
        import scipy.sparse as ss

        fixed = csc if isinstance(csc, _matrix.FixedPointMatrix) else None
        if fixed is not None:
            if acc not in (None, "i64"):
                raise ValueError("a fixed_point_matrix is summed with acc='i64'")
            acc = "i64"
        else:
            acc = acc or "f64"
            if acc == "i64":
                raise ValueError("acc='i64' needs a matrix from fixed_point_matrix")
        if acc not in ACC:
            raise ValueError("acc must be one of %s" % list(ACC))
        self.acc = acc
        super().__init__(mask, dtype, codec, pipeline, batch, howmany, thresholds, group, unblock)
        if self.kind == 2 and acc == "i64":
            raise ValueError("acc='i64' needs integer pixels")
        if fixed is not None:
            fixed.check(self.dtype, self.npix)
            nm = fixed.nm
        else:
            nm = _matrix._fold_mask(_matrix.normalise_matrix(csc, npix=self.npix), self.mask)
        self.nbins = nm.nbins
        M = ss.csc_matrix(
            (
                nm.data if fixed is None else fixed.data,
                nm.indices.astype(np.int64),
                nm.indptr.astype(np.int64),
            ),
            shape=nm.shape,
        )
        M.sort_indices()
        self.scale = None if fixed is None else fixed.scale
        data = M.data
        # the GPU wants rows = bins (CSR): one thread block per bin reads its entries, no atomics
        R = ss.csc_matrix((data, M.indices, M.indptr), shape=M.shape).tocsr()
        R.sort_indices()
        if R.nnz >= 2**31 or self.npix >= 2**31:
            raise ValueError("matrix too large for 32-bit CSR indices")
        self.d_coef, self.d_idx, self.d_ptr = _shared(
            self._dev,
            "csr-" + acc,
            [  # one copy per process and device
                R.data.astype(np.uint32 if acc == "i64" else np.float32),
                R.indices.astype(np.int32),
                R.indptr.astype(np.int32),
            ],
        )
        mean = R.nnz / max(self.nbins, 1)  # threads per bin: about 16 entries each, 32 .. 256
        self.wg = int(min(256, max(32, 2 ** int(np.ceil(np.log2(max(mean / 16, 1)))))))
        self.k_csr = {
            F: self.mod.get_function(_csr_name(self.dtype, acc, F))
            for F in sorted({1, FRAMES_PER_READ})
        }
        self.csr_info = dict(nnz=int(R.nnz), nbins=int(self.nbins), threads_per_bin=self.wg)

    @property
    def powder_scale(self):
        """The integer (2**b) an integer powder is divided by to get float; None for float
        accumulators."""
        return self.scale

    def _acc(self):
        return self.acc

    def _on_resize(self, n):
        shape, dt = (n, max(self.nbins, 1)), np.dtype(_fixed_np(self.acc))
        self.d_powder = cu.zeros(shape, dt)
        self.h_powder = np.frombuffer(
            cu.pinned(shape[0] * shape[1] * dt.itemsize), dt, shape[0] * shape[1]
        ).reshape(shape)

    def _dot(self, n):
        if self.nbins == 0:
            return
        F = FRAMES_PER_READ if n >= FRAMES_PER_READ else 1
        self.k_csr[F](
            (self.nbins, (n + F - 1) // F),
            (self.wg,),
            (
                self.d_dense,
                np.uint64(self.stride_el),
                np.int32(n),
                self.d_coef,
                self.d_idx,
                self.d_ptr,
                np.int32(self.nbins),
                self.d_powder,
            ),
        )

    def _prepare_outputs(self, nf):
        if getattr(self, "_powder_buf", None) is None or self._powder_buf.shape[0] < nf:
            self._powder_buf = np.empty(
                (nf, self.nbins), np.int64 if self.acc == "i64" else np.float64
            )
        self._powder_out = self._powder_buf[:nf]

    def _fetch_powder(self, f0, n):
        h = self.h_powder[:n]
        self.d_powder[:n].get(out=h)  # into pinned memory: no allocation per batch
        self._powder_out[f0 : f0 + n] = h[:, : self.nbins]

    def _extra(self):
        return (self._powder_out,)
