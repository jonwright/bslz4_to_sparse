"""bslz4_to_sparse: decode bitshuffle-lz4/zstd detector frames straight
into sparse pixel lists, optionally with a sparse matrix product (pyFAI
integration) in the same pass.

    chunk2sparse(mask, dtype, codec, pipeline)        sparsify
    chunk2sparseCSC(mask, csc, dtype, codec, pipeline) sparsify + matrix dot

Each object has a `pipeline` (src/_pipeline.py): one value per processing
step, resolved when the object is made (0 / None: the best guess for this
CPU, data and matrix) and kept in `.pipeline`; describe() names the values.
An automatic mask step on uint16 data with masked pixels is settled by the
first frame decoded: 'planes' when most masked pixels hold 65535 there (and
the decode block is at most 8192 pixels, a multiple of 64), 'pixel' otherwise.
Nothing is global, so objects with different pipelines can run in
different threads.

    plan(mask, csc, dtype, codec, pipeline)           the matrix of a chunk2sparseCSC, prepared once
    plan.context(howmany, thresholds)                 a chunk2sparseCSC on that plan, per thread

A chunk2sparseCSC owns a copy of its matrix (mask folded, analysed, laid out for the dot).  A plan
is that copy made once and shared: every object made from it (chunk2sparseCSC(mask, plan, ...) or
plan.context()) builds each layout at most once between them, and only reads the plan's arrays,
which are read-only.  Workspace, output buffers and the pipeline stay per object.

Threads: the GIL is released while decoding, and on free-threaded Python
(3.14t, 3.15t) the module runs without a GIL.  There are no locks (a plan has one, for building a layout), so:
  * use one chunk2sparse / chunk2sparseCSC object per thread; an object
    keeps its workspace and output arrays and reuses them on every call;
  * the arrays a call returns are those buffers (except coo(), which
    copies): the next call on the same object overwrites them;
  * leave a call's inputs alone until it returns: do not modify, resize or
    close the chunks, the from_buffer base buffer (e.g. an mmap) or the
    mask from another thread meanwhile.
Sharing an object between threads, or breaking these rules, gives wrong
results or a crash rather than an error.
"""
import json
import math
import os
import struct

import numpy as np

from .c2py_loader import load_native
from . import _pipeline
from . import _matrix
from ._matrix import fixed_point_matrix
from . import _toppix
from ._pipeline import describe, CODEC_LZ4, CODEC_ZSTD

_ext = load_native(os.path.dirname(os.path.abspath(__file__)), "_bslz4_to_sparse")
_pipeline._ext = _ext

version = "0.0.22a3"

__all__ = [
    "version",
    "bslz4_to_sparse",
    "chunk2sparse",
    "chunk2sparseCSC",
    "plan",
    "fixed_point_matrix",
    "describe",
    "detect_codec",
    "locate_chunks",
    "offsets_lengths",
    "decompress",
    "decompress_from_buffer",
    "block_starts",
    "block_starts_from_buffer",
    "build_info",
    "CODEC_LZ4",
    "CODEC_ZSTD",
]

DEFAULT_BLOCK_BYTES = 8192
BSHUF_H5FILTER = 32008


def build_info():
    """
    How the loaded native extension was built, as a dict: the package
    "version", "git" (`git describe --tags --always --dirty` of the source
    tree: a tag when built clean on a tagged commit, "-dirty" when it had
    local changes, None outside a git checkout), "modified" (the changed
    tracked files when dirty), "src_sha256" (of the compiled sources and
    headers, also for builds without git), "compiler", "platform" and
    "built_utc".
    """
    buf = np.zeros(1024, np.uint8)
    n = _ext.build_info(buf)
    if n > buf.size:
        buf = np.zeros(n, np.uint8)
        n = _ext.build_info(buf)
    return json.loads(buf[:n].tobytes().decode("ascii"))


def detect_codec(ds):
    """
    Detect which block codec (CODEC_LZ4 or CODEC_ZSTD) a dataset's
    bitshuffle filter was configured with, from its HDF5 filter pipeline
    (cd_values[4]). Falls back to CODEC_LZ4 if it can't be determined.
    """
    try:
        plist = ds.id.get_create_plist()
        for i in range(plist.get_nfilters()):
            filter_id, _flags, cd_values, _name = plist.get_filter(i)
            if filter_id == BSHUF_H5FILTER and len(cd_values) > 4:
                return cd_values[4]
    except Exception:
        pass
    return CODEC_LZ4


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _codec_of_chunk(cmp):
    """
    The block codec of one compressed bitshuffle chunk, from its bytes: an 8
    byte uncompressed size, a 4 byte block size, then blocks of (4 byte
    compressed size, data).  A zstd block is a zstd frame and starts with
    the zstd magic number; anything else is taken as lz4.
    """
    return CODEC_ZSTD if memoryview(cmp)[16:20] == _ZSTD_MAGIC else CODEC_LZ4


def _codec_from_pipeline(pipeline):
    """CODEC_LZ4 / CODEC_ZSTD if the pipeline names a decode step, else None (automatic)"""
    name = _pipeline.NAMES["decode"][_pipeline.parse(pipeline)[_pipeline.DECODE]]
    if name == "zstd":
        return CODEC_ZSTD
    return None if name == "auto" else CODEC_LZ4


def _blocksize_bytes(cmp):
    if len(cmp) < 12:
        return DEFAULT_BLOCK_BYTES
    blocksize = struct.unpack_from(">I", cmp, 8)[0]
    return blocksize if blocksize else DEFAULT_BLOCK_BYTES


_DECODE_ERRORS = {
    -2: "a chunk failed to decompress (its compressed data is corrupt)",
    -98: "a chunk's decompressed size does not fit in an int",
    -99: "a chunk's decompressed size needs more pixels than the mask has",
    -100: "the cut must not be negative or NaN, and the top_pixels request must be well formed",
    -103: "the workspace or the output buffer is too small",
    -104: "the untranspose kernel reported a failure (corrupt chunk data)",
    -105: "no chunks were given",
    -106: "the chunks do not share the same total size and block size",
    -107: "a chunk is corrupt or truncated (its header, a block length or its "
    "raw tail lies outside the chunk)",
    -108: "a chunk offset or size lies outside the buffer it refers to",
    -109: "the dot's matrix layout does not match the decode entry point, or "
          "the decoded block size differs from the layout's",
    -110: "a chunk's decompressed size is smaller than the mask (frame and mask "
          "shapes differ)",
    -111: "a pipeline step value is unknown, auto (0: resolve it first) or not valid here",
    -112: "a pipeline step value needs an instruction set this CPU/build lacks",
    -113: "the pixel dtype is out of range or unsupported",
    -114: "the cut must be one element of the pixel type",
}
_CORRUPT_CODES = (-2, -104, -107, -108)


def _decode_error(ret, batch=True):
    what = _DECODE_ERRORS.get(ret, "unknown error")
    msg = "Error decoding %s: %d: %s." % ("batch" if batch else "chunk", ret, what)
    if batch and ret in _CORRUPT_CODES:
        msg += (" At least one chunk in this batch is bad; the outputs of the whole"
                " batch are incomplete and must not be used. Decode the chunks one"
                " at a time to find which.")
    return Exception(msg)


def _gather_chunks(chunks):
    n = len(chunks)
    pointers = np.empty(n, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int32)
    for i, chunk in enumerate(chunks):
        _ext.note_chunk(chunk, i, pointers, lengths)
    return pointers, lengths


def _offset_pointers(base, offsets, lengths):
    """offsets[i] bytes into base -> pointers (checked to lie inside base)."""
    pointers = np.array(offsets, dtype=np.int64)      # a copy: converted in place
    lengths = np.ascontiguousarray(lengths, dtype=np.int32)
    if len(pointers) == 0:
        raise _decode_error(-105)
    rc = _ext.offsets_to_pointers(base, pointers, lengths)
    if rc < 0:
        raise _decode_error(rc)
    o = int(offsets[0])
    return pointers, lengths, memoryview(base)[o:o + 20]


def locate_chunks(ds):
    """
    Where each frame's compressed chunk lies in the HDF5 file: a dict
    {frame: (byte_offset, byte_size)} for a (nframes, ni, nj) dataset chunked
    one frame per chunk.  With offsets_lengths, it lets from_buffer
    read the chunks from one buffer holding the file (e.g. an np.memmap)
    with no h5py call per frame.  Raises ValueError for other chunk shapes,
    and for a chunk stored with the bitshuffle filter skipped.
    """
    chunks = tuple(ds.chunks) if ds.chunks is not None else None
    if chunks is None or len(chunks) != 3 or chunks[0] != 1:
        raise ValueError(
            "locate_chunks needs a (1, ni, nj)-chunked dataset, got chunks=%r "
            "(4-D (1,1,ni,nj) chunking is not supported yet)" % (chunks,)
        )

    offsets = {}

    def _store(frame, byte_offset, byte_size, filter_mask):
        if filter_mask & 1:
            raise ValueError(
                "chunk for frame %d has the bitshuffle filter skipped (filter_mask=%d); "
                "raw (uncompressed) chunk passthrough is not supported" % (frame, filter_mask)
            )
        offsets[frame] = (byte_offset, byte_size)

    if hasattr(ds.id, "chunk_iter"):
        def _cb(chunk_info):
            _store(chunk_info.chunk_offset[0], chunk_info.byte_offset,
                   chunk_info.size, chunk_info.filter_mask)
        ds.id.chunk_iter(_cb)
    else:
        for i in range(ds.id.get_num_chunks()):
            info = ds.id.get_chunk_info(i)
            _store(info.chunk_offset[0], info.byte_offset, info.size, info.filter_mask)

    return offsets


def offsets_lengths(frame_offsets, frames):
    """
    The (offsets int64, lengths int32) arrays from_buffer takes, for the
    given frame numbers, from locate_chunks' {frame: (offset, size)}.
    """
    n = len(frames)
    offsets = np.empty(n, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int32)
    for i, frame in enumerate(frames):
        offsets[i], lengths[i] = frame_offsets[frame]
    return offsets, lengths


def _unravel(indices, nfast, row, col):
    """row, col = divmod(indices, nfast) without a division per pixel.

    nfast is fixed for a dataset, so divide by multiplying with its
    reciprocal: magic = ceil(2**40 / nfast), row = (index * magic) >> 40,
    col = index - row * nfast.  Exact while index * (magic * nfast - 2**40)
    < 2**40, which holds for index < 2**26 (67M pixels) and nfast < 2**14
    (checked against divmod for every pixel of the Eiger 4M/16M shapes); the
    product fits in 64 bits.  Larger images fall back to np.divmod."""
    n = indices.shape[0]
    if n == 0:
        return
    if nfast >= (1 << 14) or int(indices.max()) >= (1 << 26):
        np.divmod(indices, nfast, out=(row, col), casting="unsafe")
        return
    t = indices.astype(np.uint64)
    r = t * np.uint64(-(-(1 << 40) // nfast))
    r >>= np.uint64(40)
    row[:] = r
    r *= np.uint64(nfast)
    np.subtract(t, r, out=t)
    col[:] = t


# An automatic mask step on uint16 data is settled on the first frame decoded:
# 'planes' (the mask ANDed into the bit-planes) wins only where masked pixels
# hold the maximum, which stops the byte skip of the high byte-planes; on
# zero-filled masked pixels it is pure overhead (src/steps/mask.h).
_MASK_PLANES_FRACTION = 0.5     # of the masked pixels at 65535 in the first frame


def _mask_auto_pending(pipeline, dtype, mask):
    """True when the mask step was left automatic and the first frame decides it."""
    return (_pipeline.parse(pipeline)[_pipeline.MASK] == 0 and np.dtype(dtype) == np.uint16
            and not bool((mask != 0).all()))


def _cut_array(cut, dtype):
    """The cut as a contiguous one-element array of the pixel dtype, which is how the C code takes
    it (it is compared in the pixel type).  A Python number or a 0-d / one-element array.  Integer
    pixels: a fractional cut is floored (pixel > 2.5 is pixel > 2) and a cut above the type's
    maximum becomes the maximum (nothing is above it).  Negative cuts and NaN are refused."""
    dtype = np.dtype(dtype)
    c = np.asarray(cut)
    if c.size != 1:
        raise ValueError("the cut must be one number, got %d values" % c.size)
    c = c.reshape(-1)[0]
    if dtype.kind == "f":
        v = float(c)
        if v != v:
            raise ValueError("the cut must not be NaN")
        if v < 0:
            raise _decode_error(-100, batch=False)
        return np.array([v], dtype)
    if isinstance(c, (np.integer, int, bool, np.bool_)):
        v = int(c)
    else:
        f = float(c)
        if f != f:
            raise ValueError("the cut must not be NaN")
        v = int(math.floor(f)) if abs(f) != float("inf") else (1 << 70) * (1 if f > 0 else -1)
    if v < 0:
        raise _decode_error(-100, batch=False)
    return np.array([min(v, int(np.iinfo(dtype).max))], dtype)


def _settle_mask(obj, pointers, lengths, cmp):
    """Decode the first chunk once with no mask and keep only the pixels at
    65535: if most masked pixels hold it, switch obj's mask step to 'planes'."""
    obj._mask_pending = False
    block_elems = _blocksize_bytes(cmp) // 2
    if block_elems > 8192 or block_elems % 64:      # BSLZ4_MASKPLANE_MAX_ELEMS: no planes
        return
    p = obj.pipeline.copy()
    p[_pipeline.MASK] = _pipeline.NAMES["mask"].index("none")
    p[_pipeline.ROUTE] = p[_pipeline.DOT] = 0
    values = np.empty(obj.npix, np.uint16)
    adr = np.empty(obj.npix, np.uint32)
    npx = np.empty(1, np.int32)
    ret = _ext.sparsify(
        pointers[:1],
        lengths[:1],
        np.ones(obj.npix, np.uint8),
        values,
        adr,
        npx,
        np.array([65534], np.uint16),
        np.empty(3 * _blocksize_bytes(cmp), np.uint8),
        np.empty(1, np.int64),
        _toppix.EMPTY_TOP,
        _toppix.EMPTY_OUT,
        obj._di,
        p,
    )
    if ret < 0:
        return                      # the real decode reports the error
    masked = obj.mask == 0
    at_max = int(np.count_nonzero(masked[adr[:int(npx[0])]]))
    if at_max >= _MASK_PLANES_FRACTION * int(np.count_nonzero(masked)):
        obj.pipeline[_pipeline.MASK] = _pipeline.NAMES["mask"].index("planes")


def _decompress(pointers, lengths, cmp, dtype, unshuffle, out, codec):
    dtype = np.dtype(dtype)
    if dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError("dtype %s: pixels of 1, 2, 4 or 8 bytes" % dtype)
    cmp = memoryview(cmp)[:20]
    if len(cmp) < 12:
        raise _decode_error(-107)
    total = struct.unpack(">Q", cmp[:8])[0]
    n = len(pointers)
    if unshuffle and total % dtype.itemsize:
        raise ValueError(
            "the chunks hold %d bytes, not a whole number of %s pixels" % (total, dtype)
        )
    if out is None:
        out = np.empty(
            (n, total // dtype.itemsize) if unshuffle else (n, total),
            dtype if unshuffle else np.uint8,
        )
    flat = out.reshape(-1) if out.flags.c_contiguous else None
    if flat is None or flat.nbytes < n * total or not out.flags.writeable:
        raise ValueError(
            "out must be a writeable C-contiguous array of at least %d bytes" % (n * total)
        )
    if codec is None:
        codec = _codec_of_chunk(cmp)
    names = _pipeline.NAMES
    decode = names["decode"].index("zstd" if codec == CODEC_ZSTD else "lz4-stock")
    workspace = np.empty(
        3 * min(_blocksize_bytes(cmp), total) if unshuffle else 1, np.uint8
    )  # a corrupt header cannot ask for more
    ret = _ext.decompress(
        pointers,
        lengths,
        flat.view(np.uint8),
        workspace,
        dtype.itemsize,
        decode,
        names["untranspose"].index("kcb"),
        1 if unshuffle else 0,
    )
    if ret < 0:
        raise _decode_error(ret, batch=n > 1)
    return out


def decompress(chunks, dtype, unshuffle=True, out=None, codec=None):
    """
    Decompress bitshuffle-lz4/zstd chunks (one frame each, as h5py's
    read_direct_chunk gives them) without any sparse processing: the front
    half of the pipeline alone.  The GIL is released, so threads decompress in
    parallel; nothing touches HDF5.

    dtype = the pixel dtype of the dataset
    unshuffle = True: the pixel values, array (nframes, npixels) of dtype.
            False: each block as the HDF5 filter's decompressor yields it,
            before the bit unshuffle: uint8 array (nframes, nbytes), for
            a later stage (the GPU, from_frames after a separate unshuffle)
    out = None, or a writeable C-contiguous array to fill (at least
            nframes * the decoded size in bytes), returned
    codec = None (found from the first chunk), CODEC_LZ4 or CODEC_ZSTD
    """
    if len(chunks) == 0:
        raise _decode_error(-105)
    pointers, lengths = _gather_chunks(chunks)
    return _decompress(pointers, lengths, chunks[0], dtype, unshuffle, out, codec)


def decompress_from_buffer(base, offsets, lengths, dtype, unshuffle=True, out=None, codec=None):
    """As decompress, for the chunks at byte offsets[i] (lengths[i]) inside the
    one buffer base (an mmap of the HDF5 file, or the file read into memory):
    see locate_chunks and offsets_lengths."""
    pointers, lengths, cmp = _offset_pointers(base, offsets, lengths)
    return _decompress(pointers, lengths, cmp, dtype, unshuffle, out, codec)


def _block_starts(pointers, lengths, dtype, out=None, status=None):
    """(rc, starts, status): starts (nframes, nblocks) uint32, status int32 per chunk (0 or -107);
    rc 0, -107 (see status) or another error code.  out: None or a uint32 array to fill."""
    n = len(pointers)
    status = np.empty(n, np.int32) if status is None else status
    nblocks = np.zeros(1, np.int64)
    flat = np.empty(0, np.uint32) if out is None else out.reshape(-1)
    rc = _ext.block_starts(pointers, lengths, flat, status, nblocks, np.dtype(dtype).itemsize)
    if rc == -103 and out is None:  # sized now that nblocks is known
        flat = np.empty(n * int(nblocks[0]), np.uint32)
        rc = _ext.block_starts(pointers, lengths, flat, status, nblocks, np.dtype(dtype).itemsize)
    if rc < 0 and rc != -107:
        return rc, None, status
    return rc, flat[: n * int(nblocks[0])].reshape(n, int(nblocks[0])), status


def _checked_block_starts(pointers, lengths, dtype, out):
    rc, starts, status = _block_starts(pointers, lengths, dtype, out)
    if rc == -107:  # name the bad chunks
        raise Exception("Error decoding batch: -107: %s; chunks %s." % (
            _DECODE_ERRORS[-107], np.flatnonzero(status)[:20].tolist()))
    if rc < 0:
        raise _decode_error(rc)
    return starts


def block_starts(chunks, dtype, out=None):
    """
    Where each compressed block of each chunk starts, found on the CPU
    without decoding: the block index the GPU decoder needs (cuda
    unblock='cpu').  Block b holds pixels [b * B, (b + 1) * B) with B = the
    chunk's block size in pixels.

    Returns starts, a uint32 array (nchunks, nblocks): the byte offset, from
    the start of its chunk, of each block's 4-byte big-endian compressed
    size; the compressed block follows it.  The last block may be partial,
    and up to 8 * dtype.itemsize - 1 bytes of pixels after it are stored raw
    at the end of the chunk.  lz4 or zstd.  Every chunk must have the same
    total size and block size; a chunk whose blocks do not tile it exactly
    raises (corrupt or truncated).
    out = None, or a uint32 array of nchunks * nblocks elements to fill
    """
    if len(chunks) == 0:
        raise _decode_error(-105)
    pointers, lengths = _gather_chunks(chunks)
    return _checked_block_starts(pointers, lengths, dtype, out)


def block_starts_from_buffer(base, offsets, lengths, dtype, out=None):
    """As block_starts, for the chunks at byte offsets[i] (lengths[i]) inside
    the one buffer base; the starts stay relative to each chunk."""
    pointers, lengths, _ = _offset_pointers(base, offsets, lengths)
    return _checked_block_starts(pointers, lengths, dtype, out)


class chunk2sparse:
    """
    Decode bitshuffle-lz4/zstd chunks of (ni, nj) frames into the pixels
    above a cut (and not masked).

    mask = detector mask, (ni, nj) uint8, active pixels > 0
    dtype = the pixel dtype of the dataset
    codec = None (default): found from the first chunk decoded, or CODEC_LZ4 /
            CODEC_ZSTD (detect_codec(dataset) reads it from the HDF5 file).
            Until then .codec is CODEC_LZ4 and .pipeline is resolved for lz4;
            it is resolved again if the first chunk is zstd.
    pipeline = None (best guess), or step values: see describe()
    howmany = None (default): every pixel above the cut is returned.  An
            integer: at most that many per frame, chosen as ImageD11's
            top_pixels does.  A frame with more pixels above the cut keeps
            those above the first of the sorted `thresholds` that leaves
            fewer than howmany (else above the last), and then the first
            howmany of them in pixel order.
    thresholds = None: max(cut, 1) * 2**i for i in 0..15, or up to 16 sorted
            values.  Only used when howmany is set.

    cut is compared in the pixel type: for integer pixels a fractional cut is
    floored and a cut above the type's maximum selects nothing; a negative cut
    is an error.

    c(chunk, cut) -> npixels, (values, indices) for one frame (the arrays
    are full size: slice [:npixels]); c.multi(chunks, cut) -> the same per
    frame for a batch, npixels an array and the arrays (nframes, ni*nj).
    After a call, last_cut[i] is the cut frame i ended up with (the call's
    cut unless howmany reduced it) and last_nabove[i] the number of pixels
    above that cut, before the list was truncated to howmany.
    """

    def __init__(
        self, mask, dtype=np.uint16, codec=None, pipeline=None, howmany=None, thresholds=None
    ):
        self.howmany, self.thresholds = _toppix.check(howmany, thresholds)
        self.nfast = mask.shape[1]
        self.mask = mask.ravel()
        self.npix = mask.size
        self.dtype = np.dtype(dtype)
        if codec is None:
            codec = _codec_from_pipeline(pipeline)
        self._codec_pending = codec is None
        self.codec = CODEC_LZ4 if codec is None else codec
        self._pipeline_arg = pipeline
        self.pipeline = self._resolve()
        self._mask_pending = _mask_auto_pending(pipeline, self.dtype, self.mask)
        self._di = _pipeline.DTYPE_INDEX[_pipeline.dtype_suffix(self.dtype)]
        self._nframes = 0
        self._workspace = None

    def _resolve(self):
        return _pipeline.resolve(self._pipeline_arg, self.dtype, self.codec, self.mask)

    def _settle_codec(self, cmp):
        """The first chunk decides the codec when it was not given"""
        self._codec_pending = False
        codec = _codec_of_chunk(cmp)
        if codec != self.codec:
            self.codec = codec
            self.pipeline = self._resolve()

    def _ensure_capacity(self, nframes, cmp, blocksize=None):
        if self._nframes < nframes:  # grow only: a smaller batch uses views, nothing is reallocated
            self._values_all = np.empty((nframes, self.npix), self.dtype)
            self._indices_all = np.empty((nframes, self.npix), np.uint32)
            self._npx_all = np.empty(nframes, np.int32)
            self._cursors_all = np.empty(nframes, np.int64)
            self._nframes = nframes
        self._values, self._indices = self._values_all[:nframes], self._indices_all[:nframes]
        self._npx, self._cursors = self._npx_all[:nframes], self._cursors_all[:nframes]
        need = 3 * (blocksize or _blocksize_bytes(cmp))
        if self._workspace is None or self._workspace.size < need:
            self._workspace = np.empty(need, np.uint8)

    def _run(self, pointers, lengths, cmp, cut, pixels=False):
        if pixels:
            pipe = self._pixel_pipeline()
            self._ensure_capacity(len(pointers), None, DEFAULT_BLOCK_BYTES)
        else:
            if self._codec_pending:
                self._settle_codec(cmp)
            if self._mask_pending:
                _settle_mask(self, pointers, lengths, cmp)
            self._ensure_capacity(len(pointers), cmp)
            pipe = self.pipeline  # after the settling above: the first chunk may change it
        cut_a = _cut_array(cut, self.dtype)
        req = _toppix.Request(self.howmany, self.thresholds, cut_a[0].item(), self.dtype, len(pointers))
        ret = _ext.sparsify(
            pointers,
            lengths,
            self.mask,
            self._values.ravel(),
            self._indices.ravel(),
            self._npx,
            cut_a,
            self._workspace,
            self._cursors,
            req.top,
            req.out,
            self._di,
            pipe,
        )
        if ret < 0:
            self._npx[:] = 0
            raise _decode_error(ret, batch=len(pointers) > 1)
        self.last_nabove, self.last_cut = req.finish(self._npx, self._values, self._indices)
        return self._npx, (self._values, self._indices)

    def multi(self, chunks, cut):
        """Decode a batch of chunks (one frame each) of this dataset:
        returns npixels[nframes], (values, indices) of shape (nframes, ni*nj)."""
        if len(chunks) == 0:
            raise _decode_error(-105)
        pointers, lengths = _gather_chunks(chunks)
        return self._run(pointers, lengths, chunks[0], cut)

    def _pixel_pipeline(self):
        """This object's resolved pipeline with the decode step 'pixels' (the input is uncompressed)."""
        p = self.pipeline.copy()
        p[_pipeline.DECODE] = _pipeline.NAMES["decode"].index("pixels")
        return p

    def from_frames(self, frames, cut):
        """As multi, for uncompressed frames: an array (nframes, ni, nj) or (nframes, ni*nj) of this
        object's pixel dtype.  Only the back half of the pipeline runs (mask, collect, matrix
        product, howmany): the same results as multi on the compressed frames.  A C-contiguous array
        of the right dtype is used in place, else a copy is made."""
        arr = np.require(frames, self.dtype, ["C", "A"])  # C-contiguous and aligned, else a copy
        if arr.size == 0 or arr.size % self.npix:
            raise ValueError("frames must hold whole frames of %d pixels" % self.npix)
        arr = arr.reshape(-1, self.npix)
        nbytes = self.npix * self.dtype.itemsize
        pointers = arr.ctypes.data + nbytes * np.arange(len(arr), dtype=np.int64)
        lengths = np.full(len(arr), nbytes, np.int32)
        return self._run(pointers, lengths, None, cut, pixels=True)

    def from_buffer(self, base, offsets, lengths, cut):
        """As multi, for the chunks at byte offsets[i] (lengths[i]) inside
        the one buffer base (an mmap of the HDF5 file, or the file read into
        memory): see locate_chunks and offsets_lengths."""
        pointers, lengths, cmp = _offset_pointers(base, offsets, lengths)
        return self._run(pointers, lengths, cmp, cut)

    def __call__(self, buffer, cut):
        """
        Decode one compressed chunk (one frame): returns npixels, (values,
        indices).  values and indices are full-frame-size views of this
        object's buffers: the first npixels entries are the pixels above
        cut (and not masked), in pixel order, and the next call overwrites
        them.
        """
        npx, (values, indices) = self.multi([buffer], cut)
        return int(npx[0]), (values[0], indices[0])

    def coo(self, buffer, cut):
        """
        Decode one chunk to (npixels, row, col, values): uint16 row and
        column of each pixel above cut, and its value.  These are new arrays
        (copies), unlike the buffers __call__ returns.
        """
        npixels, (values, indices) = self.__call__(buffer, cut)
        row = np.empty(npixels, np.uint16)
        col = np.empty(npixels, np.uint16)
        _unravel(indices[:npixels], self.nfast, row, col)
        return npixels, row, col, values[:npixels].copy()


def bslz4_to_sparse(ds, num, cut, mask=None, pixelbuffer=None, workspace=None, codec=None,
                    pipeline=None):
    """
    Reads a bitshuffle compressed hdf5 dataset and converts this directly
    into a sparse format (indices, values) when decoding the data.

    ds = hdf5 dataset containing [nframes, ni, nj] pixels
    num = frame number to read
    cut = threshold, pixels not above this value are ignored (compared in the pixel type)
    mask = detector mask. Active pixels > 0.
    pixelbuffer = None or (values, indices) storage space
    workspace = None or a uint8 array to use as decode scratch space
    codec = None to auto-detect lz4 vs zstd, or an explicit CODEC_LZ4 / CODEC_ZSTD
    pipeline = None (best guess), or step values: see describe()

    returns (number_of_pixels, (values, indices))
    """
    if mask is None:
        mask = np.ones((ds.shape[1], ds.shape[2]), np.uint8).ravel()
    if pixelbuffer is None:
        indices = np.empty((ds.shape[1], ds.shape[2]), np.uint32).ravel()
        values = np.empty((ds.shape[1], ds.shape[2]), ds.dtype).ravel()
    else:
        values, indices = pixelbuffer
    if codec is None:
        codec = detect_codec(ds)
    p = _pipeline.resolve(pipeline, values.dtype, codec, mask)
    filtinfo, buffer = ds.id.read_direct_chunk((num, 0, 0))
    if workspace is None:
        workspace = np.empty(3 * _blocksize_bytes(buffer), np.uint8)
    pointers, lengths = _gather_chunks([buffer])
    npx_out = np.empty(1, np.int32)
    cursors = np.empty(1, np.int64)
    ret = _ext.sparsify(
        pointers,
        lengths,
        mask,
        values,
        indices,
        npx_out,
        _cut_array(cut, values.dtype),
        workspace,
        cursors,
        _toppix.EMPTY_TOP,
        _toppix.EMPTY_OUT,
        _pipeline.DTYPE_INDEX[_pipeline.dtype_suffix(values.dtype)],
        p,
    )
    if ret < 0:
        raise _decode_error(ret, batch=False)
    return int(npx_out[0]), (values, indices)


class chunk2sparseCSC:
    """
    As chunk2sparse, and every frame also goes through a sparse matrix
    product (a pyFAI integration): powder = csc @ frame.

    mask = detector mask, (ni, nj) uint8, active pixels > 0
    csc = scipy.sparse matrix (nout, ni*nj), or a pyFAI CSC integrator
          (data, indices, indptr, bins): used as it is, float32 weights, float64
          powder.  Or a plan(mask, csc, ...), which holds the matrix prepared once (mask folded,
          analysed, layouts built on demand) for any number of objects; mask must be the one
          the plan was made with, else ValueError.  Or the result of
          fixed_point_matrix(csc, mask, dtype): integer
          weights, and the powder is the int64 sums of weight x pixel, exact, not
          converted to float: float powder = powder / powder_scale (an integer power
          of two, 2**b).  With a matrix of weight 1 per pixel (no pixel splitting) these are
          exactly the photon counts summed per bin.  Either way the matrix is not
          changed here: preparing it is a step of its own.
    dtype, codec, pipeline: as chunk2sparse; pipeline's dot step picks the
          matrix layout (auto: from what the matrix is; "csc-int" for a
          fixed_point_matrix, which also takes csc-run-int and, made with wbits=16,
          csc-run-int16).  A matrix of weight 1 and one bin per pixel (a histogram) of integer
          pixels of at most 32 bits is auto "csc-nosplit-int": the sums are accumulated in int64
          (no weights, no conversion to float) and returned as float64 like any float dot; a bin
          sum above 2**53 is rounded in that last cast
    howmany, thresholds: as chunk2sparse; they limit the pixels returned
          only: the powder always uses every unmasked pixel

    The array of the powder is the object's own and is overwritten by the next call.
    The object owns copies of the matrix and the mask, taken here (read-only): later changes to
    the arrays you passed do not reach it.  The mask is folded into a raw matrix once, here:
    masked pixels contribute nothing to the powder (fixed_point_matrix does it for the matrix it
    makes).  Objects made again from the same csc each repeat this and hold their own copy: for
    one object per thread, make a plan(mask, csc, ...) and give it to each in place of csc (or
    use plan.context()); they then share the copy and the layouts, each built once.
    c(chunk, cut) -> npixels, (values, indices), powder; c.multi(chunks, cut)
    -> the same per frame for a batch.
    """

    def __init__(
        self,
        mask,
        csc,
        dtype=np.uint16,
        codec=None,
        pipeline=None,
        howmany=None,
        thresholds=None,
    ):
        self.howmany, self.thresholds = _toppix.check(howmany, thresholds)
        # the matrix and mask are copies, so a later change to the caller's arrays changes nothing
        # here; a plan is used as it is (shared, its arrays read-only)
        if isinstance(csc, _matrix._Family):
            family = csc
            m = np.asarray(mask)
            if m.shape != family.shape or not np.array_equal(m.reshape(-1), family.mask):
                raise ValueError("the mask differs from the one the plan was made with")
        else:
            family = _matrix._Family(csc, mask)
        self._family = family
        self.nfast = family.shape[1]
        self.mask = family.mask
        self.npix = self.mask.size
        self.dtype = np.dtype(dtype)
        fixed = family.fixed
        if fixed is not None:
            fixed.check(self.dtype, self.npix)
            if pipeline is not None and not isinstance(pipeline, dict):
                raise ValueError("a fixed_point_matrix takes the pipeline as a dict (or None)")
            pipeline = dict(pipeline or {})
            pipeline.setdefault("dot", "csc-int" if fixed.data.dtype == np.uint32 else "csc-run-int16")
        nm = family.nm
        self._fixed = fixed
        if codec is None:
            codec = _codec_from_pipeline(pipeline)
        self._codec_pending = codec is None
        self.codec = CODEC_LZ4 if codec is None else codec
        self._pipeline_arg = pipeline
        self._nm = nm
        self.nbins = nm.nbins
        padded_avx2 = _pipeline.available("dot", _pipeline.NAMES["dot"].index("padded-avx2")) == 1
        self._padded_avx2 = padded_avx2
        self._dot_is_auto = _pipeline.parse(pipeline)[_pipeline.DOT] == 0
        self.pipeline = self._resolve()
        self._mask_pending = _mask_auto_pending(pipeline, self.dtype, self.mask)
        self.dot = _pipeline.NAMES["dot"][self.pipeline[_pipeline.DOT]]
        self._di = _pipeline.DTYPE_INDEX[_pipeline.dtype_suffix(self.dtype)]
        self._layout = self._build(DEFAULT_BLOCK_BYTES // self.dtype.itemsize)
        self._nframes = 0
        self._workspace = None

    @property
    def powder_scale(self):
        """The integer (2**b) an integer powder is divided by to get float; None for a float dot."""
        return self._layout.scale

    def _resolve(self):
        info, padded_avx2 = self._family.info, self._padded_avx2
        return _pipeline.resolve(
            self._pipeline_arg, self.dtype, self.codec, self.mask, matrix=True,
            dot_auto=lambda: _matrix.auto_dot(info, padded_avx2,
                                              DEFAULT_BLOCK_BYTES // self.dtype.itemsize,
                                              self._int_pixels()))

    def _int_pixels(self):
        return self.dtype.kind in "ui" and self.dtype.itemsize <= 4

    def _settle_codec(self, cmp):
        """The first chunk decides the codec when it was not given (the codec
        does not change the dot step, so the matrix layout stands)"""
        self._codec_pending = False
        codec = _codec_of_chunk(cmp)
        if codec != self.codec:
            self.codec = codec
            self.pipeline = self._resolve()

    def _build(self, block_elems):
        try:
            return self._family.layout(self.dot, block_elems, self.dtype)
        except ValueError as e:
            raise ValueError(
                "pipeline['dot']=%r cannot represent this matrix: %s (auto picks %r)"
                % (self.dot, e, _matrix.auto_dot(self._family.info, False, None, self._int_pixels()))
            )

    def _ensure_capacity(self, nframes, cmp, blocksize=None):
        lay = self._layout
        if self._nframes < nframes:  # grow only: a smaller batch uses views, nothing is reallocated
            self._values_all = np.empty((nframes, self.npix), self.dtype)
            self._indices_all = np.empty((nframes, self.npix), np.uint32)
            self._npx_all = np.empty(nframes, np.int32)
            self._cursors_all = np.empty(nframes, np.int64)
            self._nframes = nframes
        self._values, self._indices = self._values_all[:nframes], self._indices_all[:nframes]
        self._npx, self._cursors = self._npx_all[:nframes], self._cursors_all[:nframes]
        blocksize = blocksize or _blocksize_bytes(cmp)
        be = blocksize // self.dtype.itemsize
        if lay.block_elems is not None and be != lay.block_elems:
            # a block-dependent layout built at another block size: rebuild
            # (C also refuses a mismatch with -109).  An automatic dot is
            # chosen again for this block size (bsb-csr takes <= 65536 pixels).
            if self._dot_is_auto:
                dot = _matrix.auto_dot(self._family.info, self._padded_avx2, be,
                                       self._int_pixels())
                if dot != self.dot:
                    self.dot = dot
                    self.pipeline[_pipeline.DOT] = _pipeline.NAMES["dot"].index(dot)
            self._layout = self._build(be)
            lay = self._layout
        pshape = (self._nframes, lay.nout)
        if getattr(self, "_powder_all", None) is None or (
            self._powder_all.shape != pshape or self._powder_all.dtype != lay.powder_dtype
        ):
            self._powder_all = np.empty(pshape, lay.powder_dtype)
        self._powder = self._powder_all[:nframes]
        need = 3 * blocksize + be * (4 + self.dtype.itemsize)
        if self._workspace is None or self._workspace.size < need:
            self._workspace = np.empty(need, np.uint8)

    def _run(self, pointers, lengths, cmp, cut, pixels=False):
        if pixels:
            pipe = self._pixel_pipeline()
            self._ensure_capacity(len(pointers), None, DEFAULT_BLOCK_BYTES)
        else:
            if self._codec_pending:
                self._settle_codec(cmp)
            if self._mask_pending:
                _settle_mask(self, pointers, lengths, cmp)
            self._ensure_capacity(len(pointers), cmp)
            pipe = self.pipeline  # after the settling above: the first chunk may change it
        lay = self._layout
        a = lay.args()
        cut_a = _cut_array(cut, self.dtype)
        req = _toppix.Request(self.howmany, self.thresholds, cut_a[0].item(), self.dtype, len(pointers))
        head = (pointers, lengths, self.mask, self._values.ravel(), self._indices.ravel(),
                self._npx, cut_a, self._powder.ravel())
        tail = (self._di, pipe)
        if lay.entry == "csc":
            ret = _ext.sparsify_and_dot(
                *(head + a + (self._workspace, self._cursors, req.top, req.out, lay.nout) + tail)
            )
        elif lay.entry == "padded":
            ret = _ext.sparsify_and_dot_padded(
                *(
                    head
                    + a[:5]
                    + (self._workspace, self._cursors, req.top, req.out)
                    + a[5:]
                    + (lay.nout,)
                    + tail
                )
            )
        else:
            ret = _ext.sparsify_and_dot_bsbcsr(
                *(
                    head
                    + a[:8]
                    + (self._workspace, self._cursors, req.top, req.out)
                    + a[8:]
                    + (lay.nout,)
                    + tail
                )
            )
        if ret < 0:
            self._npx[:] = 0
            raise _decode_error(ret, batch=len(pointers) > 1)
        powder = self._powder
        self.last_nabove, self.last_cut = req.finish(self._npx, self._values, self._indices)
        return self._npx, (self._values, self._indices), powder

    def multi(self, chunks, cut):
        """Decode a batch of chunks (one frame each) of this dataset:
        returns npixels[nframes], (values, indices) of shape (nframes,
        ni*nj), powder (nframes, nout)."""
        if len(chunks) == 0:
            raise _decode_error(-105)
        pointers, lengths = _gather_chunks(chunks)
        return self._run(pointers, lengths, chunks[0], cut)

    def _pixel_pipeline(self):
        """This object's resolved pipeline with the decode step 'pixels' (the input is uncompressed)."""
        p = self.pipeline.copy()
        p[_pipeline.DECODE] = _pipeline.NAMES["decode"].index("pixels")
        return p

    def from_frames(self, frames, cut):
        """As multi, for uncompressed frames: an array (nframes, ni, nj) or (nframes, ni*nj) of this
        object's pixel dtype.  Only the back half of the pipeline runs (mask, collect, matrix
        product, howmany): the same results as multi on the compressed frames.  A C-contiguous array
        of the right dtype is used in place, else a copy is made."""
        arr = np.require(frames, self.dtype, ["C", "A"])  # C-contiguous and aligned, else a copy
        if arr.size == 0 or arr.size % self.npix:
            raise ValueError("frames must hold whole frames of %d pixels" % self.npix)
        arr = arr.reshape(-1, self.npix)
        nbytes = self.npix * self.dtype.itemsize
        pointers = arr.ctypes.data + nbytes * np.arange(len(arr), dtype=np.int64)
        lengths = np.full(len(arr), nbytes, np.int32)
        return self._run(pointers, lengths, None, cut, pixels=True)

    def from_buffer(self, base, offsets, lengths, cut):
        """As multi, for the chunks at byte offsets[i] (lengths[i]) inside
        the one buffer base: see locate_chunks and
        offsets_lengths."""
        pointers, lengths, cmp = _offset_pointers(base, offsets, lengths)
        return self._run(pointers, lengths, cmp, cut)

    def __call__(self, buffer, cut):
        """
        Decompress one chunk, put the pixels above cut into (values, indices) and every pixel
        into the matrix product.

        returns npixels, (values, indices), powder_sum: values and indices are full-frame-size
        views of buffers owned by this object, overwritten by the next call (slice
        values[:npixels] and indices[:npixels]; copy what you keep); powder_sum is float64, or
        int64 when this object was made with a fixed_point_matrix (divide by powder_scale for float).
        """
        npx, (values, indices), powder = self.multi([buffer], cut)
        return int(npx[0]), (values[0], indices[0]), powder[0]

    def coo(self, buffer, cut):
        """
        Decode one chunk to (npixels, row, col, values, powder): as
        chunk2sparse.coo, plus the frame's powder.  All are new arrays
        (copies), unlike the buffers __call__ returns.
        """
        npixels, (values, indices), powder = self.__call__(buffer, cut)
        row = np.empty(npixels, np.uint16)
        col = np.empty(npixels, np.uint16)
        _unravel(indices[:npixels], self.nfast, row, col)
        return npixels, row, col, values[:npixels].copy(), powder.copy()


class plan(_matrix._Family):
    """
    A matrix prepared once for any number of chunk2sparseCSC objects, one per thread: the plan
    holds its own read-only copy of csc (mask folded in, or the fixed_point_matrix copied), what
    the matrix is (for the automatic dot) and the layouts the objects ask for, each built once
    however many ask at the same time.  Later changes to the arrays passed in do not reach it.

        p = plan(mask, csc, dtype, codec, pipeline)
        c = p.context(howmany, thresholds)     # or chunk2sparseCSC(mask, p, dtype, ...)

    mask, csc, dtype, codec, pipeline: as chunk2sparseCSC.  They are the defaults of context();
    chunk2sparseCSC(mask, p, ...) takes its own.  A plan is never changed by its objects except
    by adding a built layout.
    """

    def __init__(self, mask, csc, dtype=np.uint16, codec=None, pipeline=None):
        _matrix._Family.__init__(self, csc, mask)
        self.dtype = np.dtype(dtype)
        self._defaults = (codec, dict(pipeline) if isinstance(pipeline, dict) else pipeline)
        self.context()      # refuses a pipeline or matrix that do not go together now, and makes
        #                     the first layout

    def context(self, howmany=None, thresholds=None):
        """A chunk2sparseCSC on this plan, for one thread: its own workspace, output arrays and
        pipeline; the matrix and layouts are the plan's.  howmany, thresholds: as chunk2sparse."""
        return chunk2sparseCSC(self.mask.reshape(self.shape), self, self.dtype, self._defaults[0],
                               self._defaults[1], howmany, thresholds)
