import os
import struct
import numpy as np
import ctypes
from .c2py_loader import load_native

_ext = load_native(os.path.dirname(os.path.abspath(__file__)), "_bslz4_to_sparse")

version = "0.0.20a1"

# We cast away the 'read-only' nature of python bytes.
# Not needed for the latest numpy.
if hasattr(ctypes.pythonapi, "PyMemoryView_FromMemory"):
    buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
    buffer_from_memory.restype = ctypes.py_object
    buffer_from_memory.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
else:
    def buffer_from_memory(buf, l, f):
        if type(buf) == str and buf[:6] == 'array(':
            import warnings
            warnings.warn('Bad performance from python2.7 and old h5py')
            from array import array
            return eval(buf)
        raise Exception('Unknown code path for buffer decoding ' + str(type(buf)))


def npbuf(buf):
    if isinstance(buf, np.ndarray):
        return buf
    elif isinstance(buf, memoryview):
        return np.frombuffer(buf, np.uint8)
    else:
        return np.frombuffer(buffer_from_memory(buf, len(buf), 0x200), np.uint8)


_TYPE_SUFFIXES = ("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64")

_SUFFIX_FOR_KIND_ITEMSIZE = {
    ("u", 1): "u8", ("u", 2): "u16", ("u", 4): "u32", ("u", 8): "u64",
    ("i", 1): "i8", ("i", 2): "i16", ("i", 4): "i32", ("i", 8): "i64",
    ("f", 4): "f32", ("f", 8): "f64",
}


def _suffix_for_dtype(dtype):
    dtype = np.dtype(dtype)
    try:
        return _SUFFIX_FOR_KIND_ITEMSIZE[(dtype.kind, dtype.itemsize)]
    except KeyError:
        raise TypeError("unsupported pixel dtype %r" % (dtype,))


# ---- Stage / implementation tables (mirror bslz4_registry.h) ----
# Stage ids.
_STAGE_DECOMPRESS, _STAGE_UNTRANSPOSE, _STAGE_COLLECT, _STAGE_DOT = 0, 1, 2, 3
# dtype index (plan.md section 5): u8,u16,u32,u64,i8,i16,i32,i64,f32,f64
_SUFFIX_TO_DTYPE = {"u8": 0, "u16": 1, "u32": 2, "u64": 3, "i8": 4, "i16": 5,
                    "i32": 6, "i64": 7, "f32": 8, "f64": 9}
_BACKEND_TO_ID = {"kcb": 0, "sse": 1, "neon": 2, "scal": 3}
_COLLECT_ID_TO_NAME = {0: "scalar", 1: "avx512", 2: "avx2", 3: "sse2", 4: "vsx", 5: "neon"}

DEFAULT_BLOCK_BYTES = 8192
BSHUF_H5FILTER = 32008
CODEC_LZ4 = 2
CODEC_ZSTD = 3

# ---- Module-level defaults (the shims edit these; no C globals) ----
_BACKEND_NAMES = ("kcb", "sse", "neon", "scal")
_default_backend = "kcb"
_dense_sparse_threshold = 8.0


def _backend_available(name):
    return _ext.impl_available(_STAGE_UNTRANSPOSE, _BACKEND_TO_ID[name]) > 0


def available_backends():
    """Names of the untranspose backends usable on this machine."""
    return tuple(sorted(name for name in _BACKEND_NAMES if _backend_available(name)))


def set_backend(name):
    """Select the untranspose backend (see available_backends()). None restores
    the best available default."""
    if name is not None and not _backend_available(name):
        raise ValueError(
            "backend %r is not usable in this build (available: %s)"
            % (name, ", ".join(available_backends()))
        )
    global _default_backend
    _default_backend = name if name is not None else _best_backend()


def _best_backend():
    for n in _BACKEND_NAMES:
        if _backend_available(n):
            return n
    return _BACKEND_NAMES[0]


def set_dense_sparse_threshold(x):
    """Set the compression-factor threshold routing the CSC decode between
    its dense and sparse per-(frame,block) paths."""
    global _dense_sparse_threshold
    _dense_sparse_threshold = float(x)


def get_dense_sparse_threshold():
    """Current dense/sparse routing threshold (see set_dense_sparse_threshold)."""
    return _dense_sparse_threshold


# ---- Per-tier SIMD collect shims (u16/u32 only) ----
avx512_collect_available = _ext.avx512_collect_available
get_avx512_collect = _ext.get_avx512_collect
avx2_collect_available = _ext.avx2_collect_available
get_avx2_collect = _ext.get_avx2_collect
sse2_collect_available = _ext.sse2_collect_available
get_sse2_collect = _ext.get_sse2_collect
vsx_collect_available = _ext.vsx_collect_available
get_vsx_collect = _ext.get_vsx_collect
neon_collect_available = _ext.neon_collect_available
get_neon_collect = _ext.get_neon_collect


def _collect_setter(name, tiername):
    def setter(enabled):
        if getattr(_ext, "set_%s_collect" % name)(1 if enabled else 0) < 0:
            raise RuntimeError(
                "%s collect kernel requested but this build/CPU lacks it "
                "(%s_collect_available() is False)" % (tiername, name)
            )
    return setter


set_avx512_collect = _collect_setter("avx512", "AVX-512")
set_avx2_collect = _collect_setter("avx2", "AVX2")
set_sse2_collect = _collect_setter("sse2", "SSE2")
set_vsx_collect = _collect_setter("vsx", "VSX")
set_neon_collect = _collect_setter("neon", "NEON")


def _active_collect_tier():
    for mid, name in [(1, "avx512"), (2, "avx2"), (3, "sse2"), (4, "vsx"), (5, "neon")]:
        if getattr(_ext, "get_%s_collect" % name)():
            return mid
    return 0


# ---- Pipeline packing ----
def pack_pipeline(decompress=None, untranspose=None, collect=None, dot=None, options=0):
    """
    Build the u64 pipeline word from stage ids, defaulting each stage from
    the module-level defaults (codec, untranspose backend, active collect
    tier, csc).  Unavailable selections raise NotImplementedError naming the
    stage and implementation.
    """
    if decompress is None:
        decompress = _default_codec()
    if untranspose is None:
        untranspose = _BACKEND_TO_ID[_default_backend]
    if collect is None:
        collect = _active_collect_tier()
    if dot is None:
        dot = 0
    for stage, value, name in ((_STAGE_DECOMPRESS, decompress, "decompress"),
                               (_STAGE_UNTRANSPOSE, untranspose, "untranspose"),
                               (_STAGE_COLLECT, collect, "collect"),
                               (_STAGE_DOT, dot, "dot")):
        avail = _ext.impl_available(stage, value)
        if avail < 0:
            raise NotImplementedError("unknown %s implementation id %d" % (name, value))
        if avail == 0:
            raise NotImplementedError(
                "%s implementation (id %d) is not available in this build/CPU" % (name, value)
            )
    if options & ~(1 << 0):
        raise ValueError("unknown option bits in pipeline: %r" % (options,))
    return _pack(decompress, untranspose, collect, dot, options)


def _default_codec():
    return CODEC_LZ4


def _pack(decompress, untranspose, collect, dot, options):
    # Mirrors BSLZ4_PIPE_MAKE in bslz4_common.h: 4 bits per stage id, 8 bits
    # of options, packed into the low 24 bits (c2py23 accepts a signed int,
    # so the pipeline cannot use bits >= 31).
    return ((options & 0xFF) << 16 | (dot & 0xF) << 12 |
            (collect & 0xF) << 8 | (untranspose & 0xF) << 4 |
            (decompress & 0xF))


def _pipeline_for(codec, collect_tier, dot=0):
    """Pipeline for the current module defaults, given an explicit codec, the
    collect tier for this dtype (0 for anything but u16/u32, which are the
    only types the SIMD collect tiers support) and the dot implementation id."""
    return _pack(codec, _BACKEND_TO_ID[_default_backend], collect_tier, dot, 0)


def _collect_tier_for_suffix(suffix):
    if suffix in ("u16", "u32"):
        return _active_collect_tier()
    return 0


def _make_sparsify(suffix, pipeline=None):
    di = _SUFFIX_TO_DTYPE[suffix]

    def fn(pointers, lengths, mask, output, output_adr, npx_out, threshold,
           workspace, cursors, codec):
        p = pipeline if pipeline is not None else _pipeline_for(codec, _collect_tier_for_suffix(suffix))
        return _ext.sparsify(pointers, lengths, mask, output, output_adr, npx_out,
                             threshold, workspace, cursors, di, p)
    return fn


def _make_csc(suffix, pipeline=None, dot=0):
    di = _SUFFIX_TO_DTYPE[suffix]

    def fn(pointers, lengths, mask, outpx, output_adr, npx_out, threshold, powder, data,
           indices, indptr, workspace, cursors, nout, codec):
        p = pipeline if pipeline is not None else _pipeline_for(codec, _collect_tier_for_suffix(suffix), dot)
        return _ext.sparsify_and_dot(pointers, lengths, mask, outpx, output_adr, npx_out,
                                     threshold, powder, data, indices, indptr,
                                     workspace, cursors, nout, _dense_sparse_threshold,
                                     di, p)
    return fn


def _make_csc_base(suffix, pipeline=None, dot=0):
    di = _SUFFIX_TO_DTYPE[suffix]

    def fn(base, offsets, lengths, mask, outpx, output_adr, npx_out, threshold, powder, data,
           indices, indptr, workspace, cursors, nout, codec):
        rc = _ext.offsets_to_pointers(base, offsets, lengths, len(lengths))
        if rc < 0:
            return rc
        p = pipeline if pipeline is not None else _pipeline_for(codec, _collect_tier_for_suffix(suffix), dot)
        return _ext.sparsify_and_dot(offsets, lengths, mask, outpx, output_adr, npx_out,
                                     threshold, powder, data, indices, indptr,
                                     workspace, cursors, nout, _dense_sparse_threshold,
                                     di, p)
    return fn


_BSLZ4_MULTI = {s: _make_sparsify(s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI = {s: _make_csc(s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI_BASE = {s: _make_csc_base(s) for s in _TYPE_SUFFIXES}

note_chunk = _ext.note_chunk
# Availability and test counters.
impl_available = _ext.impl_available
reset_counters = _ext.reset_counters
read_counters = _ext.read_counters


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


def _blocksize_bytes(cmp):
    if len(cmp) < 12:
        return DEFAULT_BLOCK_BYTES
    blocksize = struct.unpack(">I", cmp[8:12])[0]
    return blocksize if blocksize else DEFAULT_BLOCK_BYTES


_DECODE_ERRORS = {
    -2: "a chunk failed to decompress (its compressed data is corrupt)",
    -98: "a chunk's decompressed size does not fit in an int",
    -99: "a chunk's decompressed size needs more pixels than the mask has",
    -100: "threshold must not be negative",
    -103: "the workspace is too small",
    -104: "the untranspose kernel reported a failure (corrupt chunk data)",
    -105: "no chunks were given",
    -106: "the chunks do not share the same total size and block size",
    -107: "a chunk is corrupt or truncated (its header, a block length or its "
    "raw tail lies outside the chunk)",
    -108: "a chunk offset or size lies outside the buffer it refers to",
    -110: "the CSC matrix arrays are inconsistent (sizes/itemsize)",
    -111: "an unknown stage id or unknown option bit was requested",
    -112: "a known implementation is unavailable on this build/CPU",
    -113: "the pixel dtype is out of range or unsupported",
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
        note_chunk(chunk, i, pointers, lengths)
    return pointers, lengths


def harvest_chunk_offsets(ds):
    chunks = tuple(ds.chunks) if ds.chunks is not None else None
    if chunks is None or len(chunks) != 3 or chunks[0] != 1:
        raise ValueError(
            "harvest_chunk_offsets needs a (1, ni, nj)-chunked dataset, got chunks=%r "
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


def pack_offsets_lengths(frame_offsets, frames):
    n = len(frames)
    offsets = np.empty(n, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int32)
    for i, frame in enumerate(frames):
        offsets[i], lengths[i] = frame_offsets[frame]
    return offsets, lengths


def _workspace_bytes(cmp):
    return 3 * _blocksize_bytes(cmp)


def _workspace_bytes_csc(cmp, itemsize):
    blocksize = _blocksize_bytes(cmp)
    block_elems = blocksize // itemsize
    return 3 * blocksize + block_elems * (4 + itemsize)


# ---- Phase 3: matrix normaliser and selection hook ----
class _NormalMatrix(object):
    """A csc-like object with normalised f32 data and u32 indices/indptr."""

    def __init__(self, data, indices, indptr, nbins, npix):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = (int(nbins), int(npix))
        self.nbins = int(nbins)
        self.npix = int(npix)


def normalise_matrix(csc, npix=None):
    """
    Accept a pyFAI CSC engine, a scipy.sparse.csc_matrix, or any duck-typed
    object with .indptr/.indices/.data plus .shape or .bins; other scipy
    formats (whatever offers .tocsc()) are converted.  Converts to float32 /
    uint32 once and validates the contents:

      len(indptr) == npix + 1 (when npix is given), indptr[0] == 0,
      indptr nondecreasing, indptr[-1] == len(indices), every indices[k] < nbins.

    Returns a _NormalMatrix; raises ValueError on bad contents.
    """
    if not hasattr(csc, "indptr") and hasattr(csc, "tocsc"):
        csc = csc.tocsc()
    indptr = np.asarray(csc.indptr)
    indices = np.asarray(csc.indices)
    data = np.asarray(csc.data)
    if hasattr(csc, "shape"):
        nbins = int(np.prod(csc.shape[0]))
    elif hasattr(csc, "bins"):
        nbins = int(np.prod(csc.bins))
    else:
        raise ValueError("csc matrix has no shape or bins attribute")
    if npix is not None and len(indptr) != npix + 1:
        raise ValueError("csc indptr has %d entries, expected %d (npix + 1)"
                         % (len(indptr), npix + 1))
    if indptr.size and indptr[0] != 0:
        raise ValueError("csc indptr[0] must be 0")
    if np.any(np.diff(indptr.astype(np.int64)) < 0):
        raise ValueError("csc indptr is not nondecreasing")
    if indptr[-1] != len(indices):
        raise ValueError("csc indptr[-1] (%d) != len(indices) (%d)"
                         % (indptr[-1], len(indices)))
    if indices.size and np.any(indices.astype(np.int64) >= nbins):
        raise ValueError("a csc index is >= nbins (%d)" % nbins)
    return _NormalMatrix(
        data.astype(np.float32),
        indices.astype(np.uint32),
        indptr.astype(np.uint32),
        nbins,
        len(indptr) - 1,
    )


# Selection hook: given the matrix and a block size (bytes), decide which
# dot implementation to use.  Only csc exists today, so it always returns 0
# (plain csc); the csc-fused variant (id 1) is selected explicitly via the
# `dot=` constructor argument.
def select_dot(matrix, blocksize=0):
    """Select the dot implementation id for a matrix/block-size combo.
    Only the csc layout exists, so this returns 0; a specific variant can
    be chosen with the `dot=` argument to chunk2sparseCSC/multi."""
    return 0



class chunk2sparseMulti:
    """
    Batched plain sparse decode: decodes a series of frames from the same
    dataset in one call. Workspace is 3*blocksize, independent of frame count.
    """

    def __init__(self, mask, dtype=np.uint16, codec=CODEC_LZ4, pipeline=None):
        self.nfast = mask.shape[1]
        self.mask = mask.ravel()
        self.npix = mask.size
        self.dtype = dtype
        self.codec = codec
        self.pipeline = pipeline
        self._fn = _make_sparsify(_suffix_for_dtype(dtype), pipeline=pipeline)

        self._nframes = 0
        self._output = None
        self._output_adr = None
        self._npx_out = None
        self._cursors = None
        self._workspace = None

    def _ensure_capacity(self, nframes, cmp):
        if self._nframes != nframes:
            self._output = np.empty((nframes, self.npix), self.dtype)
            self._output_adr = np.empty((nframes, self.npix), np.uint32)
            self._npx_out = np.empty(nframes, np.int32)
            self._cursors = np.empty(nframes, np.int64)
            self._nframes = nframes
        need = _workspace_bytes(cmp)
        if self._workspace is None or self._workspace.size < need:
            self._workspace = np.empty(need, np.uint8)

    def __call__(self, buffers, cut):
        nframes = len(buffers)
        self._ensure_capacity(nframes, buffers[0])

        pointers, lengths = _gather_chunks(buffers)

        ret = self._fn(
            pointers,
            lengths,
            self.mask,
            self._output.ravel(),
            self._output_adr.ravel(),
            self._npx_out,
            cut,
            self._workspace,
            self._cursors,
            self.codec,
        )
        if ret < 0:
            self._npx_out[:] = 0
            raise _decode_error(ret)
        return self._npx_out, (self._output, self._output_adr)


class chunk2sparse:
    """Single-frame plain sparse decode: chunk2sparseMulti with nframes fixed at 1."""

    def __init__(self, mask, dtype=np.uint16, codec=CODEC_LZ4, pipeline=None):
        self._multi = chunk2sparseMulti(mask, dtype=dtype, codec=codec, pipeline=pipeline)
        self.nfast = self._multi.nfast

    def __call__(self, buffer, cut):
        npx_out, (output, output_adr) = self._multi([buffer], cut)
        return int(npx_out[0]), (output[0], output_adr[0])

    def coo(self, buffer, cut):
        npixels, (values, indices) = self.__call__(buffer, cut)
        row = np.empty(npixels, np.uint16)
        col = np.empty(npixels, np.uint16)
        np.divmod(indices[:npixels], self.nfast, out=(row, col))
        return npixels, row, col, values[:npixels].copy()


def bslz4_to_sparse(ds, num, cut, mask=None, pixelbuffer=None, workspace=None, codec=None):
    """
    Reads a bitshuffle compressed hdf5 dataset and converts this directly
    into a sparse format (indices, values) when decoding the data.

    ds = hdf5 dataset containing [nframes, ni, nj] pixels
    num = frame number to read
    cut = threshold, pixels below this value are ignored
    mask = detector mask. Active pixels > 0.
    pixelbuffer = None or (values, indices) storage space
    workspace = None or a uint8 array to use as decode scratch space
    codec = None to auto-detect lz4 vs zstd, or an explicit CODEC_LZ4 / CODEC_ZSTD

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
    filtinfo, buffer = ds.id.read_direct_chunk((num, 0, 0))
    if workspace is None:
        workspace = np.empty(_workspace_bytes(buffer), np.uint8)
    fn = _BSLZ4_MULTI[_suffix_for_dtype(values.dtype)]
    pointers, lengths = _gather_chunks([buffer])
    npx_out = np.empty(1, np.int32)
    cursors = np.empty(1, np.int64)
    ret = fn(pointers, lengths, mask, values, indices, npx_out, cut, workspace, cursors, codec)
    if ret < 0:
        raise _decode_error(ret, batch=False)
    npixels = int(npx_out[0])
    return npixels, (values, indices)


class chunk2sparseCSCmulti:
    """
    Batched CSC decode: decodes a series of frames from the same dataset in
    one call, per (frame, block) routing between a dense and a sparse CSC
    strategy based on that block's compression ratio.
    """

    def __init__(self, mask, csc, dtype=np.uint16, codec=CODEC_LZ4, pipeline=None, dot=None):
        self.nfast = mask.shape[1]
        self.mask = mask.ravel()
        nm = normalise_matrix(csc, npix=len(self.mask))
        self.cscdata = nm.data
        self.cscindices = nm.indices
        self.cscindptr = nm.indptr
        self.nbins = nm.nbins
        # dot implementation id: 0 = csc (dot then threshold), 1 = csc-fused
        # (the dense >cut sparsify interleaved into the CSC loop).
        self.dot_id = select_dot(nm) if dot is None else int(dot)

        self.npix = mask.size
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        self.codec = codec
        self.pipeline = pipeline
        self._fn = _make_csc(_suffix_for_dtype(dtype), pipeline=pipeline, dot=self.dot_id)

        self._nframes = 0
        self._outpx = None
        self._output_adr = None
        self._npx_out = None
        self._powder = None
        self._cursors = None
        self._workspace = None

    def _ensure_capacity(self, nframes, cmp):
        if self._nframes != nframes:
            self._outpx = np.empty((nframes, self.npix), self.dtype)
            self._output_adr = np.empty((nframes, self.npix), np.uint32)
            self._npx_out = np.empty(nframes, np.int32)
            self._powder = np.empty((nframes, self.nbins), np.float64)
            self._cursors = np.empty(nframes, np.int64)
            self._nframes = nframes
        need = _workspace_bytes_csc(cmp, self.itemsize)
        if self._workspace is None or self._workspace.size < need:
            self._workspace = np.empty(need, np.uint8)

    def __call__(self, buffers, cut):
        nframes = len(buffers)
        self._ensure_capacity(nframes, buffers[0])

        pointers, lengths = _gather_chunks(buffers)

        ret = self._fn(
            pointers,
            lengths,
            self.mask,
            self._outpx.ravel(),
            self._output_adr.ravel(),
            self._npx_out,
            cut,
            self._powder.ravel(),
            self.cscdata,
            self.cscindices,
            self.cscindptr,
            self._workspace,
            self._cursors,
            self.nbins,
            self.codec,
        )
        if ret < 0:
            self._npx_out[:] = 0
            raise _decode_error(ret)
        return self._npx_out, (self._outpx, self._output_adr), self._powder


class chunk2sparseCSC:
    """Single-frame CSC decode: chunk2sparseCSCmulti with nframes fixed at 1."""

    def __init__(self, mask, csc, dtype=np.uint16, codec=CODEC_LZ4, pipeline=None, dot=None):
        self._multi = chunk2sparseCSCmulti(mask, csc, dtype=dtype, codec=codec,
                                           pipeline=pipeline, dot=dot)
        self.nfast = self._multi.nfast

    def __call__(self, buffer, cut):
        npx_out, (outpx, output_adr), powder = self._multi([buffer], cut)
        return int(npx_out[0]), (outpx[0], output_adr[0]), powder[0]

    def coo(self, buffer, cut):
        npixels, (values, indices), powder = self.__call__(buffer, cut)
        row = np.empty(npixels, np.uint16)
        col = np.empty(npixels, np.uint16)
        np.divmod(indices[:npixels], self.nfast, out=(row, col))
        return npixels, row, col, values[:npixels].copy(), powder.copy()
