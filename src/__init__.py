import numpy as np
import ctypes
from . import bslz4_to_sparse as _ext

version = "0.1.0"

# c2py23's per-call timing instrumentation is compiled OUT entirely
# ("timing": False in the .c2py spec, src/bslz4_to_sparse.cpp) -- it adds
# per-call overhead (a branch plus, when enabled, two clock reads and a
# counter update) to every single decode call regardless of whether
# set_timing()/perf counters are actually in use, since the instrumented
# code path has to exist in the compiled wrapper either way. Use an
# external profiler (perf, py-spy, timeit around batches) instead.

# We cast away the 'read-only' nature of python bytes.
# Not needed for the latest numpy.
if hasattr( ctypes.pythonapi, "PyMemoryView_FromMemory"):
    buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
    buffer_from_memory.restype = ctypes.py_object
    buffer_from_memory.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
else:
    def buffer_from_memory(buf, l, f):
        if type(buf) == str and buf[:6] == 'array(':
            # python 2.7 and a rather sad feature in h5py.
            #   probably we should not support this.
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


# Pixel type suffixes c2py23's "expand" template mechanism generates one
# Python function per (see the C2PY_BEGIN block in src/bslz4_to_sparse.cpp).
# There are only two function families now, both multi-frame:
# bslz4_multi_<suffix> (plain sparse) and bslz4_csc_multi_<suffix> (CSC).
# There is no single-frame family at all -- a single frame is just
# nframes==1, both at the C++ template level (bslz4_core.hpp) and here:
# chunk2sparse/chunk2sparseCSC are thin wrappers around
# chunk2sparseMulti/chunk2sparseCSCmulti with a 1-frame batch, not a
# second, separately-maintained implementation.
_TYPE_SUFFIXES = ("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64")

# (numpy dtype.kind, itemsize) -> suffix. kind+itemsize (not a numpy/struct
# format character) is what's portable across platforms: e.g. np.int64's
# buffer format character is 'l' on Linux (64-bit long) but 'q' on Windows
# (64-bit long long, since Windows' long is 32-bit) -- kind='i',itemsize=8
# is the same on both.
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


# Per-suffix function lookups, built once from whatever c2py23's "expand"
# generated (see bslz4_to_sparse.cpp) rather than importing 20 names by
# hand. Each dict is {suffix: callable_or_rebind_fn}.
_BSLZ4_MULTI = {s: getattr(_ext, "bslz4_multi_%s" % s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI = {s: getattr(_ext, "bslz4_csc_multi_%s" % s) for s in _TYPE_SUFFIXES}
# bslz4_csc_multi_base_<suffix>: like _BSLZ4_CSC_MULTI, but for chunks
# that all share one base buffer (e.g. an mmap'ed HDF5 file) addressed by
# byte offset rather than one buffer object per frame -- see
# harvest_chunk_offsets()/pack_offsets_lengths() below. Not currently
# used by any class here (nothing in this module has opened/mmap'ed a
# whole file yet); exposed as a building block for callers who have.
_BSLZ4_CSC_MULTI_BASE = {s: getattr(_ext, "bslz4_csc_multi_base_%s" % s) for s in _TYPE_SUFFIXES}
_REBIND = {
    s: (
        getattr(_ext, "_rebind_bslz4_multi_%s" % s),
        getattr(_ext, "_rebind_bslz4_csc_multi_%s" % s),
        getattr(_ext, "_rebind_bslz4_csc_multi_base_%s" % s),
    )
    for s in _TYPE_SUFFIXES
}

# The only place a compressed chunk's address is extracted -- via
# c2py23's own buffer acquisition on note_chunk's "chunk" parameter, not
# any Python-side ctypes/numpy trick. See _gather_chunks().
note_chunk = _ext.note_chunk

get_dense_sparse_threshold = _ext.get_dense_sparse_threshold
set_dense_sparse_threshold = _ext.set_dense_sparse_threshold

# Optional SIMD mask+threshold collect kernel tiers (u16/u32 only, see
# bslz4_collect_simd.hpp): avx512, avx2, sse2, vsx, tried in that
# priority order when more than one is enabled (in practice at most one
# of {avx512,avx2,sse2} vs vsx can even be compiled into a given build,
# so the order across that x86/POWER boundary is moot). All off by
# default and never auto-enabled by CPU capability alone -- AVX-512 in
# particular can throttle clocks on some chips enough to net-lose for
# this workload, so each is a benchmarkable opt-in, the same shape as
# set_backend() for the untranspose step. sse2 is the x86-64 ABI
# baseline (always available on x86-64, no capability gap, no known
# throttling risk) -- still off by default for consistency, not because
# it's expected to be a bad idea; measure it too before relying on the
# default meaning anything about whether it's worth turning on. vsx
# (POWER8+) is real-hardware-measured (not just simulated/assumed) at
# 3.75-8.2x on sparse data, see set_vsx_collect().
avx512_collect_available = _ext.avx512_collect_available
get_avx512_collect = _ext.get_avx512_collect
avx2_collect_available = _ext.avx2_collect_available
get_avx2_collect = _ext.get_avx2_collect
sse2_collect_available = _ext.sse2_collect_available
get_sse2_collect = _ext.get_sse2_collect
vsx_collect_available = _ext.vsx_collect_available
get_vsx_collect = _ext.get_vsx_collect


def set_avx512_collect(enabled):
    """
    Enable/disable the AVX-512 collect kernel for bslz4_multi_u16/u32 and
    bslz4_csc_multi_u16/u32's sparse-route compaction. Applies immediately
    to every call already made with those functions (it's a runtime
    switch, not a per-call or per-dtype choice) -- measure on your own
    machine (A/B against it left off) before turning it on; it is not
    always a win, see the module docstring in bslz4_collect_simd.hpp.
    Tried before avx2/sse2 when more than one tier is enabled.

    Raises RuntimeError if enabled=True is requested on a CPU without the
    required AVX-512F+BW+VL feature bits (see avx512_collect_available()).
    """
    if _ext.set_avx512_collect(1 if enabled else 0) < 0:
        raise RuntimeError(
            "AVX-512 collect kernel requested but this CPU lacks the required "
            "AVX512F+AVX512BW+AVX512VL feature bits (avx512_collect_available() "
            "is False)"
        )


def set_avx2_collect(enabled):
    """
    Enable/disable the AVX2 collect kernel for bslz4_multi_u16/u32 and
    bslz4_csc_multi_u16/u32's sparse-route compaction -- see
    set_avx512_collect(), same shape. Ignored when AVX-512 collect is
    also enabled (avx512 is tried first). AVX2 carries much less
    frequency-throttling risk than AVX-512 on most chips, but measure
    before relying on that rather than assuming it.

    Raises RuntimeError if enabled=True is requested on a CPU without AVX2
    (see avx2_collect_available()).
    """
    if _ext.set_avx2_collect(1 if enabled else 0) < 0:
        raise RuntimeError(
            "AVX2 collect kernel requested but this CPU lacks AVX2 "
            "(avx2_collect_available() is False)"
        )


def set_sse2_collect(enabled):
    """
    Enable/disable the SSE2 collect kernel for bslz4_multi_u16/u32 and
    bslz4_csc_multi_u16/u32's sparse-route compaction -- see
    set_avx512_collect(), same shape. Ignored when AVX-512 or AVX2
    collect is also enabled (tried last). Unlike the other two tiers,
    SSE2 is the x86-64 ABI baseline: always available, no capability
    gap, no known throttling risk -- still off by default for
    consistency with set_backend()'s measure-first philosophy, not
    because it's expected to be a bad idea.

    Raises RuntimeError if enabled=True is requested on a non-x86-64
    build, or one whose compiler lacks GCC/Clang-style target attributes
    (see sse2_collect_available()).
    """
    if _ext.set_sse2_collect(1 if enabled else 0) < 0:
        raise RuntimeError(
            "SSE2 collect kernel requested but this build doesn't support it "
            "(sse2_collect_available() is False)"
        )


def set_vsx_collect(enabled):
    """
    Enable/disable the POWER VSX collect kernel for bslz4_multi_u16/u32
    and bslz4_csc_multi_u16/u32's sparse-route compaction -- see
    set_avx512_collect(), same shape. Real-hardware-measured on a POWER9
    box (see bslz4_collect_simd.hpp and tools/bslz4_power9_collect_probe.c
    for the full story): 3.75-8.2x faster than scalar on sparse data via
    a vec_any_gt fast-skip gate. Still off by default for consistency
    with set_backend()'s measure-first philosophy.

    Raises RuntimeError if enabled=True is requested on a non-POWER
    build, or POWER hardware/build without VSX (see
    vsx_collect_available()).
    """
    if _ext.set_vsx_collect(1 if enabled else 0) < 0:
        raise RuntimeError(
            "VSX collect kernel requested but this build/CPU doesn't support it "
            "(vsx_collect_available() is False)"
        )


DEFAULT_BLOCK_BYTES = 8192

# bitshuffle-hdf5 filter id and its cd_values[4] block-codec numbering
# (bitshuffle/src/bshuf_h5filter.h: BSHUF_H5_COMPRESS_LZ4/_ZSTD).
BSHUF_H5FILTER = 32008
CODEC_LZ4 = 2
CODEC_ZSTD = 3


def detect_codec(ds):
    """
    Detect which block codec (CODEC_LZ4 or CODEC_ZSTD) a dataset's
    bitshuffle filter was configured with, from its HDF5 filter pipeline
    (cd_values[4]). Falls back to CODEC_LZ4 if it can't be determined
    (e.g. no bitshuffle filter present).
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
    """
    The bitshuffle block size (bytes) encoded in this compressed chunk's
    stream header (bytes 8:12, big endian; 0 means the 8192 byte default).

    Works on any buffer-protocol object whose integer indexing returns
    ints -- bytes/bytearray/memoryview/numpy uint8 array, all fine in
    Python 3 as-is, no conversion needed first.
    """
    if len(cmp) < 12:
        return DEFAULT_BLOCK_BYTES
    blocksize = (int(cmp[8]) << 24) | (int(cmp[9]) << 16) | (int(cmp[10]) << 8) | int(cmp[11])
    return blocksize if blocksize else DEFAULT_BLOCK_BYTES


def _gather_chunks(chunks):
    """
    Build the (pointers, lengths) pair bslz4_multi_*/bslz4_csc_multi_*
    expect from a plain Python sequence of independent chunk buffers
    (bytes, bytearray, memoryview, mmap slices, ... any buffer-protocol
    object -- source type doesn't matter and isn't inspected).

    No ctypes: note_chunk (a genuine c2py23-wrapped function, so it's
    c2py23's own buffer acquisition doing the address extraction, not a
    Python-side reimplementation of it via e.g. .ctypes.data) writes each
    chunk's address+length into pointers/lengths directly. pointers/
    lengths are still plain int64/int32 numpy arrays, not bytearrays --
    bslz4_multi_*/bslz4_csc_multi_* infer nframes from
    compressed_ptrs.itemsize (8 for int64), which a bytearray (itemsize
    1) can't satisfy; numpy is already a hard dependency of this package
    elsewhere (masks, outputs), so this isn't adding one. The `chunks`
    sequence itself is consumed entirely here in Python -- c2py23 never
    sees more than one buffer object per call.
    """
    n = len(chunks)
    pointers = np.empty(n, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int32)
    for i, chunk in enumerate(chunks):
        note_chunk(chunk, i, pointers, lengths)
    return pointers, lengths


def harvest_chunk_offsets(ds):
    """
    Harvest {frame_index: (byte_offset, byte_size)} for every stored
    chunk of a (1, ni, nj)-chunked bitshuffle dataset, via h5py's
    chunk_iter (h5py >=3.8: one native B-tree traversal) or, if that's
    not available, a get_num_chunks()/get_chunk_info() loop.

    Raises ValueError if ds isn't chunked (1, ni, nj) -- 4-D (1,1,ni,nj)
    chunking is not supported yet -- or if any chunk has the bitshuffle
    filter (pipeline position 0) marked skipped in its filter_mask (that
    chunk's bytes would be raw pixel data, not a bitshuffle stream; in
    practice this doesn't happen for these datasets, so it's a hard
    error here rather than a case decode needs to handle).

    Pairs with pack_offsets_lengths() and bslz4_csc_multi_base_<suffix>
    (see _BSLZ4_CSC_MULTI_BASE) for callers who have the whole HDF5 file
    already open/mapped as one buffer and want to skip one
    read_direct_chunk() call per frame.
    """
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
            # must not return 0 (or any falsy-to-HDF5-iterator value that
            # h5py maps to H5_ITER_STOP) -- returning None continues.
        ds.id.chunk_iter(_cb)
    else:
        for i in range(ds.id.get_num_chunks()):
            info = ds.id.get_chunk_info(i)
            _store(info.chunk_offset[0], info.byte_offset, info.size, info.filter_mask)

    return offsets


def pack_offsets_lengths(frame_offsets, frames):
    """
    Build the (offsets, lengths) pair bslz4_csc_multi_base_<suffix>
    expects (int64/int32 numpy arrays, see _gather_chunks for why not
    bytearrays), from harvest_chunk_offsets()'s {frame_index:
    (byte_offset, byte_size)} result and the ordered list of frame
    indices to include. These are byte offsets, already plain Python
    ints from HDF5 metadata, so no buffer-address extraction
    (note_chunk) is needed here at all.
    """
    n = len(frames)
    offsets = np.empty(n, dtype=np.int64)
    lengths = np.empty(n, dtype=np.int32)
    for i, frame in enumerate(frames):
        offsets[i], lengths[i] = frame_offsets[frame]
    return offsets, lengths


def _workspace_bytes(cmp):
    """
    Bytes required for the plain sparse decode workspace (bslz4_multi_*,
    any number of frames including 1) for this compressed chunk: 3 times
    its block size (see _blocksize_bytes). Independent of frame count --
    bslz4_multi_* reuses one shared raw/scratch/block region per frame in
    turn, there is nothing to share across frames for plain sparse decode.
    """
    return 3 * _blocksize_bytes(cmp)


def _workspace_bytes_csc(cmp, itemsize):
    """
    Bytes required for the CSC decode workspace (bslz4_csc_multi_*, any
    number of frames including 1) for this compressed chunk: the same
    3*blocksize as plain decode, plus a compaction scratch pair sized to
    one block's worth of pixels (tidx: uint32 per pixel, tval: itemsize
    bytes per pixel) -- see bslz4_core.hpp's bslz4_csc_decode_multi.
    Independent of frame count: routing and compaction are both
    per-(frame, block) decisions with nothing to share across frames, so
    this is NOT (nframes+2)*blocksize like the superseded design
    (issue #12) was.
    """
    blocksize = _blocksize_bytes(cmp)
    block_elems = blocksize // itemsize
    return 3 * blocksize + block_elems * (4 + itemsize)


def available_backends():
    """
    Names of the untranspose backends compiled into this build (e.g.
    'kcb', 'sse', 'scal'). See set_backend(). Every dtype has the same
    set, so this just asks a representative one (u16).
    """
    names = [v.decode() for v in _ext._variants_bslz4_multi_u16()]
    return tuple(sorted(set(name.rsplit("_", 1)[-1] for name in names)))


def set_backend(name):
    """
    Select which untranspose (bit/byte de-shuffle) backend bslz4_to_sparse
    uses. 'kcb' (https://github.com/kalcutter/bitshuffle) is the default
    and does its own CPU dispatch; 'sse' and 'scal' are upstream
    bitshuffle's SSE2 and portable scalar reference kernels
    (https://github.com/kiyo-masui/bitshuffle). See available_backends()
    for the names compiled into this build.

    Applies to every pixel type and to all three multi-frame decode
    families (plain, CSC, and the base+offsets CSC variant -- there is no
    separate single-frame family to also update, see the module
    docstring), so the kernels can be measured against each other rather
    than one being silently fixed at build time. Pass None to restore
    auto-resolve.
    """
    for suffix, (rbm, rbcm, rbcmb) in _REBIND.items():
        rbm(None if name is None else "bslz4_multi_%s_%s" % (suffix, name))
        rbcm(None if name is None else "bslz4_csc_multi_%s_%s" % (suffix, name))
        rbcmb(None if name is None else "bslz4_csc_multi_base_%s_%s" % (suffix, name))


# Historical note (see bug_variant.md): a c2py23 code-generation bug used
# to make set_backend(None)'s auto-resolve land on the wrong variant
# regardless of "default": True in the .c2py spec. Fixed upstream
# (c2py23 branch "variant-fallback-marker", commit ed4e9f9) but not yet
# released to PyPI, which is what `pip install c2py23` still gives you --
# and our own .c2py spec is written so that fix isn't even load-bearing
# for us regardless (sse/scal are "default": False, so kcb is the sole
# unconditional default:true variant per group, unambiguous either way).
# Set explicitly anyway, to fix the *actually* released c2py23 and to
# make the intended default explicit rather than implicit.
set_backend("kcb")


class chunk2sparseMulti:
    """
    Batched plain sparse decode: decodes a series of frames from the same
    dataset in one call. Workspace is 3*blocksize, independent of frame
    count -- see bslz4_core.hpp's bslz4_decode_multi.

    All frames must come from the same dataset (same detector shape/
    dtype/block size).
    """

    def __init__(self, mask, dtype=np.uint16, codec=CODEC_LZ4):
        self.nfast = mask.shape[1]
        self.mask = mask.ravel()
        self.npix = mask.size
        self.dtype = dtype
        self.codec = codec
        self._fn = _BSLZ4_MULTI[_suffix_for_dtype(dtype)]

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
        """
        buffers = a sequence of N raw compressed chunks (one per frame,
                  e.g. from repeated ds.id.read_direct_chunk() calls),
                  all from the same dataset.
        cut = threshold, pixels below this value are ignored

        returns (npx_out, (output, output_adr)):
          npx_out     -- int32 array, length N, pixel count per frame
          output      -- (N, npix) array, output[f, :npx_out[f]] valid
          output_adr  -- (N, npix) uint32 array, same slicing

        The returned arrays are owned by this object and reused (and
        possibly reallocated) on the next call.
        """
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
            raise Exception("Error decoding batch: %d" % (ret))
        return self._npx_out, (self._output, self._output_adr)


class chunk2sparse:
    """
    Single-frame plain sparse decode: chunk2sparseMulti with nframes
    fixed at 1 -- composition, not a second implementation. See
    bslz4_core.hpp's module docstring for why there is no separate
    C-level single-frame entry point either.
    """

    def __init__(self, mask, dtype=np.uint16, codec=CODEC_LZ4):
        self._multi = chunk2sparseMulti(mask, dtype=dtype, codec=codec)
        self.nfast = self._multi.nfast

    def __call__(self, buffer, cut):
        npx_out, (output, output_adr) = self._multi([buffer], cut)
        return int(npx_out[0]), (output[0], output_adr[0])

    def coo(self, buffer, cut):
        """Computes i,j indices and MAKES COPIES"""
        npixels, (values, indices) = self.__call__(buffer, cut)
        row = np.empty(npixels, np.uint16)
        col = np.empty(npixels, np.uint16)
        np.divmod(indices[:npixels], self.nfast, out=(row, col))
        return npixels, row, col, values[:npixels].copy()


def bslz4_to_sparse(ds, num, cut, mask=None, pixelbuffer=None, workspace=None, codec=None):
    """
    Reads a bitshuffle compressed hdf5 dataset and converts this
    directly into a sparse format (indices, values) when decoding
    the data.

    ds = hdf5 dataset containing [nframes, ni, nj] pixels
    num = frame number to read
    cut = threshold, pixels below this value are ignored
    mask = detector mask. Active pixels > 0.
    pixelbuffer = None or (values, indices) storage space
    workspace = None or a uint8 array to use as decode scratch space
                (see _workspace_bytes / available_backends)
    codec = None to auto-detect lz4 vs zstd from ds's filter pipeline
            (see detect_codec), or an explicit CODEC_LZ4 / CODEC_ZSTD

    returns (number_of_pixels, (values, indices))

    Unlike chunk2sparse, this calls bslz4_multi_<suffix> directly (with a
    1-element frame batch built inline) rather than through
    chunk2sparseMulti, so that caller-supplied pixelbuffer/workspace
    arrays are used as-is instead of being replaced by ones this function
    owns -- callers reusing the same buffers across many frames (see
    test/bench1.py) rely on that.
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
    # todo : h5py malloc free version coming? see https://github.com/h5py/h5py/pull/2232
    filtinfo, buffer = ds.id.read_direct_chunk((num, 0, 0))
    # note_chunk's C parameter is read-only (const char *), so a plain
    # h5py-returned bytes object goes straight through _gather_chunks --
    # unlike numpy's .ctypes.data, no writability workaround is needed.
    if workspace is None:
        workspace = np.empty(_workspace_bytes(buffer), np.uint8)
    fn = _BSLZ4_MULTI[_suffix_for_dtype(values.dtype)]
    pointers, lengths = _gather_chunks([buffer])
    npx_out = np.empty(1, np.int32)
    cursors = np.empty(1, np.int64)
    ret = fn(pointers, lengths, mask, values, indices, npx_out, cut, workspace, cursors, codec)
    if ret < 0:
        raise Exception("Error decoding: %d" % (ret))
    npixels = int(npx_out[0])
    return npixels, (values, indices)


class chunk2sparseCSCmulti:
    """
    Batched CSC decode (issue #12, redesigned): decodes a series of
    frames from the same dataset in one call. Per (frame, block), routes
    between a dense and a sparse CSC strategy based on that block's own
    compression ratio (see bslz4_core.hpp's bslz4_csc_decode_multi and
    set_dense_sparse_threshold()) -- there is no cross-frame sharing of
    the CSC indptr/indices/data lookup (an earlier design tried that and
    measured it as a net loss against real CSC data).

    All frames must come from the same dataset (same detector shape/
    dtype/block size).
    """

    def __init__(self, mask, csc, dtype=np.uint16, codec=CODEC_LZ4):
        """
        mask = detector mask
        csc = Either scipy.sparse.csc_matrix (data, indices, indptr, shape)
              Or  pyFAI CSCIntegrator object (data, indices, indptr, bins)
        dtype = the dtype for the pixels in the dataset
        codec = CODEC_LZ4 (default) or CODEC_ZSTD, matching the dataset's
                bitshuffle filter (see detect_codec)
        """
        self.nfast = mask.shape[1]
        self.mask = mask.ravel()
        self.cscdata = csc.data
        self.cscindices = csc.indices
        self.cscindptr = csc.indptr
        assert len(csc.indptr) == len(self.mask) + 1, "csc shape must match mask"
        if hasattr(csc, "shape"):
            self.nbins = csc.shape[0]
        elif hasattr(csc, "bins"):
            self.nbins = csc.bins
        else:
            raise Exception("csc argument has no shape or bins attribute")

        self.npix = mask.size
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        self.codec = codec
        self._fn = _BSLZ4_CSC_MULTI[_suffix_for_dtype(dtype)]

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
        # Independent of nframes -- see _workspace_bytes_csc.
        need = _workspace_bytes_csc(cmp, self.itemsize)
        if self._workspace is None or self._workspace.size < need:
            self._workspace = np.empty(need, np.uint8)

    def __call__(self, buffers, cut):
        """
        buffers = a sequence of N raw compressed chunks (one per frame,
                  e.g. from repeated ds.id.read_direct_chunk() calls),
                  all from the same dataset.
        cut = threshold, pixels below this value are ignored

        returns (npx_out, (outpx, output_adr), powder):
          npx_out      -- int32 array, length N, pixel count per frame
          outpx         -- (N, npix) array, outpx[f, :npx_out[f]] valid
          output_adr    -- (N, npix) uint32 array, same slicing
          powder        -- (N, nbins) float64 array, one full CSC
                            integration per frame

        The returned arrays are owned by this object and reused (and
        possibly reallocated) on the next call.
        """
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
            raise Exception("Error decoding batch: %d" % (ret))
        return self._npx_out, (self._outpx, self._output_adr), self._powder


class chunk2sparseCSC:
    """
    Single-frame CSC decode: chunk2sparseCSCmulti with nframes fixed at
    1 -- composition, not a second implementation.
    """

    def __init__(self, mask, csc, dtype=np.uint16, codec=CODEC_LZ4):
        """
        mask = detector mask
        csc = Either scipy.sparse.csc_matrix (data, indices, indptr, shape)
              Or  pyFAI CSCIntegrator object (data, indices, indptr, bins)
        dtype = the dtype for the pixels in the dataset
        codec = CODEC_LZ4 (default) or CODEC_ZSTD, matching the dataset's
                bitshuffle filter (see detect_codec)
        """
        self._multi = chunk2sparseCSCmulti(mask, csc, dtype=dtype, codec=codec)
        self.nfast = self._multi.nfast

    def __call__(self, buffer, cut):
        """
        Decompress buffer and place pixels above cut into (vals, indices)
        All pixels go into a powder integration (csc product)

        returns npixels, (values[fullsize], indices[fullsize]), powder_sum

        You will need to slice the result values[:npixels] yourself if you need that.
        """
        npx_out, (outpx, output_adr), powder = self._multi([buffer], cut)
        return int(npx_out[0]), (outpx[0], output_adr[0]), powder[0]

    def coo(self, buffer, cut):
        """Computes i,j indices and MAKES COPIES"""
        npixels, (values, indices), powder = self.__call__(buffer, cut)
        row = np.empty(npixels, np.uint16)
        col = np.empty(npixels, np.uint16)
        np.divmod(indices[:npixels], self.nfast, out=(row, col))
        return npixels, row, col, values[:npixels].copy(), powder.copy()
