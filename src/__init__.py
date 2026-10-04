import json
import os
import struct
import numpy as np
from .c2py_loader import load_native
from . import _csc_variants

_ext = load_native(os.path.dirname(os.path.abspath(__file__)), "_bslz4_to_sparse")

version = "0.0.21a1"

# The general/documented public API.  Expert tuning knobs (available_backends,
# set_backend, set_dense_sparse_threshold, get_dense_sparse_threshold,
# set_byteskip, get_byteskip, set_plane_extract, get_plane_extract,
# set_lz4_zero, get_lz4_zero),
# test/introspection helpers and the internal decode helpers are all still
# importable directly, but are deliberately not part of `import *`.
__all__ = [
    "version",
    "bslz4_to_sparse",
    "chunk2sparse",
    "chunk2sparseCSC",
    "chunk2sparseMulti",
    "chunk2sparseCSCmulti",
    "pack_pipeline",
    "CODEC_LZ4",
    "CODEC_ZSTD",
]


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
_COLLECT_ID_TO_NAME = {0: "scalar", 1: "avx512", 2: "avx2", 3: "sse2", 4: "vsx", 5: "neon",
                       6: "avx512cs"}

# Pipeline option bits (bslz4_common.h).  NO_MASK: every pixel is valid (e.g.
# the data was zeroed at collection), so the kernels skip the mask entirely.
_OPT_NO_MASK = 1 << 1
# The untranspose byte skip (bslz4_common.h BSLZ4_OPT_BYTESKIP); on by
# default, set by set_byteskip().
_OPT_BYTESKIP = 1 << 2
# Plane extraction for the plain u8/u16/u32 sparsify (bslz4_common.h
# BSLZ4_OPT_PLANE_EXTRACT); on by default, set by set_plane_extract().
_OPT_PLANE_EXTRACT = 1 << 3
# The zero-aware lz4 block decoder (bslz4_common.h BSLZ4_OPT_LZ4_ZERO); on by
# default, set by set_lz4_zero().
_OPT_LZ4_ZERO = 1 << 4

# Dot implementations (mirrors bslz4_registry.c bslz4_dots[]): id -> (name,
# layout).  Layout is a string: "csc", "csc-run", "csc-nosplit", "padded" or
# "bsb-csr".  The three csc* layouts share the one CSC entry and differ only
# in what `indices` holds (see _run_starts / _nosplit_bins).  New schemes
# append a row here and a matching dot id in the native registry.
_DOT_TABLE = {
    0: ("csc", "csc"),
    1: ("csc-fused", "csc"),
    2: ("padded", "padded"),
    3: ("padded-sse2", "padded"),
    4: ("padded-avx2", "padded"),
    5: ("padded-avx512", "padded"),
    6: ("bsb-csr", "bsb-csr"),
    7: ("csc-run", "csc-run"),
    8: ("csc-nosplit", "csc-nosplit"),
    9: ("bsb-csr-nosplit", "bsb-csr"),
    10: ("csc-nosplit-moment", "csc-nosplit-moment"),
    # Experimental CSC-entry dots: the arrays for each are made by
    # _csc_variants.BUILDERS[name] (layout "csc-x").
    11: ("csc-nosplit-dump", "csc-x"),
    12: ("csc-nosplit-moment-dump", "csc-x"),
    13: ("csc-run-u16", "csc-x"),
    14: ("csc-nosplit-u16", "csc-x"),
    15: ("csc-run-delta", "csc-x"),
    16: ("csc-nosplit-delta", "csc-x"),
    17: ("csc-nosplit-walk", "csc-x"),
    18: ("csc-tile", "csc-x"),
    19: ("csc-run-moment", "csc-x"),
    20: ("csc-int", "csc-x"),
    21: ("csc-run-int", "csc-x"),
    22: ("csc-run-int16", "csc-x"),
    23: ("csc-permute", "csc-x"),
    24: ("csc-permute-runs", "csc-x"),
}
_DOT_IDS = {name: i for i, (name, _l) in _DOT_TABLE.items()}


def _dot_supported_dtypes(id):
    # All 10 for the current float layouts (mirrors dot dtype mask 0x3FF).
    if id in _DOT_TABLE:
        return _TYPE_SUFFIXES
    return ()


def available_dots():
    """Names of the dot implementations usable on this machine (`dot=` accepts these)."""
    return tuple(_DOT_TABLE[i][0] for i in sorted(_DOT_TABLE)
                 if _ext.impl_available(_STAGE_DOT, i) == 1)


def dot_info(name_or_id):
    """Introspect a dot implementation by name or id.

    Returns a dict with: id, name, layout (a string), dtypes (tuple of dtype
    names it accepts) and available (1 usable here, 0 known-but-not, -1 unknown).
    """
    if isinstance(name_or_id, str):
        i = _DOT_IDS.get(name_or_id)
        if i is None:
            raise ValueError("unknown dot %r (available: %s)"
                             % (name_or_id, ", ".join(available_dots())))
    else:
        i = int(name_or_id)
        if i not in _DOT_TABLE:
            raise ValueError("unknown dot id %d" % i)
    name, layout = _DOT_TABLE[i]
    return {
        "id": i,
        "name": name,
        "layout": layout,
        "dtypes": tuple(_dot_supported_dtypes(i)),
        "available": _ext.impl_available(_STAGE_DOT, i),
    }


def _resolve_dot(dot):
    """Resolve a `dot=` argument (None / "auto" / a name or an int id) to a dot
    id, raising NotImplementedError for a name not available on this build/CPU.
    None and "auto" both select "csc" for now (decided; smarter auto rules come
    later from race data)."""
    if dot is None or dot == "auto":
        return 0
    if isinstance(dot, str):
        i = _DOT_IDS.get(dot)
        if i is None:
            raise NotImplementedError("unknown dot %r" % (dot,))
    else:
        i = int(dot)
    avail = _ext.impl_available(_STAGE_DOT, i)
    if avail < 0:
        raise NotImplementedError("unknown dot id %d" % i)
    if avail == 0:
        raise NotImplementedError(
            "dot implementation %r is not available in this build/CPU" % (dot,))
    return i

DEFAULT_BLOCK_BYTES = 8192
BSHUF_H5FILTER = 32008
CODEC_LZ4 = 2
CODEC_ZSTD = 3

# ---- Module-level defaults (the shims edit these; no C globals) ----
_BACKEND_NAMES = ("kcb", "sse", "neon", "scal")
_default_backend = "kcb"
_dense_sparse_threshold = 8.0
_byteskip = True
_plane_extract = True
_lz4_zero = True


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


def set_byteskip(enabled):
    """Enable (default) or disable the untranspose byte skip: when every value
    in a u16 decode block is < 256 (e.g. low counts, masked pixels zero), only
    the low 8 bit-planes are untransposed, straight into u16 (on CPUs with
    AVX-512 VBMI and GFNI; ignored elsewhere).  Identical results either way;
    5-12 % faster on medium/dense u16 frames."""
    global _byteskip
    _byteskip = bool(enabled)


def set_plane_extract(enabled):
    """Enable (default) or disable plane extraction for the plain sparsify of
    unsigned integer pixels: for a well compressed block with only a few
    non-zero pixels, the values are read straight from the bit-planes,
    skipping the untranspose and the scan (~25 % faster on very sparse u16
    frames).  Identical results either way."""
    global _plane_extract
    _plane_extract = bool(enabled)


def set_lz4_zero(enabled):
    """Enable (default) or disable the zero-aware lz4 block decoder for well
    compressed blocks (> 24x): zero runs and the block's zero tail are not
    written, and the plane extraction / u16 untranspose are told where the
    data ends.  Identical results either way; 40-60 % faster on very sparse
    u16 frames."""
    global _lz4_zero
    _lz4_zero = bool(enabled)


def get_lz4_zero():
    """Whether the zero-aware lz4 decoder is enabled (see set_lz4_zero)."""
    return _lz4_zero


def get_plane_extract():
    """Whether plane extraction is enabled (see set_plane_extract)."""
    return _plane_extract


def get_byteskip():
    """Whether the untranspose byte skip is enabled (see set_byteskip)."""
    return _byteskip


def get_dense_sparse_threshold():
    """Current dense/sparse routing threshold (see set_dense_sparse_threshold)."""
    return _dense_sparse_threshold


# ---- Default SIMD collect tier (no per-tier flags) ----
# The tier that actually runs is chosen by the collect id in the decode
# stages (see pack_pipeline) -- that is the one way to select it.  The
# default, when the caller does not pass a pipeline, is the highest-priority
# SIMD tier this build/CPU supports, else scalar.  There are no per-tier
# set_/get_ shims or C flags state anymore.

def _active_collect_tier():
    # avx512cs (compress-store: branch-free extraction) first: 23-57 % faster
    # than avx512 when many pixels are selected, the same when few are
    # (2026-10-03, EPYC 9454); then avx512, avx2, sse2, vsx, neon
    for mid in (6, 1, 2, 3, 4, 5):
        if _ext.impl_available(_STAGE_COLLECT, mid) == 1:
            return mid
    return 0


# ---- Stage-array ("pipeline") helpers ----
def pack_pipeline(decompress=None, untranspose=None, collect=None, dot=None, options=0):
    """
    Build the stages array (a uint16 ndarray of BSLZ4_STAGES_N entries --
    one per stage/option: decompress, untranspose, collect, dot, options)
    from stage ids, defaulting each stage from the module-level defaults
    (codec, untranspose backend, active collect tier, csc).  Unavailable
    selections raise NotImplementedError naming the stage and implementation.
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
    if options & ~((1 << 0) | _OPT_NO_MASK | _OPT_BYTESKIP | _OPT_PLANE_EXTRACT | _OPT_LZ4_ZERO):
        raise ValueError("unknown option bits: %r" % (options,))
    return _stages(decompress, untranspose, collect, dot, options)


def _default_codec():
    return CODEC_LZ4


def _stages(decompress, untranspose, collect, dot, options):
    # Mirrors bslz4_common.h BSLZ4_STAGES_N: a uint16 ndarray with one entry
    # per stage/option.  c2py23 accepts int/float/buffer scalar inputs, so the
    # options travel as a buffer rather than a single integer.
    return np.array([decompress, untranspose, collect, dot, options], dtype=np.uint16)


def _pipeline_for(codec, collect_tier, dot=0, options=0):
    """Stages array for the current module defaults, given an explicit codec,
    the collect tier for this dtype (0 for anything but u16/u32, which are
    the only types the SIMD collect tiers support) and the dot implementation
    id."""
    if _byteskip:
        options |= _OPT_BYTESKIP
    if _plane_extract:
        options |= _OPT_PLANE_EXTRACT
    if _lz4_zero:
        options |= _OPT_LZ4_ZERO
    return _stages(codec, _BACKEND_TO_ID[_default_backend], collect_tier, dot, options)


def _collect_tier_for_suffix(suffix):
    if suffix in ("u16", "u32"):
        return _active_collect_tier()
    return 0


def _suffix_itemsize(suffix):
    return int(suffix[1:]) // 8      # "u8"=1, "u16"=2, "f32"=4, "f64"=8, ...


def _check_outpx_itemsize(out, suffix):
    want = _suffix_itemsize(suffix)
    got = out.itemsize
    if got != want:
        raise ValueError(
            "outpx.itemsize %d does not match dtype %s itemsize %d" % (got, suffix, want))


def _make_sparsify(suffix, pipeline=None, options=0):
    di = _SUFFIX_TO_DTYPE[suffix]

    def fn(pointers, lengths, mask, output, output_adr, npx_out, threshold,
           workspace, cursors, codec):
        _check_outpx_itemsize(output, suffix)
        p = pipeline if pipeline is not None else _pipeline_for(codec, _collect_tier_for_suffix(suffix), options=options)
        return _ext.sparsify(pointers, lengths, mask, output, output_adr, npx_out,
                             threshold, workspace, cursors, di, p)
    return fn


def _make_csc(suffix, pipeline=None, dot=0, options=0):
    di = _SUFFIX_TO_DTYPE[suffix]

    def fn(pointers, lengths, mask, outpx, output_adr, npx_out, threshold, powder, data,
           indices, indptr, workspace, cursors, nout, codec):
        _check_outpx_itemsize(outpx, suffix)
        p = pipeline if pipeline is not None else _pipeline_for(codec, _collect_tier_for_suffix(suffix), dot, options)
        return _ext.sparsify_and_dot(pointers, lengths, mask, outpx, output_adr, npx_out,
                                     threshold, powder, data, indices, indptr,
                                     workspace, cursors, nout, _dense_sparse_threshold,
                                     di, p)
    return fn


def _make_csc_base(suffix, pipeline=None, dot=0, options=0):
    di = _SUFFIX_TO_DTYPE[suffix]

    def fn(base, offsets, lengths, mask, outpx, output_adr, npx_out, threshold, powder, data,
           indices, indptr, workspace, cursors, nout, codec):
        _check_outpx_itemsize(outpx, suffix)
        rc = _ext.offsets_to_pointers(base, offsets, lengths)
        if rc < 0:
            return rc
        p = pipeline if pipeline is not None else _pipeline_for(codec, _collect_tier_for_suffix(suffix), dot, options)
        return _ext.sparsify_and_dot(offsets, lengths, mask, outpx, output_adr, npx_out,
                                     threshold, powder, data, indices, indptr,
                                     workspace, cursors, nout, _dense_sparse_threshold,
                                     di, p)
    return fn


def _make_csc_padded(suffix, pipeline=None, dot=0, options=0):
    di = _SUFFIX_TO_DTYPE[suffix]

    def fn(pointers, lengths, mask, outpx, output_adr, npx_out, threshold, powder,
           base, weights, pixels, rowmap, row_ptr, width, listed, block_elems,
           workspace, cursors, nout, codec):
        _check_outpx_itemsize(outpx, suffix)
        p = pipeline if pipeline is not None else _pipeline_for(codec, _collect_tier_for_suffix(suffix), dot, options)
        return _ext.sparsify_and_dot_padded(pointers, lengths, mask, outpx, output_adr, npx_out,
                                            threshold, powder, base, weights, pixels, rowmap,
                                            row_ptr, workspace, cursors, width, listed, block_elems,
                                            nout, _dense_sparse_threshold, di, p)
    return fn


def _make_csc_bsbcsr(suffix, pipeline=None, dot=0, options=0):
    di = _SUFFIX_TO_DTYPE[suffix]

    def fn(pointers, lengths, mask, outpx, output_adr, npx_out, threshold, powder,
           blk_ptr, bins, bin_ptr, idx, data, csc_data, csc_indices, csc_indptr, block_elems,
           workspace, cursors, nout, codec):
        _check_outpx_itemsize(outpx, suffix)
        p = pipeline if pipeline is not None else _pipeline_for(codec, _collect_tier_for_suffix(suffix), dot, options)
        return _ext.sparsify_and_dot_bsbcsr(pointers, lengths, mask, outpx, output_adr, npx_out,
                                            threshold, powder, blk_ptr, bins, bin_ptr, idx, data,
                                            csc_data, csc_indices, csc_indptr, workspace, cursors,
                                            block_elems, nout, _dense_sparse_threshold, di, p)
    return fn


def _make_csc_padded_base(suffix, pipeline=None, dot=0, options=0):
    inner = _make_csc_padded(suffix, pipeline=pipeline, dot=dot, options=options)

    def fn(base, offsets, lengths, mask, outpx, output_adr, npx_out, threshold, powder,
           mbase, weights, pixels, rowmap, row_ptr, width, listed, block_elems,
           workspace, cursors, nout, codec):
        rc = _ext.offsets_to_pointers(base, offsets, lengths)
        if rc < 0:
            return rc
        return inner(offsets, lengths, mask, outpx, output_adr, npx_out, threshold, powder,
                     mbase, weights, pixels, rowmap, row_ptr, width, listed, block_elems,
                     workspace, cursors, nout, codec)
    return fn


def _make_csc_bsbcsr_base(suffix, pipeline=None, dot=0, options=0):
    inner = _make_csc_bsbcsr(suffix, pipeline=pipeline, dot=dot, options=options)

    def fn(base, offsets, lengths, mask, outpx, output_adr, npx_out, threshold, powder,
           blk_ptr, bins, bin_ptr, idx, data, csc_data, csc_indices, csc_indptr, block_elems,
           workspace, cursors, nout, codec):
        rc = _ext.offsets_to_pointers(base, offsets, lengths)
        if rc < 0:
            return rc
        return inner(offsets, lengths, mask, outpx, output_adr, npx_out, threshold, powder,
                     blk_ptr, bins, bin_ptr, idx, data, csc_data, csc_indices, csc_indptr,
                     block_elems, workspace, cursors, nout, codec)
    return fn


_BSLZ4_MULTI = {s: _make_sparsify(s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI = {s: _make_csc(s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI_BASE = {s: _make_csc_base(s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI_PADDED = {s: _make_csc_padded(s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI_PADDED_BASE = {s: _make_csc_padded_base(s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI_BSBCSR = {s: _make_csc_bsbcsr(s) for s in _TYPE_SUFFIXES}
_BSLZ4_CSC_MULTI_BSBCSR_BASE = {s: _make_csc_bsbcsr_base(s) for s in _TYPE_SUFFIXES}

note_chunk = _ext.note_chunk
# Availability and test counters.
impl_available = _ext.impl_available
reset_counters = _ext.reset_counters
read_counters = _ext.read_counters


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
    return json.loads(bytes(buf[:n]).decode("ascii"))


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
    -109: "the selected matrix layout does not match the decode entry point, or "
          "the decoded block size differs from the layout's",
    -110: "a chunk's decompressed size is smaller than the mask (frame and mask "
          "shapes differ)",
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
    # Duck-typed conversion to CSC.  A scipy CSR/BSR (or any sparse matrix with
    # a row-wise indptr) has an indptr but it is over rows, so is actually
    # converted; only a true CSC (or an already-CSC pyFAI engine, which has no
    # .format) is left alone.
    if hasattr(csc, "tocsc") and (
        not hasattr(csc, "indptr")
        or getattr(csc, "format", None) not in ("csc", None)
    ):
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


# ---- Phase 4: derived matrix layouts (padded CSC, blockwise CSR) ----

_PADDED_MAX_WIDTH = 64


def _fold_mask(nm, mask):
    """Apply the integrator's 0/1 mask to a _NormalMatrix by dropping the
    entries of masked pixels (mask == 0) so their columns become empty.  The
    matvec kernels then need no mask test (a masked pixel contributes nothing),
    and the `>cut` collect uses the raw mask separately.  Zero, never NaN."""
    m = np.asarray(mask).reshape(-1) > 0
    if m.size != nm.npix:
        raise ValueError("mask has %d pixels, the matrix %d" % (m.size, nm.npix))
    n = np.diff(nm.indptr.astype(np.int64))
    keep = np.repeat(m, n)
    data = nm.data[keep]
    indices = nm.indices[keep]
    new_n = np.where(m, n, 0)
    indptr = np.concatenate(([0], np.cumsum(new_n))).astype(np.uint32)
    return _NormalMatrix(data, indices, indptr, nm.nbins, nm.npix)


class _PaddedLayout(object):
    """A pixel -> bin matrix as one first bin plus a fixed number of weights
    per row (the padding the name refers to).  row == pixel when listed==0,
    else a row per pixel in `pixels`.  `row_ptr` splits the rows by decode
    block (nblocks+1), replacing the pre-refactor per-block cursor."""

    def __init__(self, base, weights, pixels, rowmap, row_ptr, width, listed,
                 block_elems, nbins, npix):
        self.base = base
        self.weights = weights
        self.pixels = pixels
        self.rowmap = rowmap
        self.row_ptr = row_ptr
        self.width = width
        self.listed = listed
        self.block_elems = block_elems
        self.nbins = nbins
        self.npix = npix
        self.nrows = base.shape[0]

    @property
    def _weights_flat(self):
        return self.weights.reshape(-1)

    @property
    def _pixels_arg(self):
        return self.pixels if self.listed else np.zeros(1, dtype=np.int32)

    @property
    def _rowmap_arg(self):
        return self.rowmap if self.listed else np.zeros(1, dtype=np.int32)


class _BsbCSR(object):
    """A bit-shuffle-block-sized CSR: per decode block the active (non-empty)
    bins, and per bin the (in-block pixel index, weight) entries."""

    def __init__(self, blk_ptr, bins, bin_ptr, idx, data, block_elems):
        self.blk_ptr = blk_ptr
        self.bins = bins
        self.bin_ptr = bin_ptr
        self.idx = idx
        self.data = data
        self.block_elems = block_elems


def _padded_from_csc(nm, block_elems):
    """Build a _PaddedLayout from a _NormalMatrix, exactly (no weight is
    changed, only re-laid-out).  The integrator passes the mask-folded matrix
    (_fold_mask), so a masked pixel gets an all-zero row (implicit layout) or
    no row (listed layout), and the kernels make no mask test.  Raises
    ValueError naming how many pixels and why if the matrix cannot be padded
    (a pixel reaches bins that are not one run of consecutive bins, or the
    width exceeds the limit)."""
    indptr = nm.indptr.astype(np.int64)
    indices = nm.indices.astype(np.int64)
    data = nm.data
    npix = nm.npix
    nbins = nm.nbins
    n = np.diff(indptr)
    used = int(n.max()) if npix else 0
    width = max(used, 1)
    if width > _PADDED_MAX_WIDTH:
        raise ValueError("a padded width of %d is not supported (maximum %d); use dot='csc'"
                         % (width, _PADDED_MAX_WIDTH))
    pix = np.repeat(np.arange(npix, dtype=np.int64), n)
    if pix.size > 1:
        broken = (pix[1:] == pix[:-1]) & (np.diff(indices) != 1)
        nb = np.unique(pix[1:][broken]).size
        if nb:
            raise ValueError("%d of %d pixels reach bins that are not one ascending run of "
                             "consecutive bins, so dot='padded' cannot represent this matrix; "
                             "use dot='csc'" % (nb, int((n > 1).sum())))
    has = n > 0
    first = indptr[:-1]
    base = np.zeros(npix, dtype=np.int64)
    if indices.size:
        base[has] = indices[np.minimum(first[has], indices.size - 1)]
    new_base = np.minimum(base, max(nbins - width, 0)).astype(np.int32)
    shift = base - new_base.astype(np.int64)
    w = np.zeros((npix, width), dtype=np.float32)
    for k in range(used):
        sel = n > k
        w[sel, shift[sel] + k] = data[first[sel] + k]
    listed = bool(has.mean() < 0.5) if npix else False
    nblocks = (npix + block_elems - 1) // block_elems
    cb = np.minimum(np.arange(nblocks + 1) * block_elems, npix).astype(np.int32)
    if listed:
        rows = np.flatnonzero(has).astype(np.int32)
        rowmap = np.full(npix, -1, dtype=np.int32)
        rowmap[rows] = np.arange(rows.shape[0], dtype=np.int32)
        row_ptr = np.searchsorted(rows, cb).astype(np.int32)
        return _PaddedLayout(new_base[rows], w[rows], rows, rowmap, row_ptr,
                             width, 1, block_elems, nbins, npix)
    return _PaddedLayout(new_base, w, None, None, cb, width, 0, block_elems, nbins, npix)


_NO_BIN = np.uint32(0xFFFFFFFF)


def _run_starts(nm):
    """indices for dot='csc-run' (start + length): the first bin of each
    pixel (npix entries; 0 for an empty column).  The pixel's entries stay
    data[indptr[p]:indptr[p+1]], for the consecutive bins start, start+1, ...
    Raises ValueError if some pixel's bins are not one ascending run."""
    indptr = nm.indptr.astype(np.int64)
    indices = nm.indices.astype(np.int64)
    n = np.diff(indptr)
    pix = np.repeat(np.arange(nm.npix, dtype=np.int64), n)
    if pix.size > 1:
        broken = (pix[1:] == pix[:-1]) & (np.diff(indices) != 1)
        nb = np.unique(pix[1:][broken]).size
        if nb:
            raise ValueError("%d of %d pixels reach bins that are not one ascending run of "
                             "consecutive bins, so dot='csc-run' cannot represent this matrix; "
                             "use dot='csc'" % (nb, int((n > 1).sum())))
    starts = np.zeros(nm.npix, dtype=np.uint32)
    has = n > 0
    starts[has] = indices[indptr[:-1][has]]
    return starts


def _nosplit_bins(nm):
    """indices for dot='csc-nosplit' (a histogram): the one bin of each pixel
    (npix entries, _NO_BIN for an empty column).  data and indptr are not
    read, so every weight must be exactly 1.  Raises ValueError otherwise."""
    n = np.diff(nm.indptr.astype(np.int64))
    if (n > 1).any():
        raise ValueError("%d pixels reach more than one bin, so dot='csc-nosplit' cannot "
                         "represent this matrix; use dot='csc-run' or 'csc'" % int((n > 1).sum()))
    if nm.data.size and not (nm.data == 1).all():
        raise ValueError("dot='csc-nosplit' needs every weight to be 1 (%d are not); "
                         "use dot='csc-run' or 'csc'" % int((nm.data != 1).sum()))
    bins = np.full(nm.npix, _NO_BIN, dtype=np.uint32)
    bins[n == 1] = nm.indices[nm.indptr[:-1][n == 1]]
    return bins


def _nosplit_moment(nm):
    """(data, indices) for dot='csc-nosplit-moment': a histogram with a first
    moment beside each output, i.e. every pixel reaches either no bin or the
    pair b, b+1 with weights exactly 1 (sum I) and q (sum qI) -- the
    interleaved [I, qI, I, qI, ...] order.  indices[p] = b (_NO_BIN for none),
    data[p] = q (npix entries each).  Raises ValueError otherwise."""
    indptr = nm.indptr.astype(np.int64)
    n = np.diff(indptr)
    if ((n != 0) & (n != 2)).any():
        raise ValueError("%d pixels do not reach exactly 0 or 2 bins, so "
                         "dot='csc-nosplit-moment' cannot represent this matrix; use "
                         "dot='csc-run' or 'csc'" % int(((n != 0) & (n != 2)).sum()))
    has = n == 2
    k0 = indptr[:-1][has]
    b0 = nm.indices[k0].astype(np.int64)
    b1 = nm.indices[k0 + 1].astype(np.int64)
    w0 = nm.data[k0]
    bad = int(((b1 != b0 + 1) | (w0 != 1)).sum())
    if bad:
        raise ValueError("%d pixels are not (bin b weight 1, bin b+1), so "
                         "dot='csc-nosplit-moment' cannot represent this matrix; use "
                         "dot='csc-run' or 'csc'" % bad)
    bins = np.full(nm.npix, _NO_BIN, dtype=np.uint32)
    bins[has] = b0
    q = np.zeros(nm.npix, dtype=np.float32)
    q[has] = nm.data[k0 + 1]
    return q, bins


def _bsb_csr_from_csc(nm, block_elems):
    """Build a _BsbCSR from a _NormalMatrix: CSC entries sorted by
    (pixel//block_elems, bin), grouped per (block, bin).  General (works for
    any matrix, 2D/FAZIT included), unlike padded."""
    indptr = nm.indptr.astype(np.int64)
    indices = nm.indices.astype(np.uint32)
    data = nm.data.astype(np.float32)
    npix = nm.npix
    nbins = nm.nbins
    n = np.diff(indptr)
    pix = np.repeat(np.arange(npix, dtype=np.int64), n)
    nz = pix.size
    nblocks = (npix + block_elems - 1) // block_elems
    if nz == 0:
        return _BsbCSR(np.zeros(nblocks + 1, np.uint32), np.zeros(0, np.uint32),
                       np.zeros(1, np.uint32), np.zeros(0, np.uint16),
                       np.zeros(0, np.float32), block_elems)
    block = pix // block_elems
    order = np.lexsort((pix, indices, block))
    block_s = block[order]
    bin_s = indices[order]
    pix_s = pix[order]
    data_s = data[order]
    key = block_s.astype(np.int64) * (nbins + 1) + bin_s.astype(np.int64)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
    block_of_group = block_s[starts]
    bins_g = bin_s[starts]
    bin_ptr = np.concatenate((starts, [nz])).astype(np.uint32)
    idx = (pix_s % block_elems).astype(np.uint16)
    blk_ptr = np.searchsorted(block_of_group, np.arange(nblocks + 1)).astype(np.uint32)
    return _BsbCSR(blk_ptr, bins_g, bin_ptr, idx, data_s.copy(), block_elems)


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
        no_mask = _OPT_NO_MASK if bool((self.mask == 1).all()) else 0
        self._fn = _make_sparsify(_suffix_for_dtype(dtype), pipeline=pipeline, options=no_mask)

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

    The mask is folded into the matrix once, here: the entries of masked
    pixels (mask == 0) are dropped, so they contribute nothing to the powder
    on any route or layout.  The raw mask still selects the pixels of the
    sparse (>cut) output.  Frames must have exactly mask.size pixels.

    `dot` picks the matrix-dot layout: "auto"/None (= "csc" for now),
    "csc", "csc-fused", "csc-run" (start + length, for pixels that reach
    consecutive bins), "csc-nosplit" (one bin per pixel, weight 1: a
    histogram), "padded" (and its SIMD tiers), "bsb-csr", or
    "bsb-csr-nosplit" (bsb-csr for a histogram), "csc-nosplit-moment"
    (a histogram with sum qI interleaved after each sum I), or one of the
    experimental dots of _csc_variants (index formats, integer weights); see
    available_dots().
    """

    def __init__(self, mask, csc, dtype=np.uint16, codec=CODEC_LZ4, pipeline=None, dot=None):
        self.nfast = mask.shape[1]
        self.mask = mask.ravel()
        self._nm = normalise_matrix(csc, npix=len(self.mask))
        # Fold the mask into the matrix (drop masked pixels' entries) once, at
        # build, so the matvec kernels need no per-pixel mask test.  The raw
        # mask is kept for the shared `>cut` collect (sparse output).
        self._nm = _fold_mask(self._nm, self.mask)
        self.cscdata = self._nm.data
        self.cscindices = self._nm.indices
        self.cscindptr = self._nm.indptr
        self.nbins = self._nm.nbins
        # dot implementation: select by name or id; None/"auto" -> csc (id 0).
        self.dot_id = _resolve_dot(dot)
        self.dot = _DOT_TABLE[self.dot_id][0]
        self.layout = _DOT_TABLE[self.dot_id][1]

        self.npix = mask.size
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        self.codec = codec
        self.pipeline = pipeline
        suffix = _suffix_for_dtype(dtype)
        be = DEFAULT_BLOCK_BYTES // self.itemsize
        self._block_elems = be
        self.padded = None
        self.bsb_csr = None
        if self.layout == "csc-run":
            self.cscindices = _run_starts(self._nm)
        elif self.layout == "csc-nosplit":
            self.cscindices = _nosplit_bins(self._nm)
        elif self.layout == "csc-nosplit-moment":
            self.cscdata, self.cscindices = _nosplit_moment(self._nm)
        # experimental CSC-entry dots: their own arrays, maybe hidden dump
        # bins after the real ones (nout = nbins + extra), maybe an int64
        # powder scaled back to float (scale)
        self.variant = None
        if self.layout == "csc-x":
            self.variant = _csc_variants.BUILDERS[self.dot](self._nm, be, dtype)
        self._nout = self.nbins + (self.variant.extra if self.variant else 0)
        self._powder_dtype = np.int64 if (self.variant and self.variant.scale) else np.float64
        if self.variant is not None and self.variant.packed:
            self._powder_dtype = np.dtype(dtype)
        self._nosplit_csc = None
        if self.dot == "bsb-csr-nosplit":
            # the sparse route walks the nested csc as csc-nosplit does: one
            # bin per pixel.  csc_data is never read; it is only there because
            # the bsb-csr entry checks csc_indices.n == csc_data.n.
            bins = _nosplit_bins(self._nm)
            self._nosplit_csc = (np.ones(bins.size, np.float32), bins, self.cscindptr)
        if self.layout == "padded":
            self.padded = _padded_from_csc(self._nm, be)
        elif self.layout == "bsb-csr":
            self.bsb_csr = _bsb_csr_from_csc(self._nm, be)
        if self.layout in ("csc", "csc-run", "csc-nosplit", "csc-nosplit-moment", "csc-x"):
            self._fn = _make_csc(suffix, pipeline=pipeline, dot=self.dot_id,
                                 options=_OPT_NO_MASK if bool((self.mask == 1).all()) else 0)
            self._fn_base = _make_csc_base(suffix, pipeline=pipeline, dot=self.dot_id,
                                           options=_OPT_NO_MASK if bool((self.mask == 1).all()) else 0)
        elif self.layout == "padded":
            self._fn = _make_csc_padded(suffix, pipeline=pipeline, dot=self.dot_id,
                                        options=_OPT_NO_MASK if bool((self.mask == 1).all()) else 0)
            self._fn_base = _make_csc_padded_base(suffix, pipeline=pipeline, dot=self.dot_id,
                                                  options=_OPT_NO_MASK if bool((self.mask == 1).all()) else 0)
        else:
            self._fn = _make_csc_bsbcsr(suffix, pipeline=pipeline, dot=self.dot_id,
                                        options=_OPT_NO_MASK if bool((self.mask == 1).all()) else 0)
            self._fn_base = _make_csc_bsbcsr_base(suffix, pipeline=pipeline, dot=self.dot_id,
                                                  options=_OPT_NO_MASK if bool((self.mask == 1).all()) else 0)

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
            self._powder = np.empty((nframes, self._nout), self._powder_dtype)
            self._cursors = np.empty(nframes, np.int64)
            self._nframes = nframes
        be = (_blocksize_bytes(cmp) // self.itemsize) or self._block_elems
        if be != self._block_elems:
            self._rebuild_layout(be)
        need = _workspace_bytes_csc(cmp, self.itemsize)
        if self._workspace is None or self._workspace.size < need:
            self._workspace = np.empty(need, np.uint8)

    def _rebuild_layout(self, block_elems):
        """The decoded block size differs from the one the layout was built at:
        rebuild the derived layout, never compute wrong results (C also rejects
        a mismatch with -109)."""
        self._block_elems = block_elems
        if self.layout == "padded":
            self.padded = _padded_from_csc(self._nm, block_elems)
        elif self.layout == "bsb-csr":
            self.bsb_csr = _bsb_csr_from_csc(self._nm, block_elems)
        elif self.variant is not None and self.variant.block_dependent:
            self.variant = _csc_variants.BUILDERS[self.dot](self._nm, block_elems, self.dtype)

    def _matrix_args(self):
        if self.variant is not None:
            v = self.variant
            return (v.data, v.indices, v.indptr)
        if self.layout in ("csc", "csc-run", "csc-nosplit", "csc-nosplit-moment"):
            return (self.cscdata, self.cscindices, self.cscindptr)
        if self.layout == "padded":
            p = self.padded
            return (p.base, p._weights_flat, p._pixels_arg, p._rowmap_arg, p.row_ptr,
                    p.width, int(p.listed), p.block_elems)
        q = self.bsb_csr
        csc = self._nosplit_csc or (self.cscdata, self.cscindices, self.cscindptr)
        return (q.blk_ptr, q.bins, q.bin_ptr, q.idx, q.data) + tuple(csc) + (q.block_elems,)

    def __call__(self, buffers, cut):
        nframes = len(buffers)
        self._ensure_capacity(nframes, buffers[0])

        pointers, lengths = _gather_chunks(buffers)
        return self._decode(self._fn, (pointers, lengths), cut)

    def decode_offsets(self, base, offsets, lengths, cut):
        """
        Decode the frames whose compressed chunks lie at offsets[i] (bytes)
        with lengths[i] inside the one buffer base: an mmap of the whole
        HDF5 file, or the file read into memory. offsets/lengths come from
        harvest_chunk_offsets() and pack_offsets_lengths(), so no HDF5 call
        and no per-chunk Python object is needed per batch. Every chunk is
        checked to lie inside base before any is decoded. Returns the same
        as __call__ on those chunks.
        """
        # a copy: the C side turns the offsets into pointers in place
        offsets = np.array(offsets, dtype=np.int64)
        lengths = np.ascontiguousarray(lengths, dtype=np.int32)
        if len(offsets) == 0:
            raise _decode_error(-105)
        o = int(offsets[0])
        self._ensure_capacity(len(offsets), bytes(memoryview(base)[o:o + 12]))
        return self._decode(self._fn_base, (base, offsets, lengths), cut)

    def _decode(self, fn, chunk_args, cut):
        ret = fn(*(chunk_args + (
            self.mask,
            self._outpx.ravel(),
            self._output_adr.ravel(),
            self._npx_out,
            cut,
            self._powder.ravel(),
        ) + self._matrix_args() + (
            self._workspace,
            self._cursors,
            self._nout,
            self.codec,
        )))
        if ret < 0:
            self._npx_out[:] = 0
            raise _decode_error(ret)
        return self._npx_out, (self._outpx, self._output_adr), self._powder_out()

    def _powder_out(self):
        """The (nframes, nbins) float powder: the work buffer itself for the
        float dots, without the hidden dump bins, or the int64 fixed-point
        sums scaled back to float."""
        v = self.variant
        if v is None:
            return self._powder
        p = self._powder[:, :self.nbins]
        if v.scale is not None:
            return p * v.scale
        return p


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
