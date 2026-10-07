"""bslz4_to_sparse: decode bitshuffle-lz4/zstd detector frames straight
into sparse pixel lists, optionally with a sparse matrix product (pyFAI
integration) in the same pass.

    chunk2sparse(mask, dtype, codec, pipeline)        sparsify
    chunk2sparseCSC(mask, csc, dtype, codec, pipeline) sparsify + matrix dot

Each object has a `pipeline` (src/_pipeline.py): one value per processing
step, resolved when the object is made (0 / None: the best guess for this
CPU, data and matrix) and kept in `.pipeline`; describe() names the values.
Nothing is global, so objects with different pipelines can run in
different threads.

The GIL is released while decoding. Use one object per thread and do not
share them: there are no locks.
"""
import json
import os
import struct

import numpy as np

from .c2py_loader import load_native
from . import _pipeline
from . import _matrix
from ._pipeline import describe, CODEC_LZ4, CODEC_ZSTD

_ext = load_native(os.path.dirname(os.path.abspath(__file__)), "_bslz4_to_sparse")
_pipeline._ext = _ext

version = "0.0.21a1"

__all__ = [
    "version",
    "bslz4_to_sparse",
    "chunk2sparse",
    "chunk2sparseCSC",
    "describe",
    "detect_codec",
    "harvest_chunk_offsets",
    "pack_offsets_lengths",
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
    -109: "the dot's matrix layout does not match the decode entry point, or "
          "the decoded block size differs from the layout's",
    -110: "a chunk's decompressed size is smaller than the mask (frame and mask "
          "shapes differ)",
    -111: "a pipeline step value is unknown, auto (0: resolve it first) or not valid here",
    -112: "a pipeline step value needs an instruction set this CPU/build lacks",
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
    return pointers, lengths, bytes(memoryview(base)[o:o + 12])


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


class chunk2sparse:
    """
    Decode bitshuffle-lz4/zstd chunks of (ni, nj) frames into the pixels
    above a cut (and not masked).

    mask = detector mask, (ni, nj) uint8, active pixels > 0
    dtype = the pixel dtype of the dataset
    codec = CODEC_LZ4 or CODEC_ZSTD (detect_codec(dataset))
    pipeline = None (best guess), or step values: see describe()

    c(chunk, cut) -> npixels, (values, indices) for one frame (the arrays
    are full size: slice [:npixels]); c.multi(chunks, cut) -> the same per
    frame for a batch, npixels an array and the arrays (nframes, ni*nj).
    """

    def __init__(self, mask, dtype=np.uint16, codec=CODEC_LZ4, pipeline=None):
        self.nfast = mask.shape[1]
        self.mask = mask.ravel()
        self.npix = mask.size
        self.dtype = np.dtype(dtype)
        self.codec = codec
        self.pipeline = _pipeline.resolve(pipeline, self.dtype, codec, self.mask)
        self._di = _pipeline.DTYPE_INDEX[_pipeline.dtype_suffix(self.dtype)]
        self._nframes = 0
        self._workspace = None

    def _ensure_capacity(self, nframes, cmp):
        if self._nframes != nframes:
            self._values = np.empty((nframes, self.npix), self.dtype)
            self._indices = np.empty((nframes, self.npix), np.uint32)
            self._npx = np.empty(nframes, np.int32)
            self._cursors = np.empty(nframes, np.int64)
            self._nframes = nframes
        need = 3 * _blocksize_bytes(cmp)
        if self._workspace is None or self._workspace.size < need:
            self._workspace = np.empty(need, np.uint8)

    def _run(self, pointers, lengths, cmp, cut):
        self._ensure_capacity(len(pointers), cmp)
        ret = _ext.sparsify(pointers, lengths, self.mask, self._values.ravel(),
                            self._indices.ravel(), self._npx, cut, self._workspace,
                            self._cursors, self._di, self.pipeline)
        if ret < 0:
            self._npx[:] = 0
            raise _decode_error(ret, batch=len(pointers) > 1)
        return self._npx, (self._values, self._indices)

    def multi(self, chunks, cut):
        """Decode a batch of chunks (one frame each) of this dataset:
        returns npixels[nframes], (values, indices) of shape (nframes, ni*nj)."""
        if len(chunks) == 0:
            raise _decode_error(-105)
        pointers, lengths = _gather_chunks(chunks)
        return self._run(pointers, lengths, chunks[0], cut)

    def decode_offsets(self, base, offsets, lengths, cut):
        """As multi, for the chunks at byte offsets[i] (lengths[i]) inside
        the one buffer base (an mmap of the HDF5 file, or the file read into
        memory): see harvest_chunk_offsets and pack_offsets_lengths."""
        pointers, lengths, cmp = _offset_pointers(base, offsets, lengths)
        return self._run(pointers, lengths, cmp, cut)

    def __call__(self, buffer, cut):
        npx, (values, indices) = self.multi([buffer], cut)
        return int(npx[0]), (values[0], indices[0])

    def coo(self, buffer, cut):
        """Computes i,j indices and MAKES COPIES"""
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
    cut = threshold, pixels below this value are ignored
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
    ret = _ext.sparsify(pointers, lengths, mask, values, indices, npx_out, cut, workspace,
                        cursors, _pipeline.DTYPE_INDEX[_pipeline.dtype_suffix(values.dtype)], p)
    if ret < 0:
        raise _decode_error(ret, batch=False)
    return int(npx_out[0]), (values, indices)


class chunk2sparseCSC:
    """
    As chunk2sparse, and every frame also goes through a sparse matrix
    product (a pyFAI integration): powder = csc @ frame.

    mask = detector mask, (ni, nj) uint8, active pixels > 0
    csc = scipy.sparse matrix (nout, ni*nj), or a pyFAI CSC integrator
          (data, indices, indptr, bins)
    dtype, codec, pipeline: as chunk2sparse; pipeline's dot step picks the
          matrix layout (auto: from what the matrix is)

    The mask is folded into the matrix once, here: masked pixels contribute
    nothing to the powder.  c(chunk, cut) -> npixels, (values, indices),
    powder; c.multi(chunks, cut) -> the same per frame for a batch.
    """

    def __init__(self, mask, csc, dtype=np.uint16, codec=CODEC_LZ4, pipeline=None):
        self.nfast = mask.shape[1]
        self.mask = mask.ravel()
        self.npix = mask.size
        self.dtype = np.dtype(dtype)
        self.codec = codec
        nm = _matrix._fold_mask(_matrix.normalise_matrix(csc, npix=self.npix), self.mask)
        self._nm = nm
        self.nbins = nm.nbins
        padded_avx2 = _pipeline.available("dot", _pipeline.NAMES["dot"].index("padded-avx2")) == 1
        self._dot_auto = _pipeline.parse(pipeline)[_pipeline.DOT] == 0
        self.pipeline = _pipeline.resolve(
            pipeline, self.dtype, codec, self.mask, matrix=True,
            dot_auto=lambda: _matrix.auto_dot(_matrix.analyse(nm), padded_avx2))
        self.dot = _pipeline.NAMES["dot"][self.pipeline[_pipeline.DOT]]
        self._di = _pipeline.DTYPE_INDEX[_pipeline.dtype_suffix(self.dtype)]
        self._layout = self._build(DEFAULT_BLOCK_BYTES // self.dtype.itemsize)
        self._nframes = 0
        self._workspace = None

    def _build(self, block_elems):
        try:
            return _matrix.build(self.dot, self._nm, block_elems, self.dtype)
        except ValueError as e:
            auto = _matrix.auto_dot(_matrix.analyse(self._nm), False)
            raise ValueError("pipeline['dot']=%r cannot represent this matrix: %s%s"
                             % (self.dot, e, "" if auto == self.dot else " (auto picks %r)" % auto))

    def _ensure_capacity(self, nframes, cmp):
        lay = self._layout
        if self._nframes != nframes:
            self._values = np.empty((nframes, self.npix), self.dtype)
            self._indices = np.empty((nframes, self.npix), np.uint32)
            self._npx = np.empty(nframes, np.int32)
            self._powder = np.empty((nframes, lay.nout), lay.powder_dtype)
            self._cursors = np.empty(nframes, np.int64)
            self._nframes = nframes
        blocksize = _blocksize_bytes(cmp)
        be = blocksize // self.dtype.itemsize
        if lay.block_elems is not None and be != lay.block_elems:
            # a block-dependent layout built at another block size: rebuild
            # (C also refuses a mismatch with -109)
            if self._dot_auto and self.dot in _matrix.BSB_CSR_DOTS and be > _matrix.BSB_CSR_MAX_BLOCK_ELEMS:
                # an auto dot never fails on the data: bsb-csr cannot take
                # blocks this large, use the csc form of the same matrix
                self.dot = _matrix.BSB_CSR_DOTS[self.dot]
                self.pipeline[_pipeline.DOT] = _pipeline.NAMES["dot"].index(self.dot)
            self._layout = self._build(be)
        need = 3 * blocksize + be * (4 + self.dtype.itemsize)
        if self._workspace is None or self._workspace.size < need:
            self._workspace = np.empty(need, np.uint8)

    def _run(self, pointers, lengths, cmp, cut):
        self._ensure_capacity(len(pointers), cmp)
        lay = self._layout
        a = lay.args()
        head = (pointers, lengths, self.mask, self._values.ravel(), self._indices.ravel(),
                self._npx, cut, self._powder.ravel())
        tail = (self._di, self.pipeline)
        if lay.entry == "csc":
            ret = _ext.sparsify_and_dot(*(head + a + (self._workspace, self._cursors, lay.nout) + tail))
        elif lay.entry == "padded":
            ret = _ext.sparsify_and_dot_padded(*(head + a[:5] + (self._workspace, self._cursors) +
                                                 a[5:] + (lay.nout,) + tail))
        else:
            ret = _ext.sparsify_and_dot_bsbcsr(*(head + a[:8] + (self._workspace, self._cursors) +
                                                 a[8:] + (lay.nout,) + tail))
        if ret < 0:
            self._npx[:] = 0
            raise _decode_error(ret, batch=len(pointers) > 1)
        powder = self._powder if lay.scale is None else self._powder * lay.scale
        return self._npx, (self._values, self._indices), powder

    def multi(self, chunks, cut):
        """Decode a batch of chunks (one frame each) of this dataset:
        returns npixels[nframes], (values, indices) of shape (nframes,
        ni*nj), powder (nframes, nout)."""
        if len(chunks) == 0:
            raise _decode_error(-105)
        pointers, lengths = _gather_chunks(chunks)
        return self._run(pointers, lengths, chunks[0], cut)

    def decode_offsets(self, base, offsets, lengths, cut):
        """As multi, for the chunks at byte offsets[i] (lengths[i]) inside
        the one buffer base: see harvest_chunk_offsets and
        pack_offsets_lengths."""
        pointers, lengths, cmp = _offset_pointers(base, offsets, lengths)
        return self._run(pointers, lengths, cmp, cut)

    def __call__(self, buffer, cut):
        """
        Decompress buffer and place pixels above cut into (vals, indices)
        All pixels go into a powder integration (csc product)

        returns npixels, (values[fullsize], indices[fullsize]), powder_sum

        You will need to slice the result values[:npixels] yourself if you need that.
        """
        npx, (values, indices), powder = self.multi([buffer], cut)
        return int(npx[0]), (values[0], indices[0]), powder[0]

    def coo(self, buffer, cut):
        """Computes i,j indices and MAKES COPIES"""
        npixels, (values, indices), powder = self.__call__(buffer, cut)
        row = np.empty(npixels, np.uint16)
        col = np.empty(npixels, np.uint16)
        _unravel(indices[:npixels], self.nfast, row, col)
        return npixels, row, col, values[:npixels].copy(), powder.copy()
