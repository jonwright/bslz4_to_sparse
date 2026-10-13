"""The chunk index: where each frame's compressed chunk lies in an HDF5 file, kept beside it.

HDF5 stores this index as B-tree nodes spread through the file, so finding it takes dozens of small
scattered reads and the HDF5 library (and h5py's global lock).  The sidecar keeps it flat, once per
file: reading it needs neither.  It holds the HDF5 chunk offsets only, 12 bytes per frame (12 kB for
a 1000-frame file, 0.005 % of it), never anything that would need the data to be read to build.

    <file>.bslz4idx, or <index_dir>/<path with / replaced by __>.bslz4idx

Layout (little endian): the 8 bytes b"BSLZ4IDX", a uint32 length H, H bytes of UTF-8 JSON (dataset,
shape, dtype, chunks, codec, file_size, file_mtime_ns, nframes, declared_nframes, version), zero
padding to a multiple of 8, then nframes int64 byte offsets and nframes int32 byte
lengths.  A sidecar is stale, and refused, when the data file's size or modification time no
longer match.

Only harvest() touches HDF5 (through h5py, once per file, offline); everything else is numpy."""

import json
import os
import struct

import numpy as np

MAGIC = b"BSLZ4IDX"
VERSION = 1
SUFFIX = ".bslz4idx"


class StaleIndex(ValueError):
    """The sidecar does not describe the data file as it is now."""


class Index(object):
    """shape (nframes, ni, nj) of the frames stored (meta["declared_nframes"] is what HDF5 says;
    fewer when the acquisition stopped early), dtype, codec (CODEC_LZ4 / CODEC_ZSTD), offsets
    int64[nframes], lengths int32[nframes] (bytes of each frame's chunk in the file), dataset, meta
    (the header)."""

    def __init__(self, meta, offsets, lengths):
        self.meta = meta
        self.dataset = meta["dataset"]
        self.shape = tuple(meta["shape"])
        self.dtype = np.dtype(meta["dtype"])
        self.codec = meta["codec"]
        self.offsets, self.lengths = offsets, lengths


def sidecar_path(path, index_dir=None):
    """Where the chunk index of the data file `path` is kept: beside it, or under `index_dir`."""
    path = os.path.abspath(path)
    if index_dir is None:
        return path + SUFFIX
    return os.path.join(index_dir, path.strip("/").replace("/", "__") + SUFFIX)


def harvest(path, dset):
    """Read the chunk index of the HDF5 file path (h5py, once, serially).  Returns an Index: where each
    frame's compressed chunk lies in the file (.offsets, bytes from the start; .lengths) and what
    the frames are (.shape, .dtype, .codec).  Nothing is written: see write_index."""
    import h5py

    import bslz4_to_sparse as b

    st = os.stat(path)
    with h5py.File(path, "r") as h:
        ds = h[dset]
        if ds.chunks is None or len(ds.chunks) != 3 or ds.chunks[0] != 1:
            raise ValueError("%s: needs one frame per chunk, got chunks=%r" % (path, ds.chunks))
        n = ds.shape[0]
        off = np.zeros(n, np.int64)
        ln = np.zeros(n, np.int32)
        seen = np.zeros(n, bool)

        def cb(info):
            if info.filter_mask & 1:
                raise ValueError("%s: a chunk without the bitshuffle filter" % path)
            i = info.chunk_offset[0]
            off[i], ln[i], seen[i] = info.byte_offset, info.size, True

        ds.id.chunk_iter(cb)
        stored = int(seen.sum())
        if (
            stored < n
        ):  # an acquisition stopped early: HDF5 declares n frames, only a prefix was written
            if seen[:stored].all():
                off, ln = off[:stored], ln[:stored]
            else:
                raise ValueError(
                    "%s: %d of %d frames have no stored chunk and they are not all at the end"
                    % (path, n - stored, n)
                )
        meta = dict(
            version=VERSION,
            dataset=dset,
            shape=[stored] + list(ds.shape[1:]),
            dtype=ds.dtype.str,
            chunks=list(ds.chunks),
            codec=int(b.detect_codec(ds)),
            file_size=st.st_size,
            file_mtime_ns=st.st_mtime_ns,
            nframes=stored,
            declared_nframes=n,
        )
    return Index(meta, off, ln)


def write_index(index, path, index_dir=None):
    """Write an Index to its small sidecar file beside the data file `path` (12 bytes per frame), so
    that every later run finds the chunks without HDF5; returns the sidecar's path.  It is written
    under a temporary name and renamed, so a reader never sees a half-written index."""
    head = json.dumps(index.meta, sort_keys=True).encode("utf-8")
    pad = -(len(MAGIC) + 4 + len(head)) % 8
    out = sidecar_path(path, index_dir)
    if index_dir is not None:
        os.makedirs(index_dir, exist_ok=True)
    tmp = "%s.tmp%d" % (out, os.getpid())
    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(head)) + head + b"\0" * pad)
        f.write(np.ascontiguousarray(index.offsets, "<i8").tobytes())
        f.write(np.ascontiguousarray(index.lengths, "<i4").tobytes())
    os.replace(tmp, out)
    return out


def read_index(path, index_dir=None, check=True):
    """Load the sidecar of path (an Index).  check=True refuses one whose recorded file size or
    modification time differ from the data file's.  FileNotFoundError when there is none."""
    sc = sidecar_path(path, index_dir)
    with open(sc, "rb") as f:
        raw = f.read()
    if raw[:8] != MAGIC:
        raise ValueError("%s is not a bslz4 chunk index" % sc)
    (hlen,) = struct.unpack("<I", raw[8:12])
    meta = json.loads(raw[12 : 12 + hlen].decode("utf-8"))
    if meta.get("version") != VERSION:
        raise ValueError("%s: index version %r, this reads %d" % (sc, meta.get("version"), VERSION))
    n = meta["nframes"]
    start = (12 + hlen + 7) // 8 * 8
    if len(raw) != start + 12 * n:
        raise ValueError("%s: truncated or corrupt (%d bytes for %d frames)" % (sc, len(raw), n))
    off = np.frombuffer(raw, "<i8", n, start).astype(np.int64)
    ln = np.frombuffer(raw, "<i4", n, start + 8 * n).astype(np.int32)
    if check:
        st = os.stat(path)
        if st.st_size != meta["file_size"] or st.st_mtime_ns != meta["file_mtime_ns"]:
            raise StaleIndex(
                "%s was written for a file of %d bytes modified at %d; %s is %d bytes, "
                "modified at %d"
                % (sc, meta["file_size"], meta["file_mtime_ns"], path, st.st_size, st.st_mtime_ns)
            )
    return Index(meta, off, ln)
