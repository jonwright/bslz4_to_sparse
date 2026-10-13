"""Timing the decode: what one core does per frame, and what it means for a network link.

tools/readbench.py (put the tools directory on PYTHONPATH) times "readers": functions that each do
the whole job for a batch of frames already in memory.  They run on one pinned core, round-robin so
that drift hits all of them alike, and the best of several repeats is kept.  Timings are appended to
a JSON-lines file with the machine and build, and report() prints them as ms per frame, then how many
cores keep up with a 10 or 25 GbE link when reads overlap with compute.  Here two readers do the same
job, the pixels above a cut: bslz4_to_sparse straight from the compressed chunks, and h5py to read
the frame with numpy to find the pixels.
"""

import os

import h5py
import hdf5plugin  # registers the bitshuffle filter, so that h5py can decompress too
import numpy as np
import readbench  # tools/readbench.py

import bslz4_to_sparse

h5filename = "eiger_0000.h5"  # your file: bitshuffle-compressed, one frame per chunk
h5dataset = "/entry_0000/ESRF-ID11/eiger/data"  # the (nframes, ni, nj) dataset inside it
timings = "timings.jsonl"  # where the timings are recorded
cut, nframes = 3, 6

with h5py.File(h5filename, "r") as h5:
    ds = h5[h5dataset]
    dtype, codec = ds.dtype, bslz4_to_sparse.detect_codec(ds)
    where = bslz4_to_sparse.harvest_chunk_offsets(ds)
    frame0 = ds[0]
mask = (frame0 != np.iinfo(dtype).max).astype(np.uint8)
file_bytes = np.fromfile(h5filename, dtype=np.uint8)
offsets, lengths = bslz4_to_sparse.pack_offsets_lengths(where, range(nframes))
c2s = bslz4_to_sparse.chunk2sparse(mask, dtype=dtype, codec=codec)


def from_chunks():
    """bslz4_to_sparse: decode the chunks of the batch and keep the pixels above the cut"""
    return c2s.decode_offsets(file_bytes, offsets, lengths, cut)[0].copy()


def from_h5py_and_numpy():
    """h5py decompresses each frame, numpy finds the pixels above the cut"""
    with h5py.File(h5filename, "r") as h5:
        frames = h5[h5dataset][:nframes]
    return np.array([np.count_nonzero((f > cut) & (mask > 0)) for f in frames])


with readbench.pinned():
    times, results = readbench.time_interleaved(
        {"bslz4_to_sparse": from_chunks, "h5py + numpy": from_h5py_and_numpy}, repeats=3
    )
assert (results["bslz4_to_sparse"] == results["h5py + numpy"]).all()  # the same answer
readbench.record(
    timings,
    {(reader, "sparse"): t for reader, t in times.items()},
    nframes=nframes,
    bytes_per_frame=lengths.mean(),
    cases={"sparse": "pixels above a cut of %d" % cut},
    data={"name": os.path.basename(h5filename)},
    pipeline_used=c2s.pipeline,
    batch=nframes,
    dtype=dtype,
)
readbench.report(timings)
