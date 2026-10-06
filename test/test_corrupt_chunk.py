"""
Corrupt / truncated compressed chunks must raise, never crash, and a decode
must never read outside the chunk it was given (no cross talk into the
neighbouring chunk of a shared buffer).

Reduced from two field files (scan1107 chunk 5172, scan2169 chunk 2669: each a
single chunk in 7200 whose bitshuffle-LZ4 stream loses framing part-way, so a
per-block length word holds garbage).  No h5py needed: a tiny valid chunk is
built by hand (literal-only LZ4 blocks of zeros), then damaged.

Each case runs in a subprocess (a segfault must not take pytest down), with
the chunk copied to the very end of a mapped page followed by a PROT_NONE
page, so ANY read past the end of the buffer is a deterministic SIGSEGV.
Linux only (mprotect).
"""
from __future__ import print_function

import mmap
import os
import struct
import subprocess
import sys

import pytest

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

BS = 8192          # default bitshuffle block, bytes
NBLK = 3
NROW, NCOL = 96, 128   # 12288 uint16 pixels == NBLK blocks of BS bytes


def _lz4_zeros(n):
    """LZ4 block holding n zero bytes as one literal-only sequence."""
    ext = n - 15
    return bytes(bytearray([0xF0])) + b"\xff" * (ext // 255) + bytes(bytearray([ext % 255])) + b"\0" * n


def _good_chunk():
    blk = _lz4_zeros(BS)
    out = struct.pack(">QI", BS * NBLK, 0)
    for _ in range(NBLK):
        out += struct.pack(">I", len(blk)) + blk
    return bytearray(out)


def _variants():
    g = _good_chunk()
    blk = len(_lz4_zeros(BS))
    w1 = 12 + 4 + blk                      # block 1's length word
    w2 = 12 + 2 * (4 + blk)                # block 2's length word

    def setw(v, at):
        c = bytearray(g)
        c[at:at + 4] = struct.pack(">I", v)
        return c

    return {
        "good": g,
        "len_0xFFFFFFFF": setw(0xFFFFFFFF, w1),
        "len_0x7FFFFFFF": setw(0x7FFFFFFF, w1),
        "len_32MB": setw(0x02000000, w1),
        "len_overruns_next_word": setw(blk + 10, w1),
        "len_small": setw(10, w1),
        "len_zero": setw(0, w1),
        "last_block_len_huge": setw(0x10000000, w2),
        "truncated_mid_block": g[:w1 + 100],
        "truncated_after_header": g[:12],
        "truncated_in_header": g[:6],
    }


def _guarded(data):
    """data at the very end of a mapped page; the next page is PROT_NONE."""
    import ctypes
    import numpy as np
    pg = mmap.PAGESIZE
    n = (len(data) + pg - 1) // pg + 1
    m = mmap.mmap(-1, n * pg)
    libc = ctypes.CDLL(None, use_errno=True)
    base = ctypes.addressof(ctypes.c_char.from_buffer(m))
    assert libc.mprotect(ctypes.c_void_p(base + (n - 1) * pg), pg, 0) == 0
    off = (n - 1) * pg - len(data)
    m[off:off + len(data)] = bytes(data)
    return m, np.frombuffer(m, np.uint8, len(data), off)


def _child(name, kind):
    import numpy as np
    import bslz4_to_sparse as b
    mask = np.ones((NROW, NCOL), np.uint8)
    good = _variants()["good"]
    if kind == "batch":       # good, damaged, good in ONE batched call
        held = [_guarded(good), _guarded(_variants()[name]), _guarded(good)]
        dec = b.chunk2sparse(mask)
        bufs = [a for _, a in held]
        run = dec.multi
    else:
        held = [_guarded(_variants()[name])]
        dec = b.chunk2sparse(mask)
        bufs = held[0][1]
        run = dec
    try:
        run(bufs, 0)
        print("returned")
    except Exception as e:
        # a failed batch must not leave plausible-looking partial counts
        npx = getattr(dec, "_npx", None)
        zeroed = npx is None or not np.any(npx)
        print("raised", e, "" if zeroed else "| npx_out NOT zeroed")


def _run(name, kind):
    p = subprocess.run([sys.executable, __file__, name, kind],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return p.returncode, p.stdout.strip()


_DAMAGED = [k for k in _variants() if k != "good"]

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="guard page uses mprotect (Linux)")


@pytest.mark.parametrize("kind", ["single", "batch"])
def test_good_chunk_decodes(kind):
    rc, out = _run("good", kind)
    assert rc == 0 and out == "returned"


@pytest.mark.parametrize("kind", ["single", "batch"])
@pytest.mark.parametrize("name", _DAMAGED)
def test_damaged_chunk_raises_not_crashes(name, kind):
    rc, out = _run(name, kind)
    assert rc == 0, "crashed with signal %d" % -rc
    assert out.startswith("raised"), out
    assert "chunk" in out, out      # the message must say a chunk was bad
    assert "NOT zeroed" not in out, out
    if kind == "batch":
        assert "batch" in out and "one at a time" in out, out


# --- base-buffer variant: offsets/sizes come from HDF5 metadata -------------

def _base_call(base, offsets, lengths):
    """Run chunk2sparseCSC.decode_offsets over `base`; returns (ret,
    offsets_after) with ret 0 on success, else the error code from the
    raised message ("Error decoding ...: <code>: ...")."""
    import re
    import numpy as np
    import bslz4_to_sparse as b
    npix = NROW * NCOL
    nbins = 4
    offsets = np.array(offsets, np.int64)
    lengths = np.array(lengths, np.int32)
    indptr = np.arange(npix + 1, dtype=np.uint32)
    indices = (np.arange(npix) % nbins).astype(np.uint32)
    data = np.ones(npix, np.float32)
    csc = type("_CSC", (), {"indptr": indptr, "indices": indices, "data": data,
                            "shape": (nbins, npix)})()
    dec = b.chunk2sparseCSC(np.ones((NROW, NCOL), np.uint8), csc, dtype=np.uint16,
                            pipeline={"dot": "csc"})
    try:
        dec.decode_offsets(base, offsets, lengths, 0)
        ret = 0
    except Exception as e:
        m = re.match(r"Error decoding \w+: (-?\d+):", str(e))
        assert m, str(e)
        ret = int(m.group(1))
    return ret, offsets


def test_base_good():
    g = bytes(_good_chunk())
    ret, _ = _base_call(g, [0], [len(g)])
    assert ret == 0


@pytest.mark.parametrize("offsets,lengths", [
    ([-1], [100]),                       # negative offset
    ([0], [-5]),                         # negative size
    ([10**9], [100]),                    # offset beyond the buffer
    ([2**62], [100]),                    # absurd offset
    ([0], [10**9]),                      # size beyond the buffer
    ([100], [-1 + 2**31 - 1]),           # offset ok, offset+size beyond
])
def test_base_bounds_rejected(offsets, lengths):
    g = bytes(_good_chunk())
    ret, after = _base_call(g, offsets, lengths)
    assert ret == -108
    assert list(after) == list(offsets), "offsets must be left unmutated on rejection"


def test_base_one_bad_of_two_rejects_before_any_decode():
    g = bytes(_good_chunk())
    ret, after = _base_call(g + g, [0, len(g)], [len(g), len(g) + 1])   # 2nd overruns by 1
    assert ret == -108
    assert list(after) == [0, len(g)]


def test_short_declared_length_does_not_read_neighbour():
    """Two good chunks back to back in one buffer; the first is DECLARED shorter
    than it is. Its block words then point past the declared end, into the
    neighbour's bytes. The decode must fail, not quietly consume them."""
    g = bytes(_good_chunk())
    for cut in (1, 4, 50, 5000):
        ret, _ = _base_call(g + g, [0, len(g)], [len(g) - cut, len(g)])
        assert ret == -107, (cut, ret)


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2])
