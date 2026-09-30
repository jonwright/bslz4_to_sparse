"""
Corrupt / truncated compressed chunks must raise, never crash.

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
        dec = b.chunk2sparseMulti(mask)
        bufs = [a for _, a in held]
    else:
        held = [_guarded(_variants()[name])]
        dec = b.chunk2sparse(mask)
        bufs = held[0][1]
    try:
        dec(bufs, 0)
        print("returned")
    except Exception as e:
        print("raised", e)


def _run(name, kind):
    p = subprocess.run([sys.executable, __file__, name, kind],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return p.returncode, p.stdout.strip()


_CRASHES = ("truncated_mid_block", "truncated_after_header", "truncated_in_header")
_DAMAGED = [k for k in _variants() if k != "good"]

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="guard page uses mprotect (Linux)")


def _mark(name):
    if name in _CRASHES:
        return pytest.param(name, marks=pytest.mark.xfail(
            reason="known: decoder trusts block length words / cursor and reads "
                   "past the end of a truncated chunk (SIGSEGV)", strict=False))
    return name


@pytest.mark.parametrize("kind", ["single", "batch"])
def test_good_chunk_decodes(kind):
    rc, out = _run("good", kind)
    assert rc == 0 and out == "returned"


@pytest.mark.parametrize("kind", ["single", "batch"])
@pytest.mark.parametrize("name", [_mark(n) for n in _DAMAGED])
def test_damaged_chunk_raises_not_crashes(name, kind):
    rc, out = _run(name, kind)
    assert rc == 0, "crashed with signal %d" % -rc
    assert out.startswith("raised"), out


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2])
