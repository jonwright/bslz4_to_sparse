"""The helpers that read a few bytes of a chunk or of build_info give real bytes on every Python:
on Python 2.7 bytes(memoryview) and bytes(ndarray) are not the data (this file runs there too)."""

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np

import bslz4_to_sparse as b


def test_build_info_is_parsed():
    info = b.build_info()
    assert info["version"] == b.version and "src_sha256" in info


def test_the_codec_is_read_from_the_header_of_a_chunk_in_any_buffer():
    header = bytearray(20)
    header[16:20] = b"\x28\xb5\x2f\xfd"  # the zstd frame magic
    for chunk in (bytes(header), bytearray(header), np.frombuffer(bytes(header), np.uint8)):
        assert b._codec_of_chunk(chunk) == b.CODEC_ZSTD
    header[16:20] = b"\x00\x00\x00\x00"
    assert b._codec_of_chunk(bytes(header)) == b.CODEC_LZ4
