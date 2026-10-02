"""
The build description embedded in the native extension (bslz4_to_sparse.build_info()).
"""
from __future__ import print_function

import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import bslz4_to_sparse as b

_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")


def _so_path():
    d = os.path.dirname(b.__file__)
    return [os.path.join(d, f) for f in os.listdir(d)
            if f.startswith("_bslz4_to_sparse.") and f.endswith((".so", ".pyd"))][0]


def test_build_info_fields():
    info = b.build_info()
    assert set(info) == {"version", "git", "modified", "src_sha256", "compiler",
                         "platform", "built_utc"}
    assert info["version"] == b.version
    assert len(info["src_sha256"]) == 64
    assert isinstance(info["modified"], list)


def test_build_info_readable_without_loading():
    # setup.py decides whether to rebuild from the digest in the file itself
    sys.path.insert(0, _TOOLS)
    try:
        from build_extension import embedded_digest
    finally:
        sys.path.pop(0)
    assert embedded_digest(_so_path()) == b.build_info()["src_sha256"]
