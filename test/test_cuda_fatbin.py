"""The shipped fatbin (cuda/bslz4_kernels.fatbin) was built from the CUDA source beside it.  CI has
no CUDA toolkit: after changing src/cuda/*.cu* rebuild with tools/build_cuda_fatbin.py (see its
docstring for the toolkits) and commit the fatbin and json.  No GPU needed."""

import hashlib
import json
import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import pytest

import bslz4_to_sparse  # noqa: E402

CUDA = os.path.join(os.path.dirname(os.path.abspath(bslz4_to_sparse.__file__)), "cuda")


def test_fatbin_matches_the_source():
    if not os.path.exists(os.path.join(CUDA, "bslz4_split.cu")):
        pytest.skip("no CUDA source in this install")
    with open(os.path.join(CUDA, "bslz4_kernels.json")) as fh:
        build = json.load(fh)["_build"]
    assert os.path.getsize(os.path.join(CUDA, "bslz4_kernels.fatbin")) > 0
    h = hashlib.sha256()
    for f in build["files"]:
        with open(os.path.join(CUDA, f), "rb") as fh:
            h.update(fh.read())
    assert h.hexdigest() == build["source_sha256"], (
        "src/cuda changed since the fatbin was built: run tools/build_cuda_fatbin.py"
    )
    assert "20" in build["nvrtc"] and "35" in build["nvrtc"]  # Fermi and Kepler cubins kept
