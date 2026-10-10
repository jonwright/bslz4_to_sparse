"""The hash tools/bench_docs.py records beside the C sources' src_sha256.

_pipeline.py chooses the kernels and _matrix.py builds the matrix layouts:
both change the speed without changing the compiled sources, so the
performance page compares this hash too (tools/generate_docs.py).
"""
import hashlib
import os

PY_TUNING = ("_pipeline.py", "_matrix.py")


def py_sha256(pkg_dir):
    """sha256 over PY_TUNING, read from the package directory pkg_dir."""
    h = hashlib.sha256()
    for name in PY_TUNING:
        with open(os.path.join(pkg_dir, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()
