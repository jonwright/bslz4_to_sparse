"""The hash tools/bench_docs.py records beside the C sources' src_sha256.

_pipeline.py chooses the kernels, _matrix.py builds the matrix layouts and
__init__.py settles the automatic mask step on the first frame: all change
the speed without changing the compiled sources, so the performance pages
compare this hash too (tools/generate_docs.py).
"""
import hashlib
import os

PY_TUNING = ("_pipeline.py", "_matrix.py", "__init__.py")


def py_sha256(pkg_dir):
    """sha256 over PY_TUNING, read from the package directory pkg_dir."""
    h = hashlib.sha256()
    for name in PY_TUNING:
        with open(os.path.join(pkg_dir, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()
