"""
Make `import bslz4_to_sparse` work against an in-place (`setup.py
build_ext --inplace`) build without relying on the PYTHONPATH
environment variable -- some environments (e.g. ESRF's jupyter-slurm on
ppc64le, where site.py customization means PYTHONPATH is never even
read) don't honor it at all. Pure sys.path manipulation instead, which
pytest always sees regardless of environment/site.py behavior -- this
is that patch, done once here rather than by hand before every run.

setuptools' package_dir={"bslz4_to_sparse": "src"} (see setup.py) means
an in-place build's compiled .so lands directly in src/, next to
__init__.py -- but the directory is literally named "src", not
"bslz4_to_sparse", so a plain sys.path entry pointing at the repo root
won't resolve `import bslz4_to_sparse` on its own no matter how it gets
onto sys.path. This creates (or reuses) a bslz4_to_sparse -> src
symlink at the repo root so it does; gitignored and recreated here
rather than committed, so there's no "did the symlink survive checkout"
question to worry about.

Only kicks in if `import bslz4_to_sparse` doesn't already work as-is --
a real `pip install .` (this project's normal dev workflow elsewhere)
needs none of this, and must keep taking priority: the symlink points
at a plain source checkout, which has no compiled extension in it at
all unless an in-place build was also done, so forcing it onto
sys.path unconditionally would shadow a perfectly good installed
package with a broken one.
"""
import os
import sys

try:
    import bslz4_to_sparse  # noqa: F401 -- already importable, nothing to do
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_HERE)
    _LINK = os.path.join(_REPO_ROOT, "bslz4_to_sparse")
    _TARGET = os.path.join(_REPO_ROOT, "src")

    if not os.path.exists(_LINK) and os.path.isdir(_TARGET):
        try:
            os.symlink("src", _LINK)
        except OSError:
            pass  # e.g. no symlink support/permission here -- the import
                  # pytest does next will then fail with a clear error

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
