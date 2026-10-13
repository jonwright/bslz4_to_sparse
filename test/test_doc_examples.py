"""Every examples/ex*.py runs and passes its own checks, as the documentation
site shows it (tools/generate_docs.py runs the same scripts the same way)."""

import glob
import os
import subprocess
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import pytest

import bslz4_to_sparse

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = sorted(glob.glob(os.path.join(REPO, "examples", "ex*.py")))


def _needs(path):
    with open(path) as fh:
        src = fh.read()
    return [
        m for m in ("h5py", "hdf5plugin", "pyFAI") if "import %s" % m in src or "from %s" % m in src
    ]


@pytest.fixture(scope="session")
def standin_folder(tmp_path_factory):
    """The files the examples are written for (see tools/standin.py), in a working directory."""
    pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    pytest.importorskip("pyFAI")
    import standin

    d = str(tmp_path_factory.mktemp("examples"))
    standin.make_standin(d)
    return d


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: os.path.basename(p))
def test_example_runs(path, standin_folder):
    for mod in _needs(path):
        pytest.importorskip(mod)
    env = dict(os.environ)
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(bslz4_to_sparse.__file__)))
    env["PYTHONPATH"] = os.pathsep.join([pkg_parent] + [p for p in [env.get("PYTHONPATH")] if p])
    r = subprocess.run(
        [sys.executable, path],
        env=env,
        cwd=standin_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r.returncode == 0, r.stdout.decode()
