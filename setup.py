"""
Setup script.

bslz4_to_sparse ships a prebuilt native module treated as PACKAGE DATA
(there is no setuptools.Extension): the .so/.pyd is compiled by
tools/build_extension.py and dropped into src/ (the package dir), then
packaged as data.  This gives a version-independent wheel tagged
py2.py3-none-<plat> -- one binary serves Python 2.7 through 3.15 (c2py23
resolves the CPython API at runtime via dlsym).

How the native module gets built:
  * a developer `pip install .` (or `python -m build --wheel` on a native
    host) triggers build_py, which runs tools/build_extension.py for the
    host platform if the .so isn't already present;
  * CI cross-builds each platform's .so beforehand and sets
    BSLZ4_SKIP_NATIVE_BUILD=1 so the wheel just packages the prebuilt data
    (with BSLZ4_WHEEL_PLAT to force the wheel's platform tag).

c2py23 is NOT a build time or runtime dependency: the generated wrapper and
the runtime it needs are vendored (src/bslz4_to_sparse_wrapper.c,
c2py_runtime/).
"""
import os
import re
import sys

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - older setuptools
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:  # no bdist_wheel available (plain `setup.py build`)
        _bdist_wheel = None

HERE = os.path.abspath(os.path.dirname(__file__))

# The vendored loader is the single source of truth for the platform key, so
# the filename build_py looks for cannot drift from the one src/__init__.py
# imports at runtime.
sys.path.insert(0, os.path.join(HERE, "src"))
from c2py_loader import _platform_key  # noqa: E402

_SO_SUFFIX = ".pyd" if os.name == "nt" else ".so"
_SO_NAME = "_bslz4_to_sparse.c2py23-%s%s" % (_platform_key(), _SO_SUFFIX)
_SO_PATH = os.path.join(HERE, "src", _SO_NAME)


class PlatlibDistribution(Distribution):
    """Force a platlib (platform) wheel despite no setuptools.Extension."""

    def has_ext_modules(self):
        return True


def _ensure_native_so():
    import subprocess
    if os.path.exists(_SO_PATH):
        return
    subprocess.check_call(
        [sys.executable, os.path.join(HERE, "tools", "build_extension.py")])


class build_native(build_py):
    """Build the host .so on demand (unless prebuilt for a cross target)."""

    def run(self):
        if os.environ.get("BSLZ4_SKIP_NATIVE_BUILD") != "1":
            _ensure_native_so()
        build_py.run(self)


_cmdclass = {"build_py": build_native}

if _bdist_wheel is not None:

    class bdist_wheel_override(_bdist_wheel):
        """Tag the wheel py2.py3-none-<plat> instead of cpXY-cpXY-<plat>."""

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            _impl, _abi, plat = _bdist_wheel.get_tag(self)
            plat = os.environ.get("BSLZ4_WHEEL_PLAT", plat)
            return os.environ.get("BSLZ4_PYTHON_TAG", "py2.py3"), "none", plat

    _cmdclass["bdist_wheel"] = bdist_wheel_override


with open(os.path.join(HERE, "README.md"), "r") as f:
    readme = f.read()

# src/__init__.py is the single source of the version. Parsed rather than
# imported: importing needs the compiled extension, which may not exist yet.
with open(os.path.join(HERE, "src", "__init__.py"), "r") as f:
    version = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)

setup(
    name="bslz4_to_sparse",
    packages=["bslz4_to_sparse"],
    package_dir={"bslz4_to_sparse": "src"},
    package_data={
        "bslz4_to_sparse": [
            "_bslz4_to_sparse.c2py23-*.so",
            "_bslz4_to_sparse.c2py23-*.pyd",
        ],
    },
    distclass=PlatlibDistribution,
    cmdclass=_cmdclass,
    # numpy is a runtime dependency (src/__init__.py imports it); h5py/pyFAI
    # are test-only and intentionally absent here.
    install_requires=["numpy"],
    author="Jon Wright",
    author_email="wright@esrf.fr",
    url="http://github.com/jonwright/bslz4_to_sparse",
    version=version,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
)
