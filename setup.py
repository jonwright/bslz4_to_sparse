"""
Setup script.

Builds the bslz4_to_sparse C++ extension by compiling
src/bslz4_to_sparse_wrapper.c (the Python wrapper, committed to the repo
-- see that file's header comment and tools/regenerate_wrapper.py)
together with a vendored copy of the c2py23 runtime (c2py_runtime/, see
its README.md), the vendored lz4/bitshuffle/kcb/zstd sources, and our own
C++ core. Building this package does NOT require c2py23 to be installed;
that's only needed by tools/regenerate_wrapper.py, run by hand after
changing the C2PY_BEGIN spec embedded in src/bslz4_to_sparse.cpp.
"""
import glob
import os
import platform
import sys

from setuptools import Extension, setup

HERE = os.path.abspath(os.path.dirname(__file__))
C2PY_RUNTIME_DIR = os.path.join(HERE, "c2py_runtime")

sources = [
    "src/bslz4_to_sparse_wrapper.c",
    os.path.join(C2PY_RUNTIME_DIR, "c2py_runtime.c"),
    "src/bslz4_to_sparse.cpp",
    "kcb/src/bitshuffle.c",
    "bitshuffle/src/bitshuffle_core.c",
    "bitshuffle/src/iochain.c",
    "lz4/lib/lz4.c",
]

# zstd decompression only (issue #10): the compress/ and dictBuilder/ trees,
# and the optional huf_decompress_amd64.S fast path, are not needed.
sources += sorted(glob.glob("zstd/lib/common/*.c"))
sources += sorted(glob.glob("zstd/lib/decompress/*.c"))

include_dirs = [
    C2PY_RUNTIME_DIR,
    "lz4/lib",
    "kcb/src",
    "bitshuffle/src",
    "zstd/lib",
]

flags = ["-O2", "-std=c++11", "-DZSTD_DISABLE_ASM"]

if platform.system() == "Windows":
    flags = ["/O2", "/std:c++14", "-Drestrict=", "-DZSTD_DISABLE_ASM"]
    if sys.version_info[0] < 3:
        include_dirs += ["src/msvc_include"]

ext = Extension(
    "bslz4_to_sparse",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=flags,
    language="c++",
)

with open(os.path.join(HERE, "README.md"), "r") as f:
    readme = f.read()

setup(
    name="bslz4_to_sparse",
    packages=["bslz4_to_sparse"],
    package_dir={"bslz4_to_sparse": "src"},
    ext_package="bslz4_to_sparse",
    ext_modules=[ext],
    # c2py23 is not a dependency at all, build-time or runtime: the
    # wrapper it generates and the runtime it needs are both vendored
    # into this repo (src/bslz4_to_sparse_wrapper.c, c2py_runtime/) --
    # see tools/regenerate_wrapper.py for the one case (spec changes)
    # where a developer needs c2py23 installed, by hand.
    install_requires=["numpy", "h5py"],
    author="Jon Wright",
    author_email="wright@esrf.fr",
    url="http://github.com/jonwright/bslz4_to_sparse",
    version="0.1.0",
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
)
