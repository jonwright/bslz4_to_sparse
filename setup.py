"""
Setup script.

Builds the bslz4_to_sparse C++ extension via c2py23: the Python wrapper is
generated at build time from the C2PY_BEGIN block embedded in
src/bslz4_to_sparse.cpp, then compiled and linked together with the c2py23
runtime, the vendored lz4/bitshuffle/kcb sources, and our own C++ core.
"""
import os
import platform
import sys

from setuptools import Extension, setup

import c2py23
from c2py23.generator import generate
from c2py23.harvester import extract_from_file
from c2py23.parser import from_c2py_dict

HERE = os.path.abspath(os.path.dirname(__file__))
C2PY_RUNTIME_DIR = os.path.join(os.path.dirname(c2py23.__file__), "runtime")


def generate_wrapper():
    """Regenerate src/bslz4_to_sparse_wrapper.c from the C2PY_BEGIN block."""
    source_path = os.path.join(HERE, "src", "bslz4_to_sparse.cpp")
    spec = extract_from_file(source_path)
    module = from_c2py_dict(spec, source_path)
    code = generate(module)
    wrapper_path = os.path.join(HERE, "src", "bslz4_to_sparse_wrapper.c")
    with open(wrapper_path, "w") as f:
        f.write(code)
    return wrapper_path


wrapper_path = generate_wrapper()

sources = [
    wrapper_path,
    os.path.join(C2PY_RUNTIME_DIR, "c2py_runtime.c"),
    "src/bslz4_to_sparse.cpp",
    "kcb/src/bitshuffle.c",
    "bitshuffle/src/bitshuffle_core.c",
    "bitshuffle/src/iochain.c",
    "lz4/lib/lz4.c",
]

include_dirs = [
    C2PY_RUNTIME_DIR,
    "lz4/lib",
    "kcb/src",
    "bitshuffle/src",
]

flags = ["-O2", "-std=c++11"]

if platform.system() == "Windows":
    flags = ["/O2", "/std:c++14", "-Drestrict="]
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
    install_requires=["numpy", "h5py", "c2py23"],
    author="Jon Wright",
    author_email="wright@esrf.fr",
    url="http://github.com/jonwright/bslz4_to_sparse",
    version="0.1.0",
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
)
