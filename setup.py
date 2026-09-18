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
import re
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.abspath(os.path.dirname(__file__))
C2PY_RUNTIME_DIR = os.path.join(HERE, "c2py_runtime")

# The vendored loader is the single source of truth for the platform key,
# so the filename written by build_ext (see get_ext_filename below) cannot
# drift from the one src/__init__.py looks for at import time.
sys.path.insert(0, os.path.join(HERE, "src"))
from c2py_loader import _platform_key  # noqa: E402 -- needs the path above

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

# -std=c++11 goes in cpp_only_flags, not in the shared flags:
# BuildExtCppStd applies it to the .cpp sources alone. Passing it to a C
# source only warns on GCC, but that warning is fatal under -Wall -Werror.
flags = ["-O2", "-DZSTD_DISABLE_ASM"]
cpp_only_flags = ["-std=c++11"]

if platform.system() == "Windows":
    # MSVC takes the C++ standard flag globally without complaint, so
    # cpp_only_flags stays empty and BuildExtCppStd is a no-op here.
    flags = ["/O2", "/std:c++14", "-Drestrict=", "-DZSTD_DISABLE_ASM"]
    cpp_only_flags = []
    if sys.version_info[0] < 3:
        include_dirs += ["src/msvc_include"]
elif platform.machine() in ("ppc64le", "ppc64"):
    # VSX is ppc64le ABI baseline, so these are safe unconditionally.
    # Needed globally: the VSX collect kernel in bslz4_collect_simd.hpp
    # uses no per-function target attribute. Whether it is actually used
    # is still a runtime check (c2py_ppc64_vsx).
    #
    # NO_WARN_X86_INTRINSICS makes upstream bitshuffle's SSE2 untranspose
    # kernels real code here: GCC's x86-intrinsic compatibility headers
    # implement emmintrin.h over VSX on ppc64le, and bitshuffle guards
    # those kernels with `defined(__SSE2__) || defined(NO_WARN_X86_INTRINSICS)`
    # for exactly this. Upstream's own setup.py does the same. Without it
    # the "sse" backend is a stub returning -14 on POWER.
    flags = flags + ["-maltivec", "-mvsx", "-DNO_WARN_X86_INTRINSICS"]


class BuildExtCppStd(build_ext):
    """Applies cpp_only_flags to .cpp sources only: Extension has no
    per-source-file flag concept. No-op on MSVC, whose compiler object
    has no _compile hook."""

    def build_extensions(self):
        if cpp_only_flags and self.compiler.compiler_type != "msvc":
            original_compile = self.compiler._compile

            def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                postargs = extra_postargs
                if src.endswith(".cpp"):
                    postargs = list(extra_postargs) + cpp_only_flags
                return original_compile(obj, src, ext, cc_args, postargs, pp_opts)

            self.compiler._compile = _compile
        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        """<module>.c2py23-<os>_<arch>.so, the c2py_loader convention
        src/__init__.py loads. Tagged by platform only: c2py23 resolves
        the CPython API at runtime, so one binary serves any Python
        version, and several architectures can share one directory."""
        suffix = ".pyd" if os.name == "nt" else ".so"
        return os.path.join(*ext_name.split(".")) + ".c2py23-" + _platform_key() + suffix


ext = Extension(
    "_bslz4_to_sparse",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=flags,
    language="c++",
)

with open(os.path.join(HERE, "README.md"), "r") as f:
    readme = f.read()

# src/__init__.py is the one place the version is written. Parsed rather
# than imported: importing the package needs the compiled extension,
# which does not exist yet at this point.
with open(os.path.join(HERE, "src", "__init__.py"), "r") as f:
    version = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)

setup(
    name="bslz4_to_sparse",
    packages=["bslz4_to_sparse"],
    package_dir={"bslz4_to_sparse": "src"},
    ext_package="bslz4_to_sparse",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtCppStd},
    # c2py23 is not a dependency at all, build-time or runtime: the
    # wrapper it generates and the runtime it needs are both vendored
    # into this repo (src/bslz4_to_sparse_wrapper.c, c2py_runtime/) --
    # see tools/regenerate_wrapper.py for the one case (spec changes)
    # where a developer needs c2py23 installed, by hand.
    install_requires=["numpy", "h5py"],
    author="Jon Wright",
    author_email="wright@esrf.fr",
    url="http://github.com/jonwright/bslz4_to_sparse",
    version=version,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
)
