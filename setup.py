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
from setuptools.command.build_ext import build_ext

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

# -std=c++11 is deliberately NOT in the flags shared by every source
# below: it's meaningless for the plain C sources in this extension
# (the vendored lz4/bitshuffle/zstd sources, src/bslz4_to_sparse_wrapper.c,
# c2py_runtime.c) -- GCC only warns and ignores it there ("valid for
# C++/ObjC++ but not for C"), but some toolchains treat that warning as
# fatal under -Wall -Werror (confirmed on a ppc64le GCC), breaking the
# whole build over a flag those files never needed in the first place.
# cpp_only_flags (added back in for just the .cpp sources by
# BuildExtCppStd below) is how it actually reaches the one file that
# does need it, src/bslz4_to_sparse.cpp.
flags = ["-O2", "-DZSTD_DISABLE_ASM"]
cpp_only_flags = ["-std=c++11"]

if platform.system() == "Windows":
    # MSVC: no reported issue with a global C++-standard flag reaching
    # the C sources here, so keep it simple and global as before, unlike
    # the POSIX/GCC path -- cpp_only_flags empty means BuildExtCppStd
    # has nothing to add per-file on this platform.
    flags = ["/O2", "/std:c++14", "-Drestrict=", "-DZSTD_DISABLE_ASM"]
    cpp_only_flags = []
    if sys.version_info[0] < 3:
        include_dirs += ["src/msvc_include"]
elif platform.machine() in ("ppc64le", "ppc64"):
    # VSX is baseline on the ppc64le ABI (introduced with POWER8, which
    # already has it as a core ISA feature -- same relationship SSE2 has
    # to x86-64) so these flags are safe unconditionally, not opt-in.
    # Needed globally (not per-function target attributes, unlike the
    # x86 SIMD collect tiers in bslz4_collect_simd.hpp): GCC's PowerPC
    # target-attribute support for enabling vector types per function is
    # far less established than x86's, not something to rely on blind
    # without a way to verify it. Whether the VSX collect kernel is
    # actually *used* is still a separate runtime check (c2py_ppc64_vsx,
    # see bslz4_collect_simd.hpp) -- this only makes it compile.
    flags = flags + ["-maltivec", "-mvsx"]


class BuildExtCppStd(build_ext):
    """Adds cpp_only_flags (the C++ standard flag) only when compiling a
    .cpp source. distutils/setuptools' Extension.extra_compile_args has
    no per-source-file concept -- this is the standard way to apply a
    flag to just the C++ files in a mixed C/C++ extension (see the
    -std=c++11 comment above for why that matters here). No-op on MSVC
    (cpp_only_flags is already empty there, and its compiler object
    doesn't expose the same _compile hook UnixCCompiler-family classes
    do)."""

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
    version="0.1.0",
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
)
