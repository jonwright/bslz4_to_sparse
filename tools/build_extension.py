#!/usr/bin/env python3
"""
Build the bslz4_to_sparse native .so/.pyd directly.

The extension is treated as package data (see setup.py), so we compile it
ourselves with the platform toolchain and drop the result into the package
source dir (src/).  This replaces setuptools' build_ext and lets the same
script do native builds, cross-compiles (set CC/CXX to a cross toolchain and
pass --target-platform), and MSVC on Windows.

Usage:
  # native (host platform)
  python3 tools/build_extension.py

  # cross-compile for aarch64
  CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ \
    python3 tools/build_extension.py --target-platform linux_aarch64

  # output somewhere other than src/ (e.g. a staging dir for a wheel)
  python3 tools/build_extension.py --out /tmp/pkg/bslz4_to_sparse
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
C2PY_RUNTIME = os.path.join(REPO, "c2py_runtime")

sys.path.insert(0, os.path.join(REPO, "src"))
from c2py_loader import _platform_key  # noqa: E402


def _sources():
    s = [
        "src/bslz4_to_sparse_wrapper.c",
        os.path.join(C2PY_RUNTIME, "c2py_runtime.c"),
        "src/bslz4_to_sparse.cpp",
        "kcb/src/bitshuffle.c",
        "bitshuffle/src/bitshuffle_core.c",
        "bitshuffle/src/iochain.c",
        "lz4/lib/lz4.c",
    ]
    s += sorted(glob.glob("zstd/lib/common/*.c"))
    s += sorted(glob.glob("zstd/lib/decompress/*.c"))
    return [os.path.join(REPO, x) for x in s]


def _include_dirs():
    return [
        C2PY_RUNTIME,
        os.path.join(REPO, "lz4", "lib"),
        os.path.join(REPO, "kcb", "src"),
        os.path.join(REPO, "bitshuffle", "src"),
        os.path.join(REPO, "zstd", "lib"),
    ]


# The C++ is template expansion only: no libstdc++ types, no new/delete, no
# exceptions, no RTTI.  The one hidden libstdc++ dependency is
# __cxa_guard_acquire/release, emitted for thread-safe function-local statics
# (the idempotent SIMD capability probes), so turn that off too.  With these
# flags the .so needs only libc/libm and is linked with the C driver.
_CXX_FLAGS = ["-std=c++11", "-fno-threadsafe-statics", "-fno-exceptions", "-fno-rtti"]


def _build_gcc(srcs, incs, outpath, plat, builddir):
    """gcc- or clang-style driver (CC/CXX from the environment)."""
    cc = os.environ.get("CC") or "gcc"
    cxx = os.environ.get("CXX") or "g++"
    ppc_flags = ["-maltivec", "-mvsx", "-DNO_WARN_X86_INTRINSICS"] if plat == "linux_ppc64le" else []
    objs = []
    for s in srcs:
        obj = os.path.join(builddir, os.path.basename(s) + ".o")
        cmd = [cc if s.endswith(".c") else cxx, "-O2", "-DZSTD_DISABLE_ASM", "-fPIC"]
        cmd += ["-I%s" % i for i in incs]
        cmd += _CXX_FLAGS if s.endswith(".cpp") else []
        cmd += ppc_flags
        cmd += ["-c", s, "-o", obj]
        subprocess.check_call(cmd)
        objs.append(obj)
    link = [cc, "-shared", "-o", outpath] + objs + ["-lm"]
    if not plat.startswith("darwin"):
        link.append("-static-libgcc")
    subprocess.check_call(link)


def _build_msvc(srcs, incs, outpath, builddir):
    # Assumes the MSVC environment (vcvars) is set up by the caller. cl
    # compiles .c as C and .cpp as C++ and links in one /LD invocation.
    cl = os.environ.get("CC") or "cl"
    cmd = [cl, "/nologo", "/LD", "/O2", "/DZSTD_DISABLE_ASM", "/std:c++14",
           "/GR-", "/EHs-c-", "/Zc:threadSafeInit-"]
    cmd += ["/I%s" % i for i in incs]
    cmd += ["/Fe%s" % outpath]
    cmd += srcs
    subprocess.check_call(cmd)
    # cl honours /Fe, but if it still emitted a .dll, normalise to .pyd.
    if not os.path.exists(outpath):
        alt = os.path.splitext(outpath)[0] + ".dll"
        if os.path.exists(alt):
            os.rename(alt, outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-platform", default=None,
                    help="platform key for the output filename (e.g. linux_aarch64); default host")
    ap.add_argument("--build-dir", default=None,
                    help="scratch dir for object files (default: build/build_extension); "
                         "use a distinct one per target when building several in one tree")
    ap.add_argument("--out", default=None,
                    help="output directory (default: the package source dir src/)")
    args = ap.parse_args()

    plat = args.target_platform or _platform_key()
    if args.target_platform:
        # _platform_key() uses the host; honour a cross target explicitly.
        pass
    is_windows = plat.startswith("win") or os.name == "nt"
    suffix = ".pyd" if is_windows else ".so"
    outdir = os.path.abspath(args.out or os.path.join(REPO, "src"))
    os.makedirs(outdir, exist_ok=True)
    outname = "_bslz4_to_sparse.c2py23-%s%s" % (plat, suffix)
    outpath = os.path.join(outdir, outname)

    srcs = _sources()
    incs = _include_dirs()
    builddir = os.path.abspath(args.build_dir or os.path.join(REPO, "build", "build_extension"))
    os.makedirs(builddir, exist_ok=True)

    if is_windows and os.environ.get("MSVC_ENV", "").lower() not in ("1", "true", "yes"):
        # Default to the mixer/gcc cross toolchain if one is provided.
        if os.environ.get("CC"):
            _build_gcc(srcs, incs, outpath, plat, builddir)
        else:
            _build_msvc(srcs, incs, outpath, builddir)
    elif sys.platform == "win32":
        _build_msvc(srcs, incs, outpath, builddir)
    else:
        _build_gcc(srcs, incs, outpath, plat, builddir)

    print("wrote %s" % outpath)


if __name__ == "__main__":
    main()
