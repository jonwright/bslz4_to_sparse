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

  # extra flags for every compile / the link (gcc-style drivers), e.g. PGO
  BSLZ4_EXTRA_CFLAGS="-fprofile-generate" BSLZ4_EXTRA_LDFLAGS="-fprofile-generate" \
    python3 tools/build_extension.py --out ...
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
C2PY_RUNTIME = os.path.join(REPO, "c2py_runtime")

sys.path.insert(0, os.path.join(REPO, "src"))
from c2py_loader import _platform_key  # noqa: E402


def _sources():
    s = [
        "src/bslz4_to_sparse_wrapper.c",
        os.path.join(C2PY_RUNTIME, "c2py_runtime.c"),
        "src/bslz4_to_sparse.c",
        "src/bslz4_driver.c",
        "src/bslz4_registry.c",
        "src/kernels_generic.cpp",
        "kcb/src/bitshuffle.c",
        "bitshuffle/src/bitshuffle_core.c",
        "bitshuffle/src/iochain.c",
        "lz4/lib/lz4.c",
    ]
    s = [os.path.join(REPO, x) for x in s]
    s += sorted(glob.glob(os.path.join(REPO, "zstd", "lib", "common", "*.c")))
    s += sorted(glob.glob(os.path.join(REPO, "zstd", "lib", "decompress", "*.c")))
    return s


def _include_dirs():
    return [
        C2PY_RUNTIME,
        os.path.join(REPO, "lz4", "lib"),
        os.path.join(REPO, "kcb", "src"),
        os.path.join(REPO, "bitshuffle", "src"),
        os.path.join(REPO, "zstd", "lib"),
    ]


def _rel(path):
    return os.path.relpath(path, REPO).replace(os.sep, "/")


def _patch_files(lib):
    """patches/<lib>/*.patch, in order: applied to the submodule's sources at
    build time (the submodule itself stays at the upstream commit)."""
    return sorted(glob.glob(os.path.join(REPO, "patches", lib, "*.patch")))


def _apply_patches(src, patches, out):
    """Write `src` with the unified-diff hunks for its basename from `patches`
    applied to `out`.  Strict: every hunk's context must match exactly (near
    its stated line), or the build stops -- a submodule bump that no longer
    takes the patch must be noticed."""
    with open(src, "r", newline="") as f:
        lines = f.read().split("\n")
    name = os.path.basename(src)
    for pf in patches:
        with open(pf, "r", newline="") as f:
            plines = f.read().split("\n")
        i, active = 0, False
        while i < len(plines):
            l = plines[i]
            if l.startswith("+++ "):
                active = l[4:].strip().split("/")[-1] == name
            elif l.startswith("@@") and active:
                start = int(re.match(r"@@ -(\d+)", l).group(1)) - 1
                old, new = [], []
                i += 1
                while i < len(plines) and not plines[i].startswith(("@@", "diff ", "--- ")) \
                        and plines[i] != "-- ":                     # format-patch signature

                    h = plines[i]
                    if h.startswith(" ") or h == "":
                        old.append(h[1:]); new.append(h[1:])
                    elif h.startswith("-"):
                        old.append(h[1:])
                    elif h.startswith("+"):
                        new.append(h[1:])
                    i += 1
                while old and new and old[-1] == "" and new[-1] == "":     # trailing blank of the patch file
                    old.pop(); new.pop()
                hits = [k for k in range(max(0, start - 200), min(len(lines), start + 200))
                        if lines[k:k + len(old)] == old]
                if len(hits) != 1:
                    raise SystemExit("patch %s does not apply to %s (hunk at line %d: %d matches)"
                                     % (os.path.basename(pf), src, start + 1, len(hits)))
                k = hits[0]
                lines[k:k + len(old)] = new
                continue
            i += 1
    with open(out, "w", newline="") as f:
        f.write("\n".join(lines))
    return out


def _digest_files():
    """Every compiled source and every header next to one, sorted (and the
    patches applied at build time)."""
    files = set(_sources()) | set(_patch_files("lz4"))
    for d in [os.path.join(REPO, "src")] + _include_dirs():
        for pat in ("*.h", "*.hpp"):
            files.update(glob.glob(os.path.join(d, pat)))
    return sorted(files, key=_rel)


def source_digest():
    """sha256 over _digest_files(), names and contents.

    Identifies the code inside a .so independently of git (sdists have no
    .git), so a build can be matched to a source tree byte for byte.
    """
    h = hashlib.sha256()
    for f in _digest_files():
        with open(f, "rb") as fh:
            data = fh.read()
        h.update(("%s %d\n" % (_rel(f), len(data))).encode())
        h.update(data)
    return h.hexdigest()


def embedded_digest(sopath):
    """The src_sha256 a built .so/.pyd carries, read from the file without
    loading it; None for a build that predates the embedded build info."""
    with open(sopath, "rb") as fh:
        m = re.search(b'"src_sha256": "([0-9a-f]{64})"', fh.read())
    return m.group(1).decode() if m else None


def embedded_git(sopath):
    """The "git" field a built .so/.pyd carries (None if it has none, or was
    built outside a git checkout)."""
    with open(sopath, "rb") as fh:
        m = re.search(b'"git": "([^"]*)"', fh.read())
    return m.group(1).decode() if m else None


def _git(*args):
    # Only ask git about REPO itself: in an unpacked sdist inside some other
    # checkout, git would silently walk up and describe THAT repository.
    try:
        top = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--show-toplevel"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        if os.path.realpath(top) != os.path.realpath(REPO):
            return None
        return subprocess.check_output(["git", "-C", REPO] + list(args),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # no git, not a checkout, or "dubious ownership" in a container
        return None


def _compiler_id(cmd):
    try:
        out = subprocess.check_output([cmd, "--version"], stderr=subprocess.STDOUT)
        return out.decode("utf-8", "replace").splitlines()[0].strip()
    except Exception:
        return cmd


def git_state():
    """(describe, modified) of the compiled files: `git describe --tags
    --always`, with -dirty when a compiled file differs from that commit or
    is not yet tracked, and the list of those files.  (None, []) outside a
    git checkout."""
    # Only the files that go into the .so count: an edited notebook or test
    # does not make a build "dirty". Files inside a submodule (lz4, zstd, ...)
    # are covered by naming the submodule, which also catches a moved commit.
    subs = (_git("config", "--file", ".gitmodules", "--get-regexp", r"\.path$") or "").split()[1::2]
    paths = sorted(set(next((m for m in subs if r.startswith(m + "/")), r)
                       for r in map(_rel, _digest_files())))
    describe = _git("describe", "--tags", "--always")
    # status, not diff: a new, still untracked source or header is a change
    # too (diff HEAD does not list it).
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *paths)
    # "XY path" (or "R  old -> new"); _git() strips the output, which eats the
    # leading space of a first " M path", so split on whitespace rather than
    # slicing a fixed column.
    modified = sorted(set(line.split(None, 1)[1].split(" -> ")[-1]
                          for line in status.splitlines() if line.strip())) \
        if status else []
    if describe and modified:
        describe += "-dirty"
    return describe, modified


def build_info(compiler, plat):
    """The build description embedded in the .so (see bslz4_to_sparse.build_info())."""
    with open(os.path.join(REPO, "src", "__init__.py")) as f:
        version = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)
    describe, modified = git_state()
    return {
        "version": version,
        # a tag when built clean on a tagged commit, else tag-N-gHASH; with
        # -dirty when a compiled file differs from that commit (listed)
        "git": describe,
        "modified": modified,
        "src_sha256": source_digest(),
        "compiler": compiler,
        "platform": plat,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_BUILD_INFO_C = """\
/* Generated by tools/build_extension.py for one build; not kept in the repo. */
#include <string.h>

static const char bslz4_build_info_json[] = "%s";

int bslz4_build_info(char *out, int n) {
    int len = (int)sizeof(bslz4_build_info_json) - 1;
    if (n > 0)
        memcpy(out, bslz4_build_info_json, (size_t)(len < n ? len : n));
    return len;
}
"""


def _write_build_info_c(builddir, info):
    text = json.dumps(info, sort_keys=True, ensure_ascii=True)
    path = os.path.join(builddir, "bslz4_build_info.c")
    with open(path, "w") as f:
        f.write(_BUILD_INFO_C % text.replace("\\", "\\\\").replace('"', '\\"'))
    return path


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
        cmd += os.environ.get("BSLZ4_EXTRA_CFLAGS", "").split()     # e.g. PGO
        if os.path.basename(s) == "lz4.c" and not plat.startswith("darwin"):
            # Pin the decoder's code alignment.  Unaligned, its speed moved
            # with whatever was linked before it: one constant changed in the
            # driver made LZ4_decompress_safe ~25 % slower on real Eiger blocks
            # (start at 0x30 vs 0x10 mod 64; EPYC 9454, 2026-10-04).
            cmd += os.environ.get("BSLZ4_LZ4_CFLAGS",
                                  "-falign-functions=64 -falign-loops=64").split()
        cmd += ["-c", s, "-o", obj]
        subprocess.check_call(cmd)
        objs.append(obj)
    link = [cc, "-shared", "-o", outpath] + objs + ["-lm"]
    link += os.environ.get("BSLZ4_EXTRA_LDFLAGS", "").split()
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
    builddir_ = os.path.abspath(args.build_dir or os.path.join(REPO, "build", "build_extension"))
    os.makedirs(builddir_, exist_ok=True)
    if _patch_files("lz4"):
        # lz4 stays at the upstream release in the submodule; our patches
        # (patches/lz4) go into a copy compiled in its place
        lz4c = os.path.join(REPO, "lz4", "lib", "lz4.c")
        patched = _apply_patches(lz4c, _patch_files("lz4"), os.path.join(builddir_, "lz4.c"))
        srcs = [patched if s == lz4c else s for s in srcs]
    incs = _include_dirs()
    builddir = os.path.abspath(args.build_dir or os.path.join(REPO, "build", "build_extension"))
    os.makedirs(builddir, exist_ok=True)

    if is_windows and os.environ.get("MSVC_ENV", "").lower() not in ("1", "true", "yes"):
        # Default to the mixer/gcc cross toolchain if one is provided.
        use_msvc = not os.environ.get("CC")
    else:
        use_msvc = sys.platform == "win32"
    cc = os.environ.get("CC") or ("cl" if use_msvc else "gcc")
    info = build_info(cc if use_msvc else _compiler_id(cc), plat)
    srcs.append(_write_build_info_c(builddir, info))
    print("build info: %s" % json.dumps(info, sort_keys=True))

    if use_msvc:
        _build_msvc(srcs, incs, outpath, builddir)
    else:
        _build_gcc(srcs, incs, outpath, plat, builddir)

    print("wrote %s" % outpath)


if __name__ == "__main__":
    main()
