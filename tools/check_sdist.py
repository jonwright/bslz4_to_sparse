#!/usr/bin/env python3
"""Check that a built sdist actually contains everything the native build needs.

setup.py builds the .so with tools/build_extension.py *outside* setuptools'
Extension machinery (the extension is package data -- see the module docstring),
so setuptools no longer adds its C sources to the sdist implicitly.  Every
source and header the build opens must therefore be named in MANIFEST.in.  When
one is missing the failure only shows up at `pip install` time on a machine
with no wheel -- i.e. after the release is published.  This script catches it
before that.

It does two things:

  1. a static check that every file build_extension._sources() compiles, plus
     the files setup.py needs at build time, is present in the tarball;
  2. (unless --contents-only) `pip install`s the sdist into a throwaway venv
     and imports it.  That is the real release path: pip resolves the PEP 517
     build requires (setuptools/wheel) and install_requires (numpy) itself, so
     this script never hand-installs build or runtime dependencies.

Usage:
    python3 tools/check_sdist.py --build           # build sdist, then check + install it
    python3 tools/check_sdist.py dist/*.tar.gz     # check an existing sdist
    python3 tools/check_sdist.py --contents-only   # static contents check only
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

sys.path.insert(0, HERE)
import build_extension as be  # noqa: E402

# Files setup.py / the build path needs by name, beyond the compiled sources.
_BUILD_FILES = (
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    "src/__init__.py",
    "src/c2py_loader.py",
    "src/_csc_variants.py",
    "src/bslz4_common.h",
    "src/bslz4_codec.h",
    "src/bslz4_untranspose.h",
    "src/bslz4_collect_caps.h",
    "src/bslz4_collect_simd.hpp",
    "src/bslz4_csc_variants.hpp",
    "src/bslz4_lz4zero.h",
    "tools/build_extension.py",
)


def required_files():
    req = set(_BUILD_FILES)
    for src in be._sources():
        req.add(os.path.relpath(src, REPO))
    return sorted(req)


def sdist_members(path):
    with tarfile.open(path) as tf:
        names = tf.getnames()
    members = set()
    for name in names:
        parts = name.split("/", 1)
        if len(parts) == 2 and parts[1]:
            members.add(parts[1])
    return members


def find_sdist(explicit):
    if explicit:
        return explicit
    candidates = sorted(
        glob.glob(os.path.join(REPO, "dist", "*.tar.gz")), key=os.path.getmtime)
    if not candidates:
        raise SystemExit("no sdist found in dist/; pass one or use --build")
    return candidates[-1]


def check_contents(path):
    have = sdist_members(path)
    missing = [r for r in required_files() if r not in have]
    if missing:
        print("MISSING from %s (%d):" % (os.path.basename(path), len(missing)),
              file=sys.stderr)
        for name in missing:
            print("  " + name, file=sys.stderr)
        return False
    print("contents OK: %s has all %d required build inputs"
          % (os.path.basename(path), len(required_files())))
    return True


def install_sdist(path):
    """pip install the sdist into a throwaway venv and import it.

    Deliberately uses pip, not `setup.py build`: pip reads the sdist's
    metadata and installs install_requires (numpy) plus, in an isolated
    build, the pyproject build-system requires (setuptools/wheel).  That is
    the path a user without a wheel actually takes, and it needs no
    hand-installed dependencies here.
    """
    tmp = tempfile.mkdtemp(prefix="check_sdist_")
    try:
        venv_dir = os.path.join(tmp, "venv")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        py = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin",
                          "python")
        subprocess.check_call(
            [py, "-m", "pip", "install", "--quiet", os.path.abspath(path)])
        subprocess.check_call(
            [py, "-c",
             "import bslz4_to_sparse as b; "
             "print('installed', b.__file__); "
             "print('backends', b.available_backends())"])
        print("install OK: pip installed and imported %s" % os.path.basename(path))
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sdist", nargs="?", help="path to .tar.gz (default: newest in dist/)")
    ap.add_argument("--build", action="store_true",
                    help="run `python -m build --sdist` first")
    ap.add_argument("--contents-only", action="store_true",
                    help="skip the pip-install step (static contents check only)")
    args = ap.parse_args()

    if args.build:
        shutil.rmtree(os.path.join(REPO, "dist"), ignore_errors=True)
        subprocess.check_call(
            [sys.executable, "-m", "build", "--sdist",
             "--outdir", os.path.join(REPO, "dist")], cwd=REPO)

    path = find_sdist(args.sdist)
    ok = check_contents(path)
    if ok and not args.contents_only:
        ok = install_sdist(path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
