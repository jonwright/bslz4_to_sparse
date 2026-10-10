# c2py23 runtime (vendored)

These files are the fixed runtime support code that every c2py23-generated
wrapper links against: a CPython C-API loader (`c2py_dlsym.*`,
`c2py_pythonh.*`), the shared runtime helpers (`c2py_runtime.*`), and the
per-arch CPU-feature-detection headers (`c2py_amd64.h`, `c2py_arm64.h`,
`c2py_ppc64.h`, plus the umbrella `c2py.h`). None of it is generated from
our `.c2py` spec — it is upstream infrastructure, unchanged by anything in
this repo. `src/c2py_loader.py` is vendored from the same source and the same tag
(it has to live in the package directory to be importable at runtime);
re-vendor it alongside these files.

Vendored (not a build or runtime dependency on the `c2py23` PyPI package)
so that building this project does not require c2py23 to be installed at
all. c2py23 is still needed as a developer tool to *regenerate*
`src/bslz4_to_sparse_wrapper.c` after a change to the `.c2py` spec
embedded in `src/bslz4_to_sparse.c` -- see the C2PY_BEGIN block there
and `tools/regenerate_wrapper.py`.

Source: https://github.com/jonwright/c2py23, `v0.5.8`,
commit `ee81c975bfd05ef4649dd3d1d4ef5a9facd018aa`. Since v0.5.5:
v0.5.6 fixes the loader's "overwriting existing module" check to sample
`sys.modules` before `module_from_spec()` (this repo carried that as a
`LOCAL PATCH`; `src/c2py_loader.py` is now upstream again); v0.5.7 makes
the decode-time dtype check honor the itemsize fallback in `when:`
conditions; v0.5.8 is the generated-wrapper size work -- shared
`c2py_check_no_overlap()`/`c2py_check_contiguity()` helpers, a thin
METH_VARARGS shim over METH_FASTCALL, per-function docstrings emitted once,
and an opt-in `check_aliasing: false`. v0.5.5 fixed the `_variants_*`
callable segfaulting on Python 2.7 (the dlsym runtime resolved
`PyBytes_FromStringAndSize` with no `PyString_*` fallback). License: MIT
(see `c2py23`'s own `LICENSE`; same terms as this project).

To update: pull a newer c2py23 tag, copy its `c2py23/runtime/*.c` and
`*.h` files over these, update the commit hash above and in the wrapper's
header comment, regenerate the wrapper, and confirm the test suite still
passes.
