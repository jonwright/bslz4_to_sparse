# c2py23 runtime (vendored)

These files are the fixed runtime support code that every c2py23-generated
wrapper links against: a CPython C-API loader (`c2py_dlsym.*`,
`c2py_pythonh.*`), the shared runtime helpers (`c2py_runtime.*`), and the
per-arch CPU-feature-detection headers (`c2py_amd64.h`, `c2py_arm64.h`,
`c2py_ppc64.h`, plus the umbrella `c2py.h`). None of it is generated from
our `.c2py` spec — it is upstream infrastructure, unchanged by anything in
this repo. `src/c2py_loader.py` is vendored from the same source and the
same tag (it has to live in the package directory to be importable at
runtime); re-vendor it alongside these files.

`src/c2py_loader.py` carries one local patch, marked `LOCAL PATCH` in the
file: its "overwriting existing module" check samples `sys.modules` before
`module_from_spec()`. Drop it once that is fixed upstream.

Vendored (not a build or runtime dependency on the `c2py23` PyPI package)
so that building this project does not require c2py23 to be installed at
all. c2py23 is still needed as a developer tool to *regenerate*
`src/bslz4_to_sparse_wrapper.c` after a change to the `.c2py` spec
embedded in `src/bslz4_to_sparse.cpp` -- see the C2PY_BEGIN block there
and `tools/regenerate_wrapper.py`.

Source: https://github.com/jonwright/c2py23, tag `v0.5.4`,
commit `3aeeb8bb62ad750721ef626b04e5a4f9def15303`. License: MIT (see
`c2py23`'s own `LICENSE`; same terms as this project).

To update: pull a newer c2py23 tag, copy its `c2py23/runtime/*.c` and
`*.h` files over these, update the commit hash above and in the wrapper's
header comment, regenerate the wrapper, and confirm the test suite still
passes.
