# docs/AGENTS.md: how this documentation stays true

The site is built by running the code.  Nobody, human or LLM, types API
text, example output or a benchmark number into a Markdown file.

## Where each kind of content comes from

| Content | Source of truth | Made into a page by |
|---|---|---|
| Overview | `docs/index.md` (hand-written, conceptual only) | mkdocs |
| Examples and their output | `examples/ex*.py` (runnable scripts, written for the reader's own `h5filename`/`h5dataset`, that assert their own results) | `tools/generate_docs.py` turns each into a notebook (`tools/make_notebooks.py`), runs it on the stand-in files of `tools/standin.py` and shows its output and figures |
| API reference | docstrings and signatures in `src/` | `tools/generate_docs.py` (introspection) |
| Pipeline step values | `VALUES` in `src/_pipeline.py` | `tools/generate_docs.py` |
| Decode error codes | `_DECODE_ERRORS` in `src/__init__.py` | `tools/generate_docs.py` |
| Acknowledgements, references, licences | the `## Acknowledgements` and `## References` sections of `README.md`, and `licenses/` | `tools/generate_docs.py` |
| Performance numbers | `docs/bench/real_data_<cpu>.json`, one per machine, measured there by `tools/bench_docs.py` and committed | `tools/generate_docs.py` |
| Benchmark data | the two example files, `mask.h5` and `calib.poni` (downloaded; `tools/example_data.py` makes the 8 kB lz4 and masked variants from them) | `tools/bench_docs.py`, `tools/bench_kernels.py` |
| Kernel comparison | `docs/bench/kernels_<cpu>.json`, measured by `tools/bench_kernels.py` and committed | `tools/generate_docs.py` |

Generated pages go to `docs/gen/`, which is git-ignored: if a file is not
in the repository, it cannot be edited by hand.

## Rules

1. **Never write generated content into a Markdown file.**  A function
   signature, a parameter description, program output or a timing in a
   committed `.md` file is a bug.
2. **Change the docs by changing the code.**  Wrong API text: edit the
   docstring.  New example: add `examples/exN_name.py` (a module docstring
   whose first line is the title, then code that asserts what it shows) and
   add it to the nav in `docs/mkdocs.yml`.  `test/test_doc_examples.py` runs
   every example, so a broken one fails the tests.
   An example never creates its own data: it starts from the reader's file
   (`h5filename`, `h5dataset`, and `poni_filename` where pyFAI is used).  The
   tests and the build write stand-ins with those names (`tools/standin.py`).
   A figure is plain matplotlib in the script; the notebook keeps it.
3. **Benchmarks are measured, never typed and never run in CI.**  Run
   `tools/bench_docs.py --out docs/bench/real_data_<cpu>.json` on the machine
   the numbers are for, on one core, and commit the JSON; every file is
   shown, side by side, so all must measure the same cases.  The JSON records the
   C sources' sha256 and a hash of `_pipeline.py`, `_matrix.py` and `__init__.py`; when either
   differs from the code the site is built from, the performance page says
   so in a warning box.
4. **`docs/index.md` is conceptual.**  No code blocks, no signatures, no
   numbers with units; `tools/verify_docs.py` rejects them.
5. **Run `tools/verify_docs.py` before committing docs changes.**  CI runs
   it, generates the site from the CI-built wheel and builds it with
   `mkdocs build --strict`; any failure stops the deploy.

## Building locally

    BSLZ4_TO_SPARSE_PATH=<dir holding a built bslz4_to_sparse> python3 tools/verify_docs.py
    BSLZ4_TO_SPARSE_PATH=<the same> python3 tools/generate_docs.py   # lib/ if built with
                                                       # setup.py build --build-lib=lib
    mkdocs build --strict -f docs/mkdocs.yml        # site/ (git-ignored)

Needs h5py, hdf5plugin, pyFAI, scipy, mkdocs and mkdocs-material.
