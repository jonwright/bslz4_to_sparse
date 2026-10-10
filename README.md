
Decompress Dectris bitshuffle lz4 data directly to indices/value arrays.

The C/C++ core is a Python extension through
[c2py23](https://github.com/jonwright/c2py23).  Each decoder object has a
`pipeline`: one value per processing step (decode, mask, untranspose,
collect, and for matrix objects route and dot), chosen automatically for the
CPU, data and matrix unless given; `bslz4_to_sparse.describe()` lists the
values and which this machine can run.

After git clone:

    git submodule init
    git submodule update
    python3 -m pip install .

To run it at ESRF:

    cd test
    python3 test_vs_hdf5plugin.py
    python3 bench1.py

To test a local build without installing:

    python3 setup.py build --build-lib=lib
    cd test
    BSLZ4_TO_SPARSE_PATH=$(pwd)/../lib python3 -m pytest -v
    BSLZ4_TO_SPARSE_PATH=$(pwd)/../lib python3 bench1.py

Every script in `test/` prepends `BSLZ4_TO_SPARSE_PATH` to `sys.path`, so
it wins over an installed `bslz4_to_sparse` and works where `PYTHONPATH`
is ignored (ESRF `jupyter-slurm`).

The extension is tagged by platform only
(`_bslz4_to_sparse.c2py23-linux_ppc64le.so`); c2py23 resolves the CPython
API at runtime, so one binary serves any Python version. Run the build
once per machine: a single `lib/` on a shared filesystem holds all of
them. `C2PY_TRACE=1` reports which file the loader picks.

Threads and free-threaded Python (3.14t, 3.15t): the decode entry points
release the GIL, and the extension declares itself free-threading safe, so
importing it does not re-enable the GIL. Nothing is locked; the rule is one
`chunk2sparse` / `chunk2sparseCSC` object per thread. Each object reuses its
own workspace and output arrays, so the arrays a call returns are
overwritten by that object's next call, and a call's inputs (chunks, mmap
base, mask) must not be changed or closed by another thread while it runs.
Breaking these rules gives wrong results or a crash, not an error.
`_testing.read_counters()` (test instrumentation) is only exact while one
decode runs at a time.

SIMD collect tiers the machine cannot run are skipped:
`test_vsx_collect_matches_scalar` on x86_64, `test_neon_collect_matches_scalar`
on x86_64/POWER, the avx512/avx2/sse2 ones on POWER/ARM.

