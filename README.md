# bslz4_to_sparse

Decompress Dectris Eiger bitshuffle-LZ4 data directly to sparse
index/value arrays.

Documentation, with runnable examples and speed on real data:
https://jonwright.github.io/bslz4_to_sparse/

This repository brings together upstream code from bitshuffle [1],
the DECTRIS bitshuffle SIMD kernels [2], LZ4 [3] and
Zstandard [4] to process detector data as a
stream (in cache), rather than writing whole images to RAM and reading them
back again. Versions before 0.0.16 were mostly written by @jonwright; later
versions include a large amount of LLM-generated code (DeepSeek and Claude).

It is used in ImageD11 [5] to convert images directly to sparse
format, and it can also do radial integrations as sparse matrix products
(for example with the integration matrices of pyFAI [6]). The C/C++ core
is a Python extension through c2py23 [7]. Each decoder object has a
`pipeline`: one value per processing step (decode, mask, untranspose,
collect, and for matrix objects route and dot), chosen automatically for the
CPU, data and matrix unless given; `bslz4_to_sparse.describe()` lists the
values and which ones this machine can run.

After `git clone`:

    git submodule init
    git submodule update
    python3 -m pip install .

To run it at the ESRF:

    cd test
    python3 test_vs_hdf5plugin.py
    python3 bench1.py

To test a local build without installing:

    python3 setup.py build --build-lib=lib
    cd test
    BSLZ4_TO_SPARSE_PATH=$(pwd)/../lib python3 -m pytest -v
    BSLZ4_TO_SPARSE_PATH=$(pwd)/../lib python3 bench1.py

Every script in `test/` prepends `BSLZ4_TO_SPARSE_PATH` to `sys.path`, so
it wins over any installed `bslz4_to_sparse` and works where `PYTHONPATH`
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
`_testing.counters()` (test instrumentation) is only exact while one
decode runs at a time.

Collect values the machine cannot run (vsx on x86_64, neon on x86_64/POWER,
avx512cs/avx2cs/sse2 on POWER/ARM) are skipped by
`test_simd_collect_matches_scalar`.

The licences of the bundled upstream code are in `licenses/`.

## Acknowledgements

This package is mostly a regrouping of other people's work, put together
so that detector frames can be decoded and reduced in one pass. The
bitshuffle format and its reference implementation come from K. Masui and
co-authors [1]; the fast SIMD bit-transpose kernels from K. Conley
at DECTRIS [2]; the LZ4 and Zstandard decompressors from Y. Collet
and the Zstandard contributors [3], [4]. The integration matrices are
pyFAI's [6], and the sparse output serves ImageD11 [5].
The portable Python wrappers come from c2py23 [7]: they let one
compiled module load on every CPython from 2.7 to 3.15, free-threaded
builds included, without compiling against any Python headers. Our thanks
to all of these projects; please cite them as well as this one.

## References

[1] bitshuffle, K. Masui et al., https://github.com/kiyo-masui/bitshuffle;
   K. Masui et al., "A compression scheme for radio data in high performance
   computing", Astronomy and Computing 12, 181-190 (2015),
   https://doi.org/10.1016/j.ascom.2015.07.002

[2] bitshuffle SIMD bit-transpose kernels ("kcb"), K. Conley (DECTRIS),
   https://github.com/kalcutter/bitshuffle

[3] LZ4, Y. Collet, https://github.com/lz4/lz4

[4] Zstandard, Y. Collet and Meta Platforms, https://github.com/facebook/zstd;
   Y. Collet and M. Kucherawy (ed.), "Zstandard Compression and the
   'application/zstd' Media Type", RFC 8878 (2021),
   https://doi.org/10.17487/RFC8878

[5] ImageD11, https://github.com/FABLE-3DXRD/ImageD11

[6] pyFAI, https://github.com/silx-kit/pyFAI; G. Ashiotis, A. Deschildre,
   Z. Nawaz, J. P. Wright, D. Karkoulis, F. E. Picca and J. Kieffer, "The
   fast azimuthal integration Python library: pyFAI", J. Appl. Cryst. 48,
   510-519 (2015), https://doi.org/10.1107/S1600576715004306

[7] c2py23, https://github.com/jonwright/c2py23
