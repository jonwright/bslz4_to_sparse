# Third-party code in bslz4_to_sparse

The compiled extension (`_bslz4_to_sparse.c2py23-*.so` / `.pyd`) contains
code from the projects below, built from the git submodules of this
repository (or the copies of their sources in the sdist).  Their licences are
reproduced in this directory and ship in every wheel and sdist
(`license_files`, see `setup.cfg`).  bslz4_to_sparse itself is MIT
(`../LICENSE`).

| Component | Files compiled in | Version | Licence | File here |
|---|---|---|---|---|
| [LZ4](https://github.com/lz4/lz4) | `lz4/lib/lz4.c` (with `patches/lz4/*.patch` applied at build time) | v1.10.0 | BSD 2-Clause (`lib/LICENSE`) | `LZ4-LICENSE` |
| [Zstandard](https://github.com/facebook/zstd) | `zstd/lib/common/*.c`, `zstd/lib/decompress/*.c` | v1.5.7 | BSD 3-Clause (dual licensed BSD / GPLv2; used under BSD) | `ZSTD-LICENSE` |
| [bitshuffle](https://github.com/kiyo-masui/bitshuffle) | `bitshuffle/src/bitshuffle_core.c`, `iochain.c` | 526440a | MIT | `BITSHUFFLE-LICENSE` |
| [bitshuffle (Kal Conley, DECTRIS)](https://github.com/kalcutter/bitshuffle) ("kcb") | `kcb/src/bitshuffle.c` (with `patches/kcb/*.patch`), and its bit-transpose kernels adapted in `src/steps/lowplanes.h` and `src/steps/lowplanes_avx2.hpp` | 7be5b20 | MIT or Apache-2.0 (dual) | `KCB-LICENSE-MIT`, `KCB-LICENSE-APACHE` |
| [c2py23](https://github.com/jonwright/c2py23) runtime | `c2py_runtime/*.c`, `src/c2py_loader.py` | v0.5.8 | MIT | `C2PY23-LICENSE` |

`test/test_licenses.py` checks that these copies still match the
submodules' own licence files, so a submodule bump cannot leave a stale text
here.
