
Decompress Dectris bitshuffle lz4 data directly to indices/value arrays.

The C++ core is generated as a Python extension via
[c2py23](https://github.com/jonwright/c2py23) and compiles in two
untranspose backends: [kcb](https://github.com/kalcutter/bitshuffle)
(the default) and the portable scalar kernel from upstream
[bitshuffle](https://github.com/kiyo-masui/bitshuffle). Use
`bslz4_to_sparse.available_backends()` / `set_backend()` to compare them.

After git clone:

    git submodule init
    git submodule update
    python3 -m pip install .

To run it at ESRF:

    cd test
    python3 test_vs_hdf5plugin.py
    python3 bench1.py

