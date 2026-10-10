# bslz4_to_sparse

Decode Dectris Eiger frames, stored as bitshuffle-LZ4 (or zstd) compressed
HDF5 chunks, straight into sparse pixel lists, and optionally into azimuthal
integrations, without building the dense frame in Python.  It is used in
ImageD11 to turn diffraction images into sparse peak data.

## What it does

An Eiger writes each frame as one compressed chunk.  Most pixels of a
diffraction frame hold zero or background, so the useful content is a short
list of pixels above a threshold.  This package reads the compressed chunk
and produces that list, the pixel values and their positions, in a single
pass in C, honouring a detector mask.

The same pass can also multiply the frame by a sparse matrix.  A pyFAI
integration engine is such a matrix, so a powder pattern (1D, or a 2D
radial-by-azimuth cake) comes out of the decompression together with the
sparse pixels.

## How it is organised

Each processor object fixes what it is given when it is made: the detector
mask, the pixel type, the codec, and for integration the matrix.  It then
decodes one frame, or a batch of frames, per call.  Inside, decoding is a
pipeline of steps: decompress, apply the mask, undo the bit shuffle, collect
the pixels above the cut, and for integration route each block to a dense
or sparse product with the matrix.  Each step has several implementations
(portable C, and SIMD versions for x86, POWER and ARM); the best available
one is chosen automatically and can be overridden.

One compiled module serves every Python version from 2.7 to 3.15, including
the free-threaded builds.  Decoding releases the GIL, and objects are not
shared between threads: use one per thread.

## Where to look

- **Examples**: short programs that run as shown; each page includes the
  output of running it when the site was built.
- **Performance**: speed on three real data sets, measured on a stated
  machine and recorded in the repository.
- **Reference**: the API, read from the docstrings, every pipeline step
  value, and the acknowledgements and licences.  This package is mostly a
  regrouping of other people's work: bitshuffle, LZ4, Zstandard, pyFAI's
  matrices and the c2py23 wrappers; please cite them too.

The source is on [GitHub](https://github.com/jonwright/bslz4_to_sparse).
