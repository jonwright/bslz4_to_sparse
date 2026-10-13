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
mask, the pixel type, and for integration the matrix (the codec, lz4 or
zstd, is found from the first chunk decoded unless it is given).  It then
decodes one frame, or a batch of frames, per call.  Inside, decoding is a
pipeline of steps: decompress, apply the mask, undo the bit shuffle, collect
the pixels above the cut, and for integration route each block to a dense
or sparse product with the matrix.  Each step has several implementations
(portable C, and SIMD versions for x86, POWER and ARM); the best available
one is chosen automatically and can be overridden.

One compiled module serves every Python version from 2.7 to 3.15, including
the free-threaded builds (the GPU decoder and the conveyor need Python 3).
Decoding releases the GIL, and objects are not
shared between threads: use one per thread.

## More than sparse pixels

A processor can limit how many pixels a frame returns.  The pixels are chosen as
ImageD11 does it: a frame with too many above the cut moves to the first of a
list of higher thresholds that leaves few enough.  The selection happens inside
the decode, so a crowded frame costs little.

The two halves of the work can also be used alone.  The front half decompresses
chunks, with or without undoing the bit shuffle, without any sparse step and
without HDF5, releasing the GIL.  The back half takes frames that were never
compressed, so every collecting and integrating step can be tested, and used, on
plain images.  The same processing exists on a CUDA GPU behind the same calls
(Python 3 with cupy).

## Moving many files

The decoders do not need anything around them, but a worked example shows them at scale. For a
whole data set the conveyor keeps decoders busy: each worker owns two
buffers and a reader, and the reader fills one with a unit of work (consecutive frames of one file, a whole file when it
fits), in large reads,
while the worker decodes the other.  Where every chunk lies in a file is kept
once in a small index file beside it, so reading never calls HDF5, which takes a
global lock.  A faster worker simply takes more files, and the only
synchronisation is about once per file.  The conveyor is pure Python 3.

Every run reports the compressed bytes read, the bytes of results and the rates.
A guard sink measures the sparse pixels compressed as they would be stored and, for a
file where they are not sparse, does not write them and says so, while the powder and
ring sums of the same frames are written as usual.  A policy states what
the conveyor agrees to write, and results can also be taken in memory as a plain
iterator, which is how to read them without choosing a file format.  For writing there is
one sparse file per input file in ImageD11's layout, with the powder kept, when the decoder
is asked for exact sums, as integer sums and a power-of-two scale, and a small integrator file (geometry, sparse
matrix, normalisation) that turns the sums into pyFAI's intensities.  For texture work there is a ring writer: per frame, for
every ring and azimuth bin, the sum over the ring's window and over a background
window either side, in the frame order of each input file.

## Where to look

- **Examples**: short programs that run as shown; each page includes the
  output of running it when the site was built.
- **Performance**: speed on three real data sets, and the time of every pipeline of
  kernels, measured on stated machines and recorded in the repository.
- **Reference**: the API, the GPU decoder and the conveyor, all read from the
  docstrings, every pipeline step value, and the acknowledgements and licences.  This package is mostly a
  regrouping of other people's work: bitshuffle, LZ4, Zstandard, NVIDIA's nvcomp (the
  optional GPU decoder), pyFAI's matrices and the c2py23 wrappers; please cite them too.

The source is on [GitHub](https://github.com/jonwright/bslz4_to_sparse).
