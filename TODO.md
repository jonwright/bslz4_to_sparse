# To do

- **u32 precision, maxval and wtmax.** The exact sums are the default for 8 and 16-bit pixels and
  float for 32-bit ones, because the fixed-point weights must leave int64 headroom for the largest
  pixel value: for u32 at its dtype maximum a 1D matrix gets 15 to 17 fractional bits (the
  weights' step 2^-15 to 2^-17, bin sums off by up to a few 1e-6), for 16-bit pixels 31 or 32
  (tools/fixed_point_precision.py prints this for your matrix and data).  `maxval=` gives the
  weights more bits when the data cannot reach the dtype maximum, as a promise: pixels above it can
  overflow the sums silently.  A safe form needs the u32 kernels to clip pixel values inside the
  dot (ImageD11's wtmax); until then wtmax is applied afterwards, bit for bit, with
  `clip_exact_powder` (needs every pixel above wtmax in the sparse output).  Decide whether u32
  data with a known wtmax should be exact by default.
- **dummy / dynamic mask.** Kept in the sparse output and removed later. Dropping a pixel
  changes the normalisation per frame: the Integrator would need a per-pixel normalisation
  image, not only the per-bin sums.
- **Benchmarks on the ID11 example data.** `tools/bench_docs.py` and `tools/bench_kernels.py` still
  use the kevlar and eiger4m data sets and cached local matrices; point them at the two example files
  (CeO2, u32 and dense; silicon, u16 and sparse, in /data/id11/jon/hdftest/bslz4_to_sparse_examples,
  with calib.poni and mask.h5), so that anyone can reproduce the tables of the performance and
  kernels pages on their own computer, and re-measure docs/bench/*.json on both machines (they
  predate the current code, and still list the removed csc-nosplit-moment dot).
- **Dark subtraction and error bars** (pyFAI's variance models): not in the sums.

## Future work: a cut that is an image

Not for 0.0.22.  Since 0.0.22 the cut crosses the C boundary as a buffer of the pixel dtype (one
element, `ncut` = 1 is checked in `read_cut`, src/pipeline/driver.c), which is the room left for
this.

Meaning, decided:

- a cut of one number: `data.flat[j] > cut`, in the pixel type (done);
- a cut that is an image of the pixel type and shape: `data.flat[j] > cut.flat[j]`;
- `howmany` and `thresholds` do not combine with an image.  Their series of cut-offs
  `data.flat[j] - background[j] >= thresholds[k]` would need thresholds that follow the
  background (sqrt(background) * 2**k, say), which is a statistical choice with more than one right
  answer, and low and high signal regions want different behaviour.  Whoever supplies an image
  has the means to choose its own selection, so the combination is refused, not guessed.

What it takes (an estimate from reading the code, nothing written):

- a new collect kernel, not a variant of the scalar one: a per-pixel compare in the generic
  scalar loop, and in each ISA loop (AVX-512, AVX2, SSE2, NEON, VSX) a vector load of the cut beside
  each value vector.  The scalar loops stay as they are for one number;
- the fused u16 low-planes collect compares plane-wise with one number: it is not used with an
  image (an image that is flat and zero is the same as `cut = 0`);
- the dot kernels (`dot_*.hpp`, about 18 places) read the cut as one number to skip pixels that cannot
  pass; with an image the early exit `bslz4_cut_above_max` would need a per-block maximum;
- `read_cut` keeps a pointer when `ncut == NIJ` and each block gets its slice of the image next to
  its slice of the mask; Python `_cut_array` accepts an image of the frame's shape;
- GPU: `above()` in src/cuda/bslz4_split.cu takes the cut as scalars today; an image is a pointer
  and a per-pixel load;
- tests: the scalar path is the reference for every SIMD variant and the GPU, with images that
  are flat, random and zero, in every pixel dtype.
