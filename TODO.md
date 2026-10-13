# To do

- **u32 precision and wtmax.** The exact (fixed-point) dot leaves the weights 2^-17 to 2^-22
  for u32 pixels, because the scale must cover the full u32 range; u16 gets 2^-31 or finer.
  Setting wtmax (ImageD11's largest allowed intensity) lower would give finer weights, but needs
  the u32 kernels changed to clip the pixel values inside the dot (the GPU `maxval` argument
  already takes the bound for the weights). Until then wtmax is applied afterwards,
  bit for bit, with `clip_exact_powder` (needs every pixel above wtmax in the sparse output).
- **dummy / dynamic mask.** Kept in the sparse output and removed later. Dropping a pixel
  changes the normalisation per frame: the Integrator would need a per-pixel normalisation
  image, not only the per-bin sums.
- **Ring ranges.** The rings matrix and its ring ranges (2theta windows per Bragg peak, bins per
  azimuth) in the Integrator file.
- **`exact` as the default** for integer pixels: a results change, to be measured first.
- **Trim the kernels at the end.** Among the cases to review: the `csc-nosplit-moment` dot (a
  moment folded into the matrix) is superseded by the stacked matrix of `IntegratorStack` and
  `Integrator.first_moment`; removing it takes the C enum and registry, the pipeline names, the
  layout tests, the benchmark cases and a rerun of the committed benchmark JSON on both machines.
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
- the dot kernels (`dot_*.hpp`, 17 places) read the cut as one number to skip pixels that cannot
  pass; with an image the early exit `bslz4_cut_above_max` would need a per-block maximum;
- `read_cut` keeps a pointer when `ncut == NIJ` and each block gets its slice of the image next to
  its slice of the mask; Python `_cut_array` accepts an image of the frame's shape;
- GPU: `above()` in src/cuda/bslz4_split.cu takes the cut as scalars today; an image is a pointer
  and a per-pixel load;
- tests: the scalar path is the reference for every SIMD variant and the GPU, with images that
  are flat, random and zero, in every pixel dtype.
