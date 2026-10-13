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
- **Dark subtraction and error bars** (pyFAI's variance models): not in the sums.
