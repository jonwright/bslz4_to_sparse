# Pipeline case space and minimum API (2026-10-06)

Status: decided and implemented on branch cleanup-licence-api (see §7 for
what was measured after the review and what the code now does).

This note sets out the axes, the steps, the implementations, the measured
winners, the candidates to drop and the proposed API. It changes no code.
The measurements are cited from `notes_dot_layouts.md`, abbreviated **NDL**,
with the § naming the section.

## 1. Axes of the case space

| Axis | Values | Known when |
|---|---|---|
| A. Operation | sparsify only · sparsify + matrix dot | construction (matrix given or not) |
| B. Source sparsity | per block, the compression ratio r: dense r < 8 · mid 8–24 · sparse 24–48 · very sparse > 48 | run time, per block |
| C. Pixel maximum per block | < 256 (high byte-planes empty, u16) · ≥ 256 | run time, per block |
| D. Mask | none (all valid) · masked pixels hold 0 · masked pixels hold the maximum (65535) | construction (+ the first frame for the value) |
| E. Weights | all 1 (no split) · general f32 · fixed-point integer | construction (matrix) |
| F. Entries per pixel | 0/1 · one consecutive run (bbox/full: 1–3) · general | construction (matrix) |
| G. nout vs nin | nout ≪ nin (1D) · ≈ npix/4 (2D) · = npix, one entry each (permutation, FAZIT) | construction (matrix) |
| H. Pixel occupancy | most pixels have entries · few (rings, < 10 %) | construction (matrix) |
| I. Extra outputs | none · q-moment pair per bin | construction (matrix) |
| J. dtypes | pixel: u16 (Eiger), u32 primary; u8, i*, f* rare · weight: f32, u32/u16 fixed point · output: f64, i64 (fixed point), pixel dtype (permute) | construction |
| K. CPU | Zen 4/5 (AVX-512 + VBMI2/GFNI) · Cascade Lake (AVX-512 without VBMI; AVX2 paths faster, NDL §hpc5) · AVX2 (Zen 3, Coffee Lake) · POWER8+ VSX · aarch64 NEON · scalar / MSVC build | import (CPU probe) |

Axes B and C change from block to block, so they are never a pipeline
choice. A step value may *contain* a per-block rule, such as the lz4 band
or the dense/sparse route. Every other axis is fixed when the object is
constructed.

## 2. The steps and the intermediate forms between them

```
compressed block --decode--> planes (bit-shuffled bytes + nz_end) --mask?--> planes
planes --untranspose--> dense block --collect--> pixel list (index, value)
planes ------------- fused untranspose+collect ---> pixel list
dense block / pixel list --dot (dense route / sparse route)--> powder
pixel list --> sparse output (values, addresses)
```

| Step | Input → output | Today selected by |
|---|---|---|
| decode | compressed → planes | the codec (stages[0], from the data) + the option bit LZ4_ZERO + the compile-time band 8×/24× (x86) or 1 (other) |
| mask | planes → planes, or applied in collect | the option bits NO_MASK (auto from mask == 1) and MASK_PLANES (global) |
| untranspose | planes → block | stages[1] backend + the option bit BYTESKIP + CPU probe + `#if` |
| fused untranspose+collect | planes → list | **implicit**: BYTESKIP + u16 + CPU probe picks mode 1 (VBMI/GFNI), 2 (AVX2) or generic/VSX; `#define`s FUSED_U8, LOWPLANES_FUSED, LOWPLANES_SKIP(_RATIO), FUSED_TWOPASS |
| plane extract | planes → list | the option bit PLANE_EXTRACT (global, off) |
| collect | block → list | stages[2] |
| route | per block: dense (dot over the block) or sparse (dot over the list) | the float `dense_sparse_x` (global 8.0, passed per call) |
| dot | block/list → powder | stages[3] **and** `dot=`, separately; the layout built from `dot=` |

## 3. Implementations, measured winners, proposed ids

The proposed ids put the best measured first: 0 = auto, 1 = best. An id
marked † is a drop candidate (§4).

### decode
| Proposed | Today | Wins where | Ref |
|---|---|---|---|
| 1 lz4-band | LZ4_ZERO + band 8×/24× | x86: Zen 3/4/5, Coffee Lake (stock lz4 is better at 8–24×) | NDL §hpc8, §Windows 4, §To do |
| 2 lz4-zero | LZ4_ZERO, ratio 1 | POWER9 (stock slower in every band); Cascade Lake | NDL §POWER9 4, §hpc5 |
| 3 lz4-stock | no LZ4_ZERO | nowhere as a whole; it is the middle band of 1 | — |
| — zstd | codec from data | the only zstd decoder | — |

Selecting lz4-* on zstd data is an error. Auto picks 1 on x86 and 2
elsewhere. Cascade Lake: the band was not re-measured after the band went
in (gap).

### mask
| Proposed | Today | Wins where | Ref |
|---|---|---|---|
| 1 per-pixel | default | mask = 0 data (pure overhead otherwise: +2..+17 %) | NDL §Real data WAu |
| 2 planes | MASK_PLANES | masked pixels = 65535: −1..−5 % at cut 0 (lets the byte skip apply) | same |
| 3 none | NO_MASK | mask all ones (auto today) | — |

Auto: 3 if the mask is all ones, else 1. Choosing 2 needs the first frame
(axis D), so it stays an explicit choice for now.

### untranspose (and fused untranspose+collect)
| Proposed | Today | Wins where | Ref |
|---|---|---|---|
| 1 lowplanes-vbmi | fused mode 1 (+ group skip > 48×) | Zen 4/5, u16 | NDL §CSC cut 0, §hpc8 |
| 2 lowplanes-avx2 | fused mode 2 | Zen 3, Cascade Lake, Coffee Lake, u16 | NDL §hpc6 4, §hpc5 |
| 3 lowplanes-vsx | FUSED_GENERIC + VSX | POWER8+, u16 | NDL §POWER9 2–3 |
| 4 lowplanes-generic | FUSED_GENERIC, SWAR collect | aarch64 (not measured: gap) | — |
| 5 kcb | backend 0, no byte skip | every dtype other than u16, and u16 blocks ≥ 256 (the fallback inside 1–4) | — |
| 6 scalar | backend 3 | — (reference) | — |
| 7 neon | backend 2 (bitshuffle's NEON) | **keep**: kcb has no NEON code, so on aarch64 this is the only SIMD transpose (aarch64 not measured: gap) | — |
| † sse | backend 1 (bitshuffle's SSE2) | kcb dispatches SSE2/AVX2/AVX-512 itself | — |
| (part of 1/2) byteskip into a block | `bslz4_untranspose_block` with BYTESKIP | **still used**: dot objects on the dense route (planes → dense block, low planes only, widened). Not a drop candidate; it is the planes → block form of 1/2 | NDL §u16 three steps 2b |
| † FUSED_U8=1 | kcb bytes + avx2cs collect | lost to mode 2 "almost everywhere" | NDL §hpc6 4 |
| † one-pass fused | FUSED_TWOPASS=0 | lost (−28..−34 % at cut 0 for two-pass) | NDL §CSC cut 0 |
| † plane-extract | PLANE_EXTRACT | loses on real data even at 16 px | NDL §Real data WAu |

Values 1–4 are "planes → list" for u16 and fall back to 5 per block.
Validation: 1–4 need u16 and the ISA. The fallback is a per-block rule
inside the value, so it does not break "no silent ISA fallback".

### collect (block → list; used when the fused step does not apply)
| Proposed | Today id | Wins where | Ref |
|---|---|---|---|
| 1 avx512cs | 6 | Zen 4/5 (23–57 % over avx512 at many hits) | NDL §u16 three steps 1 |
| 2 avx2cs | 7 | Zen 3, Cascade Lake (beat vpcompress there), Coffee Lake | NDL §hpc6 1, §hpc5 |
| 3 vsx | 4 | POWER | — |
| 4 neon | 5 | aarch64 | — |
| 5 sse2 | 3 | fallback for old x86 without AVX2 | — |
| 6 scalar | 0 | every dtype; the only choice for non-u16/u32 | — |
| † avx512 | 1 | superseded by avx512cs | NDL §u16 three steps 1 |
| † avx2 | 2 | superseded by avx2cs | NDL §hpc6 |

### route (dot objects only)
| Proposed | Today | Note |
|---|---|---|
| 1 auto | ratio > 8 → sparse | per block |
| 2 dense | threshold = ∞ | benchmarks; forced dense |
| 3 sparse | threshold = 0 | tests |

### dot (dot objects only); each id's matrix condition is checked at construction
| Proposed | Today | Condition (validation) | Wins where (4M, ms/frame) | Ref |
|---|---|---|---|---|
| 1 csc | 0 | any | 2D split: nothing faster (14.5) | NDL §Dense |
| 2 csc-run | 7 | each pixel's entries are one consecutive run | 1D bbox/full (8.89 vs 10.0) | NDL §Dense |
| 3 csc-nosplit | 8 | ≤ 1 entry per pixel, weight 1 | 1D/2D no split (4.81 vs 7.12) | NDL §Dense |
| 4 bsb-csr | 6 | any (built per block) | rings: few occupied pixels (2.19 vs 4.79) | NDL §Dense |
| 5 bsb-csr-nosplit | 9 | as 3 | rings no split (2.10) | NDL §Dense |
| 6 csc-permute | 23 | nout = npix, one entry per pixel, weight 1, unique bins; output in the pixel dtype | FAZIT (8.19 vs 27.7) | NDL §FAZIT |
| 7 csc-nosplit-moment | 10 | pairs (b, b+1) with weights (1, q) | rings + moment no split | NDL §Later ideas |
| 8 csc-fused | 1 | any | — (dense route + sparse output in one pass; not benchmarked separately: gap) | — |
| † padded ×4 | 2–5 | run of ≤ W | beaten by bsb-csr on rings and by csc-run on 1D dense; won 1D split on the *default* route in the first timings only (4.05 vs 4.68, NDL §First timings), before the staging change. **Re-measure before dropping** | |
| 9 csc-int, 10 csc-run-int, 11 csc-run-int16 | 20–22 | integer pixels; weights ≥ −1e-6·max; fixed point, int64 output | **keep** (2026-10-06): exact integer sums, so sparse pixels can later be subtracted from the integration without loss; also 1–2 % over csc-run | NDL §Dense |
| † csc-nosplit-u16, csc-run-u16 | 13, 14 | nbins < 65535 | equal to the u32 forms | NDL §Experimental |
| † csc-nosplit-dump, moment-dump | 11, 12 | | 2D no split 7.20 vs 7.35 (2 %); a trap on mostly empty matrices (16.2 vs 5.9) | NDL §Experimental, §Dense |
| † delta, walk, tile | 15–18 | | all slower than csc / nosplit | NDL §Experimental |
| † csc-run-moment | 19 | | not shown to beat csc-run on the interleaved matrix | NDL §Experimental |
| † csc-permute-runs | 24 | | loses (runs ~3.3 px) | NDL §FAZIT tiles |

**Auto rule for dot** (today auto is always csc), with the matrix analysis
in order:
1. Permutation → 6.
2. No split → occupancy < ~10 % ? 5 : 3.
3. Moment pairs with no split → 7.
4. Consecutive runs → occupancy < ~10 % ? 4 : 2.
5. Otherwise → occupancy < ~10 % ? 4 : 1.

The 10 % cut-off is a guess between the rings (8 %) and the full-detector
cases. It needs measuring: a gap.

## 4. Drop candidates (the user decides)

Decided 2026-10-06: keep csc-permute and the integer dots (20–22); drop the
other experimental dots (11–19, 24). With that:
- Dot ids go from 25 to 11.
- Collect tiers go from 8 to 6.
- The untranspose backend sse goes (neon stays).
- The option bits go: all six (DROP_NEGATIVES was never implemented).
- `#define`s whose losing branch goes: FUSED_U8, FUSED_TWOPASS,
  LOWPLANES_FUSED, EXTRACT_MAX/RATIO.
- `#define`s that become untranspose values 2–4 instead of switches:
  BYTESKIP_GENERIC (also needed on the dense route), FUSED_GENERIC,
  FUSED_VSX.
- `#define`s that stay as named constants: LZ4ZERO_RATIO/DENSE,
  LOWPLANES_SKIP_RATIO, DOT_PREFETCH, DOT_STAGE.
- Files: `src/_csc_variants.py` and `src/bslz4_csc_variants.hpp` shrink to
  permute and the fixed-point dots. `src/bslz4_padded.hpp` goes if padded
  goes.

## 5. Gaps to measure before deciding

1. padded vs csc-run, on the default route, after the staging change (1D bbox/full, 4M/16M).
2. The decode band on Cascade Lake (2 vs 1) and Zen 4 (hpc7, still open from NDL §To do).
3. The occupancy cut-off for bsb-csr vs csc / csc-run / csc-nosplit.
4. csc-fused vs csc.
5. aarch64: nothing measured.

## 6. Minimum API (proposal, revised 2026-10-06 after review)

The 0.0.16 names and call signatures stay:

```python
c = bslz4_to_sparse.chunk2sparse(mask, dtype=np.uint16, pipeline=None)
npx, (vals, adr) = c(chunk, cut);  c.coo(chunk, cut)
d = bslz4_to_sparse.chunk2sparseCSC(mask, csc, dtype=np.uint16, pipeline=None)
npx, (vals, adr), powder = d(chunk, cut)
bslz4_to_sparse.bslz4_to_sparse(ds, num, cut, mask=None, pixelbuffer=None)
c.pipeline                     # set on creation: resolved uint16[6], no zeros
bslz4_to_sparse.describe(pipeline)   # step names for a pipeline (or the
                                     # options this CPU and build offer)
detect_codec(ds); harvest_chunk_offsets(ds); build_info()
```

- **`pipeline`:** `None`, or 6 ints `[decode, mask, untranspose, collect,
  route, dot]`, each 0 = auto, or a dict of names (`{"collect": "avx2cs"}`)
  where missing keys are auto. No bitfields. `pipeline` is the only new
  keyword. `codec` stays as it was.
- **Batches of frames (decided 2026-10-06):** a `.multi(chunks, cut)`
  method on `chunk2sparse` and `chunk2sparseCSC`. It returns per-frame
  `npx`, `(vals, adr)` arrays of shape (nframes, npix) and, for CSC,
  `powder` (nframes, nout). The unreleased `chunk2sparseMulti` /
  `chunk2sparseCSCmulti` classes go. How chunks are fed (bytes objects,
  or one buffer with offsets and lengths, as `harvest_chunk_offsets`
  gives) is open: `.multi` can take either.
- **The codec** comes from the data. lz4-* on zstd data → error.
- **Errors** (`ValueError`) give the step, the value, the reason and the
  alternatives. For example: `pipeline[collect]=avx512cs: this CPU lacks
  AVX512_VBMI2 (available: avx2cs, sse2, scalar)`.
- **Removed (never in a release):** `set_*`/`get_*`, `pack_pipeline`,
  `dot=` (now the pipeline's dot step), `available_dots`, `dot_info` and
  `available_backends` (merged into `describe`). The counters and
  `impl_available` move to `bslz4_to_sparse._testing`. `npbuf` (0.0.16)
  stays removed.
- **Per object only:** two objects with different pipelines can run in
  two threads.


## 7. Decided and measured (2026-10-06)

Gaps of §5, measured on gpid11-nice02 (EPYC 9454, Zen 4), 4M frames, two
rounds, every powder checked against csc (scratch gaps.py):

- decode band (Zen 4): keep 24.  Zero decoder for every block: 9 % frames
  +16..+21 %; ratio 48: WAu0008 +17..+20 %.  Cascade Lake still open.
- 1D bbox, default route: padded best on mid/sparse/real data at every
  occupancy (9 % frames 3.13 vs csc 3.93 vs csc-run 4.07; WAu0012 0.79 vs
  0.99 vs 0.91); bsb-csr best only on dense frames at occupancy <= 75 %
  (dense occ 8 %: 2.34 vs padded-avx2 2.52 vs csc 5.54).  So padded stays;
  padded-sse2 / padded-avx512 equal padded-avx2 and go.
- 1D no split: bsb-csr-nosplit never slower than csc-nosplit, 2x on dense
  frames with few occupied pixels (occ 8 %: 1.93 vs 4.08).
- csc-fused: never faster than csc, dense 14.2 vs 9.9: dropped.
- csc-run: not strictly best in this set (1D x5 not measured); kept.

Dropped: dots csc-fused, padded-sse2, padded-avx512, 11-19 and 24 (kept:
csc-permute and the integer dots, for lossless subtraction of sparse
pixels); collect tiers avx512, avx2 (avx2cs keeps avx2's u32 kernel); the
SSE untranspose backend; plane extraction; FUSED_U8 modes 0/1; the one-pass
fused collect; every option bit and global; dot=; set_*/get_*;
pack_pipeline; available_*; dot_info; chunk2sparseMulti/CSCmulti (now the
.multi method); tools/bench_dots.py, bench_real_wau.py, compress_matrices.py.

The pipeline is uint16[6] = decode, mask, untranspose, collect, route, dot
(src/pipeline/pipeline.h, src/_pipeline.py); 0 = auto per step; route
values are ratio / dense / sparse.  Auto dot (src/_matrix.py auto_dot):
histogram -> bsb-csr-nosplit; moment pairs -> csc-nosplit-moment; one run
of <= 8 bins -> padded(-avx2); longer runs -> csc-run; occupancy < 50 % ->
bsb-csr; else csc.  Auto never picks csc-permute or the integer dots (they
change the output type).

Source layout: src/pipeline/ (common.h, pipeline.h, registry.c, driver.c)
and src/steps/ (decode.h + lz4zero.h, mask.h, lowplanes.h +
lowplanes_avx2.hpp, untranspose.h, collect.hpp, dot_csc/padded/bsbcsr/
fixed.hpp, caps.h, kernels.cpp).  Templates only for the pixel types, with
explicit instantiations in kernels.cpp; the macro generators
(BSLZ4_DTYPE_KERNELS, BSLZ4_LOWPLANES_GROUP, BSLZ4_COLLECT_LOW,
BSLZ4_PADDED_DENSE_CASE, BSLZ4_CUT_ABOVE_MAX, BSLZ4_PREFETCH) became inline
functions or explicit switches.

### Known limit: bsb-csr and large decode blocks (2026-10-07)

bsb-csr and bsb-csr-nosplit hold each entry's position in its decode block
as uint16, so they need blocks of at most 65536 pixels.  Larger blocks (256 kB
and 1 MB bitshuffle blocks were tried) are refused with a ValueError when the
layout is built for that block size (src/_matrix.py BSB_CSR_MAX_BLOCK_ELEMS),
and the C entry checks it too.  Every other dot, and every sparsify step, is
correct at those sizes (test/test_block_sizes.py).  Auto picks bsb-csr and
bsb-csr-nosplit from the matrix alone, before a block size is known, so on
such data pass pipeline={"dot": ...} explicitly.

To do, when large blocks need bsb-csr: add a builder at a divisor of the
decode block (e.g. 65536 pixels), with the dot walking the layout blocks of
each decode block.  Same memory as today.  A uint32 idx was rejected: +33 %
memory and dot bandwidth for the default block size too.
