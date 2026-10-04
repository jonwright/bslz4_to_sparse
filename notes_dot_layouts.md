# Notes: matrix-dot layouts, what to clean out (2026-10-03)

Aim: a compact API. Every matrix-dot kernel goes through the one CSC entry,
`sparsify_and_dot(data, indices, indptr)`, and is chosen by a dot id in
`stages[]` (plus option bits). A new kernel is a row in `bslz4_dots[]`
(`bslz4_registry.c`), a leaf in `kernels_generic.cpp`, and a builder in
`__init__.py`. It needs no new c2py entry, driver entry or matrix struct.

## What is wrong with the merged padded / bsb-csr (PR #26, 4febc0d)

Both got their own c2py entry, native entry, driver entry and `bslz4_mat_*`
struct (plan_matrix_v2.md §3.3 "one c2py entry per layout"). That plan
assumed a layout cannot be packed into the three CSC arrays. For padded that
is false.

**padded** is two layouts, chosen silently by `_padded_from_csc`:

- *implicit* (`has.mean() >= 0.5`): `base[npix]` + `weights[npix*W]`, W = max
  entries per pixel, zero padded. This is the intended "single offset +
  npixels x nsplit weights". It fits the CSC entry as `indices = base`
  (n = npix), `data = weights` (n = npix*W), W = data.n / indices.n.
  `row_ptr` here is just `min(b*block_elems, npix)`: redundant.
- *listed* (`has.mean() < 0.5`, e.g. the rings matrix): rows only for pixels
  with entries, plus `pixels` (row -> pixel), `rowmap` (pixel -> row, used by
  the sparse route) and `row_ptr` (first row per block). These are all
  derivable from the CSC `indptr`: `pixels = flatnonzero(diff(indptr))`,
  `rowmap[p] = indptr[p] / W`, `row_ptr[b] = indptr[b*block_elems] / W`. So
  five arrays were never needed. Ported from branch
  `jonwright/padded-csc-api` without that question being asked.
- `base` is clamped to `nbins - W` and the weights shifted inside the row.
  Correct, but another wrinkle.
- Wasted work: every pixel does W multiply-adds. Start+length (`csc-run`)
  does exactly the entries there are, with no padding, and so supersedes the
  listed form.

**bsb-csr** is a transpose (per-block CSR) and keeps the full CSC as well
for the sparse route: 8 arrays plus `block_elems`. It does not pack into
three arrays as it stands. It won the forced-dense route in the notebook run
(2026-10-02), but sparse data rarely takes that route. Decide on real
benchmarks whether it earns an entry.

## The one spec change

`src/bslz4_to_sparse.c` check `indices.n == data.n` encoded "one index per
entry". It is relaxed to `indices.n == data.n or indices.n == mask.n`, so a
dot id can mean "one index per pixel". C cannot see `indices.n`, so a
per-pixel dot given an ordinary CSC whose nnz < npix would read past
`indices`. The Python builders always pass the right arrays, and the CSC
kernels already trust `indptr`'s contents in the same way.

## New dots on the CSC entry (this change)

| id | name | indices | data | indptr | reads per pixel |
|---|---|---|---|---|---|
| 7 | `csc-run` | start bin per pixel (npix) | weights, exactly nnz | standard | indptr x2, start, len weights |
| 8 | `csc-nosplit` | bin per pixel, 0xFFFFFFFF = none | not read (must be 1) | not read | one u32 |

pyFAI checks (Eiger2 4M, dummy.poni, 1500 bins): `no` split is one entry
per pixel, all weights exactly 1.0. `bbox` and `full` are 1-3 entries per
pixel, always consecutive bins (lengths 1/2/3 ~ 3.8k / 3.2M / 1.0M).

## To confirm before removing anything

- [ ] Real-geometry benchmark suite (Eiger2 4M/16M, beam centre middle and
      corner, 1D no/bbox/full, 2D, 8 rings < 10% of pixels, rings + first
      moment). See memory note on real benchmarks.
- [ ] csc-run vs padded (all tiers) on 1D bbox/full and rings. If csc-run
      matches or beats padded, remove the listed form, then padded's own
      entry. A fixed-W SIMD kernel can come back as a dot id on the CSC entry
      (`indices` = base, `data` = npix x W).
- [ ] bsb-csr: keep only if it wins on a real case that matters, then look
      for a packing onto the CSC entry or drop it.
- [ ] Who uses `sparsify_and_dot_padded` / `_bsbcsr` / `_PaddedLayout` /
      `_BsbCSR`: tests (test_padded.py, test_bsb_csr.py, test_layouts.py,
      test_dot.py), the notebook, tools/bench_dots.py, plan_matrix_v2.md.

## Later ideas (not in this change)

- **bsb-csr no-split**: DONE as dot 9 `bsb-csr-nosplit` (bsb-csr entry).
  Dense route: per active bin, sum px[idx[k]] (4 accumulators), no data
  read, no multiply. Sparse route: the nested csc carries one bin per pixel
  (csc-nosplit's loop). Wart: `csc_data` must be an unread npix float32
  array because the bsb-csr spec checks csc_indices.n == csc_data.n.
- Checked in the -O2 object code (gcc 13.3): csc-nosplit's dense and sparse
  loops and bsb-csr-nosplit's dense loop have no mulsd; `data` is not loaded.
- Idea from that object code: integer pixels pay a cvtsi2sd per pixel. A
  histogram over integer dtypes could sum in int64 per bin (bsb-csr-nosplit)
  and convert once per bin.

- Packed lengths (i8/i4) and u16 block-relative starts: blocked by the
  4-byte format check on indices/indptr.
- 2D corner + len1 + len2 tile; walk encoding (start + 2-bit steps).
- Ring first moment: no API change, just extra matrix rows built in Python
  (GQ = G @ diag(q_pixel)). Interleaved [sum I, sum qI] keeps each pixel's
  entries one consecutive run (csc-run, bsb-csr, csc take it); blocked order
  gives two runs per pixel (csc, bsb-csr only).
  - Split (bbox): nothing new needed.
  - No-split: DONE as dot 10 `csc-nosplit-moment` (CSC entry): indices[p] =
    the pair's first bin b, data[p] = q_p (npix), indptr unread. out[b] += v
    (no multiply), out[b+1] += v*q (one multiply). Interleaved order only.
    Checked in the -O2 object code: exactly one mulsd per pixel.
  - Not done: a split moment kernel storing w per entry plus one q per pixel
    (out[2k] += w v, out[2k+1] += w v q) to avoid doubling `data`. It needs
    a fourth array, or q appended to data after nnz (data[indptr[npix] + p]).
  - The c2py doc string for sparsify_and_dot names csc-run and csc-nosplit
    as the per-pixel dots; it does not mention csc-nosplit-moment (left to
    avoid another wrapper regeneration).

## First timings (2026-10-03, one geometry only: notebook dummy.poni, Eiger2 4M)

ms/frame, pinned, 200 frames of synthetic_rings.h5, batches of 25, 1000 bins.
csc matches the 2026-10-02 notebook run (unchanged kernels).

| case | route | csc | csc-run | csc-nosplit | padded | padded-sse2 | bsb-csr |
|---|---|---|---|---|---|---|---|
| 1D no | dense | 8.28 | 7.93 | 5.93 | 9.09 | 7.28 | 6.08 |
| 1D bbox | dense | 10.37 | 9.55 | - | 11.12 | 9.53 | 8.31 |
| 1D full | dense | 10.36 | 9.55 | - | 11.13 | 9.53 | 8.34 |
| rings | dense | 5.88 | 6.98 | - | 3.75 | 3.91 | 3.12 |
| 1D no | default | 4.07 | 3.81 | 3.54 | 3.73 | 3.73 | 4.06 |
| 1D bbox | default | 4.96 | 4.68 | - | 4.05 | 4.05 | 4.96 |
| 1D full | default | 4.98 | 4.68 | - | 4.04 | 4.05 | 4.93 |
| rings | default | 3.82 | 3.77 | - | 3.54 | 3.59 | 3.81 |

Not yet explained (hypotheses to test, not conclusions):
- csc-run loses to padded on 1D split (4.68 vs 4.05 default). Lengths are a
  random mix of 2 and 3 (3.2M / 1.0M), so the variable trip count may cost
  branch mispredicts, and the length comes from two indptr loads before the
  weights can be read. Padded has a fixed trip count and no indptr.
- csc-run loses to csc on rings dense (6.98 vs 5.88): ~92% of pixels are
  empty, and csc-run also reads start[p] for each of them.

## Experimental CSC-entry dots 11-22 (2026-10-03)

All go through `sparsify_and_dot(data, indices, indptr)`; the dot id says how
the arrays are read. Kernels: `src/bslz4_csc_variants.hpp` (new; the older
kernels are untouched). Builders: `src/_csc_variants.py`.

API change, types only (same Python signature and arguments):
- `powder`: 'd', or int64 ('q'/'l') for the fixed-point dots.
- `data`: 'f', or u32 ('I'/'L'), or u16 ('H').
- `indices`: 4-byte ints, or u16 ('H', one per pixel), or a byte stream ('B')
  that starts with a header (magic, block_elems, nblocks, stride, tables);
  the driver checks magic/block size (-109).
- C prototype: `void *powder, const void *weights, const void *indices`.
- Trust: C cannot see array lengths; a stream or u16 array that does not
  match the dot id is the builder's responsibility (as indptr already is).

| id | name | indices | data | notes |
|---|---|---|---|---|
| 11 | csc-nosplit-dump | u32 bin, empty -> nbins | unread | branch free; powder nbins+1 |
| 12 | csc-nosplit-moment-dump | u32 pair start, empty -> nbins | q/pixel | powder nbins+2 |
| 13 | csc-run-u16 | u16 start | f32 | nbins <= 65536 |
| 14 | csc-nosplit-u16 | u16 bin, 0xFFFF none | unread | nbins < 65535 |
| 15 | csc-run-delta | int8 delta / 0x80+abs u32 | f32 | per-block table |
| 16 | csc-nosplit-delta | same, dump bin | unread | |
| 17 | csc-nosplit-walk | 4-bit (dr,da) + exception list | unread | stride inferred |
| 18 | csc-tile | base, a0, l1<<16/l2 per pixel | f32 l1 x l2 | 2D, azimuth wraps |
| 19 | csc-run-moment | u32 start (pair index) | [q, w...] per column | ~1e-7 vs stored w*q |
| 20 | csc-int | u32 per entry | u32 fixed point | int64 powder |
| 21 | csc-run-int | u32 start | u32 fixed point | |
| 22 | csc-run-int16 | u32 start | u16 fixed point | 15 bits for w <= 1 |

Fixed point: b fractional bits = min(fit in the weight type, keep every
int64 bin sum exact for the pixel dtype); each pixel's rounding residual goes
on its largest weight so its weights sum to round(sum w * 2**b). pyFAI
"full" has weights ~-3e-9: clamped to 0; anything below -1e-6 * max refused.

First look (smoke run, 4M mid, 25 frames, dense route, ms/frame):
- 1D no: csc 8.30, nosplit 5.98, nosplit-u16 5.98, nosplit-dump 6.65,
  nosplit-delta 8.75, nosplit-walk 12.41.
- 1D bbox x1: csc 11.26, run 10.21, run-u16 10.52, run-delta 11.98,
  run-int 9.94, run-int16 10.33, csc-int 10.74.
- 2D no: walk stream 0.52 bytes/pixel (0.4 % escapes) vs delta 3.38
  (59 % escapes), yet walk 15.86 vs nosplit 8.40: decoding cost, not traffic.
- 2D bbox: csc-tile 26.24 vs csc 15.74 (no padding, no wrap needed here).
- Dump bin on mostly-empty matrices is a trap: rings+moment no-split
  dense 16.20 (dump) vs 5.91 (sentinel). ~92 % of pixels add into the same
  slot: a store-to-load dependency chain, not just wasted adds.
- Stream decoders (delta/walk) cost the same on the default route as on the
  dense one: they step through every pixel's code to reach the non-zero ones.
- With batches of 25 the block's matrix slice sits in L2: at 4M these loops
  look bound by decode/loop work, not memory traffic. These are first,
  straightforward decoders (one call per pixel, a branch per code); a
  block-at-a-time decode into a small buffer is the obvious next variant.

## FAZIT (2026-10-03)

FAZIT (Peter Boesecke's fast radial regrouping; ImageD11/sandbox/fazit.py
is one implementation of it) is a permutation, not a sum:
lut = argsort(round(r/step) + chi/2pi), each pixel to its own slot, so the
output is the image in (radial bin, azimuth) order. Built here from pyFAI:
radial bin of one pixel width (rArray / pixel1), lexsort((chi, rbin)) over
the unmasked pixels; as a matrix it is no-split, weight 1, nbins = npix.
New dot 23 `csc-permute`: out[bin[p]] = v (store) in the pixel dtype
(powder may be any 1/2/4/8-byte type; the driver takes the output element
size from the pixel dtype when the dot table says out_size 0).

ms/frame (one core):
| | csc-permute (u16 out) | best double-output (csc-nosplit / dump) | csc |
|---|---|---|---|
| 4M mid dense / default | 9.46 / 5.25 | 20.43 / 9.35 | 29.03 / 10.67 |
| 4M corner | 8.81 / 4.72 | 19.85 / 8.76 | 26.52 / 9.94 |
| 16M mid | 36.52 / 17.72 | 79.89 / 32.60 | 115.85 / 37.33 |
| 16M corner | 31.35 / 11.58 | 75.59 / 23.61 | 101.30 / 25.41 |
The double powder for a permutation is 8 bytes per pixel to zero and add
(34 MB/frame at 4M, 135 MB at 16M); csc-permute writes 2 bytes per pixel.
Delta/walk streams are useless here (~60 % escapes: neighbours in an image
row land in different radial bins).

## Single-core bandwidth (tools/membw.c, EPYC 9454, pinned)

L1 32 kB, L2 1 MB per core, L3 32 MB per 6-core CCD. GB/s:
| working set | read | write | copy | triad |
|---|---|---|---|---|
| <= 32 kB (L1) | 170-210 | 105-114 | 125-210 | 120-165 |
| 64-512 kB (L2) | 115-121 | 117-121 | 105-121 | 100-121 |
| 1-8 MB (L3) | 101-104 | 99-104 | 93-98 | 81-97 |
| >= 256 MB (DRAM) | 46-48 | 28 | 34-39 | 32-37 |

## Where the time goes (perf, 2026-10-03; tools/bench_perf.py)

Data sets: sparse (0.1 % active, x104), medium (9 %, x13), dense (x2.7).
Single-core DRAM confirmed by single cold passes over 8 GiB (tools/membw_cold.c):
read 47-48 GB/s, write 28 GB/s. No kernel config exceeds ~2.5 GB/s of
counted DRAM traffic: nothing here is bandwidth bound.

ms/frame, best kernel per case (perf record, cycles:u, grouped by symbol):
| | lz4 | untranspose | non-zero collect | dot | total |
|---|---|---|---|---|---|
| 4M sparse | 0.55 | 0.25 | 0.16 | 0.04-0.15 | 1.0-1.2 |
| 4M medium | 1.15-1.25 | 0.28 | 1.45 | 0.5-2.5 | 3.5-5.5 |
| 4M dense | 1.14 | 0.28 | - | 0.3-8.4 | 1.8-9.8 |
| 16M sparse | 2.2 | 1.0-1.2 | 0.6 | 0.1-0.3 | 4.0-4.5 |
| 16M medium | 3.8 | 1.0-1.2 | 5.2 | 1.7-7.2 | 12-17.5 |
| 16M dense | 9.7 | 1.0 | - | 1.2-33 | 12-44 |
("non-zero collect" = bslz4_collect_avx512_u16, the sparse-route compaction.)

- lz4 is the largest fixed cost (45-80 % of decode); bitshuffle ~1/4 of it.
- Medium data: the compaction costs as much as lz4 (~6 GB/s over the
  decoded block: slow for an AVX-512 scan). IPC 2.1-2.7, 5-8 branch
  misses / kinstr: the sparse route is branch bound.
- Dense data: IPC 4-5, few mispredicts: instruction bound; only fewer
  instructions per pixel (SIMD, simpler loops) helps.
- Open: 16M dense lz4 is 2x slower per pixel than 4M dense (2.2 vs 1.15
  cycles/px, same 2.3 instr/px, same clock 3.76 GHz, same DRAM fills per
  input byte, single NUMA node, not batch size: 10.0 ms at batch 1). Needs
  a C-level lz4 microbench on the same chunks.

## FAZIT tiles and frame-at-a-time (csc-permute, u16 output)

| | 4M dense b25 | 4M dense b1 | 4M default | 16M dense b25 | 16M dense b1 | 16M default |
|---|---|---|---|---|---|---|
| exact | 9.91 | 7.48 | 5.40 | 37.1 | 36.0 | 18.0 |
| 4x4 az | 8.07 | 7.03 | 5.04 | 30.7 | 32.0 | 17.8 |
| 4x4 rad | 8.04 | 6.83 | 5.11 | 30.9 | 29.0 | 18.0 |
| 2x8 az | 8.52 | 7.08 | 5.06 | 32.9 | 33.4 | 17.9 |
- Tiles (stable sort, ~4x4 px bins) help the dense route 10-20 %;
  frame-at-a-time helps 4M (8 MB output fits L3), not 16M (34 MB).
- csc-permute-runs (memcpy per run) loses: runs are only ~3.3 px long.

## lz4: the 8-15 offset slow path (2026-10-03)

- Root cause of 16M dense lz4 being 2x slower per pixel: bitshuffled blocks
  whose top bit-planes are all zero end in one ~5 kB zero run that lz4
  encodes as a match with offset 8..15. lz4 1.10.0 (and dev) copies that 8
  bytes at a time through a store-to-load chain: ~2.6 GB/s vs 12-30 GB/s
  for everything else. Depends on the per-block max (< 64 vs >= 64), not on
  density or detector. Known upstream (#126 2015, #411 2017); PRs #747,
  #1221, #1222 touch this code but none fixes offsets 9-15 (measured: all
  2.6 GB/s on the slow case).
- Patch (doubling copy for offsets 8..15, memset when the pattern is all
  zero): slow case 2.6 -> ~33 GB/s; 16M dense frame 4.1 -> 26.5 GB/s;
  medium frames unchanged; -3 % on Poisson-1000. Pre-zeroing the block costs
  10-20 % on full blocks; a zero-skipping decoder is wrong as written (lz4
  wildcopies write past each copy's end) and slower than memset: dropped.
- Validation of the doubling patch: lz4 full suite, no-fast-loop tests,
  six libFuzzer harnesses (clang ASan/UBSan), targeted stress (5M matches
  with offset 8..15), `make usan`: pass. Combined ASan+UBSan fuzzer reports
  are pre-existing in stock (compressor pointer arithmetic, low-address test).
  gcc-ASan jobs hang in an ASan DEADLYSIGNAL loop at process start here.
- Plan: one commit on a branch from v1.10.0 in the lz4 submodule, a fork to
  push it to (location to be decided by the user), and the same change as
  patches/lz4/*.patch in this repo.

## To do

- DONE (tools/compress_matrices.py): CSC arrays through hdf5plugin lz4,
  bitshuffle+lz4, bitshuffle+zstd (stored/raw, 4M mid; 16M the same):
  1D index per pixel 0.11-0.16; start delta i16 0.055-0.07 (~0.1 B/px);
  CSC indices 0.11-0.20; indptr: lz4 0.95, bitshuffle 0.05-0.21; 2D
  indices 0.27-0.39; rings starts 0.02-0.04. Weights barely compress:
  f32 x1 0.82-1.00, fixed u16 0.88-0.93, x5 ~0.5. With compressed indices
  a split matrix is ~all weights (~7.5 B/px at 1D bbox x1 vs < 1 B/px of
  indices); only coarser weights would cut that. Histograms have no
  weights: the index stream is the whole matrix.
- Re-profile per stage with the patched lz4 in the build.
- Untranspose: when the second half of a u16 block is all zero (pixels <
  256), untranspose the first 4 kB as elem_size 1 and zero-extend.
- c2py23: add runtime CPU flags for the instruction sets the new paths need
  (avx512vbmi, avx512vbmi2, gfni) next to the existing c2py_amd64_* ones.
  The avx512cs collect tier and the u16 low-planes transpose use gcc/clang's
  __builtin_cpu_supports for them today (not available with MSVC);
  popcnt already has c2py_amd64_popcnt (used by plane extraction).

## u16 sparsify for very sparse data: the three steps (2026-10-03/04)

All measured off/on within one run (machine speed drifts +-20 % between
runs), EPYC 9454, one core, u16, batches of 25.

1. Compress-store collect (collect id 6 "avx512cs", default where AVX-512
   VBMI2): one compare per 32 px, skip empty groups, vpcompressw/d into
   registers, plain stores, advance by popcnt. 23-57 % faster when many
   pixels are selected (cut 0 on medium/dense, sparse-route compaction),
   neutral on very sparse data. The old tier 1 used a per-bit scalar loop
   (ctz per selected pixel): branch bound, not popcnt.
2. a) Plane extraction (default on): for blocks > 24x compressed, OR the
      bit-planes into a pixel bitmap; if <= 48 set bits (hardware popcount),
      read the values straight from the planes; no untranspose, no scan.
      Very sparse: ~20-25 % faster. (A software popcount (libgcc) and a
      count-then-abort loop first made it slower; medium data needed the 24x
      filter.)
   b) Byte skip, u16 only (default on, AVX-512 VBMI + GFNI): when the high
      byte-planes are zero, kcb's GFNI bit transpose on the 8 low planes,
      zero-extended straight to u16. Medium/dense u16 5-12 % faster. The
      generic version (untranspose low bytes + separate widen + scalar zero
      scan) was 10-17 % slower and was removed.
3. Zero-aware lz4 block decoder (src/bslz4_lz4zero.h, default on for blocks
   > 24x): tracks the known-zero region; matches whose source lies in it and
   all-zero literal runs are not written; the zero tail is never written and
   *nz_end tells extraction (OR fewer planes) and the u16 transpose (no zero
   scan). Very sparse u16: 4M 0.72 -> 0.42 ms/frame, 16M 2.94 -> 1.41
   (40-52 %); medium 1-3 % faster, dense 2-5 % faster. At an 8x filter medium
   frames were 50-60 % slower (stock lz4's loop is better for many short
   sequences).
   Validated against LZ4_decompress_safe with libFuzzer (ASan+UBSan), seeded
   with real blocks: 113M + 25M + (final run) executions; identical output
   wherever stock decodes, and it rejects whatever stock rejects (lz4's
   end-of-block rules enforced) except offset 0, which stock 1.10.0 accepts
   (lz4 issue #1631) and this decoder rejects.

Net for the focus case (4M very sparse u16 sparsify, cut 0): ~0.92 ms/frame
before step 1 -> ~0.42 ms/frame with all three (same-run comparisons chained).
