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

## Real data: WAu5um (2026-10-04)

/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001, Eiger2 4M
u16, frames 400-599 of eiger_0008 (largest file, 303 kB/frame, x29, ~1 %
non-zero) and eiger_0012 (typical, 204 kB/frame, x44, 0.34 %).  Masked
pixels hold 65535; frame 0 == 65535 used as this scan's stand-in for the
fixed mask (special case: 65535 is also saturation / dynamic masking and is
never dropped by the library).  Copies with those pixels set to 0 in
/tmp/$USER/bslz4_bench/WAu5um_eiger*_maskzero.h5.

- Every 4096-px block crosses a module gap: with mask = 65535 no block has
  zero high byte-planes (byte skip never applies).  Real frames have ~10-35
  non-zero px per block (photon noise), not ~4 as the synthetic "sparse".
- Defaults tuned on synthetic data lost 8-13 % on these frames.  Per feature:
  plane extraction loses even at a 16-px limit (now OFF by default); the
  zero-aware lz4 decoder lost up to 6 % at a 24x filter (now 48x); byte skip
  is the clear real-data win with mask = 0 (6-12 %).
- New defaults (avx512cs, byte skip, zero lz4 at 48x; extraction and mask
  planes off), within-run vs the old pipeline: mask = 65535 -4 % .. +3.6 %;
  mask = 0 -2.6 % .. -20 %; synthetic sparse -39 %, medium -10/-34 %, dense
  -15/-58 %.
- Mask in the bitshuffled domain (set_mask_planes, off by default): the
  caller's fixed mask packed into the bit-plane layout once per block per
  batch and AND-ed into the planes after lz4 (blocks with nothing masked
  skip it).  First version had an integer division per 64-byte chunk (~1
  us/block); fixed.  Now: mask = 65535 cut 0 -1..-5 % (it lets the byte skip
  apply), cut 2 ~0, mask = 0 +2..+17 % (pure overhead).  To be enabled only
  when the first frame shows masked pixels hold the dtype maximum.

## Divisions (2026-10-04)

No division per pixel/entry remains in the C/C++ hot loops: block index
i0 / block_elems once per block, sizes once per call, constant /8 -> shifts.
coo(): np.divmod(index, nfast) replaced by a reciprocal multiply (_unravel:
magic = ceil(2**40 / nfast), row = index * magic >> 40; exact for index <
2**26 and nfast < 2**14, checked for every pixel of the 4M/16M shapes),
1.6-1.7x faster; np.divmod fallback beyond.

## Real mask=0 data: decoder copies and the fused collect (2026-10-04)

perf on WAu0012 mask=0 showed ~40 % in libc memmove, called from the zero
decoder.  Counting its copy paths on the real blocks (53585 blocks of 0012):
~6 overlap copies per block averaging 1.2 kB, all repeating a zero pattern
(offset 2) that the decoder did not know was zero -- after a copied match
zero_from was set to op.  Now it is set after the copy's trailing zeros, so
those are skipped (7 skipped matches/block, ~1.1 kB each); short matches
(<= 64 B) are one fixed load+store, zero fills <= 512 B fixed 64 B stores,
short literals a fixed 32 B copy.  Copies may write into `slack` bytes past
the block (the scratch buffer follows raw).  libc calls left: 0.5/block
(0012), 0.1 (0008).  Fuzzed against LZ4_decompress_safe with garbage-filled
output and slack 0/31/64/8192, real blocks as seeds.  The fuzzer also found
that stock accepts a block ending in a match (its literal<=14/match<=18
shortcut skips the end-of-block check); ours rejects it -- stricter, the
same before this change, never written by a compressor.

Fused low-planes transpose + collect (bslz4_lowplanes_collect_u16,
BSLZ4_LOWPLANES_FUSED): u16 blocks with empty high planes are compared,
masked and compressed straight from the GFNI transpose register; the block
is never written.  Plain sparsify writes the >cut pixels to the output; the
CSC sparse route gets the non-zero list (w->precompacted, read through
bslz4_collect_nz_w; not csc-permute-runs, whose sparse route reads the
block).  An early exit for all-zero 64-px groups lost (mispredicts on
photon noise); so did a fully branch-free loop at cut 2; kept: skip a group
when nothing passes, branch-free after that.

WAu mask=0, ms/frame, min of 3 interleaved runs, one core (EPYC 9454):

| case | HEAD 45f830d | + decoder | + fused |
|---|---|---|---|
| 0008 sparsify cut 0 | 1.621 | 1.629 | 1.592 |
| 0008 sparsify cut 2 | 1.015 | 1.034 | 0.854 |
| 0008 csc 1D bbox | 2.580 | 2.605 | 2.551 |
| 0008 csc 2D bbox | 3.255 | 3.299 | 3.242 |
| 0012 sparsify cut 0 | 1.008 | 0.944 | 0.952 |
| 0012 sparsify cut 2 | 0.782 | 0.713 | 0.563 |
| 0012 csc 1D bbox | 1.547 | 1.489 | 1.490 |
| 0012 csc 2D bbox | 1.942 | 1.888 | 1.897 |

(+fused column from a second run against the +decoder build: -17/-21 % at
cut 2, -2..+1 % at cut 0.)  The CSC (cut 0 compaction) gains little: at
cut 0 nearly every 64-px group holds a noise pixel, and the dot itself
dominates.

## CSC cut 0 and the wider check (2026-10-04, later)

perf of the CSC at cut 0 (WAu mask=0): collect ~30 %, dot 30-44 %, decode
~25-30 %.  Changes, each A/B'd on real data:

- Fused collect, two passes (BSLZ4_FUSED_TWOPASS): pass 1 transposes all
  64-px groups and records bytes + selection, pass 2 emits only groups with
  pixels by bit scan.  The one-pass `if (!k)` mispredicted on noise (19-44 %
  of groups hold a pixel at cut 0).  Cut 0: -28..-34 % sparsify, -15..-21 %
  CSC; cut 2 on very sparse frames +4-6 %.  Choosing the mode per block from
  the previous block's hit rate recovered neither.
- Dot prefetch (BSLZ4_DOT_PREFETCH 8): indptr 16 pixels ahead, entries 8
  ahead; the indices[k] load was 43 % of the dot.  -4..-5 %; 16 no gain.
- Staged entries (BSLZ4_DOT_STAGE 128) for blocks averaging 1.5-4 entries
  per pixel: each pixel writes 8 (entry, value) slots and advances by its
  count, then one loop applies them.  9 % frames: 1D bbox x1 -12 %, 2D bbox
  -9 %; other cases flat.  A 4-wide masked walk instead (extra weight-0
  updates) was +8..+15 %; staging for 1 or ~7 entries per pixel lost.
- lz4.c built with -falign-functions=64 -falign-loops=64: unaligned, its
  speed moved ~25 % with the size of code linked before it (one #define in
  the driver did it).  That explained a non-monotonic BSLZ4_LZ4ZERO_RATIO
  sweep; re-swept aligned, 48 stays best (lower: 0008 +3..+10 %).

Committed build (3f7a549) vs this, ms/frame, 4M, one core (selected):

| data | sparsify c0 | 1D bbox x1 | 2D bbox | rings |
|---|---|---|---|---|
| dense 94 % | 1.75 -> 1.79 | 10.0 -> 10.1 | 13.9 -> 14.0 | 5.78 -> 4.76 |
| mid 9 % | 1.86 -> 1.64 | 4.62 -> 3.92 | 5.76 -> 5.13 | 3.06 -> 3.03 |
| sparse 0.1 % | 0.44 -> 0.44 | 0.67 -> 0.64 | 0.88 -> 0.87 | 0.53 -> 0.53 |
| WAu0008 | 1.51 -> 1.01 | 2.35 -> 1.72 | 2.98 -> 2.36 | 1.73 -> 1.22 |
| WAu0012 | 0.95 -> 0.69 | 1.48 -> 1.14 | 1.86 -> 1.50 | 1.11 -> 0.83 |

(The 'mid' and 'dense' columns chain two runs: 3f7a549 -> lz4/twopass/
prefetch, then -> staging.)

## Tried and dropped (2026-10-04)

Measured against the build before each, real and synthetic frames:

- Two-pass collect chosen per block from the previous hit count (1/8,
  1/16, 1/32 of groups): never better than always two-pass; even its
  one-pass mode stayed 2-4 % behind a one-pass build on 0.1 % frames.
  Always two-pass costs 3-8 % on 0.1 %-occupied 16M frames and 4-6 % at
  cut 2 on WAu (0.03-0.1 ms); it saves 0.3-0.5 ms at cut 0 on noisy data.
- Dense CSC route, expanding each pixel's value over its (contiguous)
  entries then one flat loop: 2-3x slower -- 64 B of stores per pixel,
  most pixels having 0-3 entries.  The dense dot runs at IPC ~2.6 (not
  mispredict-bound); its cost is the out[] read-modify-write per entry.
  Only a smaller layout (csc-run, integer weights, ...) cuts it.
- lz4.c at -O3, -mavx2 -mbmi2, or both: within +-2 %.
- A branch-light lz4 decoder (16-byte literal copy, 32-byte match copy
  with a per-offset byte shuffle chosen by mask, careful path for the
  rest): 3x fewer mispredicts, but +6 % (9 % frames) to +20 % (WAu0008):
  16-byte loads of just-written match sources span several earlier stores
  and miss store forwarding.  Stock lz4 on 9 % frames: ~120 sequences per
  8 kB block, literals median 1 B, matches median 6 B, 46 % of offsets
  < 16; IPC ~2.0, a quarter of its time in mispredicts.  Kept in the
  session scratch only.

16M (Eiger2 16M, 50 frames), 3f7a549 -> 437a20b, ms/frame: dense rings
23.0 -> 18.9; 9 % sparsify 6.29 -> 5.53, 1D bbox x1 13.9 -> 12.4, 1D bbox x5
20.5 -> 18.6; 0.1 % frames +2..+10 % (two-pass), all within 0.1 ms.

## Dense frames (94 % non-zero): layout ranking (2026-10-04, d20b557)

tools/bench_suite.py --frames-suffix _dense, beam mid, all core dots, best
of the forced-dense and default routes (they differ by a few %), ms/frame,
one core.  Top layouts vs csc:

| case | 4M csc | 4M best | 16M csc | 16M best |
|---|---|---|---|---|
| 1D no x1 | 7.12 | bsb-csr-nosplit 4.67, csc-nosplit 4.81 | 28.0 | csc-nosplit 18.7 |
| 1D bbox x1 | 10.0 | csc-run 8.89 (int16 8.74) | 39.6 | csc-run 35.1 (int16 34.5) |
| 1D bbox x5 | 20.4 | csc-run 16.9 (int16 16.5) | 81.1 | csc-run 67.1 (int16 65.8) |
| 2D no | 10.4 | csc-nosplit-dump 7.20, csc-nosplit 7.35 | 39.4 | csc-nosplit 26.7 |
| 2D bbox | 14.5 | csc (nothing faster) | 56.9 | csc |
| fazit | 27.7 | csc-permute 8.19 | 112 | csc-permute 32.8 |
| rings | 4.79 | bsb-csr 2.19, padded-sse2 2.66 | 18.9 | bsb-csr 8.01 |
| rings no | 4.67 | bsb-csr-nosplit 2.10 | 18.8 | bsb-csr-nosplit 7.71 |
| rings+mom inter | 4.88 | bsb-csr 2.51, padded-sse2 2.56 | 19.3 | bsb-csr 9.11 |

The int/int16 weights (fixed point) buy only 1-2 % over csc-run.  Floor
for comparison: sparsify alone on these frames 1.8 ms (4M).

## Compilers and PGO (2026-10-04, tools/compiler_suite.py)

Six builds of 1b4d4ed: gcc 13.3, clang 18.1, zig 0.16 (clang 21, baseline
CPU + evex512), each with and without PGO (trained on the benchmark itself,
so a best case).  All pass the tests.  5 datasets x 7 cases, 3 interleaved
rounds, one core (EPYC 9454).  Results in
/tmp_14_days/wright/bslz4_compiler_suite/<host>/summary.txt.

- Geometric mean vs gcc: gcc-pgo 0.991, clang 0.987, clang-pgo 1.025, zig
  0.995, zig-pgo 0.985.  No compiler or PGO is worth more than ~1.5 % overall.
- Median noise between rounds 1.3 %; median spread between builds 6 %:
  per-case differences are real but small.
- Outliers: dense 1D csc-run gcc 8.90 vs gcc-pgo 10.58 (+19 %; PGO made it
  worse); clang-pgo is slowest on most decode-heavy cases (WAu0008 sparsify
  +14-17 %) yet fastest on WAu0012; zig-pgo/gcc-pgo best on WAu sparsify.
- Reading: the timings are mostly a property of the code; the remaining
  build-to-build differences are of the same kind as the lz4 alignment
  effect (where the hot loops land), not missed optimisation.

## Compilers across machines (2026-10-04, tools/compiler_suite.py compare)

Same six builds on each host (p9: gcc 11.4, clang 10, zig; no zig-pgo);
all pass the tests.  Geometric mean vs gcc on that host:

| host | CPU | gcc-pgo | clang | clang-pgo | zig | zig-pgo | spread |
|---|---|---|---|---|---|---|---|
| hpc5-0303 | Xeon Gold 6248 (CLX: AVX-512, no VBMI/GFNI) | 1.060 | 1.008 | 0.977 | 0.996 | 0.972 | 6.8 % |
| hpc6-05 | EPYC 7543 (Zen 3, no AVX-512) | 1.038 | 0.989 | 0.986 | 0.996 | 0.973 | 4.6 % |
| hpc7-01 | EPYC 9454 (Zen 4) | 1.001 | 0.997 | 1.023 | 0.996 | 0.985 | 6.9 % |
| hpc8-61 | EPYC 9655 (Zen 5) | 0.972 | 0.997 | 0.970 | 0.986 | 0.974 | 8.7 % |
| p9-02 | POWER9 | 0.736 | 0.821 | 0.812 | 0.880 | - | 30.1 % |

Fastest build, ms/frame:

| case | hpc5 | hpc6 | hpc7 | hpc8 | p9 |
|---|---|---|---|---|---|
| WAu0012 sparsify cut 0 | 1.99 | 2.03 | 0.64 | 0.48 | 8.74 |
| WAu0012 1D bbox csc | 2.62 | 2.53 | 1.12 | 0.93 | 9.20 |
| WAu0008 2D bbox csc | 5.82 | 4.10 | 2.36 | 1.71 | 13.3 |
| mid 9 % sparsify cut 0 | 5.84 | 4.37 | 1.62 | 1.31 | 18.4 |
| dense sparsify cut 0 | 7.61 | 5.97 | 1.84 | 1.45 | 16.2 |
| dense 1D bbox csc-run | 21.5 | 15.1 | 8.91 | 6.43 | 37.5 |
| sparse 0.1 % sparsify | 1.45 | 1.65 | 0.44 | 0.32 | 8.37 |

- On x86 the compiler is worth <= 3 % overall: the code, not the compiler.
- The CPU features are worth 2.5-4x: hpc5/hpc6 lack AVX-512 VBMI + GFNI,
  so the fused low-planes collect, the GFNI transpose and the avx512cs
  collect all fall back; those fallback paths are also the most compiler
  sensitive (zig-pgo/clang-pgo up to 10-16 % faster than gcc there).
- POWER9: 4-20x slower than Zen 4, and 30 % between builds: the generic C
  paths there depend on the compiler vectorising them.  Even 0.1 % frames
  take 8 ms, so a fixed per-block cost dominates -- needs a profile on p9.

## hpc6 (EPYC 7543, Zen 3, no AVX-512): AVX2 paths (2026-10-05)

perf is locked on hpc6 (perf_event_paranoid 4): profiles from py-spy
--native as the parent of the benchmark (ptrace of a child is allowed).
Before: kcb's AVX2 16-plane bit transpose 20-48 % of sparsify, the avx2
collect (per-pixel bit loop) 27-70 %, no byte skip (it needed VBMI+GFNI).

1. Collect tier 7 "avx2cs" (default after avx512cs): per 8 pixels a
   256-entry lane table packs indices (vpermd) and u16 values (pshufb),
   full-width stores, popcount advance; a 16-group with one pixel keeps the
   bit loop (threshold 1 of 1/2/4 best: no loss on WAu, keeps the dense gain).
2. Found doing this: bslz4_counters had 16 slots per stage but dots reach
   24 -- dots 16-24 counted past the array, into the new lane table (the
   integer dots then gave wrong powders).  32 slots + _Static_assert.
3. Byte skip without VBMI+GFNI: kcb's low planes as bytes, widened.
4. Fused (BSLZ4_FUSED_U8 2): bslz4_lowplanes_collect_avx2 -- kcb's AVX2 bit
   transpose for the 8 low planes, empty planes read from a zero buffer (no
   memset), two passes (record 64-px groups + selections; emit only groups
   with pixels).  Beat kcb + a u8 collect (mode 1) almost everywhere.

hpc6, ms/frame, ce9504c -> b0dd89b+fused:

| data | sparsify c0 | 1D bbox csc | 2D bbox | rings bsb-csr |
|---|---|---|---|---|
| dense 94 % | 6.65 -> 2.41 | 16.9 -> 13.3 | 21.9 -> 18.0 | 6.98 -> 3.22 |
| mid 9 % | 4.37 -> 2.54 | 6.96 -> 5.25 | 8.70 -> 7.00 | 5.90 -> 4.11 |
| sparse 0.1 % | 1.67 -> 0.95 | 1.88 -> 1.16 | 2.14 -> 1.42 | 1.74 -> 1.01 |
| WAu0008 | 2.57 -> 1.53 | 3.45 -> 2.46 | 4.09 -> 3.11 | 2.82 -> 1.78 |
| WAu0012 | 2.07 -> 1.06 | 2.64 -> 1.62 | 3.06 -> 2.03 | 2.26 -> 1.22 |

5. Group skipping (75f5b8d): the planes ORed 32 bytes at a time give a
   bitmap of 64-px groups holding data; pass A transposes/records only
   those, pass B packs only groups with selected pixels.
6. Zero decoder (a2965f9): no perf, so an in-process SIGPROF PC sampler +
   addr2line on a harness of real blocks.  26 % was the byte scan of a
   copied match's trailing zeros, 10 % the 64-byte load of a just-written
   source.  16-byte chunks with a non-zero mask: 1077 -> 781 cycles/block.
   Tried and lost: SIMD literal mask (+3 %), SIMD 255-run length (+6 %).
   Cutoff 32x instead of 48x on hpc6: WAu -8..-24 %, 0.1 % synthetic
   +10..+16 % (on Zen 4 last week 32x lost on WAu0008) -- left at 48x.

hpc6 now (a2965f9), ms/frame vs ce9504c: WAu0012 sparsify 2.07 -> 0.61,
WAu0008 2.57 -> 1.23, mid 4.37 -> 2.45, sparse 1.67 -> 0.58, dense
6.65 -> 2.40; WAu0012 CSC 1D 2.64 -> 1.19, rings 2.25 -> 0.76.

## hpc5 (Xeon Gold 6248, Cascade Lake: AVX-512 F/BW/DQ/VL, no VBMI/GFNI)

The hpc6 AVX2 work runs there unchanged (fused mode 2, collect tier 7):
vs ce9504c, WAu0012 -34..-52 %, WAu0008 -22..-27 %, 0.1 % frames
-29..-41 %, dense -10..-37 %, 9 % frames -8..-19 %.  Tried and dropped,
both against that: vpcompressd packing instead of the lane table (256-bit
+6..+16 %, 512-bit +4..+8 % on WAu/mid); kcb's AVX-512BW mask-add bit
transpose in the fused pass A (256-bit +4..+13 %, 512-bit +-3 % with dense
dots +5..+8 %).  The AVX2 path stays the one for Cascade Lake.

Zero decoder on hpc5 (d3ce41e, 6f9d43d): in the 32-48x band it was 2.2x
faster than stock on WAu blocks but 1.7x slower on synthetic sparse ones
-- ~15 non-overlapping match copies of ~184 B per block took memcpy + a
byte scan for their trailing zeros.  Chunked copies with a last-non-zero
chunk fixed it (4924 -> 2091 cycles/block; stock 2941).  Per-band timing
then showed it beats stock on almost every block, dense ones included (it
never writes the empty high planes: dense 2-8x 2894 -> 1406), so it now
decodes every compressed block (cutoff 48x -> 1).  hpc5 end to end vs 48x:
WAu0008 -17..-36 %, 0.1 % frames -6..-13 %, dense sparsify -15 %, dense
rings -17 %.  Its line profile is flat on Intel (~50 cycles/sequence of
branchy work, no line > 7 %).  Needs confirming on Zen 3/4/5.

## hpc8 (EPYC 9655, Zen 5) check (2026-10-05)

vs last week's build (ce9504c): the zero decoder for every block (6f9d43d)
made 9 % frames 8-30 % slower here -- stock lz4 is ~2.5x faster on Zen 5
than on Cascade Lake and wins on 8-24x blocks (many short sequences:
8-16x 2408 vs 3142 cycles/block).  Fixed by a band (4acc32a): zero decoder
below 8x and above 24x.  Then group skipping on the AVX-512 fused path
(c973137, blocks > 48x).  hpc8 now vs ce9504c (medians, node loaded to 45-95):
WAu0012 sparsify ~0.55 -> 0.42, 0.1 % frames 0.35 -> 0.23, WAu0008
0.85 -> 0.73, 9 % frames level, dense -2..-12 %.

## To do (2026-10-04)

- Confirm the 8x/24x zero-decoder band on hpc7 (Zen 4).  hpc6 (Zen 3,
  2026-10-05, idle node) confirmed it: best or equal everywhere -- vs
  "always", 9 % frames -4..-15 %; vs 48x, WAu0008 -10..-20 %, dense -8 %.
  hpc6 now vs ce9504c: WAu0012 sparsify 2.08 -> 0.55, WAu0008 2.58 -> 1.00,
  9 % 4.38 -> 2.43, 0.1 % 1.67 -> 0.49, dense 6.63 -> 2.15; CSC 1D WAu0012
  2.65 -> 1.12, rings 2.25 -> 0.70.

- First-frame check (Eiger): if the fixed-masked pixels hold the dtype
  maximum, enable mask planes for that dataset.
- Plane extraction: only worth it for nearly empty blocks; maybe a limit of
  ~4 px or drop it.
