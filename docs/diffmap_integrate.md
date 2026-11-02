# Integrating a Bliss mesh scan into an xrdmap / diffmap-viewer file

`tools/diffmap_integrate.py` reads a read-only Bliss master, integrates every
frame of a 2D **mesh** scan into a **2θ powder pattern**, and writes one
**pyFAI-diffmap-layout HDF5** that all three consumers open with defaults:

- **xrdmap** (`RefineMapConfig` + `build_inputs`),
- **diffmap-refine** (`read_map`),
- **pyFAI-diffmap-viewer** (`pyFAI-diffmap-view`).

The integration uses bslz4's sparse matrix product against a pyFAI geometry;
saturated pixels (`value >= exposure * 1e7`) are clipped exactly and listed
under `/entry_0000/pyFAI/clipped_pixels`.

## Setup

Use the `xrdmap` conda python and our bslz4 build:

```bash
export PATH=/cvmfs/hpc.esrf.fr/software/packages/ubuntu24.04/x86_64/xrdmap/latest/bin:$PATH
export PYTHONPATH=$HOME/wright/git/bslz4_to_sparse-dev/lib
```

## Find a mesh scan in a master

A master can hold several scans (`1.1`, `2.1`, `3.1`). The mapping ones are
`amesh`/`dmesh` (two motors). List the titles:

```bash
for e in $(h5ls MASTER.h5 | awk '{print $1}'); do
  echo "$e: $(h5dump -d /$e/title MASTER.h5 2>/dev/null | sed -n 's/.*: *"\(.*\)"/\1/p')"
done
```

Pick the mapping entry (e.g. `2.1`). `amesh`/`dmesh` titles name the two motors
(fast/slow), so nothing else is needed; for a fly scan (`fscan`/`fscan2d`) pass
`--fast <motor> --slow <motor>` (the error message tells you if they are missing).

## Run

```bash
python tools/diffmap_integrate.py \
    /data/.../SAMPLE/SAMPLE_g1/SAMPLE_g1.h5 \      # read-only master
    out/SAMPLE_g1.h5 \                              # .h5 to write
    --entry 2.1 \                                   # the mesh scan
    --poni /data/.../PROCESSED_DATA/CeO2_calib.poni # geometry
    --n-bins 2000 \
    --split full \
    --threads 16
```

Options: `--n-bins` (default 2000), `--split` (`no`/`bbox`/`full`, default `full`:
each pixel contributes to ~2-3 radial bins, so a 2000-bin CSC has ~2-3 entries per
pixel), `--threads` (default `min(16, os.cpu_affinity // 2)`; pinned to distinct
cores unless `--no-pin`), `--batch` (frames per `multi()` call, default 2),
`--mask FILE|auto` (detector mask file, pyFAI: 1 = masked; `auto`/omitted derives a
static mask from the first frame via pyFAI `dynamic_mask`), `--fast`/`--slow`
(fly-scan motor names), `--map-shape rows cols`, `--max-frames N` (debug),
`--input-file FILE::DSET` (force the chunk source, else auto-detected from the VDS),
`--unit` (default `2th_deg`; the radial space is written as `2th`).
A `.poni` copy is written next to the output (used by the raw-frame viewer marker).

## Plan re-use

The first run builds a "--plan" recipe that stores the scan geometry, the pyFAI CSC
matrix, the detector mask and the worker config, so later runs skip the whole pyFAI
setup (no `import pyFAI`, no `integrate1d` CSC build).  By default it is written as
`<poni>_plan.h5` beside the `.poni`; give `--plan PATH` to choose the location (or
to point at an existing recipe, e.g. one in the writable output directory).

```bash
# first run: build everything (and write the recipe)
python tools/diffmap_integrate.py MASTER out/SAMPLE_g1.h5 \
    --entry 2.1 --poni /data/.../CeO2_calib.poni --n-bins 2000 --split full
# later runs: only OUTPUT --plan, every other arg comes from the recipe
python tools/diffmap_integrate.py out/SAMPLE_g1.h5 --plan out/SAMPLE_g1_plan.h5
```

At the end of a run the timing is printed: `setup`, `integrate` (with a live
`frame/s` progress bar) and `shutdown/save`, then `total`.

Placement is from the **measured** per-frame motor values
(`instrument/<motor>/value`), ranked onto the sorted-unique positions, so it is
safe for serpentine/fly-scan orderings and depends only on knowing the two
scanning motors.

### Frame order

The `intensity` cube is stored in **map-grid** order `(slow, fast)` (what the
viewers draw), with `map_ptr[cell]` giving the **file-order frame id** so raw
frames are fetched in acquisition order. To plot frame number directly, the file
also carries the per-frame **acquisition-order** block under
`/entry_0000/pyFAI/result/`:

- `frame` (0..n-1), `slow_value[i]`, `fast_value[i]` = measured motor positions,
- `intensity_frame[i, :]` / `sum_intensity_frame[i, :]` = the powder per frame,

stored as virtual (row-permuted) views — no bytes are duplicated. This is
correct for rasters and serpentine scans alike.

## Open the result

```bash
diffmap-refine out/SAMPLE_g1.h5                      # diffmap-refine
pyFAI-diffmap-view out/SAMPLE_g1.h5                  # pyFAI-diffmap-view
```

For xrdmap, set the config paths to the datasets above (`intensity_flat`,
`2th`, `dim0`, `dim1`), e.g.:

```yaml
filename: out/SAMPLE_g1.h5
intensity_path: /entry_0000/pyFAI/result/intensity_flat
intensity_errors_path: null
ttheta_path: /entry_0000/pyFAI/result/2th
dim0_path: /entry_0000/pyFAI/dim0
dim1_path: /entry_0000/pyFAI/dim1
raw_path: /entry_0000/measurement/images_0001
integrate_config_path: /entry_0000/pyFAI/configuration/integrate_config
```

## Example (verified)

```bash
python tools/diffmap_integrate.py \
  /data/visitor/hc6739/id11/20261009/RAW_DATA/LD163/LD163_g1/LD163_g1.h5 \
  test/demo_LD163.h5 \
  --entry 1.1 \
  --poni /data/visitor/hc6739/id11/20261009/PROCESSED_DATA/CeO2_calib.poni \
  --n-bins 800 --threads 8
```

(`LD163_g1` is `amesh py 38..62 24 pz 38..62 24` → 25×25 = 625 points.)

## Notes

- The master is **read-only**; only `out/*.h5` + `.poni` are written.
- Output `.h5` are gitignored (`test/*.h5`); put demos under `test/` or `out/`.
- `intensity` is stored as the 3D cube `(slow, fast, radial)`; xrdmap's 2D flat
  `intensity_flat` is a **virtual dataset** of that cube, so no bytes are duplicated.
- Saturated clipping uses `wtmax = exposure * 1e7` (from `acq_expo_time`).
