# Integrating a Bliss mesh scan into an xrdmap / diffmap-viewer file

`bs_diffmap_integrate` (from `bslz4_to_sparse.conveyor.diffmap_integrate`, run as
`python -m bslz4_to_sparse.conveyor.diffmap_integrate`) reads a read-only Bliss master, integrates every
frame of a 2D **mesh** scan into a **2θ powder pattern**, and writes one
**pyFAI-diffmap-layout HDF5** that all three consumers open with defaults:

- **xrdmap** (`RefineMapConfig` + `build_inputs`),
- **diffmap-refine** (`read_map`),
- **pyFAI-diffmap-viewer** (`pyFAI-diffmap-view`).

The integration uses bslz4's sparse matrix product against a pyFAI geometry;
saturated pixels (above `wtmax = exposure * 1e7`) are clipped to `wtmax` exactly and
listed under `/entry_0000/pyFAI/clipped_pixels`.  It is part of the conveyor, the worked
example: install it with `pip install bslz4_to_sparse[diffmap]`.  The options and their
defaults are in `bs_diffmap_integrate --help`.

## Find a mesh scan in a master

A master can hold several scans (`1.1`, `2.1`, `3.1`). The mapping ones are
`amesh`/`dmesh` (two motors). List the titles:

```bash
for e in $(h5ls MASTER.h5 | awk '{print $1}'); do
  echo "$e: $(h5dump -d /$e/title MASTER.h5 2>/dev/null | sed -n 's/.*: *"\(.*\)"/\1/p')"
done
```

Pick the mapping entry. `amesh`/`dmesh` titles name the two motors (fast/slow) and
their number of positions, so nothing else is needed; for a fly scan (`fscan`/`fscan2d`)
pass `--fast <motor> --slow <motor>`, and `--map-shape rows cols` if the measured
positions alone do not show the grid.

## Run

```bash
# MASTER is read only; OUT.h5 is written, with OUT.poni and OUT_plan.h5 beside it
bs_diffmap_integrate MASTER.h5 OUT.h5 --entry 2.1 --poni CALIB.poni
```

## Plan re-use

Every run that builds the setup writes a recipe ("plan") holding the scan geometry,
the pyFAI CSC matrix, the detector mask and the worker config: to `--plan PATH` if
given, else to `<OUTPUT stem>_plan.h5` beside the output.  A recipe is read only when
`--plan` names a file that exists; the run then skips the whole pyFAI setup, and every
setting comes from the recipe.  A setting also given on the command line must agree
with the recipe's, or the run is refused.

```bash
bs_diffmap_integrate MASTER.h5 OUT.h5 --entry 2.1 --poni CALIB.poni   # writes OUT_plan.h5
bs_diffmap_integrate AGAIN.h5 --plan OUT_plan.h5                       # the same, from the plan
```

## Placement

Placement is from the **measured** per-frame motor values
(`instrument/<motor>/value`), grouped onto the grid lines, so it is safe for
serpentine/fly-scan orderings and for jitter in the measured positions.  The frames
are read from every file of the scan's virtual dataset.

### Frame order

The `intensity` cube is stored in **map-grid** order `(slow, fast)` (what the
viewers draw), with `map_ptr[cell]` giving the **file-order frame id** so raw
frames are fetched in acquisition order. To plot frame number directly, the file
also carries the per-frame **acquisition-order** block under
`/entry_0000/pyFAI/result/`:

- `frame` (0..n-1), `slow_value[i]`, `fast_value[i]` = the motor position of the frame's grid line,
- `intensity_frame[i, :]` / `sum_intensity_frame[i, :]` = the powder per frame,

stored as virtual (row-permuted) views — no bytes are duplicated. This is
correct for rasters and serpentine scans alike.

## Open the result

```bash
diffmap-refine OUT.h5        # diffmap-refine
pyFAI-diffmap-view OUT.h5    # pyFAI-diffmap-view
```

For xrdmap, set the config paths to the datasets above (`intensity_flat`,
`2th`, `dim0`, `dim1`), e.g.:

```yaml
filename: OUT.h5
intensity_path: /entry_0000/pyFAI/result/intensity_flat
intensity_errors_path: null
ttheta_path: /entry_0000/pyFAI/result/2th
dim0_path: /entry_0000/pyFAI/dim0
dim1_path: /entry_0000/pyFAI/dim1
raw_path: /entry_0000/measurement/images_0001
integrate_config_path: /entry_0000/pyFAI/configuration/integrate_config
```

## Notes

- The master is **read-only**; only the output, its `.poni` and its plan are written.
- `intensity` is stored as the 3D cube `(slow, fast, radial)`; xrdmap's 2D flat
  `intensity_flat` is a **virtual dataset** of that cube, so no bytes are duplicated.
