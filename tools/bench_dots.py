"""Compare decode+integrate throughput of the matrix-dot layouts.

Three matrix cases:
  1d    -- a generic 1D binning (each pixel reaches WIDTH consecutive bins), so
           every dot runs (padded/bsb-csr/csc), and the SIMD padded tiers show.
  2d    -- a synthetic 2D radial-major matrix (active pixels reach two
           NON-consecutive bins), so padded is refused; bsb-csr is the general
           fallback.  (This is a strawman -- see `rings` for the real harvest.)
  rings -- the real case-3 G = (rings x azimuth) texture harvest from the
           pyFAI notebook: only ~7-10% of pixels are involved, and only the
           bitshuffle blocks that cross a ring carry entries.  Padded cannot
           represent it; this is where bsb-csr is meant to win.

Reports frames/core/s per dot on the forced-dense and forced-sparse routes,
and (for 2d/rings) the per-bitshuffle-block entry histogram so a
block-crossing-a-ring is never mistaken for "2 reads".

Usage:
  PYTHONPATH=build-lib python3 tools/bench_dots.py [1d|2d|rings]
"""
import os, sys, timeit, tempfile
import numpy as np
import h5py, hdf5plugin

path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if path:
    sys.path.insert(0, path)
import bslz4_to_sparse as b
import scipy.sparse as sp

NI, NJ = 2162, 2068          # Eiger2 4M-ish; ~4.47M pixels
NPIX = NI * NJ
NBINS = 1500
NFR = 8
CASE = sys.argv[1] if len(sys.argv) > 1 else "1d"

print(f"case {CASE}: image {NI}x{NJ} = {NPIX:,} px, nbins={NBINS}, {NFR} frames")

rng = np.random.default_rng(1)


def build_1d():
    WIDTH = 3
    base = rng.integers(0, NBINS - WIDTH + 1, NPIX)
    indptr = np.arange(0, WIDTH * NPIX + 1, WIDTH, dtype=np.int32)
    indices = (base[:, None] + np.arange(WIDTH, dtype=np.int32)[None, :]).reshape(-1).astype(np.int32)
    data = np.random.default_rng(2).uniform(0.2, 1.0, (NPIX, WIDTH)).astype(np.float32)
    data /= data.sum(axis=1, keepdims=True)
    return sp.csc_matrix((data.reshape(-1).astype(np.float32), indices, indptr),
                         shape=(NBINS, NPIX))


def build_2d():
    # Synthetic radial-major 2D: a radial-edge pixel reaches bin d and d+NAZ
    # (non-consecutive), so padded is refused; bsb-csr is the general route.
    NAZ = 20
    yy, xx = np.mgrid[0:NI, 0:NJ]
    rr = np.sqrt((yy - NI / 2) ** 2 + (xx - NJ / 2) ** 2)
    active = ((rr.astype(np.int32) % 6) == 0).ravel()
    rad = (rr.astype(np.int32) // 8).ravel()
    azi = rng.integers(0, NAZ, NPIX)
    img = np.nonzero(active)[0]
    indices = np.concatenate([(rad * NAZ + azi)[img], ((rad + 1) * NAZ + azi)[img]]).astype(np.int32)
    indptr = np.zeros(NPIX + 1, np.int32)
    c = np.zeros(NPIX, np.int32); c[img] = 2; indptr[1:] = np.cumsum(c)
    data = np.concatenate([np.full(img.size, 0.6, np.float32), np.full(img.size, 0.4, np.float32)])
    n = int(rad.max() + 1) * NAZ + NAZ
    return sp.csc_matrix((data, indices, indptr), shape=(n, NPIX)), img


def build_rings():
    """Real case-3 G from the examples/pyfai_csc_integration.ipynb notebook."""
    import pyFAI
    from pyFAI.method_registry import IntegrationMethod
    from pyFAI.calibrant import get_calibrant

    poni = '''# Nota: units are meter and radian
poni_version: 2
Detector: Eiger2_4M
Detector_config: {}
Distance: 0.25
Poni1: 0.081
Poni2: 0.078
Rot1: 0.01
Rot2: 0.02
Rot3: 0.03
Wavelength: 2.85e-11
'''
    tf = tempfile.NamedTemporaryFile("w", suffix=".poni", delete=False)
    tf.write(poni); tf.close()
    try:
        ai = pyFAI.load(tf.name)
    finally:
        os.remove(tf.name)

    NR, NA = 600, 90
    METHOD = ('bbox', 'csc', 'cython')
    def get_engine(dim):
        return ai.engines[IntegrationMethod.select_one_available(METHOD, dim=dim)].engine
    def to_csc(e, n):
        return sp.csc_matrix((e.data, e.indices, e.indptr), shape=(n, len(e.indptr) - 1))

    ones = np.ones(ai.detector.shape, np.float32)
    one2 = ai.integrate2d(ones, NR, NA, method=METHOD)
    M2 = to_csc(get_engine(2), NR * NA)

    q = one2.radial
    lab6 = get_calibrant('LaB6'); lab6.wavelength = ai.wavelength
    qp = np.array(lab6.get_peaks('q_nm^-1'))[:4]
    rings = [(a * (1 - w), a * (1 + w)) for a, w in zip(qp, (0.03, 0.04, 0.05, 0.06))]
    W = np.array([(q >= lo) & (q < hi) for lo, hi in rings], dtype=np.float64)
    G = (sp.kron(sp.csr_matrix(W), sp.identity(NA), format='csr') @ M2).tocsc()
    G = sp.csc_matrix((G.data.astype(np.float32), G.indices.astype(np.int32),
                       G.indptr.astype(np.int32)), shape=G.shape)
    # active pixels = the columns that reach a ring bin
    act = np.diff(G.indptr) > 0
    return G, np.nonzero(act)[0], ai.detector.shape


def block_histogram(csc):
    """(per-bitshuffle-block entry counts) for the u16 8192-byte default."""
    deg = np.diff(csc.indptr)
    be = 8192 // 2
    pix = np.repeat(np.arange(csc.shape[1]), deg)
    blk = np.bincount(pix // be, minlength=(csc.shape[1] + be - 1) // be)
    nz = blk > 0
    s = ("  blocks with entries: %d of %d (%.1f%%); those have min %d, p25 %d, "
         "median %d, p75 %d, p90 %d, max %d entries" %
         (nz.sum(), len(blk), 100 * nz.sum() / len(blk), blk[nz].min(),
          *np.percentile(blk[nz], [25, 50, 75, 90]).astype(int), blk[nz].max()))
    return s


def build_frames(shape):
    mean = np.full(shape, 0.001, np.float32)
    if CASE == "rings":
        # "rings full, background empty": put the signal on the ring pixels
        mean_flat = mean.ravel()
        mean_flat[ACTIVE] = 20.0
        mean = mean_flat.reshape(shape)
    frames = np.random.default_rng(3).poisson(mean, size=(NFR,) + shape).astype(np.uint16)
    fn = "bench.h5"
    with h5py.File(fn, "w") as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + shape,
                              **hdf5plugin.Bitshuffle(nelems=0, cname="lz4", clevel=0))
        bufs = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(frames.shape[0])]
    os.remove(fn)
    return bufs


if CASE == "1d":
    csc = build_1d()
    ACTIVE = None
elif CASE == "2d":
    csc, ACTIVE = build_2d()
else:
    csc, ACTIVE, SHAPE = build_rings()   # SHAPE == (NI, NJ)
    SHAPE = SHAPE or (NI, NJ)

print(f"matrix nnz = {csc.nnz:,}")
if CASE in ("2d", "rings"):
    print(f"  active px = {len(ACTIVE):,} ({len(ACTIVE)/NPIX:.1%})")
    print(block_histogram(csc))
else:
    print("  per-pixel matrices (1 pixels -> WIDTH consecutive bins)")

bufs = build_frames((NI, NJ)) if CASE != "rings" else build_frames(SHAPE)
mask = (np.random.default_rng(4).random((NI, NJ)) > 0.05).astype(np.uint8)
print("backends:", b.available_backends())

DOTS = [n for n in ("csc", "csc-fused", "padded", "padded-sse2", "padded-avx2",
                    "padded-avx512", "bsb-csr") if b.dot_info(n)["available"] == 1]
if CASE == "2d":
    # the synthetic 2D matrix reaches non-consecutive bins, so padded is refused
    DOTS = [n for n in DOTS if not n.startswith("padded")]


def best(fn, repeats=3):
    return min(timeit.repeat(fn, number=1, repeat=repeats))


for route_label, route in (("dense", 1e9), ("default", None), ("sparse", 0.0)):
    if route is None:
        b.set_dense_sparse_threshold(8.0)
    else:
        b.set_dense_sparse_threshold(route)
    print(f"\n=== route {route_label} ===")
    print(f"{'dot':<14} {'ms/call':>9} {'fps/core':>9}  vs csc")
    ref = None
    for name in DOTS:
        integ = b.chunk2sparseCSCmulti(mask, csc, dtype=np.uint16, dot=name)
        integ(bufs, 1)
        t = best(lambda: integ(bufs, 1))
        fps = len(bufs) / t
        if ref is None:
            ref = t; pltxt = "1.00x"
        else:
            pltxt = f"{ref / t:.2f}x"
        print(f"{name:<14} {t*1e3:9.3f} {fps:9.1f}  {pltxt}")

b.set_dense_sparse_threshold(8.0)
