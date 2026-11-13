"""The kernel inventory: one row per value of every pipeline step.

What each kernel needs, what it gives, and what it costs, in one table, so that choosing a
kernel can become: keep the rows whose requirements hold, keep those with the output contract
the caller asked for, then pick by the cost on this machine.  Nothing chooses from it yet: the
automatic choices are still _pipeline.auto_step, _matrix.auto_dot, the fixed-point default in
chunk2sparseCSC and the conveyor's _prepared_matrix (see test/test_inventory.py, which checks the
rows against the code).

A row (a dict) has:

  step, name     the pipeline step and value name, as in _pipeline.VALUES (same order)
  cpu            instruction sets it needs, or None (portable); available() decides at run time
  dtypes         pixel dtypes it takes (names as _pipeline.DTYPE_INDEX), None = any
  codec          "lz4", "zstd" or None (any)
  matrix         dots only: what the matrix must be (see MATRIX), else None
  contract       dots only: what the powder is (see CONTRACTS), else None
  routes         dots only: the routes it has ("dense", "sparse")
  reads          dots only: per route, what one pixel (dense: every pixel of the block; sparse:
                 every listed non-zero pixel) reads and writes, as (array, bytes, indexed by,
                 access); see the dot kernels in src/steps/dot_*.hpp
  matrix_bytes   dots only: a function of the matrix sizes (see sizes()) giving the bytes of the
                 arrays the dot reads: its working set beside the powder (nbins * 8 bytes)
  scales_with    non-dot steps: what their work grows with, per block
  tuning         settings inside the kernel (constants in the C sources) with their values
  source         where the kernel is
  measured       per machine profile, filled by the benchmarks (empty here)

Access: "gather" = at a place given by the pixel list (random in the matrix on the sparse
route), "stream" = in pixel order, "run" = consecutive from a gathered start, "local" = in the
8 kB decode block or in registers.  Nothing in this file is a measurement.  Where each kernel's
loops are in the source, and what each is paid per, is tools/kernel_loops.py (for the timing
model; docs/perf_model.md).
"""

# what a matrix must be for a dot (checked against _matrix.analyse(), and the matrix type)
MATRIX = {
    "any": "any CSC matrix with float weights",
    "runs<=64": "each pixel's bins one ascending run of consecutive bins, at most 64 wide",
    "runs": "each pixel's bins one ascending run of consecutive bins",
    "nosplit": "at most one bin per pixel, weight exactly 1 (a histogram)",
    "permutation": "nosplit, and each bin used by one pixel at most",
    "fixed32": "a fixed_point_matrix with u32 weights",
    "fixed32-runs": "a fixed_point_matrix with u32 weights whose pixels' bins are runs",
    "fixed16-runs": "a fixed_point_matrix with u16 weights (wbits=16) whose pixels' bins are runs",
}

# what the powder of one frame is
CONTRACTS = {
    "float64": "float64 sums of float32 weight x pixel",
    "float64-of-int64": "int64 sums of the pixels (weight 1), cast to float64 once per frame: "
                        "exact below 2**53",
    "int64-scaled": "int64 sums of fixed-point weight x pixel, exact (modulo 2**64); "
                    "divide by powder_scale (2**b) for float",
    "pixels": "the pixel values re-ordered, in the pixel dtype (not a sum)",
}

INT = ("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64")
INT32 = ("u8", "u16", "u32", "i8", "i16", "i32")


def sizes(nm, block_elems=4096):
    """The sizes the matrix_bytes functions take, from a mask-folded _matrix._NormalMatrix."""
    import numpy as np
    n = np.diff(nm.indptr.astype(np.int64))
    occupied = int((n > 0).sum())
    return {"npix": int(nm.npix), "nnz": int(n.sum()), "nbins": int(nm.nbins),
            "width": max(int(n.max()) if n.size else 0, 1), "occupied": occupied,
            "listed": occupied < 0.5 * nm.npix, "block_elems": int(block_elems)}


def _csc_bytes(wbytes):
    return lambda s: 4 * (s["npix"] + 1) + 4 * s["nnz"] + wbytes * s["nnz"]


def _run_bytes(wbytes):
    return lambda s: 4 * (s["npix"] + 1) + 4 * s["npix"] + wbytes * s["nnz"]


def _nosplit_bytes(s):
    return 4 * s["npix"]


def _padded_bytes(s):
    rows = s["occupied"] if s["listed"] else s["npix"]
    extra = 4 * s["npix"] + 4 * rows if s["listed"] else 0   # rowmap, pixels
    return rows * (4 * s["width"] + 4) + extra


def _bsbcsr_bytes(s):
    # idx (u16) + data (f32) per entry, about one (bin, bin_ptr) per entry at most, + nested csc
    return 6 * s["nnz"] + 8 * s["nnz"] + _csc_bytes(4)(s)


def _bsbcsr_nosplit_bytes(s):
    return 2 * s["nnz"] + 8 * s["nnz"] + _nosplit_bytes(s)


# per pixel reads on each route: (array, bytes, indexed by, access); e = entries of the pixel,
# W = the padded width.  "out" is the powder (read-modify-write unless said).
_CSC_SPARSE = (("indptr", "8", "pixel", "gather"), ("indices", "4e", "entry", "run"),
               ("data", "4e", "entry", "run"), ("out", "8e", "bin", "gather rmw"))
_CSC_DENSE = (("indptr", "4", "pixel", "stream"), ("indices", "4e", "entry", "stream"),
              ("data", "4e", "entry", "stream"), ("out", "8e", "bin", "near rmw"))


def _run_reads(wbytes, route):
    acc = "gather" if route == "sparse" else "stream"
    return (("indptr", "8" if route == "sparse" else "4", "pixel", acc),
            ("start", "4", "pixel", acc), ("data", "%de" % wbytes, "entry", "run"),
            ("out", "8e", "bin", "run rmw"))


def _nosplit_reads(route, out="8"):
    acc = "gather" if route == "sparse" else "stream"
    return (("bin", "4", "pixel", acc), ("out", out, "bin", "gather rmw" if route == "sparse"
                                         else "near rmw"))


def _row(step, name, cpu=None, dtypes=None, codec=None, matrix=None, contract=None,
         reads=None, matrix_bytes=None, scales_with=None, tuning=None, source=None):
    return {"step": step, "name": name, "cpu": cpu, "dtypes": dtypes, "codec": codec,
            "matrix": matrix, "contract": contract,
            "routes": tuple(sorted(reads)) if reads else None, "reads": reads,
            "matrix_bytes": matrix_bytes, "scales_with": scales_with,
            "tuning": tuning or {}, "source": source, "measured": {}}


KERNELS = [
    # ---- decode: compressed block -> bit-planes ---------------------------------------------
    _row("decode", "lz4-band", codec="lz4",
         scales_with=("compressed bytes", "zero runs"),
         tuning={"zero-aware below/above": "8x / 24x compression (stock lz4 between)"},
         source="src/steps/decode.h, lz4zero.h"),
    _row("decode", "lz4-zero", codec="lz4", scales_with=("compressed bytes", "zero runs"),
         source="src/steps/lz4zero.h"),
    _row("decode", "lz4-stock", codec="lz4", scales_with=("decompressed bytes",),
         source="lz4/ (LZ4_decompress_safe)"),
    _row("decode", "zstd", codec="zstd", scales_with=("decompressed bytes",), source="zstd/"),
    _row("decode", "pixels", scales_with=(),
         source="no decode: from_frames on uncompressed frames"),
    # ---- mask ---------------------------------------------------------------------------------
    _row("mask", "pixel", scales_with=("pixels collected",), source="src/steps/mask.h"),
    _row("mask", "planes", scales_with=("bit-planes",),
         source="src/steps/mask.h (wins only if masked pixels hold the maximum)"),
    _row("mask", "none", scales_with=(), source="no mask tests"),
    # ---- untranspose: bit-planes -> pixels --------------------------------------------------
    _row("untranspose", "lowplanes-vbmi", cpu="avx512vbmi vbmi2 gfni", dtypes=("u16",),
         scales_with=("bit-planes in use",), source="src/steps/lowplanes.h"),
    _row("untranspose", "lowplanes-avx2", cpu="avx2", dtypes=("u16",),
         scales_with=("bit-planes in use",), source="src/steps/lowplanes_avx2.hpp"),
    _row("untranspose", "lowplanes-vsx", cpu="power8 vsx", dtypes=("u16",),
         scales_with=("bit-planes in use",), source="src/steps/lowplanes.h"),
    _row("untranspose", "lowplanes-c", dtypes=("u16",), scales_with=("bit-planes in use",),
         source="src/steps/lowplanes.h"),
    _row("untranspose", "kcb", scales_with=("bit-planes",),
         source="kcb/ (SSE2/AVX2/AVX-512 dispatched inside)"),
    _row("untranspose", "neon", cpu="aarch64 neon", scales_with=("bit-planes",),
         source="bitshuffle/ NEON kernels"),
    _row("untranspose", "scalar", scales_with=("bit-planes",), source="bitshuffle/ scalar"),
    # ---- collect: pixels -> (index, value) above the cut -------------------------------------
    _row("collect", "avx512cs", cpu="avx512bw avx512vl vbmi2", dtypes=("u16", "u32"),
         scales_with=("pixels", "pixels kept"), source="src/steps/collect.hpp"),
    _row("collect", "avx2cs", cpu="avx2", dtypes=("u16", "u32"),
         scales_with=("pixels", "pixels kept"), source="src/steps/collect.hpp"),
    _row("collect", "vsx", cpu="power vsx", dtypes=("u16", "u32"),
         scales_with=("pixels", "pixels kept"), source="src/steps/collect.hpp"),
    _row("collect", "neon", cpu="aarch64 neon", dtypes=("u16", "u32"),
         scales_with=("pixels", "pixels kept"), source="src/steps/collect.hpp"),
    _row("collect", "sse2", cpu="sse2", dtypes=("u16", "u32"),
         scales_with=("pixels", "pixels kept"), source="src/steps/collect.hpp"),
    _row("collect", "scalar", scales_with=("pixels", "pixels kept"),
         source="src/steps/collect.hpp"),
    # ---- route: per block, the dot over every pixel (dense) or the non-zero list (sparse) ----
    _row("route", "ratio", tuning={"sparse when compressed more than": "8x (BSLZ4_ROUTE_RATIO)"},
         source="src/pipeline/driver.c"),
    _row("route", "dense", source="src/pipeline/driver.c"),
    _row("route", "sparse", source="src/pipeline/driver.c"),
    _row("route", "bpp", tuning={"sparse below": "2 compressed bits per pixel (BSLZ4_ROUTE_BPP), "
                                                 "commit 242185a"},
         source="src/pipeline/driver.c"),
    # ---- dot: the matrix product ------------------------------------------------------------
    _row("dot", "csc", matrix="any", contract="float64",
         reads={"sparse": _CSC_SPARSE, "dense": _CSC_DENSE}, matrix_bytes=_csc_bytes(4),
         tuning={"sparse prefetch distance": "8 pixels (BSLZ4_DOT_PREFETCH)",
                 "sparse staging": "128 pixels, when a block averages 1.5-4 entries per pixel "
                                   "(BSLZ4_DOT_STAGE)"},
         source="src/steps/dot_csc.hpp"),
    _row("dot", "padded", matrix="runs<=64", contract="float64",
         reads={"sparse": (("rowmap (listed)", "4", "pixel", "gather"),
                           ("base", "4", "row", "gather"), ("weights", "4W", "row", "run"),
                           ("out", "8W", "bin", "run rmw")),
                "dense": (("base", "4", "row", "stream"), ("weights", "4W", "row", "stream"),
                          ("out", "8W", "bin", "near rmw"))},
         matrix_bytes=_padded_bytes, tuning={"dense tiles": "128 rows, 16 apart per pass"},
         source="src/steps/dot_padded.hpp"),
    _row("dot", "padded-avx2", cpu="avx2 fma", matrix="runs<=64", contract="float64",
         reads={"sparse": (("rowmap (listed)", "4", "pixel", "gather"),
                           ("base", "4", "row", "gather"), ("weights", "4W", "row", "run"),
                           ("out", "8W", "bin", "run rmw")),
                "dense": (("base", "4", "row", "stream"), ("weights", "4W", "row", "stream"),
                          ("out", "8W", "bin", "near rmw"))},
         matrix_bytes=_padded_bytes,
         tuning={"dense tiles": "128 rows, 16 apart per pass", "simd widths": "<= 8",
                 "sparse route": "the scalar walk of padded"},
         source="src/steps/dot_padded.hpp"),
    _row("dot", "csc-run", matrix="runs", contract="float64",
         reads={"sparse": _run_reads(4, "sparse"), "dense": _run_reads(4, "dense")},
         matrix_bytes=_run_bytes(4), source="src/steps/dot_csc.hpp"),
    _row("dot", "csc-nosplit", matrix="nosplit", contract="float64",
         reads={"sparse": _nosplit_reads("sparse"), "dense": _nosplit_reads("dense")},
         matrix_bytes=_nosplit_bytes, source="src/steps/dot_csc.hpp"),
    _row("dot", "bsb-csr", matrix="any", contract="float64",
         reads={"sparse": _CSC_SPARSE,
                "dense": (("bins, bin_ptr", "8", "bin of block", "stream"),
                          ("idx", "2", "entry", "stream"), ("data", "4", "entry", "stream"),
                          ("pixel", "T", "entry", "local"), ("out", "8", "bin", "store once"))},
         matrix_bytes=_bsbcsr_bytes,
         tuning={"block pixels": "<= 65536 (u16 in-block index)",
                 "dense accumulators": "4 per bin"},
         source="src/steps/dot_bsbcsr.hpp"),
    _row("dot", "bsb-csr-nosplit", matrix="nosplit", contract="float64",
         reads={"sparse": _nosplit_reads("sparse"),
                "dense": (("bins, bin_ptr", "8", "bin of block", "stream"),
                          ("idx", "2", "entry", "stream"), ("pixel", "T", "entry", "local"),
                          ("out", "8", "bin", "store once"))},
         matrix_bytes=_bsbcsr_nosplit_bytes,
         tuning={"block pixels": "<= 65536 (u16 in-block index)",
                 "sparse route": "the loop of csc-nosplit"},
         source="src/steps/dot_bsbcsr.hpp"),
    _row("dot", "csc-permute", matrix="permutation", contract="pixels",
         reads={"sparse": (("bin", "4", "pixel", "gather"), ("out", "T", "bin", "store")),
                "dense": (("bin", "4", "pixel", "stream"), ("out", "T", "bin", "store"))},
         matrix_bytes=_nosplit_bytes, source="src/steps/dot_fixed.hpp"),
    _row("dot", "csc-int", dtypes=INT, matrix="fixed32", contract="int64-scaled",
         reads={"sparse": (("indptr", "8", "pixel", "gather"), ("indices", "4e", "entry", "run"),
                           ("data (u32)", "4e", "entry", "run"),
                           ("out (i64)", "8e", "bin", "gather rmw")),
                "dense": (("indptr", "4", "pixel", "stream"), ("indices", "4e", "entry", "stream"),
                          ("data (u32)", "4e", "entry", "stream"),
                          ("out (i64)", "8e", "bin", "near rmw"))},
         matrix_bytes=_csc_bytes(4), tuning={"sparse prefetch, staging": "none (csc has both)",
                                             "64-bit pixels": "fixed_point_matrix needs maxval"},
         source="src/steps/dot_fixed.hpp"),
    _row("dot", "csc-run-int", dtypes=INT, matrix="fixed32-runs", contract="int64-scaled",
         reads={"sparse": _run_reads(4, "sparse"), "dense": _run_reads(4, "dense")},
         matrix_bytes=_run_bytes(4), tuning={"64-bit pixels": "fixed_point_matrix needs maxval"},
         source="src/steps/dot_fixed.hpp"),
    _row("dot", "csc-run-int16", dtypes=INT, matrix="fixed16-runs", contract="int64-scaled",
         reads={"sparse": _run_reads(2, "sparse"), "dense": _run_reads(2, "dense")},
         matrix_bytes=_run_bytes(2), tuning={"64-bit pixels": "fixed_point_matrix needs maxval"},
         source="src/steps/dot_fixed.hpp"),
    _row("dot", "csc-nosplit-int", dtypes=INT32, matrix="nosplit", contract="float64-of-int64",
         reads={"sparse": _nosplit_reads("sparse"), "dense": _nosplit_reads("dense")},
         matrix_bytes=_nosplit_bytes,
         tuning={"dense tiles": "128 pixels, 16 apart per pass"},
         source="src/steps/dot_fixed.hpp"),
]


def rows(step=None):
    """The rows, of one step or all."""
    return [k for k in KERNELS if step is None or k["step"] == step]


def row(step, name):
    """The row of one step value."""
    for k in KERNELS:
        if k["step"] == step and k["name"] == name:
            return k
    raise KeyError((step, name))
