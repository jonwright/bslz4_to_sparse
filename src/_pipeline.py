"""The pipeline: one value per processing step, chosen per object.

A pipeline is six step values, in this order:

    decode       compressed block -> bit-planes
    mask         how the pixel mask is applied
    untranspose  bit-planes -> pixels (the low-planes values: u16 blocks
                 whose values are all < 256, fused with the collect)
    collect      pixels -> list of (index, value) above the cut
    route        matrix objects: per block, dot over the block (dense) or
                 over the list of non-zero pixels (sparse)
    dot          matrix objects: the layout of the matrix

Within each step 0 means "auto" and the values are numbered best first, by
measurement (notes_pipeline_cases.md).  A value needing an instruction set
is its own value and is refused, with the reason, on a CPU without it.

The C side (src/pipeline/pipeline.h, registry.c) mirrors these tables and
checks again.
"""
import numpy as np

STEPS = ("decode", "mask", "untranspose", "collect", "route", "dot")
NSTEPS = len(STEPS)
DECODE, MASK, UNTRANSPOSE, COLLECT, ROUTE, DOT = range(NSTEPS)

CODEC_LZ4 = 2
CODEC_ZSTD = 3

# (name, what it needs / what it is) per value; index 0 is auto
VALUES = {
    "decode": (
        "auto",
        ("lz4-band", "lz4: zero-aware decoder below 8x and above 24x, stock lz4 in between"),
        ("lz4-zero", "lz4: zero-aware decoder for every block"),
        ("lz4-stock", "lz4: LZ4_decompress_safe"),
        ("zstd", "zstd"),
    ),
    "mask": (
        "auto",
        ("pixel", "tested per pixel when collecting"),
        ("planes", "ANDed into the bit-planes after decode (wins only if masked pixels hold the maximum)"),
        ("none", "every pixel valid: no mask tests"),
    ),
    "untranspose": (
        "auto",
        ("lowplanes-vbmi", "u16 pixels; AVX-512 VBMI, VBMI2 and GFNI"),
        ("lowplanes-avx2", "u16 pixels; AVX2"),
        ("lowplanes-vsx", "u16 pixels; POWER8 VSX"),
        ("lowplanes-c", "u16 pixels; portable C"),
        ("kcb", "kcb's bit transpose (SSE2/AVX2/AVX-512 dispatched inside)"),
        ("neon", "bitshuffle's NEON kernels; aarch64"),
        ("scalar", "bitshuffle's scalar kernels"),
    ),
    "collect": (
        "auto",
        ("avx512cs", "u16/u32 pixels; AVX-512 BW/VL and VBMI2"),
        ("avx2cs", "u16/u32 pixels; AVX2"),
        ("vsx", "u16/u32 pixels; POWER VSX"),
        ("neon", "u16/u32 pixels; aarch64 NEON"),
        ("sse2", "u16/u32 pixels; SSE2"),
        ("scalar", "any pixel type"),
    ),
    "route": (
        "auto",
        ("ratio", "per block: sparse when compressed more than 8x"),
        ("dense", "dot over every pixel of every block"),
        ("sparse", "dot over the non-zero pixels of every block"),
    ),
    "dot": (
        "auto",
        ("csc", "any matrix"),
        ("padded", "each pixel's bins one consecutive run (at most 64)"),
        ("padded-avx2", "as padded; AVX2 + FMA on the dense route"),
        ("csc-run", "each pixel's bins one consecutive run"),
        ("csc-nosplit", "at most one bin per pixel, weight 1 (a histogram)"),
        ("bsb-csr", "any matrix; for few occupied pixels (rings)"),
        ("bsb-csr-nosplit", "as csc-nosplit"),
        ("csc-nosplit-moment", "0 or 2 bins per pixel: (b, 1), (b+1, q)"),
        ("csc-permute", "a permutation: one bin per pixel, weight 1, each bin used once; output in the pixel dtype"),
        ("csc-int", "integer pixels; fixed-point u32 weights, exact int64 sums"),
        ("csc-run-int", "integer pixels; as csc-run with fixed-point u32 weights"),
        ("csc-run-int16", "integer pixels; as csc-run with fixed-point u16 weights"),
    ),
}

NAMES = {s: tuple(v if isinstance(v, str) else v[0] for v in VALUES[s]) for s in STEPS}

# the matrix layouts the dots read (src/_matrix.py) and the C entry each uses
DOT_ENTRY = {
    "csc": "csc", "csc-run": "csc", "csc-nosplit": "csc", "csc-nosplit-moment": "csc",
    "csc-permute": "csc", "csc-int": "csc", "csc-run-int": "csc", "csc-run-int16": "csc",
    "padded": "padded", "padded-avx2": "padded",
    "bsb-csr": "bsb-csr", "bsb-csr-nosplit": "bsb-csr",
}

DTYPE_INDEX = {"u8": 0, "u16": 1, "u32": 2, "u64": 3, "i8": 4, "i16": 5, "i32": 6, "i64": 7,
               "f32": 8, "f64": 9}
_SUFFIX = {("u", 1): "u8", ("u", 2): "u16", ("u", 4): "u32", ("u", 8): "u64",
           ("i", 1): "i8", ("i", 2): "i16", ("i", 4): "i32", ("i", 8): "i64",
           ("f", 4): "f32", ("f", 8): "f64"}


def dtype_suffix(dtype):
    dtype = np.dtype(dtype)
    try:
        return _SUFFIX[(dtype.kind, dtype.itemsize)]
    except KeyError:
        raise TypeError("unsupported pixel dtype %r" % (dtype,))


_ext = None          # the native module, set by __init__


def available(step, value):
    """1 usable here, 0 known but this CPU/build lacks what it needs, -1 unknown."""
    return _ext.step_available(STEPS.index(step), int(value))


def _usable(step):
    return [n for i, n in enumerate(NAMES[step]) if i and available(step, i) == 1]


def _value(step, v):
    """A step value given as a name or an int -> its int (0 = auto)."""
    names = NAMES[step]
    if v is None:
        return 0
    if isinstance(v, str):
        if v not in names:
            raise ValueError("pipeline[%r]=%r is unknown (values: %s)"
                             % (step, v, ", ".join(names[1:])))
        return names.index(v)
    try:
        i = int(v)
    except (TypeError, ValueError):
        raise ValueError("pipeline[%r]=%r is neither a name nor a number" % (step, v))
    if i < 0 or i >= len(names):
        raise ValueError("pipeline[%r]=%d is unknown (values 1..%d: %s)"
                         % (step, i, len(names) - 1, ", ".join(names[1:])))
    return i


def parse(pipeline):
    """None, 0, a sequence of 6 values or a {step: value} dict (values as
    names or ints, 0 / None = auto) -> a list of 6 ints."""
    if pipeline is None or (np.ndim(pipeline) == 0 and not isinstance(pipeline, dict)
                            and int(pipeline) == 0):
        return [0] * NSTEPS
    if isinstance(pipeline, dict):
        bad = [k for k in pipeline if k not in STEPS]
        if bad:
            raise ValueError("pipeline has unknown steps %s (steps: %s)"
                             % (", ".join(map(repr, bad)), ", ".join(STEPS)))
        return [_value(s, pipeline.get(s)) for s in STEPS]
    seq = list(pipeline)
    if len(seq) != NSTEPS:
        raise ValueError("a pipeline has %d values (%s), got %d"
                         % (NSTEPS, ", ".join(STEPS), len(seq)))
    return [_value(s, v) for s, v in zip(STEPS, seq)]


def _check_cpu(step, i):
    a = available(step, i)
    if a == 1:
        return
    name, need = VALUES[step][i]
    raise ValueError("pipeline[%r]=%r needs %s, which this CPU/build lacks (usable here: %s)"
                     % (step, name, need, ", ".join(_usable(step))))


def auto_step(step, suffix, codec, mask_all_valid):
    """The best guess for one step (not dot) on this CPU."""
    if step == "decode":
        if codec == CODEC_ZSTD:
            return NAMES[step].index("zstd")
        import platform
        x86 = platform.machine().lower() in ("x86_64", "amd64")
        return NAMES[step].index("lz4-band" if x86 else "lz4-zero")
    if step == "mask":
        return NAMES[step].index("none" if mask_all_valid else "pixel")
    if step == "untranspose":
        if suffix == "u16":
            for n in ("lowplanes-vbmi", "lowplanes-avx2", "lowplanes-vsx", "lowplanes-c"):
                if available(step, NAMES[step].index(n)) == 1:
                    return NAMES[step].index(n)
        return NAMES[step].index("kcb")
    if step == "collect":
        if suffix in ("u16", "u32"):
            for n in ("avx512cs", "avx2cs", "vsx", "neon", "sse2"):
                if available(step, NAMES[step].index(n)) == 1:
                    return NAMES[step].index(n)
        return NAMES[step].index("scalar")
    if step == "route":
        return NAMES[step].index("ratio")
    raise ValueError(step)


def resolve(pipeline, dtype, codec, mask, dot_auto=None, matrix=False):
    """Fill the auto steps and check every value for this object.

    dtype: the pixel dtype; codec: CODEC_LZ4 or CODEC_ZSTD; mask: the flat
    pixel mask; matrix: whether the object has a matrix (route and dot used);
    dot_auto: callable giving the dot name to use when dot is auto.  Returns
    a uint16 array of 6; route and dot are 0 without a matrix.  Raises
    ValueError naming the step, the value and why."""
    vals = parse(pipeline)
    suffix = dtype_suffix(dtype)
    mask_all_valid = bool((np.asarray(mask) != 0).all())
    if not matrix:
        for s in (ROUTE, DOT):
            if vals[s]:
                raise ValueError("pipeline[%r]=%r: this object has no matrix (use chunk2sparseCSC)"
                                 % (STEPS[s], NAMES[STEPS[s]][vals[s]]))
    for s, step in enumerate(STEPS):
        if s == DOT or (not matrix and s == ROUTE):
            continue
        if vals[s] == 0:
            vals[s] = auto_step(step, suffix, codec, mask_all_valid)
        _check_cpu(step, vals[s])
    # what the data and the dtype allow
    dname = NAMES["decode"][vals[DECODE]]
    if codec == CODEC_ZSTD and dname != "zstd":
        raise ValueError("pipeline['decode']=%r: the data is zstd compressed (use 'zstd')" % dname)
    if codec != CODEC_ZSTD and dname == "zstd":
        raise ValueError("pipeline['decode']='zstd': the data is lz4 compressed "
                         "(use lz4-band, lz4-zero or lz4-stock)")
    if NAMES["mask"][vals[MASK]] == "none" and not mask_all_valid:
        raise ValueError("pipeline['mask']='none': the mask has %d masked pixels (use 'pixel' or 'planes')"
                         % int((np.asarray(mask) == 0).sum()))
    uname = NAMES["untranspose"][vals[UNTRANSPOSE]]
    if uname.startswith("lowplanes") and suffix != "u16":
        raise ValueError("pipeline['untranspose']=%r is for u16 pixels, not %s (use kcb)"
                         % (uname, np.dtype(dtype)))
    cname = NAMES["collect"][vals[COLLECT]]
    if cname != "scalar" and suffix not in ("u16", "u32"):
        raise ValueError("pipeline['collect']=%r is for u16/u32 pixels, not %s (use scalar)"
                         % (cname, np.dtype(dtype)))
    if matrix:
        if vals[DOT] == 0:
            vals[DOT] = NAMES["dot"].index(dot_auto())
        _check_cpu("dot", vals[DOT])
        dot = NAMES["dot"][vals[DOT]]
        if dot in ("csc-int", "csc-run-int", "csc-run-int16") and suffix[0] not in "ui":
            raise ValueError("pipeline['dot']=%r needs integer pixels, not %s"
                             % (dot, np.dtype(dtype)))
    return np.array(vals, dtype=np.uint16)


def describe(pipeline=None):
    """The step names of a pipeline (a dict step -> name; None for the route
    and dot of a plain sparsify), or, with no argument, every step's values
    with whether this CPU/build can run them (a dict step -> list of
    (value, name, usable, what it needs))."""
    if pipeline is None:
        return {s: [(i, VALUES[s][i][0], available(s, i) == 1, VALUES[s][i][1])
                    for i in range(1, len(VALUES[s]))] for s in STEPS}
    vals = parse(pipeline)
    return {s: (NAMES[s][v] if v else None) for s, v in zip(STEPS, vals)}
