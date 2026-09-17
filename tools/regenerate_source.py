#!/usr/bin/env python3
"""
Regenerate src/bslz4_to_sparse.cpp.

That file is ~2300 lines of which almost all is mechanical: one extern "C"
instantiation per (dtype, untranspose backend, decode family) -- 10 x 4 x 3
-- wrapping the C++ templates in bslz4_core.hpp behind the plain symbols
c2py23 needs, plus the C2PY_BEGIN spec block describing them. Editing that
by hand is not realistic, so it is generated from the tables at the top of
this file: DTYPES and BACKENDS.

The full pipeline, both steps run by hand after changing either table:

    python3 tools/regenerate_source.py     # -> src/bslz4_to_sparse.cpp
    python3 tools/regenerate_wrapper.py    # -> src/bslz4_to_sparse_wrapper.c

Both outputs are committed, so neither this script nor c2py23 is needed to
build or install the package -- only to change the spec. Edit here, never
in the generated files: regenerating silently reverts anything hand-edited
there.
"""
import os

HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUT_PATH = os.path.join(REPO_ROOT, "src", "bslz4_to_sparse.cpp")

DTYPES = [
    ("u8", "uint8_t"), ("u16", "uint16_t"), ("u32", "uint32_t"), ("u64", "uint64_t"),
    ("i8", "int8_t"), ("i16", "int16_t"), ("i32", "int32_t"), ("i64", "int64_t"),
    ("f32", "float"), ("f64", "double"),
]
# (variant name, untranspose adapter, "when" guard or None, default,
#  preprocessor condition under which the kernel is real code or None if
#  it always is). c2py23's _variants_() lists every compiled symbol, and
#  _rebind_() accepts any of them, so the "when" guard only steers the
#  automatic choice -- backend_<name>_available() below is what tells
#  set_backend() whether a name is anything but a stub on this build.
BACKENDS = [
    ("kcb", "untranspose_kcb", None, True, None),
    # default:True but guarded, so auto-resolve picks it on POWER only:
    # measured there at 1.56x scal against kcb's 1.43x, while on x86 kcb's
    # own AVX2/AVX-512 dispatch is far ahead of this SSE2 kernel (5.72x vs
    # 2.59x), and kcb stays the unconditional default everywhere else.
    ("sse", "untranspose_bshuf_sse", "c2py_ppc64_vsx", True,
     "defined(__SSE2__) || defined(NO_WARN_X86_INTRINSICS)"),
    # Selectable but NOT a default: measured on a Cortex-A72 (Pinebook,
    # aarch64) this NEON kernel is slower than kcb's portable C -- 8.63 vs
    # 6.24 ms/frame, against scal's 11.10 -- so kcb stays the default on
    # ARM. Worth re-measuring on other ARM cores before changing that.
    ("neon", "untranspose_bshuf_neon", "c2py_arm64_asimd", False,
     "defined(__ARM_NEON) && defined(__aarch64__)"),
    ("scal", "untranspose_bshuf_scal", None, False, None),
]

HEADER = '''/*
 * bslz4_to_sparse -- decode a bitshuffle-LZ4/zstd compressed HDF5 chunk
 * directly into a masked/thresholded sparse (values, indices) pair,
 * optionally also accumulating a CSC powder integration.
 *
 * Stage 1 of the c2py23 port (see GitHub issue #11): the type-variant
 * macro-repetition (bshuf.c / bshufdot.c, #include'd once per DATATYPE) is
 * replaced by C++ templates (bslz4_core.hpp), instantiated explicitly below
 * behind plain extern "C" names for c2py23 to call.
 *
 * Four untranspose backends are wired in and compiled together: kcb
 * (https://github.com/kalcutter/bitshuffle, does its own CPU dispatch,
 * the default, x86-only internally), scal (the portable scalar reference
 * kernel), and two from upstream bitshuffle
 * (https://github.com/kiyo-masui/bitshuffle) whose ISA is baseline for
 * the architecture, so they are real code under the project's plain -O2
 * build with no -m flags: sse (guarded __SSE2__, x86-64 baseline; also
 * reached on ppc64le, where setup.py passes -DNO_WARN_X86_INTRINSICS and
 * GCC's x86-intrinsic compatibility headers implement it over VSX, as
 * upstream bitshuffle's own setup.py does) and neon (guarded __ARM_NEON
 * && __aarch64__, ARMv8-A baseline).
 *
 * avx2/avx512 adapters exist in bslz4_backends.hpp but are NOT wired in:
 * upstream gates their real code behind __AVX2__/__AVX512F__, set only by
 * -mavx2/-mavx512f, and falls back to a stub returning -14 otherwise, so
 * under this single -O2-for-everything build they would always fail.
 * Wiring them in needs per-ISA translation units built with the matching
 * -m flags, as c2bslz4's meson build does, not just a spec change.
 *
 * Dtype dispatch is a Python-level choice, not a C-level one: each dtype
 * gets its own generated Python function (bslz4_multi_u8, bslz4_multi_u16,
 * ..., bslz4_csc_multi_u8, ...) via c2py23's "expand" template mechanism,
 * rather than one polymorphic function branching on a buffer's runtime
 * format string. Backend choice (kcb/sse/neon/scal) stays an orthogonal
 * per-call C-level "variants" switch, set once via
 * set_backend()/_rebind_bslz4_multi_<suffix> and applied identically to
 * every dtype.
 *
 * There is no separate single-frame family: bslz4_decode/bslz4_csc_decode
 * (single-frame C++ templates, bslz4_core.hpp) went unused once
 * src/__init__.py's chunk2sparse/chunk2sparseCSC/bslz4_to_sparse() were
 * changed to call bslz4_multi_<suffix>/bslz4_csc_multi_<suffix> directly
 * with nframes=1 -- there is no reason for a single frame to be a
 * different code path, at the Python level or the C level, so those two
 * templates and their 60 single-frame extern "C" instantiations were
 * removed rather than left as dead code.
 *
 * Issue #10: the block-level codec (lz4 or zstd, see bslz4_codec.hpp) is
 * an ordinary runtime "codec" parameter, not a c2py23 variant -- which
 * codec a dataset used is HDF5 filter metadata the Python caller already
 * has (cd_values[4]: 2 = lz4, 3 = zstd), and can differ between two
 * datasets decoded with the same compiled function in the same process.
 *
 * No malloc, no VLA: the workspace buffer is owned and sized by the Python
 * caller (see bslz4_core.hpp for why the scratch size can't be a compile
 * time constant).
 *
 * note_chunk() and bslz4_csc_multi_base_${SUFFIX} (issue #16, originally
 * developed against the pre-expand/pre-redesign code and re-ported here)
 * are an alternative, ctypes/numpy-free way to feed multi-frame calls:
 * note_chunk() lets a plain Python loop batch independent chunk objects
 * (bytes, bytearray, memoryview, mmap slices, ...) into compressed_ptrs/
 * compressed_lengths using c2py23's own buffer acquisition for the
 * address, instead of e.g. numpy's .ctypes.data; bslz4_csc_multi_base_*
 * additionally takes one shared "base" buffer plus byte offsets into it
 * (e.g. an mmap'ed HDF5 file) rather than one buffer object per frame,
 * for callers who already have the whole file mapped. See
 * _gather_chunks()/harvest_chunk_offsets()/pack_offsets_lengths() in
 * __init__.py.
 *
 * {avx512,avx2,sse2,vsx,neon}_collect_available()/get_*_collect()/set_*_collect()
 * control the optional SIMD mask+threshold collect kernel tiers for
 * u16/u32 (bslz4_collect_simd.hpp). The best available tier defaults on
 * automatically (avx512 > avx2 > sse2 on x86, vsx on POWER, neon on
 * aarch64) -- that default is decided in bslz4_collect_simd.hpp itself
 * (each bslz4_<tier>_collect_enabled()'s own static initializer), not
 * here or in __init__.py; these functions just expose the runtime
 * override. AVX-512 in particular can
 * throttle clocks on some chips enough to net-lose for this workload --
 * set_avx512_collect(False) if that turns out to matter on yours.
 */

#include "c2py_amd64.h"
#include "c2py_ppc64.h"
#include "bslz4_backends.hpp"
#include "bslz4_core.hpp"

using namespace bslz4;

extern "C" {

double bslz4_get_dense_sparse_threshold_impl() {
    return bslz4_csc_dense_sparse_threshold();
}

void bslz4_set_dense_sparse_threshold_impl(double x) {
    bslz4_csc_dense_sparse_threshold() = x;
}

int bslz4_avx512_collect_available_impl() {
    return bslz4_avx512_collect_capable() ? 1 : 0;
}

int bslz4_get_avx512_collect_impl() {
    return bslz4_avx512_collect_enabled() ? 1 : 0;
}

int bslz4_set_avx512_collect_impl(int enabled) {
    if (enabled && !bslz4_avx512_collect_available_impl())
        return -1;
    bslz4_avx512_collect_enabled() = (enabled != 0);
    return 0;
}

int bslz4_avx2_collect_available_impl() {
    return bslz4_avx2_collect_capable() ? 1 : 0;
}

int bslz4_get_avx2_collect_impl() {
    return bslz4_avx2_collect_enabled() ? 1 : 0;
}

int bslz4_set_avx2_collect_impl(int enabled) {
    if (enabled && !bslz4_avx2_collect_available_impl())
        return -1;
    bslz4_avx2_collect_enabled() = (enabled != 0);
    return 0;
}

int bslz4_sse2_collect_available_impl() {
    return bslz4_sse2_collect_capable() ? 1 : 0;
}

int bslz4_get_sse2_collect_impl() {
    return bslz4_sse2_collect_enabled() ? 1 : 0;
}

int bslz4_set_sse2_collect_impl(int enabled) {
    if (enabled && !bslz4_sse2_collect_available_impl())
        return -1;
    bslz4_sse2_collect_enabled() = (enabled != 0);
    return 0;
}

int bslz4_vsx_collect_available_impl() {
    return bslz4_vsx_collect_capable() ? 1 : 0;
}

int bslz4_get_vsx_collect_impl() {
    return bslz4_vsx_collect_enabled() ? 1 : 0;
}

int bslz4_set_vsx_collect_impl(int enabled) {
    if (enabled && !bslz4_vsx_collect_available_impl())
        return -1;
    bslz4_vsx_collect_enabled() = (enabled != 0);
    return 0;
}

int bslz4_neon_collect_available_impl() {
    return bslz4_neon_collect_capable() ? 1 : 0;
}

int bslz4_get_neon_collect_impl() {
    return bslz4_neon_collect_enabled() ? 1 : 0;
}

int bslz4_set_neon_collect_impl(int enabled) {
    if (enabled && !bslz4_neon_collect_available_impl())
        return -1;
    bslz4_neon_collect_enabled() = (enabled != 0);
    return 0;
}

/* Write chunk's address+length into pointers[index]/lengths[index]. The
 * only place a compressed chunk's address is extracted -- via c2py23's
 * own buffer acquisition on the "chunk" parameter below, not any
 * Python-side ctypes/numpy trick -- so a plain Python loop can batch
 * independent chunk objects (bytes, bytearray, memoryview, mmap slices,
 * network buffers, ...) into compressed_ptrs/compressed_lengths without
 * either dependency. See _gather_chunks() in __init__.py. */
void bslz4_note_chunk(const char *chunk, size_t chunk_len, int index,
                       int64_t *pointers, int32_t *lengths) {
    pointers[index] = (int64_t) (intptr_t) chunk;
    lengths[index] = (int32_t) chunk_len;
}

'''

extern_blocks = []

# backend_<name>_available(): is this backend real code in this build, or
# the stub upstream bitshuffle compiles when the ISA is absent? Needed
# because _variants_()/_rebind_() work on compiled symbols alone, so
# set_backend() would otherwise happily select a stub.
for be, _untr, _when, _default, real in BACKENDS:
    if real is None:
        body = "    return 1;"
    else:
        body = "#if %s\n    return 1;\n#else\n    return 0;\n#endif" % real
    extern_blocks.append("int bslz4_backend_%s_available_impl() {\n%s\n}\n" % (be, body))

for suffix, ctype in DTYPES:
    for be, untr, _when, _default, _real in BACKENDS:
        extern_blocks.append(f"""int bslz4_multi_{suffix}_{be}(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        {ctype} *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {{
    return bslz4_decode_multi<{ctype}, {untr}>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}}
""")

for suffix, ctype in DTYPES:
    for be, untr, _when, _default, _real in BACKENDS:
        extern_blocks.append(f"""int bslz4_csc_multi_{suffix}_{be}(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              {ctype} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {{
    return bslz4_csc_decode_multi<{ctype}, {untr}>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}}
""")

for suffix, ctype in DTYPES:
    for be, untr, _when, _default, _real in BACKENDS:
        extern_blocks.append(f"""int bslz4_csc_multi_base_{suffix}_{be}(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              {ctype} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {{
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {{
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }}
    return bslz4_csc_decode_multi<{ctype}, {untr}>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}}
""")

EXTERN_C_BODY = HEADER + "\n".join(extern_blocks) + "\n} /* extern \"C\" */\n"

SUFFIXES = [d[0] for d in DTYPES]
TYPES = [d[1] for d in DTYPES]

def _variant_line(be, when, default):
    # Braces are doubled: this is .format()ed later with name=/csig=.
    parts = ['"sig": "{name}_${{SUFFIX}}_%s({csig}) -> int"' % be]
    if when:
        parts.append('"when": "%s"' % when)
    parts.append('"default": %s' % default)
    return "                {{%s}}," % ", ".join(parts)


VARIANTS_TEMPLATE = (
    '            "variants": [\n'
    + "\n".join(_variant_line(be, when, default) for be, _untr, when, default, _real in BACKENDS)
    + "\n            ],"
)

EXPAND_DICT = f'''            "expand": {{
                "SUFFIX": {SUFFIXES!r},
                "TYPE": {TYPES!r},
            }},'''

def spec_family2():
    csig = ("const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, "
            "const uint8_t *mask, int NIJ, ${TYPE} *output, uint32_t *output_adr, int32_t *npx_out, "
            "int threshold, uint8_t *workspace, size_t workspace_len, int64_t *cursors")
    return f'''        {{
            "py_sig": "bslz4_multi_${{SUFFIX}}(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, output: buffer, output_adr: buffer, npx_out: buffer, threshold: int, workspace: buffer, cursors: buffer, codec: int = 2) -> int",
            "doc": "Decode nframes bitshuffle-LZ4/zstd chunks (${{TYPE}} pixels) from the same dataset into per-frame masked/thresholded sparse (output, output_adr, npx_out). The single-frame case is just nframes==1 -- there is no separate single-frame entry point.",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I'",
                "npx_out.format == 'i'",
                "workspace.format == 'B'",
                "compressed_ptrs.itemsize == 8",
                "compressed_lengths.itemsize == 4",
                "cursors.itemsize == 8",
            ],
            "c_overloads": [
                {{
                    "map": {{
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "compressed_ptrs.n",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "npx_out": "npx_out.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                        "cursors": "cursors.ptr",
                    }},
{VARIANTS_TEMPLATE.format(name="bslz4_multi", csig=csig)}
                }},
            ],
{EXPAND_DICT}
            "default_raise": "TypeError: unsupported output dtype for bslz4_multi_${{SUFFIX}}",
        }},
'''

def spec_family4():
    csig = ("const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, "
            "const uint8_t *mask, int NIJ, ${TYPE} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, "
            "double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, "
            "uint8_t *workspace, size_t workspace_len, int64_t *cursors")
    return f'''        {{
            "py_sig": "bslz4_csc_multi_${{SUFFIX}}(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, cursors: buffer, nout: int, codec: int = 2) -> int",
            "doc": "Decode a batch of bitshuffle-LZ4/zstd chunks (${{TYPE}} pixels) from the same dataset into per-frame sparse (outpx, output_adr, npx_out) and per-frame CSC powder integrations (powder). Routes each (frame, block) between a dense and a sparse CSC strategy based on that block's own compression ratio -- see bslz4_core.hpp's bslz4_csc_decode_multi.",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I'",
                "npx_out.format == 'i'",
                "powder.format == 'd'",
                "data.format == 'f'",
                "indices.format == 'I' or indices.format == 'i'",
                "indptr.format == 'I' or indptr.format == 'i'",
                "workspace.format == 'B'",
                "compressed_ptrs.itemsize == 8",
                "compressed_lengths.itemsize == 4",
                "cursors.itemsize == 8",
            ],
            "c_overloads": [
                {{
                    "map": {{
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "compressed_ptrs.n",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "npx_out": "npx_out.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "nout",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                        "cursors": "cursors.ptr",
                    }},
{VARIANTS_TEMPLATE.format(name="bslz4_csc_multi", csig=csig)}
                }},
            ],
{EXPAND_DICT}
            "default_raise": "TypeError: unsupported outpx dtype for bslz4_csc_multi_${{SUFFIX}}",
        }},
'''

def spec_backends():
    out = []
    for be, _untr, _when, _default, real in BACKENDS:
        if real is None:
            doc = ("1 always: the %s untranspose backend is portable C, real on every "
                   "build. See set_backend()." % be)
        else:
            doc = ("1 if the %s untranspose backend is real code in this build, 0 if it is "
                   "the stub upstream bitshuffle compiles when the ISA is absent (it is "
                   "guarded by %s). available_backends() filters on this and set_backend() "
                   "refuses a backend reporting 0, because the compiled symbol exists "
                   "either way." % (be, real))
        out.append('''        {
            "py_sig": "backend_%s_available() -> int",
            "doc": "%s",
            "c_overloads": [
                {"sig": "bslz4_backend_%s_available_impl() -> int", "map": {}},
            ],
        },
''' % (be, doc, be))
    return "".join(out)


def spec_note_chunk():
    return '''        {
            "py_sig": "note_chunk(chunk: buffer, index: int, pointers: buffer, lengths: buffer) -> void",
            "doc": "Write chunk's raw address and byte length into pointers[index]/lengths[index]. The only place a buffer's address is extracted -- via c2py23's own buffer acquisition, not any Python-side ctypes/numpy trick -- so a plain Python loop can batch independent chunk objects (bytes, bytearray, memoryview, mmap slices, ...) into bslz4_multi_* or bslz4_csc_multi_*'s compressed_ptrs/compressed_lengths arrays. See _gather_chunks() in __init__.py.",
            "c_overloads": [
                {
                    "sig": "bslz4_note_chunk(const char *chunk, size_t chunk_len, int index, int64_t *pointers, int32_t *lengths)",
                    "map": {
                        "chunk": "chunk.ptr",
                        "chunk_len": "chunk.len",
                        "index": "index",
                        "pointers": "pointers.ptr",
                        "lengths": "lengths.ptr",
                    },
                },
            ],
        },
'''


def spec_family6():
    csig = ("const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, "
            "const uint8_t *mask, int NIJ, ${TYPE} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, "
            "double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, "
            "uint8_t *workspace, size_t workspace_len, int64_t *cursors")
    return f'''        {{
            "py_sig": "bslz4_csc_multi_base_${{SUFFIX}}(base: buffer, offsets: buffer, lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, cursors: buffer, nout: int, codec: int = 2) -> int",
            "doc": "Like bslz4_csc_multi_${{SUFFIX}}, but for chunks that all share one base buffer (e.g. an mmap'ed HDF5 file): offsets are byte offsets from base rather than absolute addresses, avoiding any Python-side pointer arithmetic -- see harvest_chunk_offsets()/pack_offsets_lengths() in __init__.py. NOTE: offsets is mutated in place into absolute pointers by this call (reused as scratch, like cursors/workspace already are).",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I'",
                "npx_out.format == 'i'",
                "powder.format == 'd'",
                "data.format == 'f'",
                "indices.format == 'I' or indices.format == 'i'",
                "indptr.format == 'I' or indptr.format == 'i'",
                "workspace.format == 'B'",
                "offsets.itemsize == 8",
                "lengths.itemsize == 4",
                "cursors.itemsize == 8",
            ],
            "c_overloads": [
                {{
                    "map": {{
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "offsets.n",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "npx_out": "npx_out.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "nout",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                        "cursors": "cursors.ptr",
                    }},
{VARIANTS_TEMPLATE.format(name="bslz4_csc_multi_base", csig=csig)}
                }},
            ],
{EXPAND_DICT}
            "default_raise": "TypeError: unsupported outpx dtype for bslz4_csc_multi_base_${{SUFFIX}}",
        }},
'''


C2PY_BLOCK = '''
/* C2PY_BEGIN
{
    "module": "_bslz4_to_sparse",
    "source": ["bslz4_to_sparse.cpp"],
    "headers": ["c2py_amd64.h", "c2py_arm64.h", "c2py_ppc64.h"],
    "timing": False,
    "functions": [
        {
            "py_sig": "get_dense_sparse_threshold() -> float",
            "doc": "Current compression-factor threshold routing bslz4_csc_multi_* between its dense and sparse per-(frame,block) paths -- see bslz4_core.hpp's bslz4_csc_dense_sparse_threshold(). Placeholder default, not calibrated against real detector data.",
            "c_overloads": [
                {"sig": "bslz4_get_dense_sparse_threshold_impl() -> double", "map": {}},
            ],
        },
        {
            "py_sig": "set_dense_sparse_threshold(x: float) -> void",
            "doc": "Set the compression-factor threshold routing bslz4_csc_multi_* between its dense and sparse per-(frame,block) paths. A block's (blocksize / compressed_bytes) above this uses the sparse route, at or below it uses dense.",
            "c_overloads": [
                {"sig": "bslz4_set_dense_sparse_threshold_impl(double x) -> void", "map": {"x": "x"}},
            ],
        },
        {
            "py_sig": "avx512_collect_available() -> int",
            "doc": "1 if this CPU has the AVX-512 subset (F+BW+VL) the u16/u32 collect kernel needs, 0 otherwise -- independent of whether it's currently enabled, see get/set_avx512_collect().",
            "c_overloads": [
                {"sig": "bslz4_avx512_collect_available_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "get_avx512_collect() -> int",
            "doc": "1 if the AVX-512 mask+threshold collect kernel (u16/u32 only) is currently enabled, 0 otherwise. Defaults to 1 iff avx512_collect_available() (it's the highest-priority tier, tried first) -- see set_avx512_collect().",
            "c_overloads": [
                {"sig": "bslz4_get_avx512_collect_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "set_avx512_collect(enabled: int) -> int",
            "doc": "Enable/disable the AVX-512 mask+threshold collect kernel for bslz4_multi_u16/u32 and bslz4_csc_multi_u16/u32's sparse-route compaction (bslz4_collect_simd.hpp). Returns 0 on success, -1 if enabled=1 was requested but avx512_collect_available() is false (the flag is left unchanged in that case). On by default when available (it's the highest-priority tier) -- AVX-512 can throttle clocks on some chips enough to net-lose for this workload, so if that turns out to matter on yours, set_avx512_collect(False) it and try avx2/sse2/scalar explicitly (disabling one tier does not auto-promote the next one).",
            "c_overloads": [
                {"sig": "bslz4_set_avx512_collect_impl(int enabled) -> int", "map": {"enabled": "enabled"}},
            ],
        },
        {
            "py_sig": "avx2_collect_available() -> int",
            "doc": "1 if this CPU has AVX2 (what the u16/u32 collect kernel's AVX2 tier needs), 0 otherwise -- independent of whether it's currently enabled, see get/set_avx2_collect().",
            "c_overloads": [
                {"sig": "bslz4_avx2_collect_available_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "get_avx2_collect() -> int",
            "doc": "1 if the AVX2 mask+threshold collect kernel (u16/u32 only) is currently enabled, 0 otherwise. Defaults to 1 iff avx2_collect_available() and avx512 isn't (avx2 is second priority) -- see set_avx2_collect().",
            "c_overloads": [
                {"sig": "bslz4_get_avx2_collect_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "set_avx2_collect(enabled: int) -> int",
            "doc": "Enable/disable the AVX2 mask+threshold collect kernel for bslz4_multi_u16/u32 and bslz4_csc_multi_u16/u32's sparse-route compaction (bslz4_collect_simd.hpp). Returns 0 on success, -1 if enabled=1 was requested but avx2_collect_available() is false. Ignored when avx512 collect is also enabled (avx512 is tried first). On by default when available and avx512 isn't -- AVX2 carries much less frequency-throttling risk than AVX-512 on most chips, but measure before relying on that rather than assuming it.",
            "c_overloads": [
                {"sig": "bslz4_set_avx2_collect_impl(int enabled) -> int", "map": {"enabled": "enabled"}},
            ],
        },
        {
            "py_sig": "sse2_collect_available() -> int",
            "doc": "1 on any x86-64 build with a compiler that supports the SSE2 collect kernel tier (SSE2 itself is the x86-64 ABI baseline, always present -- this just reflects whether bslz4_collect_simd.hpp compiled that tier in), 0 otherwise (non-x86 builds, or a compiler without GCC/Clang-style target attributes).",
            "c_overloads": [
                {"sig": "bslz4_sse2_collect_available_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "get_sse2_collect() -> int",
            "doc": "1 if the SSE2 mask+threshold collect kernel (u16/u32 only) is currently enabled, 0 otherwise. Defaults to 1 iff sse2_collect_available() and neither avx512 nor avx2 is (sse2 is third priority, the fallback tier on any x86-64 build) -- see set_sse2_collect().",
            "c_overloads": [
                {"sig": "bslz4_get_sse2_collect_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "set_sse2_collect(enabled: int) -> int",
            "doc": "Enable/disable the SSE2 mask+threshold collect kernel for bslz4_multi_u16/u32 and bslz4_csc_multi_u16/u32's sparse-route compaction (bslz4_collect_simd.hpp). Returns 0 on success, -1 if enabled=1 was requested but sse2_collect_available() is false. Ignored when avx512 or avx2 collect is also enabled (tried last). On by default whenever neither of those is -- unlike them, SSE2 is always safe to enable on any x86-64 CPU (no capability gap, no known throttling risk).",
            "c_overloads": [
                {"sig": "bslz4_set_sse2_collect_impl(int enabled) -> int", "map": {"enabled": "enabled"}},
            ],
        },
        {
            "py_sig": "vsx_collect_available() -> int",
            "doc": "1 if this CPU has POWER VSX (what the u16/u32 collect kernel's VSX tier needs), 0 otherwise -- independent of whether it's currently enabled, see get/set_vsx_collect(). Always 0 on non-POWER builds.",
            "c_overloads": [
                {"sig": "bslz4_vsx_collect_available_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "get_vsx_collect() -> int",
            "doc": "1 if the VSX mask+threshold collect kernel (u16/u32 only, POWER8+) is currently enabled, 0 otherwise. Defaults to 1 iff vsx_collect_available() -- see set_vsx_collect().",
            "c_overloads": [
                {"sig": "bslz4_get_vsx_collect_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "set_vsx_collect(enabled: int) -> int",
            "doc": "Enable/disable the VSX mask+threshold collect kernel for bslz4_multi_u16/u32 and bslz4_csc_multi_u16/u32's sparse-route compaction (bslz4_collect_simd.hpp). Returns 0 on success, -1 if enabled=1 was requested but vsx_collect_available() is false. On by default when available -- real-hardware-measured on a POWER9 box: 3.75-8.2x faster than scalar on sparse data via a vec_any_gt fast-skip gate (see the file comment in bslz4_collect_simd.hpp and tools/bslz4_power9_collect_probe.c for how that number was reached).",
            "c_overloads": [
                {"sig": "bslz4_set_vsx_collect_impl(int enabled) -> int", "map": {"enabled": "enabled"}},
            ],
        },
        {
            "py_sig": "neon_collect_available() -> int",
            "doc": "1 if this CPU has ARM NEON/ASIMD (what the u16/u32 collect kernel's NEON tier needs), 0 otherwise -- independent of whether it's currently enabled, see get/set_neon_collect(). Always 0 on non-aarch64 builds.",
            "c_overloads": [
                {"sig": "bslz4_neon_collect_available_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "get_neon_collect() -> int",
            "doc": "1 if the NEON mask+threshold collect kernel (u16/u32 only, AArch64) is currently enabled, 0 otherwise. Defaults to 1 iff neon_collect_available() -- see set_neon_collect().",
            "c_overloads": [
                {"sig": "bslz4_get_neon_collect_impl() -> int", "map": {}},
            ],
        },
        {
            "py_sig": "set_neon_collect(enabled: int) -> int",
            "doc": "Enable/disable the NEON mask+threshold collect kernel for bslz4_multi_u16/u32 and bslz4_csc_multi_u16/u32's sparse-route compaction (bslz4_collect_simd.hpp). Returns 0 on success, -1 if enabled=1 was requested but neon_collect_available() is false. On by default when available -- real-hardware-measured on a Cortex-A72 (Pinebook): 4.18x (u16) / 4.38x (u32) faster than scalar via a vmaxvq any-match fast-skip gate (see the file comment in bslz4_collect_simd.hpp and tools/bslz4_neon_collect_probe.c for how that number was reached).",
            "c_overloads": [
                {"sig": "bslz4_set_neon_collect_impl(int enabled) -> int", "map": {"enabled": "enabled"}},
            ],
        },
''' + spec_backends() + spec_note_chunk() + spec_family2() + spec_family4() + spec_family6() + '''    ],
}
C2PY_END */
'''

with open(OUT_PATH, "w") as f:
    f.write(EXTERN_C_BODY)
    f.write(C2PY_BLOCK)

print("wrote", len(EXTERN_C_BODY.splitlines()) + len(C2PY_BLOCK.splitlines()), "lines")
print("extern C functions:", len(extern_blocks) + 2)
