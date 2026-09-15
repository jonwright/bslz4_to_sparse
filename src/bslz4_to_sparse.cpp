/*
 * bslz4_to_sparse -- decode a bitshuffle-LZ4/zstd compressed HDF5 chunk
 * directly into a masked/thresholded sparse (values, indices) pair,
 * optionally also accumulating a CSC powder integration.
 *
 * Stage 1 of the c2py23 port (see GitHub issue #11): the type-variant
 * macro-repetition (bshuf.c / bshufdot.c, #include'd once per DATATYPE) is
 * replaced by C++ templates (bslz4_core.hpp), instantiated explicitly below
 * behind plain extern "C" names for c2py23 to call.
 *
 * Three untranspose backends are wired in and compiled together: kcb
 * (https://github.com/kalcutter/bitshuffle, does its own CPU dispatch,
 * the default), sse (upstream bitshuffle's SSE2 kernel -- real code, SSE2
 * is x86-64 baseline so this compiles under the project's plain -O2 build
 * with no extra flags), and scal, the portable scalar reference kernel
 * (also upstream bitshuffle). avx2/avx512/neon variants exist in
 * bslz4_backends.hpp but are NOT wired in here: upstream bitshuffle gates
 * their real SIMD code behind compiler-predefined macros (__AVX2__ etc,
 * only set by -mavx2/-mavx512f) and falls back to a stub returning -14
 * otherwise (bitshuffle_core.c) -- calling them under today's single
 * -O2-for-everything build would silently always fail. Wiring them in
 * needs per-ISA compiled objects first (separate translation units built
 * with the matching -m flags per tier, as c2bslz4's meson build does),
 * not just a spec change.
 *
 * Dtype dispatch is a Python-level choice, not a C-level one: each dtype
 * gets its own generated Python function (bslz4_multi_u8, bslz4_multi_u16,
 * ..., bslz4_csc_multi_u8, ...) via c2py23's "expand" template mechanism,
 * rather than one polymorphic function branching on a buffer's runtime
 * format string. Backend choice (kcb/sse/scal) stays an orthogonal
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
 * {avx512,avx2,sse2,vsx}_collect_available()/get_*_collect()/set_*_collect()
 * control the optional SIMD mask+threshold collect kernel tiers for
 * u16/u32 (bslz4_collect_simd.hpp). The best available tier defaults on
 * automatically (avx512 > avx2 > sse2, or vsx on POWER) -- that default
 * is decided in bslz4_collect_simd.hpp itself (each bslz4_<tier>_collect_
 * enabled()'s own static initializer), not here or in __init__.py; these
 * functions just expose the runtime override. AVX-512 in particular can
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

int bslz4_multi_u8_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint8_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint8_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u8_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint8_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint8_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u8_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint8_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint8_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u16_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint16_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint16_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u16_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint16_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint16_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u16_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint16_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint16_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint32_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint32_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u32_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint32_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint32_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint32_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint32_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u64_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint64_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint64_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u64_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint64_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint64_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_u64_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        uint64_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<uint64_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i8_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int8_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int8_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i8_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int8_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int8_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i8_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int8_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int8_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i16_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int16_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int16_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i16_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int16_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int16_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i16_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int16_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int16_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int32_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int32_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i32_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int32_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int32_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int32_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int32_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i64_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int64_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int64_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i64_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int64_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int64_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_i64_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        int64_t *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<int64_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_f32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        float *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<float, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_f32_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        float *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<float, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_f32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        float *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<float, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_f64_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        double *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<double, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_f64_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        double *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<double, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_multi_f64_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                        int nframes, int codec,
                        const uint8_t *mask, int NIJ,
                        double *output, uint32_t *output_adr, int32_t *npx_out, int threshold,
                        uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_decode_multi<double, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        output, output_adr, npx_out, threshold,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u8_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint8_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u8_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint8_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u8_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint8_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u16_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint16_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u16_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint16_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u16_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint16_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint32_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u32_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint32_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint32_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u64_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint64_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u64_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint64_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_u64_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<uint64_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i8_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int8_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i8_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int8_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i8_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int8_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i16_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int16_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i16_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int16_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i16_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int16_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int32_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i32_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int32_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int32_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i64_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int64_t, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i64_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int64_t, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_i64_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<int64_t, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_f32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<float, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_f32_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<float, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_f32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<float, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_f64_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<double, untranspose_kcb>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_f64_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<double, untranspose_bshuf_sse>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_f64_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    return bslz4_csc_decode_multi<double, untranspose_bshuf_scal>(
        compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u8_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint8_t, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u8_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint8_t, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u8_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint8_t, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u16_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint16_t, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u16_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint16_t, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u16_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint16_t, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u32_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint32_t, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u32_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint32_t, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u32_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint32_t, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u64_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint64_t, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u64_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint64_t, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_u64_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              uint64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint64_t, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i8_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int8_t, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i8_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int8_t, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i8_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int8_t, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i16_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int16_t, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i16_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int16_t, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i16_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int16_t, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i32_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int32_t, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i32_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int32_t, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i32_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int32_t, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i64_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int64_t, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i64_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int64_t, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_i64_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              int64_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int64_t, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_f32_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<float, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_f32_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<float, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_f32_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<float, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_f64_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<double, untranspose_kcb>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_f64_sse(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<double, untranspose_bshuf_sse>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

int bslz4_csc_multi_base_f64_scal(const char *base, int64_t *offsets, const int32_t *lengths,
                              int nframes, int codec,
                              const uint8_t *mask, int NIJ,
                              double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                              double *output, int NOUT,
                              const float *data, const uint32_t *indices, const uint32_t *indptr,
                              uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in
     * place, avoiding a second (nframes-sized) allocation just to hold
     * them -- same trick bslz4_csc_decode_multi already plays with
     * cursors/workspace. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<double, untranspose_bshuf_scal>(
        offsets, lengths, nframes, codec, mask, NIJ,
        outpx, output_adr, npx_out, threshold,
        output, NOUT, data, indices, indptr,
        workspace, workspace_len, cursors);
}

} /* extern "C" */

/* C2PY_BEGIN
{
    "module": "bslz4_to_sparse",
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
        {
            "py_sig": "bslz4_multi_${SUFFIX}(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, output: buffer, output_adr: buffer, npx_out: buffer, threshold: int, workspace: buffer, cursors: buffer, codec: int = 2) -> int",
            "doc": "Decode nframes bitshuffle-LZ4/zstd chunks (${TYPE} pixels) from the same dataset into per-frame masked/thresholded sparse (output, output_adr, npx_out). The single-frame case is just nframes==1 -- there is no separate single-frame entry point.",
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
                {
                    "map": {
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
                    },
            "variants": [
                {"sig": "bslz4_multi_${SUFFIX}_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *output, uint32_t *output_adr, int32_t *npx_out, int threshold, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "default": True},
                {"sig": "bslz4_multi_${SUFFIX}_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *output, uint32_t *output_adr, int32_t *npx_out, int threshold, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "when": "c2py_amd64_sse2", "default": False},
                {"sig": "bslz4_multi_${SUFFIX}_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *output, uint32_t *output_adr, int32_t *npx_out, int threshold, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "default": False},
            ],
                },
            ],
            "expand": {
                "SUFFIX": ['u8', 'u16', 'u32', 'u64', 'i8', 'i16', 'i32', 'i64', 'f32', 'f64'],
                "TYPE": ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'float', 'double'],
            },
            "default_raise": "TypeError: unsupported output dtype for bslz4_multi_${SUFFIX}",
        },
        {
            "py_sig": "bslz4_csc_multi_${SUFFIX}(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, cursors: buffer, nout: int, codec: int = 2) -> int",
            "doc": "Decode a batch of bitshuffle-LZ4/zstd chunks (${TYPE} pixels) from the same dataset into per-frame sparse (outpx, output_adr, npx_out) and per-frame CSC powder integrations (powder). Routes each (frame, block) between a dense and a sparse CSC strategy based on that block's own compression ratio -- see bslz4_core.hpp's bslz4_csc_decode_multi.",
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
                {
                    "map": {
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
                    },
            "variants": [
                {"sig": "bslz4_csc_multi_${SUFFIX}_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "default": True},
                {"sig": "bslz4_csc_multi_${SUFFIX}_sse(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "when": "c2py_amd64_sse2", "default": False},
                {"sig": "bslz4_csc_multi_${SUFFIX}_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "default": False},
            ],
                },
            ],
            "expand": {
                "SUFFIX": ['u8', 'u16', 'u32', 'u64', 'i8', 'i16', 'i32', 'i64', 'f32', 'f64'],
                "TYPE": ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'float', 'double'],
            },
            "default_raise": "TypeError: unsupported outpx dtype for bslz4_csc_multi_${SUFFIX}",
        },
        {
            "py_sig": "bslz4_csc_multi_base_${SUFFIX}(base: buffer, offsets: buffer, lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, cursors: buffer, nout: int, codec: int = 2) -> int",
            "doc": "Like bslz4_csc_multi_${SUFFIX}, but for chunks that all share one base buffer (e.g. an mmap'ed HDF5 file): offsets are byte offsets from base rather than absolute addresses, avoiding any Python-side pointer arithmetic -- see harvest_chunk_offsets()/pack_offsets_lengths() in __init__.py. NOTE: offsets is mutated in place into absolute pointers by this call (reused as scratch, like cursors/workspace already are).",
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
                {
                    "map": {
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
                    },
            "variants": [
                {"sig": "bslz4_csc_multi_base_${SUFFIX}_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "default": True},
                {"sig": "bslz4_csc_multi_base_${SUFFIX}_sse(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "when": "c2py_amd64_sse2", "default": False},
                {"sig": "bslz4_csc_multi_base_${SUFFIX}_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, ${TYPE} *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int", "default": False},
            ],
                },
            ],
            "expand": {
                "SUFFIX": ['u8', 'u16', 'u32', 'u64', 'i8', 'i16', 'i32', 'i64', 'f32', 'f64'],
                "TYPE": ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'float', 'double'],
            },
            "default_raise": "TypeError: unsupported outpx dtype for bslz4_csc_multi_base_${SUFFIX}",
        },
    ],
}
C2PY_END */
