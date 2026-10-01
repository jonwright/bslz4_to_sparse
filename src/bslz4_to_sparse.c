/*
 * bslz4_to_sparse.c -- native entry points for the refactored core.
 *
 * Two dtype-agnostic decode entry points (bslz4_sparsify,
 * bslz4_sparsify_and_dot) plus bslz4_offsets_to_pointers, bslz4_note_chunk,
 * availability (bslz4_impl_available) and test counters.  The dtype-generic
 * work is the inner per-block call reached through bslz4_resolve; the
 * block/tail loop is bslz4_driver.c.  The c2py spec block at the bottom is
 * the hand-written one (regenerate the wrapper with
 * tools/regenerate_wrapper.py).
 */

#include "bslz4_common.h"
#include "bslz4_codec.h"
#include "bslz4_untranspose.h"
#include "bslz4_collect_caps.h"
#include "bslz4_registry.h"

#include <stdint.h>

int bslz4_sparsify(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes,
                   const uint8_t *mask, int NIJ,
                   void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                   uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                   int dtype, uint64_t pipeline) {
    bslz4_stage st;
    int rc = bslz4_resolve(dtype, pipeline, &st);
    if (rc) return rc;
    int codec = BSLZ4_PIPE_DECOMPRESS(pipeline);
    rc = bslz4_driver_sparsify(compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
                               outpx, output_adr, npx_out, threshold,
                               workspace, workspace_len, cursors, &st);
    if (rc == 0) {
        bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, BSLZ4_PIPE_DECOMPRESS(pipeline));
        bslz4_counters_bump(BSLZ4_STAGE_UNTRANSPOSE, BSLZ4_PIPE_UNTRANSPOSE(pipeline));
        bslz4_counters_bump(BSLZ4_STAGE_COLLECT, bslz4_active_collect_tier());
        bslz4_counters_bump(BSLZ4_STAGE_DOT, BSLZ4_PIPE_DOT(pipeline));
    }
    return rc;
}

int bslz4_sparsify_and_dot(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                           int nframes, const uint8_t *mask, int NIJ,
                           void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                           double *powder, int nout,
                           const float *weights, const uint32_t *indices, const uint32_t *indptr,
                           double route_threshold,
                           uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                           int dtype, uint64_t pipeline) {
    bslz4_stage st;
    int rc = bslz4_resolve(dtype, pipeline, &st);
    if (rc) return rc;
    int codec = BSLZ4_PIPE_DECOMPRESS(pipeline);
    rc = bslz4_driver_sparsify_and_dot(compressed_ptrs, compressed_lengths, nframes, codec, mask, NIJ,
                                       outpx, output_adr, npx_out, threshold,
                                       powder, nout, weights, indices, indptr, route_threshold,
                                       workspace, workspace_len, cursors, &st);
    if (rc == 0) {
        bslz4_counters_bump(BSLZ4_STAGE_DECOMPRESS, BSLZ4_PIPE_DECOMPRESS(pipeline));
        bslz4_counters_bump(BSLZ4_STAGE_UNTRANSPOSE, BSLZ4_PIPE_UNTRANSPOSE(pipeline));
        bslz4_counters_bump(BSLZ4_STAGE_COLLECT, bslz4_active_collect_tier());
        bslz4_counters_bump(BSLZ4_STAGE_DOT, BSLZ4_PIPE_DOT(pipeline));
    }
    return rc;
}

int bslz4_offsets_to_pointers(const char *base, size_t base_len, int64_t *offsets,
                              const int32_t *lengths, int nframes) {
    /* Every (offset, length) must lie inside the base_len bytes of base.
     * All checked before any is turned into a pointer, so a bad one leaves
     * offsets untouched.  Written to be overflow-safe. */
    for (int f = 0; f < nframes; f++) {
        if (offsets[f] < 0 || lengths[f] < 0 ||
            (uint64_t) offsets[f] > (uint64_t) base_len ||
            (uint64_t) lengths[f] > (uint64_t) base_len - (uint64_t) offsets[f])
            return BSLZ4_ERR_BAD_CHUNK_BOUNDS;
    }
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return 0;
}

void bslz4_note_chunk(const char *chunk, size_t chunk_len, int index,
                      int64_t *pointers, int32_t *lengths) {
    pointers[index] = (int64_t) (intptr_t) chunk;
    lengths[index] = chunk_len > (size_t) INT32_MAX ? -1 : (int32_t) chunk_len;
}

/* C2PY_BEGIN
{
    "module": "_bslz4_to_sparse",
    "source": ["bslz4_to_sparse.c"],
    "headers": ["c2py_amd64.h", "c2py_arm64.h", "c2py_ppc64.h"],
    "timing": False,
    "functions": [
        {
            "py_sig": "sparsify(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, workspace: buffer, cursors: buffer, dtype: int, pipeline: int) -> int",
            "doc": "Decode nframes bitshuffle-LZ4/zstd chunks from the same dataset into per-frame masked/thresholded sparse (outpx, output_adr, npx_out). dtype is the pixel dtype index (0..9); pipeline packs the stage ids (decompress/untranspose/collect/dot + options).",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I' or output_adr.format == 'L'",
                "npx_out.format == 'i' or npx_out.format == 'l'",
                "workspace.format == 'B'",
                "compressed_ptrs.itemsize == 8",
                "compressed_lengths.itemsize == 4",
                "compressed_lengths.n == compressed_ptrs.n",
                "cursors.itemsize == 8",
            ],
            "c_overloads": [
                {"sig": "bslz4_sparsify(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, const uint8_t *mask, int NIJ, void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, uint8_t *workspace, size_t workspace_len, int64_t *cursors, int dtype, uint64_t pipeline) -> int", "map": {"compressed_ptrs": "compressed_ptrs.ptr", "compressed_lengths": "compressed_lengths.ptr", "nframes": "compressed_ptrs.n", "mask": "mask.ptr", "NIJ": "mask.n", "outpx": "outpx.ptr", "output_adr": "output_adr.ptr", "npx_out": "npx_out.ptr", "threshold": "threshold", "workspace": "workspace.ptr", "workspace_len": "workspace.len", "cursors": "cursors.ptr", "dtype": "dtype", "pipeline": "pipeline"}},
            ],
        },
        {
            "py_sig": "sparsify_and_dot(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, cursors: buffer, nout: int, route_threshold: float, dtype: int, pipeline: int) -> int",
            "doc": "Decode a batch of chunks into per-frame sparse (outpx, output_adr, npx_out) and per-frame CSC powder integrations (powder). dtype is the pixel dtype index; pipeline packs the stage ids.",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I' or output_adr.format == 'L'",
                "npx_out.format == 'i' or npx_out.format == 'l'",
                "powder.format == 'd'",
                "workspace.format == 'B'",
                "compressed_ptrs.itemsize == 8",
                "compressed_lengths.itemsize == 4",
                "compressed_lengths.n == compressed_ptrs.n",
                "cursors.itemsize == 8",
                "data.format == 'f'",
                "indices.format == 'I' or indices.format == 'i' or indices.format == 'L' or indices.format == 'l'",
                "indptr.format == 'I' or indptr.format == 'i' or indptr.format == 'L' or indptr.format == 'l'",
                "indices.n == data.n",
                "indptr.n == mask.n + 1",
            ],
            "c_overloads": [
                {"sig": "bslz4_sparsify_and_dot(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, const uint8_t *mask, int NIJ, void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *powder, int nout, const float *weights, const uint32_t *indices, const uint32_t *indptr, double route_threshold, uint8_t *workspace, size_t workspace_len, int64_t *cursors, int dtype, uint64_t pipeline) -> int", "map": {"compressed_ptrs": "compressed_ptrs.ptr", "compressed_lengths": "compressed_lengths.ptr", "nframes": "compressed_ptrs.n", "mask": "mask.ptr", "NIJ": "mask.n", "outpx": "outpx.ptr", "output_adr": "output_adr.ptr", "npx_out": "npx_out.ptr", "threshold": "threshold", "powder": "powder.ptr", "nout": "nout", "weights": "data.ptr", "indices": "indices.ptr", "indptr": "indptr.ptr", "route_threshold": "route_threshold", "workspace": "workspace.ptr", "workspace_len": "workspace.len", "cursors": "cursors.ptr", "dtype": "dtype", "pipeline": "pipeline"}},
            ],
        },
        {
            "py_sig": "offsets_to_pointers(base: buffer, offsets: buffer, lengths: buffer, nframes: int) -> int",
            "doc": "Convert byte offsets (into base) to absolute pointers in place, after checking every (offset, length) lies inside base. Returns 0, or -108 leaving offsets untouched on a bad one.",
            "checks": [
                "offsets.itemsize == 8",
                "lengths.itemsize == 4",
                "lengths.n == offsets.n",
            ],
            "c_overloads": [
                {"sig": "bslz4_offsets_to_pointers(const char *base, size_t base_len, int64_t *offsets, const int32_t *lengths, int nframes) -> int", "map": {"base": "base.ptr", "base_len": "base.len", "offsets": "offsets.ptr", "lengths": "lengths.ptr", "nframes": "offsets.n"}},
            ],
        },
        {
            "py_sig": "note_chunk(chunk: buffer, index: int, pointers: buffer, lengths: buffer) -> void",
            "doc": "Write chunk's raw address and byte length into pointers[index]/lengths[index].",
            "c_overloads": [
                {"sig": "bslz4_note_chunk(const char *chunk, size_t chunk_len, int index, int64_t *pointers, int32_t *lengths) -> void", "map": {"chunk": "chunk.ptr", "chunk_len": "chunk.len", "index": "index", "pointers": "pointers.ptr", "lengths": "lengths.ptr"}},
            ],
        },
        {
            "py_sig": "impl_available(stage: int, id: int) -> int",
            "doc": "1 if a known implementation is available here, 0 if known but not usable on this CPU/build, -1 if the id is unknown.",
            "c_overloads": [
                {"sig": "bslz4_impl_available(int stage, int id) -> int", "map": {"stage": "stage", "id": "id"}},
            ],
        },
        {
            "py_sig": "reset_counters() -> void",
            "doc": "Zero all per-implementation block counters (test instrumentation).",
            "c_overloads": [
                {"sig": "bslz4_reset_counters() -> void", "map": {}},
            ],
        },
        {
            "py_sig": "read_counters(out: buffer) -> int",
            "doc": "Fill out (uint64 array) with the flattened [stage][impl] counters; returns the number of entries written.",
            "c_overloads": [
                {"sig": "bslz4_read_counters(uint64_t *out, int n) -> int", "map": {"out": "out.ptr", "n": "out.n"}},
            ],
        },
        {
            "py_sig": "avx512_collect_available() -> int",
            "doc": "1 if the AVX-512 collect kernel is usable here, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_available_avx512_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "get_avx512_collect() -> int",
            "doc": "1 if the AVX-512 collect tier is enabled, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_get_avx512_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "set_avx512_collect(enabled: int) -> int",
            "doc": "Enable/disable the AVX-512 collect tier; returns 0, or -1 if enabling an unavailable tier.",
            "c_overloads": [{"sig": "bslz4_set_avx512_collect(int enabled) -> int", "map": {"enabled": "enabled"}}],
        },
        {
            "py_sig": "avx2_collect_available() -> int",
            "doc": "1 if the AVX2 collect kernel is usable here, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_available_avx2_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "get_avx2_collect() -> int",
            "doc": "1 if the AVX2 collect tier is enabled, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_get_avx2_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "set_avx2_collect(enabled: int) -> int",
            "doc": "Enable/disable the AVX2 collect tier; returns 0, or -1 if enabling an unavailable tier.",
            "c_overloads": [{"sig": "bslz4_set_avx2_collect(int enabled) -> int", "map": {"enabled": "enabled"}}],
        },
        {
            "py_sig": "sse2_collect_available() -> int",
            "doc": "1 if the SSE2 collect kernel is usable here, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_available_sse2_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "get_sse2_collect() -> int",
            "doc": "1 if the SSE2 collect tier is enabled, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_get_sse2_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "set_sse2_collect(enabled: int) -> int",
            "doc": "Enable/disable the SSE2 collect tier; returns 0, or -1 if enabling an unavailable tier.",
            "c_overloads": [{"sig": "bslz4_set_sse2_collect(int enabled) -> int", "map": {"enabled": "enabled"}}],
        },
        {
            "py_sig": "vsx_collect_available() -> int",
            "doc": "1 if the VSX collect kernel is usable here, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_available_vsx_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "get_vsx_collect() -> int",
            "doc": "1 if the VSX collect tier is enabled, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_get_vsx_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "set_vsx_collect(enabled: int) -> int",
            "doc": "Enable/disable the VSX collect tier; returns 0, or -1 if enabling an unavailable tier.",
            "c_overloads": [{"sig": "bslz4_set_vsx_collect(int enabled) -> int", "map": {"enabled": "enabled"}}],
        },
        {
            "py_sig": "neon_collect_available() -> int",
            "doc": "1 if the NEON collect kernel is usable here, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_available_neon_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "get_neon_collect() -> int",
            "doc": "1 if the NEON collect tier is enabled, 0 otherwise.",
            "c_overloads": [{"sig": "bslz4_get_neon_collect() -> int", "map": {}}],
        },
        {
            "py_sig": "set_neon_collect(enabled: int) -> int",
            "doc": "Enable/disable the NEON collect tier; returns 0, or -1 if enabling an unavailable tier.",
            "c_overloads": [{"sig": "bslz4_set_neon_collect(int enabled) -> int", "map": {"enabled": "enabled"}}],
        },
    ],
}
C2PY_END */
