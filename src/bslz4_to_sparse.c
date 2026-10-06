/*
 * bslz4_to_sparse.c -- the C entry points (called through the generated
 * c2py23 wrapper, bslz4_to_sparse_wrapper.c): check the pipeline
 * (src/pipeline/registry.c), describe the matrix layout and run the frame /
 * block loop (src/pipeline/driver.c).  The c2py spec at the bottom is
 * hand-written; regenerate the wrapper with tools/regenerate_wrapper.py.
 */

#include "pipeline/pipeline.h"

#include <stdint.h>

int bslz4_sparsify(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes,
                   const uint8_t *mask, int NIJ,
                   void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                   uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                   int dtype, const uint16_t *pipeline) {
    bslz4_pipe p;
    int rc = bslz4_resolve(dtype, pipeline, -1, &p);
    if (rc) return rc;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, mask, NIJ,
                            outpx, output_adr, npx_out, threshold, NULL, 0, NULL,
                            workspace, workspace_len, cursors, &p);
}

int bslz4_sparsify_and_dot(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                           int nframes, const uint8_t *mask, int NIJ,
                           void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                           void *powder, int nout,
                           const void *weights, const void *indices, const uint32_t *indptr,
                           uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                           int dtype, const uint16_t *pipeline) {
    bslz4_pipe p;
    int rc = bslz4_resolve(dtype, pipeline, BSLZ4_LAYOUT_CSC, &p);
    if (rc) return rc;
    bslz4_mat_csc m;
    m.data = weights;
    m.indices = (const uint32_t *) indices;
    m.indptr = indptr;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, mask, NIJ,
                            outpx, output_adr, npx_out, threshold, powder, nout, &m,
                            workspace, workspace_len, cursors, &p);
}

int bslz4_sparsify_and_dot_padded(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                                  int nframes, const uint8_t *mask, int NIJ,
                                  void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                                  double *powder, int nout,
                                  const int32_t *base, const float *weights,
                                  const int32_t *pixels, const int32_t *rowmap,
                                  const int32_t *row_ptr, int nrow_ptr, int width, int listed,
                                  size_t block_elems,
                                  uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                                  int dtype, const uint16_t *pipeline) {
    bslz4_pipe p;
    int rc = bslz4_resolve(dtype, pipeline, BSLZ4_LAYOUT_PADDED, &p);
    if (rc) return rc;
    bslz4_mat_padded m;
    m.base = base;
    m.weights = weights;
    m.pixels = listed ? pixels : NULL;
    m.rowmap = listed ? rowmap : NULL;
    m.row_ptr = row_ptr;
    m.row_ptr_n = nrow_ptr < 0 ? 0 : (size_t) nrow_ptr;
    m.width = width;
    m.listed = listed;
    m.block_elems = block_elems;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, mask, NIJ,
                            outpx, output_adr, npx_out, threshold, powder, nout, &m,
                            workspace, workspace_len, cursors, &p);
}

int bslz4_sparsify_and_dot_bsbcsr(const int64_t *compressed_ptrs, const int32_t *compressed_lengths,
                                  int nframes, const uint8_t *mask, int NIJ,
                                  void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                                  double *powder, int nout,
                                  const uint32_t *blk_ptr, int nblk_ptr, const uint32_t *bins,
                                  const uint32_t *bin_ptr, const uint16_t *idx,
                                  const float *data,
                                  const float *csc_data, const uint32_t *csc_indices,
                                  const uint32_t *csc_indptr, size_t block_elems,
                                  uint8_t *workspace, size_t workspace_len, int64_t *cursors,
                                  int dtype, const uint16_t *pipeline) {
    bslz4_pipe p;
    int rc = bslz4_resolve(dtype, pipeline, BSLZ4_LAYOUT_BSBCSR, &p);
    if (rc) return rc;
    bslz4_mat_bsbcsr m;
    m.blk_ptr = blk_ptr;
    m.blk_ptr_n = nblk_ptr < 0 ? 0 : (size_t) nblk_ptr;
    m.bins = bins;
    m.bin_ptr = bin_ptr;
    m.idx = idx;
    m.data = data;
    m.csc.data = csc_data;
    m.csc.indices = csc_indices;
    m.csc.indptr = csc_indptr;
    m.block_elems = block_elems;
    return bslz4_driver_run(compressed_ptrs, compressed_lengths, nframes, mask, NIJ,
                            outpx, output_adr, npx_out, threshold, powder, nout, &m,
                            workspace, workspace_len, cursors, &p);
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
            "py_sig": "sparsify(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, workspace: buffer, cursors: buffer, dtype: int, pipeline: buffer) -> int",
            "doc": "Decode nframes bitshuffle-LZ4/zstd chunks from the same dataset into per-frame masked/thresholded sparse (outpx, output_adr, npx_out). dtype is the pixel dtype index (0..9); pipeline is a uint16 array of 6 step values (decode, mask, untranspose, collect, route, dot; route and dot 0), all resolved (src/pipeline/pipeline.h).",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I' or output_adr.format == 'L'",
                "npx_out.format == 'i' or npx_out.format == 'l'",
                "workspace.format == 'B'",
                "compressed_ptrs.itemsize == 8",
                "compressed_lengths.itemsize == 4",
                "compressed_lengths.n == compressed_ptrs.n",
                "cursors.itemsize == 8",
                "pipeline.format == 'H'",
                "pipeline.n == 6",
            ],
            "c_overloads": [
                {"sig": "bslz4_sparsify(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, const uint8_t *mask, int NIJ, void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, uint8_t *workspace, size_t workspace_len, int64_t *cursors, int dtype, const uint16_t *pipeline) -> int", "map": {"compressed_ptrs": "compressed_ptrs.ptr", "compressed_lengths": "compressed_lengths.ptr", "nframes": "compressed_ptrs.n", "mask": "mask.ptr", "NIJ": "mask.n", "outpx": "outpx.ptr", "output_adr": "output_adr.ptr", "npx_out": "npx_out.ptr", "threshold": "threshold", "workspace": "workspace.ptr", "workspace_len": "workspace.len", "cursors": "cursors.ptr", "dtype": "dtype", "pipeline": "pipeline.ptr"}},
            ],
        },
        {
            "py_sig": "sparsify_and_dot(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, cursors: buffer, nout: int, dtype: int, pipeline: buffer) -> int",
            "doc": "Decode a batch of chunks into per-frame sparse (outpx, output_adr, npx_out) and per-frame CSC powder integrations (powder). The matrix must be mask-folded: masked pixels have empty columns (the dense route makes no mask test; the mask selects the sparse output and the sparse-route pixels). Frames must have exactly mask.n pixels. The dot says what indices holds: a bin per entry (indices.n == data.n), or, for the per-pixel dots (csc-run and its integer forms: the first bin of a run; csc-nosplit, csc-nosplit-moment, csc-permute: the one bin), a bin per pixel (indices.n == mask.n); a per-pixel dot must be given per-pixel indices. The dot also says the element types: data f32, or u32/u16 fixed-point weights with an int64 (q) powder, or (csc-permute) a powder in the pixel dtype. dtype is the pixel dtype index; pipeline is a uint16 array of 6 resolved step values (src/pipeline/pipeline.h).",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I' or output_adr.format == 'L'",
                "npx_out.format == 'i' or npx_out.format == 'l'",
                "powder.format == 'd' or ((powder.format == 'q' or powder.format == 'l') and powder.itemsize == 8) or powder.itemsize == 1 or powder.itemsize == 2 or powder.itemsize == 4 or powder.itemsize == 8",
                "workspace.format == 'B'",
                "compressed_ptrs.itemsize == 8",
                "compressed_lengths.itemsize == 4",
                "compressed_lengths.n == compressed_ptrs.n",
                "cursors.itemsize == 8",
                "data.format == 'f' or ((data.format == 'I' or data.format == 'L') and data.itemsize == 4) or data.format == 'H'",
                "(indices.format == 'I' or indices.format == 'i' or indices.format == 'L' or indices.format == 'l') and indices.itemsize == 4",
                "(indptr.format == 'I' or indptr.format == 'i' or indptr.format == 'L' or indptr.format == 'l') and indptr.itemsize == 4",
                "indices.n == data.n or indices.n == mask.n",
                "indptr.n == mask.n + 1",
                "pipeline.format == 'H'",
                "pipeline.n == 6",
            ],
            "c_overloads": [
                {"sig": "bslz4_sparsify_and_dot(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, const uint8_t *mask, int NIJ, void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, void *powder, int nout, const void *weights, const void *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors, int dtype, const uint16_t *pipeline) -> int", "map": {"compressed_ptrs": "compressed_ptrs.ptr", "compressed_lengths": "compressed_lengths.ptr", "nframes": "compressed_ptrs.n", "mask": "mask.ptr", "NIJ": "mask.n", "outpx": "outpx.ptr", "output_adr": "output_adr.ptr", "npx_out": "npx_out.ptr", "threshold": "threshold", "powder": "powder.ptr", "nout": "nout", "weights": "data.ptr", "indices": "indices.ptr", "indptr": "indptr.ptr", "workspace": "workspace.ptr", "workspace_len": "workspace.len", "cursors": "cursors.ptr", "dtype": "dtype", "pipeline": "pipeline.ptr"}},
            ],
        },
        {
            "py_sig": "sparsify_and_dot_padded(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, base: buffer, weights: buffer, pixels: buffer, rowmap: buffer, row_ptr: buffer, workspace: buffer, cursors: buffer, width: int, listed: int, block_elems: int, nout: int, dtype: int, pipeline: buffer) -> int",
            "doc": "Decode a batch of chunks into per-frame sparse and a padded-CSC powder integration. base/weights/pixels/rowmap/row_ptr describe the padded layout (see bslz4_mat_padded), built from a mask-folded matrix (masked pixels have zero-weight rows or no row); width and listed are scalars; block_elems is the decode block size in pixels. dtype is the pixel dtype index; pipeline is a uint16 array of 6 resolved step values.",
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
                "(base.format == 'i' or base.format == 'l') and base.itemsize == 4",
                "weights.format == 'f'",
                "(pixels.format == 'i' or pixels.format == 'l') and pixels.itemsize == 4",
                "(rowmap.format == 'i' or rowmap.format == 'l') and rowmap.itemsize == 4",
                "(row_ptr.format == 'i' or row_ptr.format == 'l') and row_ptr.itemsize == 4",
                "width >= 1 and width <= 64",
                "listed == 0 or listed == 1",
                "weights.n == base.n * width",
                "listed == 0 or pixels.n == base.n",
                "listed == 0 or rowmap.n == mask.n",
                "listed == 1 or base.n == mask.n",
                "block_elems >= 1",
                "pipeline.format == 'H'",
                "pipeline.n == 6",
            ],
            "c_overloads": [
                {"sig": "bslz4_sparsify_and_dot_padded(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, const uint8_t *mask, int NIJ, void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *powder, int nout, const int32_t *base, const float *weights, const int32_t *pixels, const int32_t *rowmap, const int32_t *row_ptr, int nrow_ptr, int width, int listed, size_t block_elems, uint8_t *workspace, size_t workspace_len, int64_t *cursors, int dtype, const uint16_t *pipeline) -> int", "map": {"compressed_ptrs": "compressed_ptrs.ptr", "compressed_lengths": "compressed_lengths.ptr", "nframes": "compressed_ptrs.n", "mask": "mask.ptr", "NIJ": "mask.n", "outpx": "outpx.ptr", "output_adr": "output_adr.ptr", "npx_out": "npx_out.ptr", "threshold": "threshold", "powder": "powder.ptr", "nout": "nout", "base": "base.ptr", "weights": "weights.ptr", "pixels": "pixels.ptr", "rowmap": "rowmap.ptr", "row_ptr": "row_ptr.ptr", "nrow_ptr": "row_ptr.n", "width": "width", "listed": "listed", "block_elems": "block_elems", "workspace": "workspace.ptr", "workspace_len": "workspace.len", "cursors": "cursors.ptr", "dtype": "dtype", "pipeline": "pipeline.ptr"}},
            ],
        },
        {
            "py_sig": "sparsify_and_dot_bsbcsr(compressed_ptrs: buffer, compressed_lengths: buffer, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, blk_ptr: buffer, bins: buffer, bin_ptr: buffer, idx: buffer, data: buffer, csc_data: buffer, csc_indices: buffer, csc_indptr: buffer, workspace: buffer, cursors: buffer, block_elems: int, nout: int, dtype: int, pipeline: buffer) -> int",
            "doc": "Decode a batch of chunks into per-frame sparse and a bit-shuffle-block CSR powder integration. blk_ptr/bins/bin_ptr/idx/data describe the bsb-csr layout (see bslz4_mat_bsbcsr); csc_data/csc_indices/csc_indptr is the CSC used for the sparse route. Both must be built from a mask-folded matrix (masked pixels have no entries). block_elems is the decode block size in pixels. dtype is the pixel dtype index; pipeline is a uint16 array of 6 resolved step values.",
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
                "(blk_ptr.format == 'I' or blk_ptr.format == 'L') and blk_ptr.itemsize == 4",
                "(bins.format == 'I' or bins.format == 'L') and bins.itemsize == 4",
                "(bin_ptr.format == 'I' or bin_ptr.format == 'L') and bin_ptr.itemsize == 4",
                "idx.format == 'H'",
                "data.format == 'f'",
                "csc_data.format == 'f'",
                "(csc_indices.format == 'I' or csc_indices.format == 'i' or csc_indices.format == 'L' or csc_indices.format == 'l') and csc_indices.itemsize == 4",
                "(csc_indptr.format == 'I' or csc_indptr.format == 'i' or csc_indptr.format == 'L' or csc_indptr.format == 'l') and csc_indptr.itemsize == 4",
                "bin_ptr.n == bins.n + 1",
                "idx.n == data.n",
                "csc_indices.n == csc_data.n",
                "csc_indptr.n == mask.n + 1",
                "block_elems >= 1 and block_elems <= 65536",
                "pipeline.format == 'H'",
                "pipeline.n == 6",
            ],
            "c_overloads": [
                {"sig": "bslz4_sparsify_and_dot_bsbcsr(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, const uint8_t *mask, int NIJ, void *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *powder, int nout, const uint32_t *blk_ptr, int nblk_ptr, const uint32_t *bins, const uint32_t *bin_ptr, const uint16_t *idx, const float *data, const float *csc_data, const uint32_t *csc_indices, const uint32_t *csc_indptr, size_t block_elems, uint8_t *workspace, size_t workspace_len, int64_t *cursors, int dtype, const uint16_t *pipeline) -> int", "map": {"compressed_ptrs": "compressed_ptrs.ptr", "compressed_lengths": "compressed_lengths.ptr", "nframes": "compressed_ptrs.n", "mask": "mask.ptr", "NIJ": "mask.n", "outpx": "outpx.ptr", "output_adr": "output_adr.ptr", "npx_out": "npx_out.ptr", "threshold": "threshold", "powder": "powder.ptr", "nout": "nout", "blk_ptr": "blk_ptr.ptr", "nblk_ptr": "blk_ptr.n", "bins": "bins.ptr", "bin_ptr": "bin_ptr.ptr", "idx": "idx.ptr", "data": "data.ptr", "csc_data": "csc_data.ptr", "csc_indices": "csc_indices.ptr", "csc_indptr": "csc_indptr.ptr", "block_elems": "block_elems", "workspace": "workspace.ptr", "workspace_len": "workspace.len", "cursors": "cursors.ptr", "dtype": "dtype", "pipeline": "pipeline.ptr"}},
            ],
        },
        {
            "py_sig": "offsets_to_pointers(base: buffer, offsets: buffer, lengths: buffer) -> int",
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
            "py_sig": "step_available(step: int, value: int) -> int",
            "doc": "1 if a pipeline step value can run here, 0 if it is known but this CPU/build lacks what it needs, -1 if it is unknown (0, auto, included).",
            "c_overloads": [
                {"sig": "bslz4_step_available(int step, int value) -> int", "map": {"step": "step", "value": "value"}},
            ],
        },
        {
            "py_sig": "reset_counters() -> void",
            "doc": "Zero the per step value block counters (test instrumentation).",
            "c_overloads": [
                {"sig": "bslz4_reset_counters() -> void", "map": {}},
            ],
        },
        {
            "py_sig": "read_counters(out: buffer) -> int",
            "doc": "Fill out (uint64 array) with the flattened [step][value] block counters; returns the number of entries written.",
            "c_overloads": [
                {"sig": "bslz4_read_counters(uint64_t *out, int n) -> int", "map": {"out": "out.ptr", "n": "out.n"}},
            ],
        },
        {
            "py_sig": "build_info(out: buffer) -> int",
            "doc": "Copy the JSON build description (version, git describe, source sha256, compiler, platform, time) into out (uint8 array); returns its full length in bytes, which may exceed out.len.",
            "checks": [
                "out.format == 'B'",
            ],
            "c_overloads": [
                {"sig": "bslz4_build_info(char *out, int n) -> int", "map": {"out": "out.ptr", "n": "out.len"}},
            ],
        },
    ],
}
C2PY_END */
