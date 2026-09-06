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
 * Two untranspose backends are wired in and compiled together: kcb
 * (https://github.com/kalcutter/bitshuffle, does its own CPU dispatch,
 * the default) and the portable scalar kernel from upstream bitshuffle
 * (https://github.com/kiyo-masui/bitshuffle). Both are selectable from
 * Python via the generated _rebind_/_variants_ functions, so the two can
 * be benchmarked against each other rather than one being silently fixed
 * at build time. SSE2/AVX2/AVX512/NEON variants of the upstream kernel are
 * a follow-up (they need per-ISA compiled objects; see issue #11 notes).
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
 */

#include "bslz4_backends.hpp"
#include "bslz4_core.hpp"

using namespace bslz4;

extern "C" {

int bslz4_u8_kcb(const char *compressed, int compressed_length, int codec,
                  const uint8_t *mask, int NIJ,
                  uint8_t *output, uint32_t *output_adr, int threshold,
                  uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint8_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u8_scal(const char *compressed, int compressed_length, int codec,
                   const uint8_t *mask, int NIJ,
                   uint8_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint8_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u16_kcb(const char *compressed, int compressed_length, int codec,
                   const uint8_t *mask, int NIJ,
                   uint16_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint16_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u16_scal(const char *compressed, int compressed_length, int codec,
                    const uint8_t *mask, int NIJ,
                    uint16_t *output, uint32_t *output_adr, int threshold,
                    uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint16_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u32_kcb(const char *compressed, int compressed_length, int codec,
                   const uint8_t *mask, int NIJ,
                   uint32_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint32_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u32_scal(const char *compressed, int compressed_length, int codec,
                    const uint8_t *mask, int NIJ,
                    uint32_t *output, uint32_t *output_adr, int threshold,
                    uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint32_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_csc_u8_kcb(const char *compressed, int compressed_length, int codec,
                      const uint8_t *mask, int NIJ,
                      uint8_t *outpx, uint32_t *output_adr, int threshold,
                      double *output, int NOUT,
                      const float *data, const uint32_t *indices, const uint32_t *indptr,
                      uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint8_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u8_scal(const char *compressed, int compressed_length, int codec,
                       const uint8_t *mask, int NIJ,
                       uint8_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint8_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u16_kcb(const char *compressed, int compressed_length, int codec,
                       const uint8_t *mask, int NIJ,
                       uint16_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint16_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u16_scal(const char *compressed, int compressed_length, int codec,
                        const uint8_t *mask, int NIJ,
                        uint16_t *outpx, uint32_t *output_adr, int threshold,
                        double *output, int NOUT,
                        const float *data, const uint32_t *indices, const uint32_t *indptr,
                        uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint16_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u32_kcb(const char *compressed, int compressed_length, int codec,
                       const uint8_t *mask, int NIJ,
                       uint32_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint32_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u32_scal(const char *compressed, int compressed_length, int codec,
                        const uint8_t *mask, int NIJ,
                        uint32_t *outpx, uint32_t *output_adr, int threshold,
                        double *output, int NOUT,
                        const float *data, const uint32_t *indices, const uint32_t *indptr,
                        uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint32_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_i8_kcb(const char *compressed, int compressed_length, int codec,
                  const uint8_t *mask, int NIJ,
                  int8_t *output, uint32_t *output_adr, int threshold,
                  uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<int8_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_i8_scal(const char *compressed, int compressed_length, int codec,
                   const uint8_t *mask, int NIJ,
                   int8_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<int8_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_i16_kcb(const char *compressed, int compressed_length, int codec,
                  const uint8_t *mask, int NIJ,
                  int16_t *output, uint32_t *output_adr, int threshold,
                  uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<int16_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_i16_scal(const char *compressed, int compressed_length, int codec,
                   const uint8_t *mask, int NIJ,
                   int16_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<int16_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_i32_kcb(const char *compressed, int compressed_length, int codec,
                  const uint8_t *mask, int NIJ,
                  int32_t *output, uint32_t *output_adr, int threshold,
                  uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<int32_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_i32_scal(const char *compressed, int compressed_length, int codec,
                   const uint8_t *mask, int NIJ,
                   int32_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<int32_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_f32_kcb(const char *compressed, int compressed_length, int codec,
                  const uint8_t *mask, int NIJ,
                  float *output, uint32_t *output_adr, int threshold,
                  uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<float, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_f32_scal(const char *compressed, int compressed_length, int codec,
                   const uint8_t *mask, int NIJ,
                   float *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<float, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_f64_kcb(const char *compressed, int compressed_length, int codec,
                  const uint8_t *mask, int NIJ,
                  double *output, uint32_t *output_adr, int threshold,
                  uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<double, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_f64_scal(const char *compressed, int compressed_length, int codec,
                   const uint8_t *mask, int NIJ,
                   double *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<double, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_csc_i8_kcb(const char *compressed, int compressed_length, int codec,
                      const uint8_t *mask, int NIJ,
                      int8_t *outpx, uint32_t *output_adr, int threshold,
                      double *output, int NOUT,
                      const float *data, const uint32_t *indices, const uint32_t *indptr,
                      uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<int8_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_i8_scal(const char *compressed, int compressed_length, int codec,
                       const uint8_t *mask, int NIJ,
                       int8_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<int8_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_i16_kcb(const char *compressed, int compressed_length, int codec,
                      const uint8_t *mask, int NIJ,
                      int16_t *outpx, uint32_t *output_adr, int threshold,
                      double *output, int NOUT,
                      const float *data, const uint32_t *indices, const uint32_t *indptr,
                      uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<int16_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_i16_scal(const char *compressed, int compressed_length, int codec,
                       const uint8_t *mask, int NIJ,
                       int16_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<int16_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_i32_kcb(const char *compressed, int compressed_length, int codec,
                      const uint8_t *mask, int NIJ,
                      int32_t *outpx, uint32_t *output_adr, int threshold,
                      double *output, int NOUT,
                      const float *data, const uint32_t *indices, const uint32_t *indptr,
                      uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<int32_t, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_i32_scal(const char *compressed, int compressed_length, int codec,
                       const uint8_t *mask, int NIJ,
                       int32_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<int32_t, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_f32_kcb(const char *compressed, int compressed_length, int codec,
                      const uint8_t *mask, int NIJ,
                      float *outpx, uint32_t *output_adr, int threshold,
                      double *output, int NOUT,
                      const float *data, const uint32_t *indices, const uint32_t *indptr,
                      uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<float, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_f32_scal(const char *compressed, int compressed_length, int codec,
                       const uint8_t *mask, int NIJ,
                       float *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<float, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_f64_kcb(const char *compressed, int compressed_length, int codec,
                      const uint8_t *mask, int NIJ,
                      double *outpx, uint32_t *output_adr, int threshold,
                      double *output, int NOUT,
                      const float *data, const uint32_t *indices, const uint32_t *indptr,
                      uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<double, untranspose_kcb>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_f64_scal(const char *compressed, int compressed_length, int codec,
                       const uint8_t *mask, int NIJ,
                       double *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<double, untranspose_bshuf_scal>(
        compressed, compressed_length, codec, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
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

void bslz4_note_chunk(const char *chunk, size_t chunk_len, int index,
                       int64_t *pointers, int32_t *lengths) {
    pointers[index] = (int64_t) (intptr_t) chunk;
    lengths[index] = (int32_t) chunk_len;
}

int bslz4_csc_multi_base_u8_kcb(const char *base, int64_t *offsets, const int32_t *lengths,
                                   int nframes, int codec,
                                   const uint8_t *mask, int NIJ,
                                   uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold,
                                   double *output, int NOUT,
                                   const float *data, const uint32_t *indices, const uint32_t *indptr,
                                   uint8_t *workspace, size_t workspace_len, int64_t *cursors) {
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint8_t, untranspose_kcb>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint16_t, untranspose_kcb>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint32_t, untranspose_kcb>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<uint32_t, untranspose_bshuf_scal>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int8_t, untranspose_kcb>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int16_t, untranspose_kcb>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int32_t, untranspose_kcb>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<int32_t, untranspose_bshuf_scal>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<float, untranspose_kcb>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
    for (int f = 0; f < nframes; f++) {
        offsets[f] = (int64_t) (intptr_t) (base + offsets[f]);
    }
    return bslz4_csc_decode_multi<double, untranspose_kcb>(
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
    /* offsets is caller-owned scratch: turn it into absolute pointers in place,
     * avoiding a second (nframes-sized) allocation just to hold them. */
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
    "timing": True,
    "functions": [
        {
            "py_sig": "bslz4(compressed: buffer, mask: buffer, output: buffer, output_adr: buffer, workspace: buffer, threshold: int, codec: int = 2) -> int",
            "doc": "Decode one bitshuffle-LZ4 chunk into masked/thresholded sparse (output, output_adr).",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I'",
                "workspace.format == 'B'",
            ],
            "c_overloads": [
                {
                    "when": "output.format == 'B'",
                    "group": "u8",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_u8_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint8_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_u8_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint8_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'H'",
                    "group": "u16",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_u16_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint16_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_u16_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint16_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'I'",
                    "group": "u32",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_u32_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint32_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_u32_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint32_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'b'",
                    "group": "i8",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_i8_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int8_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_i8_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int8_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'h'",
                    "group": "i16",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_i16_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int16_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_i16_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int16_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'i'",
                    "group": "i32",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_i32_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int32_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_i32_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int32_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'f'",
                    "group": "f32",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_f32_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, float *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_f32_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, float *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'd'",
                    "group": "f64",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "output": "output.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_f64_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, double *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_f64_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, double *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
            ],
            "default_raise": "TypeError: expected uint8, int8, uint16, int16, uint32, int32, float32 or float64 output buffer",
        },
        {
            "py_sig": "bslz4_csc(compressed: buffer, mask: buffer, outpx: buffer, output_adr: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, codec: int = 2) -> int",
            "doc": "Decode one bitshuffle-LZ4 chunk into sparse (outpx, output_adr) and accumulate a CSC powder integration into powder.",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I'",
                "powder.format == 'd'",
                "data.format == 'f'",
                "indices.format == 'I' or indices.format == 'i'",
                "indptr.format == 'I' or indptr.format == 'i'",
                "workspace.format == 'B'",
            ],
            "c_overloads": [
                {
                    "when": "outpx.format == 'B'",
                    "group": "csc_u8",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "powder.n",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_csc_u8_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint8_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_u8_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint8_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'H'",
                    "group": "csc_u16",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "powder.n",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_csc_u16_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint16_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_u16_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint16_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'I'",
                    "group": "csc_u32",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "powder.n",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_csc_u32_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint32_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_u32_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, uint32_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'b'",
                    "group": "csc_i8",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "powder.n",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_csc_i8_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int8_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_i8_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int8_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'h'",
                    "group": "csc_i16",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "powder.n",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_csc_i16_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int16_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_i16_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int16_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'i'",
                    "group": "csc_i32",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "powder.n",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_csc_i32_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int32_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_i32_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, int32_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'f'",
                    "group": "csc_f32",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "powder.n",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_csc_f32_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, float *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_f32_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, float *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'd'",
                    "group": "csc_f64",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
                        "codec": "codec",
                        "mask": "mask.ptr",
                        "NIJ": "mask.n",
                        "outpx": "outpx.ptr",
                        "output_adr": "output_adr.ptr",
                        "threshold": "threshold",
                        "output": "powder.ptr",
                        "NOUT": "powder.n",
                        "data": "data.ptr",
                        "indices": "indices.ptr",
                        "indptr": "indptr.ptr",
                        "workspace": "workspace.ptr",
                        "workspace_len": "workspace.len",
                    },
                    "variants": [
                        {
                            "sig": "bslz4_csc_f64_kcb(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, double *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_f64_scal(const char *compressed, int compressed_length, int codec, const uint8_t *mask, int NIJ, double *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
            ],
            "default_raise": "TypeError: expected uint8, int8, uint16, int16, uint32, int32, float32 or float64 outpx buffer",
        },
        {
            "py_sig": "bslz4_csc_multi(compressed_ptrs: buffer, compressed_lengths: buffer, nframes: int, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, cursors: buffer, nout: int, codec: int = 2) -> int",
            "doc": "Decode a batch of bitshuffle-LZ4/zstd chunks from the same dataset into per-frame sparse (outpx, output_adr, npx_out) and per-frame CSC powder integrations (powder), amortizing the CSC lookup across frames for cache locality (issue #12).",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I'",
                "npx_out.format == 'i'",
                "powder.format == 'd'",
                "data.format == 'f'",
                "indices.format == 'I' or indices.format == 'i'",
                "indptr.format == 'I' or indptr.format == 'i'",
                "workspace.format == 'B'",
            ],
            "c_overloads": [
                {
                    "when": "outpx.format == 'B'",
                    "group": "multi_u8",
                    "map": {
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_u8_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_u8_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'H'",
                    "group": "multi_u16",
                    "map": {
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_u16_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_u16_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'I'",
                    "group": "multi_u32",
                    "map": {
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_u32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_u32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'b'",
                    "group": "multi_i8",
                    "map": {
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_i8_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_i8_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'h'",
                    "group": "multi_i16",
                    "map": {
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_i16_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_i16_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'i'",
                    "group": "multi_i32",
                    "map": {
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_i32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_i32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'f'",
                    "group": "multi_f32",
                    "map": {
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_f32_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_f32_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'd'",
                    "group": "multi_f64",
                    "map": {
                        "compressed_ptrs": "compressed_ptrs.ptr",
                        "compressed_lengths": "compressed_lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_f64_kcb(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_f64_scal(const int64_t *compressed_ptrs, const int32_t *compressed_lengths, int nframes, int codec, const uint8_t *mask, int NIJ, double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
            ],
            "default_raise": "TypeError: expected uint8, int8, uint16, int16, uint32, int32, float32 or float64 outpx buffer",
        },
        {
            "py_sig": "note_chunk(chunk: buffer, index: int, pointers: buffer, lengths: buffer) -> void",
            "doc": "Write chunk's raw address and byte length into pointers[index]/lengths[index]. The only place a buffer's address is extracted -- via c2py23's own buffer acquisition, not any Python-side ctypes/numpy trick -- so a plain Python loop can batch independent chunk objects (network packets, multiple files, ...) into bslz4_csc_multi's compressed_ptrs/compressed_lengths arrays.",
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
            "py_sig": "bslz4_csc_multi_base(base: buffer, offsets: buffer, lengths: buffer, nframes: int, mask: buffer, outpx: buffer, output_adr: buffer, npx_out: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer, cursors: buffer, nout: int, codec: int = 2) -> int",
            "doc": "Like bslz4_csc_multi, but for chunks that share one base buffer (e.g. an mmap'ed HDF5 file): offsets are byte offsets from base rather than absolute addresses, avoiding any Python-side pointer arithmetic. NOTE: offsets is mutated in place into absolute pointers by this call (reused as scratch, like cursors/workspace already are).",
            "checks": [
                "mask.format == 'B' or mask.format == 'b'",
                "output_adr.format == 'I'",
                "npx_out.format == 'i'",
                "powder.format == 'd'",
                "data.format == 'f'",
                "indices.format == 'I' or indices.format == 'i'",
                "indptr.format == 'I' or indptr.format == 'i'",
                "workspace.format == 'B'",
            ],
            "c_overloads": [
                {
                    "when": "outpx.format == 'B'",
                    "group": "multi_base_u8",
                    "map": {
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_base_u8_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_base_u8_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'H'",
                    "group": "multi_base_u16",
                    "map": {
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_base_u16_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_base_u16_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'I'",
                    "group": "multi_base_u32",
                    "map": {
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_base_u32_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_base_u32_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, uint32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'b'",
                    "group": "multi_base_i8",
                    "map": {
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_base_i8_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_base_i8_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int8_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'h'",
                    "group": "multi_base_i16",
                    "map": {
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_base_i16_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_base_i16_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int16_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'i'",
                    "group": "multi_base_i32",
                    "map": {
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_base_i32_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_base_i32_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, int32_t *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'f'",
                    "group": "multi_base_f32",
                    "map": {
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_base_f32_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_base_f32_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, float *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'd'",
                    "group": "multi_base_f64",
                    "map": {
                        "base": "base.ptr",
                        "offsets": "offsets.ptr",
                        "lengths": "lengths.ptr",
                        "nframes": "nframes",
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
                        {
                            "sig": "bslz4_csc_multi_base_f64_kcb(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_multi_base_f64_scal(const char *base, int64_t *offsets, const int32_t *lengths, int nframes, int codec, const uint8_t *mask, int NIJ, double *outpx, uint32_t *output_adr, int32_t *npx_out, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len, int64_t *cursors) -> int",
                        },
                    ],
                },
            ],
            "default_raise": "TypeError: expected uint8, int8, uint16, int16, uint32, int32, float32 or float64 outpx buffer",
        },
    ],
}
C2PY_END */
