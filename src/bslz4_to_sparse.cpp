/*
 * bslz4_to_sparse -- decode a bitshuffle-LZ4 compressed HDF5 chunk directly
 * into a masked/thresholded sparse (values, indices) pair, optionally also
 * accumulating a CSC powder integration.
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
 * No malloc, no VLA: the workspace buffer is owned and sized by the Python
 * caller (see bslz4_core.hpp for why the scratch size can't be a compile
 * time constant).
 */

#include "bslz4_backends.hpp"
#include "bslz4_core.hpp"

using namespace bslz4;

extern "C" {

int bslz4_u8_kcb(const char *compressed, int compressed_length,
                  const uint8_t *mask, int NIJ,
                  uint8_t *output, uint32_t *output_adr, int threshold,
                  uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint8_t, untranspose_kcb>(
        compressed, compressed_length, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u8_scal(const char *compressed, int compressed_length,
                   const uint8_t *mask, int NIJ,
                   uint8_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint8_t, untranspose_bshuf_scal>(
        compressed, compressed_length, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u16_kcb(const char *compressed, int compressed_length,
                   const uint8_t *mask, int NIJ,
                   uint16_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint16_t, untranspose_kcb>(
        compressed, compressed_length, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u16_scal(const char *compressed, int compressed_length,
                    const uint8_t *mask, int NIJ,
                    uint16_t *output, uint32_t *output_adr, int threshold,
                    uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint16_t, untranspose_bshuf_scal>(
        compressed, compressed_length, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u32_kcb(const char *compressed, int compressed_length,
                   const uint8_t *mask, int NIJ,
                   uint32_t *output, uint32_t *output_adr, int threshold,
                   uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint32_t, untranspose_kcb>(
        compressed, compressed_length, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_u32_scal(const char *compressed, int compressed_length,
                    const uint8_t *mask, int NIJ,
                    uint32_t *output, uint32_t *output_adr, int threshold,
                    uint8_t *workspace, size_t workspace_len) {
    return bslz4_decode<uint32_t, untranspose_bshuf_scal>(
        compressed, compressed_length, mask, NIJ, output, output_adr, threshold,
        workspace, workspace_len);
}

int bslz4_csc_u8_kcb(const char *compressed, int compressed_length,
                      const uint8_t *mask, int NIJ,
                      uint8_t *outpx, uint32_t *output_adr, int threshold,
                      double *output, int NOUT,
                      const float *data, const uint32_t *indices, const uint32_t *indptr,
                      uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint8_t, untranspose_kcb>(
        compressed, compressed_length, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u8_scal(const char *compressed, int compressed_length,
                       const uint8_t *mask, int NIJ,
                       uint8_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint8_t, untranspose_bshuf_scal>(
        compressed, compressed_length, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u16_kcb(const char *compressed, int compressed_length,
                       const uint8_t *mask, int NIJ,
                       uint16_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint16_t, untranspose_kcb>(
        compressed, compressed_length, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u16_scal(const char *compressed, int compressed_length,
                        const uint8_t *mask, int NIJ,
                        uint16_t *outpx, uint32_t *output_adr, int threshold,
                        double *output, int NOUT,
                        const float *data, const uint32_t *indices, const uint32_t *indptr,
                        uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint16_t, untranspose_bshuf_scal>(
        compressed, compressed_length, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u32_kcb(const char *compressed, int compressed_length,
                       const uint8_t *mask, int NIJ,
                       uint32_t *outpx, uint32_t *output_adr, int threshold,
                       double *output, int NOUT,
                       const float *data, const uint32_t *indices, const uint32_t *indptr,
                       uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint32_t, untranspose_kcb>(
        compressed, compressed_length, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

int bslz4_csc_u32_scal(const char *compressed, int compressed_length,
                        const uint8_t *mask, int NIJ,
                        uint32_t *outpx, uint32_t *output_adr, int threshold,
                        double *output, int NOUT,
                        const float *data, const uint32_t *indices, const uint32_t *indptr,
                        uint8_t *workspace, size_t workspace_len) {
    return bslz4_csc_decode<uint32_t, untranspose_bshuf_scal>(
        compressed, compressed_length, mask, NIJ, outpx, output_adr, threshold,
        output, NOUT, data, indices, indptr, workspace, workspace_len);
}

} /* extern "C" */

/* C2PY_BEGIN
{
    "module": "bslz4_to_sparse",
    "source": ["bslz4_to_sparse.cpp"],
    "timing": True,
    "functions": [
        {
            "py_sig": "bslz4(compressed: buffer, mask: buffer, output: buffer, output_adr: buffer, workspace: buffer, threshold: int) -> int",
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
                            "sig": "bslz4_u8_kcb(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint8_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_u8_scal(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint8_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'H'",
                    "group": "u16",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
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
                            "sig": "bslz4_u16_kcb(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint16_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_u16_scal(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint16_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "output.format == 'I'",
                    "group": "u32",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
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
                            "sig": "bslz4_u32_kcb(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint32_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_u32_scal(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint32_t *output, uint32_t *output_adr, int threshold, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
            ],
            "default_raise": "TypeError: expected uint8, uint16 or uint32 output buffer",
        },
        {
            "py_sig": "bslz4_csc(compressed: buffer, mask: buffer, outpx: buffer, output_adr: buffer, threshold: int, powder: buffer, data: buffer, indices: buffer, indptr: buffer, workspace: buffer) -> int",
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
                            "sig": "bslz4_csc_u8_kcb(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint8_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_u8_scal(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint8_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'H'",
                    "group": "csc_u16",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
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
                            "sig": "bslz4_csc_u16_kcb(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint16_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_u16_scal(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint16_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
                {
                    "when": "outpx.format == 'I'",
                    "group": "csc_u32",
                    "map": {
                        "compressed": "compressed.ptr",
                        "compressed_length": "compressed.len",
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
                            "sig": "bslz4_csc_u32_kcb(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint32_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                            "default": True,
                        },
                        {
                            "sig": "bslz4_csc_u32_scal(const char *compressed, int compressed_length, const uint8_t *mask, int NIJ, uint32_t *outpx, uint32_t *output_adr, int threshold, double *output, int NOUT, const float *data, const uint32_t *indices, const uint32_t *indptr, uint8_t *workspace, size_t workspace_len) -> int",
                        },
                    ],
                },
            ],
            "default_raise": "TypeError: expected uint8, uint16 or uint32 outpx buffer",
        },
    ],
}
C2PY_END */
