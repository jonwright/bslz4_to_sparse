#pragma once
/*
 * Experimental CSC-entry dots (ids 11..22): different readings of the three
 * CSC arrays (data, indices, indptr), chosen by dot id.  Included by
 * kernels_generic.cpp after the collect helpers; nothing here is shared with
 * the csc / csc-run / csc-nosplit kernels, which stay as they are while
 * these are measured.
 *
 * Each dot is a "start source" (where a pixel's first output bin comes from)
 * and a "body" (what it adds for one pixel value).  The dense route calls the
 * body for every pixel of the block in order; the sparse route calls it for
 * the compacted non-zero pixels, also in ascending order, so a sequential
 * start source (delta, walk) only ever moves forward within a block.
 *
 * Start sources
 *   Start32     indices[p], u32
 *   Start16     indices[p], u16 (nbins < 65536)
 *   StartDelta  byte stream: a byte c != 0x80 adds (int8) c to the previous
 *               pixel's start; 0x80 is followed by an absolute u32 (LE).
 *               Each block starts with an absolute.
 *   StartWalk   one 4-bit code per pixel (two per byte, low nibble first):
 *               bits 0-1 the azimuth step, bits 2-3 the radial step, each
 *               0 -> 0, 1 -> +1, 2 -> -1, 3 -> escape; flat step =
 *               dr * stride + da.  An escape takes the next absolute bin
 *               from the exception list.  Each block starts with an escape.
 *
 * Stream header (StartDelta, StartWalk, csc-tile): `indices` is a byte
 * buffer whose first u32 words are
 *   h[0] BSLZ4_STREAM_MAGIC   h[1] header length in u32 words (payload at h + h[1])
 *   h[2] block_elems (0: any) h[3] nblocks
 *   h[4] stride (2D: azimuth bins)
 *   h[5] walk: byte offset of the exception list (u32[]) in the payload
 *   h[6], h[7] reserved (0)
 *   h[8 .. 8+nblocks] per block: delta -> byte offset of its first code;
 *               walk -> index of its first exception
 * The driver checks h[0], h[2] and h[3] before any kernel runs.
 */

#include <string.h>

#define BSLZ4_STREAM_MAGIC 0x58495342u   /* "BSIX" */
#define BSLZ4_DUMP_NONE16 0xFFFFu

namespace bslz4v {

static inline uint32_t rd32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline const uint32_t *hdr(const bslz4_mat_csc *m) {
    return (const uint32_t *) (const void *) m->indices;
}

/* ---- start sources -------------------------------------------------- */

struct Start32 {
    const uint32_t *BSLZ4_RESTRICT s;
    inline void init(const bslz4_work *w) { s = ((const bslz4_mat_csc *) w->mat)->indices; }
    inline uint32_t at(size_t p) { return s[p]; }
};

struct Start16 {
    const uint16_t *BSLZ4_RESTRICT s;
    inline void init(const bslz4_work *w) {
        s = (const uint16_t *) (const void *) ((const bslz4_mat_csc *) w->mat)->indices;
    }
    inline uint32_t at(size_t p) { return s[p]; }
};

struct StartDelta {
    const uint8_t *BSLZ4_RESTRICT c;
    size_t cur;
    uint32_t start;
    inline void step() {
        uint8_t b = *c++;
        if (BSLZ4_UNLIKELY(b == 0x80u)) {
            start = rd32(c);
            c += 4;
        } else {
            start += (uint32_t) (int32_t) (int8_t) b;
        }
    }
    inline void init(const bslz4_work *w) {
        const uint32_t *h = hdr((const bslz4_mat_csc *) w->mat);
        const uint8_t *payload = (const uint8_t *) (h + h[1]);
        const size_t blk = w->i0 / h[2];
        c = payload + h[8 + blk];
        cur = w->i0;
        start = 0;
        step();
    }
    inline uint32_t at(size_t p) {
        while (cur < p) {
            step();
            cur++;
        }
        return start;
    }
};

struct StartWalk {
    const uint8_t *BSLZ4_RESTRICT nib;
    const uint32_t *BSLZ4_RESTRICT exc;
    size_t cur;
    uint32_t start, stride;
    inline void step(size_t p) {
        const unsigned code = (nib[p >> 1] >> ((p & 1) * 4)) & 15u;
        const unsigned da = code & 3u, dr = code >> 2;
        if (BSLZ4_UNLIKELY(da == 3u || dr == 3u)) {
            start = *exc++;
        } else {
            static const int32_t st[3] = {0, 1, -1};
            start += (uint32_t) (st[dr] * (int32_t) stride + st[da]);
        }
    }
    inline void init(const bslz4_work *w) {
        const uint32_t *h = hdr((const bslz4_mat_csc *) w->mat);
        const uint8_t *payload = (const uint8_t *) (h + h[1]);
        const size_t blk = w->i0 / h[2];
        nib = payload;
        exc = (const uint32_t *) (const void *) (payload + h[5]) + h[8 + blk];
        stride = h[4];
        cur = w->i0;
        start = 0;
        step(cur);
    }
    inline uint32_t at(size_t p) {
        while (cur < p) {
            cur++;
            step(cur);
        }
        return start;
    }
};

/* ---- bodies: what one pixel value adds ------------------------------- */

/* split, start + length, float weights: data[indptr[p]..indptr[p+1]] */
template<typename S>
struct RunF32 {
    S src;
    const float *BSLZ4_RESTRICT data;
    const uint32_t *BSLZ4_RESTRICT indptr;
    double *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
        src.init(w);
        data = (const float *) m->data;
        indptr = m->indptr;
        out = (double *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t s = src.at(p);
        const uint32_t k0 = indptr[p], k1 = indptr[p + 1];
        double *BSLZ4_RESTRICT o = out + s;
        const float *BSLZ4_RESTRICT d = data + k0;
        const double pv = (double) v;
        for (uint32_t k = 0; k < k1 - k0; k++) o[k] += (double) d[k] * pv;
    }
};

/* split, start + length, fixed-point integer weights W (u32 / u16), int64
 * powder: exact integer sums of w * v. */
template<typename S, typename W>
struct RunInt {
    S src;
    const W *BSLZ4_RESTRICT data;
    const uint32_t *BSLZ4_RESTRICT indptr;
    int64_t *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
        src.init(w);
        data = (const W *) m->data;
        indptr = m->indptr;
        out = (int64_t *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t s = src.at(p);
        const uint32_t k0 = indptr[p], k1 = indptr[p + 1];
        int64_t *BSLZ4_RESTRICT o = out + s;
        const W *BSLZ4_RESTRICT d = data + k0;
        const int64_t pv = (int64_t) v;
        for (uint32_t k = 0; k < k1 - k0; k++) o[k] += (int64_t) d[k] * pv;
    }
};

/* general csc, one u32 bin per entry, u32 fixed-point weights, int64 powder */
struct CscInt {
    const uint32_t *BSLZ4_RESTRICT data;
    const uint32_t *BSLZ4_RESTRICT indices;
    const uint32_t *BSLZ4_RESTRICT indptr;
    int64_t *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
        data = (const uint32_t *) m->data;
        indices = m->indices;
        indptr = m->indptr;
        out = (int64_t *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t k0 = indptr[p], k1 = indptr[p + 1];
        const int64_t pv = (int64_t) v;
        for (uint32_t k = k0; k < k1; k++) out[indices[k]] += (int64_t) data[k] * pv;
    }
};

/* histogram, every pixel has a bin (empty pixels -> the dump bin nbins):
 * no branch, no weight, no multiply */
template<typename S>
struct NosplitDump {
    S src;
    double *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        src.init(w);
        out = (double *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) { out[src.at(p)] += (double) v; }
};

/* histogram with u16 bins, 0xFFFF = no bin */
struct Nosplit16 {
    const uint16_t *BSLZ4_RESTRICT bin;
    double *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        bin = (const uint16_t *) (const void *) ((const bslz4_mat_csc *) w->mat)->indices;
        out = (double *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t b = bin[p];
        if (b != BSLZ4_DUMP_NONE16) out[b] += (double) v;
    }
};

/* histogram + first moment, dump bin: bins b (sum I, no multiply) and b+1
 * (sum qI); data[p] = q */
struct MomentDump {
    const uint32_t *BSLZ4_RESTRICT bin;
    const float *BSLZ4_RESTRICT q;
    double *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
        bin = m->indices;
        q = (const float *) m->data;
        out = (double *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t b = bin[p];
        const double pv = (double) v;
        out[b] += pv;
        out[b + 1] += pv * (double) q[p];
    }
};

/* split + first moment: column p is [q, w_0 .. w_{L-1}] at indptr[p]; bins
 * start .. start+L-1 of the underlying matrix, written as the interleaved
 * pairs (2b: sum w I, 2b+1: sum w q I). */
struct RunMoment {
    const float *BSLZ4_RESTRICT data;
    const uint32_t *BSLZ4_RESTRICT start;
    const uint32_t *BSLZ4_RESTRICT indptr;
    double *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
        data = (const float *) m->data;
        start = m->indices;
        indptr = m->indptr;
        out = (double *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t k0 = indptr[p], k1 = indptr[p + 1];
        if (k1 == k0) return;
        const double q = (double) data[k0];
        const double pv = (double) v;
        double *BSLZ4_RESTRICT o = out + 2 * (size_t) start[p];
        for (uint32_t k = k0 + 1; k < k1; k++, o += 2) {
            const double wv = (double) data[k] * pv;
            o[0] += wv;
            o[1] += wv * q;
        }
    }
};

/* 2D tile: pixel p covers radial bins r0 .. r0+len1-1 by azimuth bins
 * a0 .. a0+len2-1 (wrapping at stride), flat bin = r * stride + a.  The
 * payload holds three u32 per pixel: base = r0*stride + a0, a0,
 * len1 << 16 | len2.  Weights data[indptr[p]..] are the len1 x len2
 * rectangle, radial-major, zero where pyFAI had no entry. */
struct Tile {
    const uint32_t *BSLZ4_RESTRICT px3;
    const float *BSLZ4_RESTRICT data;
    const uint32_t *BSLZ4_RESTRICT indptr;
    double *BSLZ4_RESTRICT out;
    uint32_t stride;
    inline void init(const bslz4_work *w) {
        const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
        const uint32_t *h = hdr(m);
        px3 = h + h[1];
        stride = h[4];
        data = (const float *) m->data;
        indptr = m->indptr;
        out = (double *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t *t = px3 + 3 * p;
        const uint32_t base = t[0], a0 = t[1], l1 = t[2] >> 16, l2 = t[2] & 0xFFFFu;
        const float *BSLZ4_RESTRICT d = data + indptr[p];
        const double pv = (double) v;
        for (uint32_t i = 0; i < l1; i++) {
            const uint32_t row = base + i * stride;
            uint32_t j = 0;
            const uint32_t nowrap = (a0 + l2 <= stride) ? l2 : stride - a0;
            for (; j < nowrap; j++) out[row + j] += (double) d[j] * pv;
            for (; j < l2; j++) out[row + j - stride] += (double) d[j] * pv;
            d += l2;
        }
    }
};

/* permutation (FAZIT): every pixel has its own output slot, b = indices[p]
 * (BSLZ4_NO_BIN: none); the value is stored, not added, in the pixel dtype,
 * so the output is the image itself in (radius, azimuth) order.  Slots of
 * pixels that are zero (sparse route) keep the driver's zero fill. */
template<typename TP>
struct Permute {
    const uint32_t *BSLZ4_RESTRICT bin;
    TP *BSLZ4_RESTRICT out;
    inline void init(const bslz4_work *w) {
        bin = ((const bslz4_mat_csc *) w->mat)->indices;
        out = (TP *) w->powder;
    }
    template<typename T> inline void add(size_t p, T v) {
        const uint32_t b = bin[p];
        if (b != 0xFFFFFFFFu) out[b] = (TP) v;
    }
};

/* permutation as runs (approximate / tiled FAZIT): the pixels of a block are
 * cut into runs that go to consecutive output slots.  The stream payload is
 * two u32 per run, (output start or 0xFFFFFFFF for masked pixels, length),
 * in pixel order, split at block boundaries; h[8 + block] is the block's first
 * run.  Every pixel of the block is covered, so both routes copy every run
 * (a zero pixel is written as zero) and the >cut collect runs over the block. */
template<typename T>
static inline int permute_runs(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    if (n == 0) return 0;
    const bslz4_mat_csc *m = (const bslz4_mat_csc *) w->mat;
    const uint32_t *h = hdr(m);
    const uint32_t *runs = h + h[1];
    const size_t blk = i0 / h[2];
    T *BSLZ4_RESTRICT out = (T *) w->powder;
    size_t j = 0;
    for (uint32_t r = h[8 + blk]; j < n; r++) {
        const uint32_t o = runs[2 * r], len = runs[2 * r + 1];
        if (o != 0xFFFFFFFFu) memcpy(out + o, px + j, (size_t) len * sizeof(T));
        j += len;
    }
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect_id,
                               (T *) w->out_vals, w->out_adr);
}

/* ---- the two loops ---------------------------------------------------- */

template<typename T, typename Body>
static inline int dense(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    if (n == 0) return 0;
    Body body;
    body.init(w);
    for (size_t j = 0; j < n; j++) body.add(j + i0, px[j]);
    if (BSLZ4_CUT_ABOVE_MAX(w)) return 0;
    return bslz4_collect_gt<T>(px, mask, i0, n, cut, w->collect_id,
                               (T *) w->out_vals, w->out_adr);
}

template<typename T, typename Body>
static inline int sparse(const bslz4_work *BSLZ4_RESTRICT w) {
    const T cut = (T) w->threshold;
    const T *px = (const T *) w->block;
    const uint8_t *BSLZ4_RESTRICT mask = w->no_mask ? NULL : w->mask;
    const size_t i0 = w->i0, n = w->n;
    if (n == 0) return 0;
    T *tv = (T *) w->tval;
    uint32_t *BSLZ4_RESTRICT tidx = w->tidx;
    int nz = bslz4_collect_nz_w<T>(w, px, mask, i0, n, tv, tidx);
    if (nz > 0) {
        Body body;
        body.init(w);
        for (int kk = 0; kk < nz; kk++) body.add(tidx[kk], tv[kk]);
    }
    int npx = 0;
    T *ov = (T *) w->out_vals;
    uint32_t *BSLZ4_RESTRICT out_adr = w->out_adr;
    const int skip = BSLZ4_CUT_ABOVE_MAX(w);
    for (int kk = 0; kk < nz; kk++) {
        T val = tv[kk];
        if (!skip && BSLZ4_UNLIKELY(val > cut)) {
            ov[npx] = val;
            out_adr[npx] = tidx[kk];
            npx++;
        }
    }
    return npx;
}

template<typename T, typename Body>
static inline int run(const bslz4_work *BSLZ4_RESTRICT w) {
    return w->route ? sparse<T, Body>(w) : dense<T, Body>(w);
}

/* dot ids 11..22 -> kernel; returns BSLZ4_ERR_BAD_LAYOUT for any other id */
template<typename T>
static inline int dispatch(const bslz4_work *BSLZ4_RESTRICT w) {
    switch (w->dot_id) {
    case 11: return run<T, NosplitDump<Start32> >(w);      /* csc-nosplit-dump */
    case 12: return run<T, MomentDump>(w);                 /* csc-nosplit-moment-dump */
    case 13: return run<T, RunF32<Start16> >(w);           /* csc-run-u16 */
    case 14: return run<T, Nosplit16>(w);                  /* csc-nosplit-u16 */
    case 15: return run<T, RunF32<StartDelta> >(w);        /* csc-run-delta */
    case 16: return run<T, NosplitDump<StartDelta> >(w);   /* csc-nosplit-delta */
    case 17: return run<T, NosplitDump<StartWalk> >(w);    /* csc-nosplit-walk */
    case 18: return run<T, Tile>(w);                       /* csc-tile */
    case 19: return run<T, RunMoment>(w);                  /* csc-run-moment */
    case 20: return run<T, CscInt>(w);                     /* csc-int */
    case 21: return run<T, RunInt<Start32, uint32_t> >(w); /* csc-run-int */
    case 22: return run<T, RunInt<Start32, uint16_t> >(w); /* csc-run-int16 */
    case 23: return run<T, Permute<T> >(w);                /* csc-permute */
    case 24: return permute_runs<T>(w);                    /* csc-permute-runs */
    default: return BSLZ4_ERR_BAD_LAYOUT;
    }
}

} /* namespace bslz4v */
