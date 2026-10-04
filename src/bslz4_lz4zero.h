#ifndef BSLZ4_LZ4ZERO_H
#define BSLZ4_LZ4ZERO_H
/*
 * An lz4 block decoder for bitshuffled detector blocks, which are mostly
 * zeros.  Standard lz4 block format (as LZ4_decompress_safe); the difference
 * is how zeros are written.
 *
 * It tracks zero_from: the bytes [zero_from, op) of the output are known to
 * be zero.  A match whose source starts at or after zero_from can only copy
 * zeros (for an overlapping match the repeated pattern lies in that region;
 * for a plain one the whole source does), so it is not written at all: the
 * output position just advances.  An all-zero literal run is skipped the
 * same way.  After a literal run or a copied match, zero_from is set past
 * its last non-zero byte, so a following long zero run (typically offset 2)
 * is skipped too.  The skipped bytes, [written, op), are materialised with one
 * memset just before the next non-zero write (a later match may read them).
 * The zero tail of the block is never written: *nz_end returns how many
 * leading bytes are valid; the rest are zero and left untouched.
 *
 * Returns cap (the block must decode to exactly cap bytes) or a negative
 * value for a malformed block.  Every read is bounds checked against the
 * input.  Short copies are done as fixed 32/64-byte chunks (inline, no
 * memcpy/memset call): they may write up to `slack` bytes past dst + cap
 * (the caller guarantees that space, e.g. the scratch buffer that follows),
 * and leave stray bytes after the last real write; those lie at or after
 * *nz_end or inside a pending zero run, which is zero-filled before any later
 * write reads it.  With slack < 64 every copy is exact.  The lz4
 * end-of-block rules are enforced as LZ4_decompress_safe does (a match ends
 * at least 5 bytes before the end; literals reaching the last 12 bytes end
 * the block), so a block it rejects is rejected here too; and an offset of
 * 0 is rejected (lz4 1.10.0 accepts it, lz4 issue #1631).
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define BSLZ4_LZ4ZERO_SSE2 1
#else
#define BSLZ4_LZ4ZERO_SSE2 0
#endif

/* Copy an overlapping match [op - off, ...) of length m (off < m) by
 * doubling: [src, op) repeats with period off, so copying it onto op doubles
 * the periodic region; every copy is non-overlapping. */
static inline void bslz4_copy_overlap(uint8_t *op, const uint8_t *src, uint8_t *end) {
    while (op < end) {
        size_t n = (size_t) (op - src);
        if (n > (size_t) (end - op)) n = (size_t) (end - op);
        memcpy(op, src, n);
        op += n;
    }
}

/* Zero n bytes at p; a run of up to 512 bytes as fixed 64-byte stores when
 * the space up to `limit` allows (the extra zeros land on bytes not yet
 * holding data). */
static inline void bslz4_zero_fill(uint8_t *p, size_t n, const uint8_t *limit) {
    const size_t n64 = (n + 63) & ~(size_t) 63;
    if (n <= 512 && (size_t) (limit - p) >= n64) {
        for (size_t k = 0; k < n64; k += 64) memset(p + k, 0, 64);
    } else {
        memset(p, 0, n);
    }
}

static int bslz4_lz4_decode_zero(const uint8_t *src, size_t srcsize, uint8_t *dst, size_t cap,
                                 size_t slack, size_t *nz_end) {
    const uint8_t *ip = src;
    const uint8_t *const iend = src + srcsize;
    uint8_t *op = dst;
    uint8_t *const oend = dst + cap;
    const uint8_t *const olim = oend + slack;        /* writable up to here */
    uint8_t *written = dst;      /* [dst, written) holds real data; [written, op) is zero, unwritten */
    uint8_t *zero_from = dst;    /* [zero_from, op) is zero */
    for (;;) {
        if (ip >= iend) return -1;
        const unsigned token = *ip++;
        size_t lit = token >> 4;
        if (lit == 15) {
            unsigned b;
            do {
                if (ip >= iend) return -1;
                b = *ip++;
                lit += b;
                if (lit > cap) return -1;
            } while (b == 255);
        }
        if (lit > (size_t) (iend - ip) || lit > (size_t) (oend - op)) return -1;
        /* As LZ4_decompress_safe: literals that reach into the last 12 output
         * bytes (MFLIMIT), or the last 8 input bytes, must end the block. */
        if (((size_t) (oend - op) < 12 || lit > (size_t) (oend - op) - 12 ||
             (size_t) (iend - ip) < 8 || lit > (size_t) (iend - ip) - 8) &&
            ip + lit != iend) return -1;
        if (lit) {
            const uint8_t *q = ip + lit;
            while (q > ip && q[-1] == 0) q--;          /* last non-zero literal */
            if (q != ip) {
                if (written < op) bslz4_zero_fill(written, (size_t) (op - written), olim);
                if (lit <= 32 && (size_t) (iend - ip) >= 32 && (size_t) (olim - op) >= 32)
                    memcpy(op, ip, 32);                /* fixed size: inlined */
                else
                    memcpy(op, ip, lit);
                written = op + lit;
                zero_from = op + (q - ip);
            }
            op += lit;
            ip += lit;
        }
        if (ip == iend) break;                         /* the last sequence is literals only */
        if (iend - ip < 2) return -1;
        const size_t off = (size_t) ip[0] | ((size_t) ip[1] << 8);
        ip += 2;
        size_t mlen = token & 15;
        if (mlen == 15) {
            unsigned b;
            do {
                if (ip >= iend) return -1;
                b = *ip++;
                mlen += b;
                if (mlen > cap) return -1;
            } while (b == 255);
        }
        mlen += 4;
        if (off == 0 || off > (size_t) (op - dst)) return -1;
        if (mlen > (size_t) (oend - op) || (size_t) (oend - op) - mlen < 5) return -1;   /* last 5 bytes are literals */
        uint8_t *match = op - off;
        if (match >= zero_from) {                      /* copies zeros only: skip */
            op += mlen;
            continue;
        }
        if (written < op) bslz4_zero_fill(written, (size_t) (op - written), olim);
        /* the copy's trailing zeros (q): a later match (often offset 2, length
         * thousands) repeating them is then skipped instead of copied */
        const uint8_t *q = op + mlen;
#if BSLZ4_LZ4ZERO_SSE2
        if (off >= mlen && mlen <= 64 && (size_t) (olim - op) >= 64) {
            /* 16-byte chunks, as many as mlen needs (a 64-byte load of a source
             * just written by smaller stores stalls on store forwarding), the
             * non-zero bytes gathered as a mask on the way: the trailing zeros
             * without a byte loop (that loop was 26 % of the decoder on WAu
             * blocks).  Bytes of a chunk at or past mlen are don't-care. */
            uint64_t nz = 0;
            for (size_t c = 0; c < mlen; c += 16) {
                const __m128i v = _mm_loadu_si128((const __m128i *) (const void *) (match + c));
                _mm_storeu_si128((__m128i *) (void *) (op + c), v);
                nz |= (uint64_t) (~(uint32_t) _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_setzero_si128())) & 0xFFFFu) << c;
            }
            if (mlen < 64) nz &= ((uint64_t) 1 << mlen) - 1;
            q = nz ? op + (64 - __builtin_clzll(nz)) : op;
        } else
#endif
        if (off >= mlen) {
            if (mlen <= 64 && (size_t) (olim - op) >= 64) {
                /* one fixed 64-byte load, then store: the first mlen bytes come
                 * from before op (off >= mlen); bytes past mlen are don't-care */
                uint8_t t[64];
                memcpy(t, match, 64);
                memcpy(op, t, 64);
            } else {
                memcpy(op, match, mlen);
            }
            while (q > op && q[-1] == 0) q--;
        } else {
            /* periodic with period off: if its last off bytes are zero, so is
             * the whole copy */
            bslz4_copy_overlap(op, match, op + mlen);
            const uint8_t *const stop = q - off;
            while (q > stop && q[-1] == 0) q--;
            if (q == stop) q = op;
        }
        op += mlen;
        written = op;
        zero_from = (uint8_t *) q;
    }
    if (op != oend) return -1;
    *nz_end = (size_t) (written - dst);
    return (int) cap;
}

#endif /* BSLZ4_LZ4ZERO_H */
