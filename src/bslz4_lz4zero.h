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
 * same way.  The skipped bytes, [written, op), are materialised with one
 * memset just before the next non-zero write (a later match may read them).
 * The zero tail of the block is never written: *nz_end returns how many
 * leading bytes are valid; the rest are zero and left untouched.
 *
 * Returns cap (the block must decode to exactly cap bytes) or a negative
 * value for a malformed block.  Every read and write is bounds checked
 * against the input and output; nothing is written past dst + cap.  The lz4
 * end-of-block rules are enforced as LZ4_decompress_safe does (a match ends
 * at least 5 bytes before the end; literals reaching the last 12 bytes end
 * the block), so a block it rejects is rejected here too; and an offset of
 * 0 is rejected (lz4 1.10.0 accepts it, lz4 issue #1631).
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

static int bslz4_lz4_decode_zero(const uint8_t *src, size_t srcsize, uint8_t *dst, size_t cap,
                                 size_t *nz_end) {
    const uint8_t *ip = src;
    const uint8_t *const iend = src + srcsize;
    uint8_t *op = dst;
    uint8_t *const oend = dst + cap;
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
                if (written < op) memset(written, 0, (size_t) (op - written));
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
        if (written < op) memset(written, 0, (size_t) (op - written));
        if (off >= mlen) memcpy(op, match, mlen);
        else bslz4_copy_overlap(op, match, op + mlen);
        op += mlen;
        written = op;
        zero_from = op;                                /* conservative */
    }
    if (op != oend) return -1;
    *nz_end = (size_t) (written - dst);
    return (int) cap;
}

#endif /* BSLZ4_LZ4ZERO_H */
