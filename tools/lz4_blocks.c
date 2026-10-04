/*
 * Time LZ4_decompress_safe alone over the blocks of bitshuffle-LZ4 chunks,
 * and describe each chunk's lz4 stream, to see which stream shapes are slow.
 *
 * Input (written by tools/lz4_scan.py): a file of records
 *   u32 label length, label bytes, u64 chunk length, chunk bytes
 * where a chunk is the HDF5 bitshuffle-LZ4 chunk (12-byte header, then per
 * block a big-endian u32 size and the lz4 block).
 *
 * Output, one line per chunk: label, output MB, ms, GB/s of output, ns per
 * output byte, then the stream shape: sequences per block, mean match
 * length, % of output bytes that are literals, and % of output bytes from
 * matches with offset < 8 and < 16 (the overlapping-copy path).
 *
 * Build (the vendored lz4, at the extension's -O2):
 *   cc -O2 -Ilz4/lib -o lz4_blocks tools/lz4_blocks.c lz4/lib/lz4.c
 * -DBSLZ4_PREZERO zeroes the output buffer before every block (timed), for
 * decoders that skip writing zero runs.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lz4.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16) | ((uint32_t) p[2] << 8) | p[3];
}

static uint64_t be64(const uint8_t *p) {
    return ((uint64_t) be32(p) << 32) | be32(p + 4);
}

/* lz4 block sequence statistics */
static void shape(const uint8_t *b, size_t n, double *seqs, double *lit, double *mbytes,
                  double *m8, double *m16) {
    size_t i = 0;
    while (i < n) {
        unsigned tok = b[i++];
        size_t L = tok >> 4;
        if (L == 15) { unsigned x; do { x = b[i++]; L += x; } while (x == 255); }
        i += L;
        *lit += L;
        if (i >= n) break;
        unsigned off = b[i] | (b[i + 1] << 8);
        i += 2;
        size_t M = tok & 15;
        if (M == 15) { unsigned x; do { x = b[i++]; M += x; } while (x == 255); }
        M += 4;
        *seqs += 1;
        *mbytes += M;
        if (off < 8) *m8 += M;
        if (off < 16) *m16 += M;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s records.bin [repeats]\n", argv[0]); return 1; }
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    printf("%-34s %7s %8s %7s %8s | %8s %8s %6s %6s %6s  %s\n", "chunk", "out MB", "ms", "GB/s",
           "ns/B", "seq/blk", "match", "lit%", "<8%", "<16%", "output hash");
    for (;;) {
        uint32_t ll;
        if (fread(&ll, 4, 1, f) != 1) break;
        char label[256] = {0};
        if (ll >= sizeof label || fread(label, 1, ll, f) != ll) return 1;
        uint64_t cl;
        if (fread(&cl, 8, 1, f) != 1) return 1;
        uint8_t *c = malloc(cl);
        if (fread(c, 1, cl, f) != cl) return 1;
        uint64_t total = be64(c);
        uint32_t bs = be32(c + 8);
        if (bs == 0) bs = 8192;
        char *dst = malloc(bs);
        /* stream shape, once */
        double seqs = 0, lit = 0, mb = 0, m8 = 0, m16 = 0;
        size_t nblk = 0;
        for (size_t p = 12; p + 4 <= cl && nblk < total / bs; nblk++) {
            uint32_t nb = be32(c + p);
            shape(c + p + 4, nb, &seqs, &lit, &mb, &m8, &m16);
            p += 4 + nb;
        }
        /* FNV-1a hash of the decompressed frame: different builds/patches must agree */
        uint64_t hash = 1469598103934665603ULL;
        {
            size_t p = 12;
            for (size_t k = 0; k < nblk; k++) {
                uint32_t nb = be32(c + p);
#ifdef BSLZ4_PREZERO
                memset(dst, 0, bs);
#endif
                int got = LZ4_decompress_safe((const char *) c + p + 4, dst, (int) nb, (int) bs);
                if (got != (int) bs) { fprintf(stderr, "%s: block %zu failed (%d)\n", label, k, got); return 1; }
                for (uint32_t q = 0; q < bs; q++) hash = (hash ^ (uint8_t) dst[q]) * 1099511628211ULL;
                p += 4 + nb;
            }
        }
        /* timing: every full block, best of reps */
        double best = 1e30;
        volatile int sink = 0;
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            size_t p = 12;
            for (size_t k = 0; k < nblk; k++) {
                uint32_t nb = be32(c + p);
#ifdef BSLZ4_PREZERO
                memset(dst, 0, bs);   /* a decoder that skips zero runs needs a zeroed buffer */
#endif
                sink += LZ4_decompress_safe((const char *) c + p + 4, dst, (int) nb, (int) bs);
                p += 4 + nb;
            }
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        double out = (double) nblk * bs;
        printf("%-34s %7.2f %8.3f %7.2f %8.3f | %8.1f %8.1f %6.1f %6.1f %6.1f  %016llx\n", label,
               out / 1e6, best * 1e3, out / best / 1e9, best * 1e9 / out, seqs / nblk,
               mb / (seqs > 0 ? seqs : 1), 100 * lit / out, 100 * m8 / out, 100 * m16 / out,
               (unsigned long long) hash);
        fflush(stdout);
        free(dst);
        free(c);
        (void) sink;
    }
    return 0;
}
