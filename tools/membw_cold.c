/*
 * Single-pass DRAM read/write rate on one core, no best-of: each pass is
 * over a freshly allocated and first-touched buffer far larger than L3, and
 * is timed once.  Cross-check for tools/membw.c's DRAM plateau.
 *   cc -O3 -march=native -o membw_cold tools/membw_cold.c && taskset -c 7 ./membw_cold
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(void) {
    const size_t bytes = (size_t) 8 << 30, n = bytes / sizeof(double);
    double sink = 0;
    for (int pass = 0; pass < 3; pass++) {
        double *a = aligned_alloc(64, bytes);
        if (!a) return 1;
        for (size_t i = 0; i < n; i++) a[i] = (double) (i & 7);      /* first touch */
        /* evict: touch another 1 GB so nothing of `a` is left in L3 */
        char *e = malloc((size_t) 1 << 30);
        memset(e, 1, (size_t) 1 << 30);
        double t0 = now();
        __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
        for (size_t i = 0; i + 16 <= n; i += 16) {
            s0 = _mm256_add_pd(s0, _mm256_load_pd(a + i));
            s1 = _mm256_add_pd(s1, _mm256_load_pd(a + i + 4));
            s2 = _mm256_add_pd(s2, _mm256_load_pd(a + i + 8));
            s3 = _mm256_add_pd(s3, _mm256_load_pd(a + i + 12));
        }
        double tr = now() - t0;
        double r[4];
        _mm256_storeu_pd(r, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
        sink += r[0] + r[1] + r[2] + r[3] + e[12345];
        t0 = now();
        const __m256d v = _mm256_set1_pd(2.0);
        for (size_t i = 0; i + 4 <= n; i += 4) _mm256_store_pd(a + i, v);
        double tw = now() - t0;
        printf("pass %d: read %.1f GB/s, write %.1f GB/s (8 GiB, single pass each)\n",
               pass, bytes / tr / 1e9, bytes / tw / 1e9);
        fflush(stdout);
        free(e);
        free(a);
    }
    printf("(sink %g)\n", sink);
    return 0;
}
