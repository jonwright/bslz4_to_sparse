/*
 * Single-core memory bandwidth by working-set size (a small STREAM-like
 * probe), to place the matrix-dot kernels against the L1 / L2 / L3 / DRAM
 * limits of the machine.  Run it pinned to one core:
 *
 *   cc -O3 -march=native -o membw tools/membw.c && taskset -c 7 ./membw [max [min]]
 *
 * For each size it repeats the kernel over the same buffer until ~0.2 s has
 * passed and reports the best GB/s:
 *   read   sum of doubles (8 independent AVX accumulators)
 *   write  store a constant
 *   copy   b[i] = a[i]               (bytes counted: read + write)
 *   triad  c[i] = a[i] + s * b[i]    (bytes counted: 2 reads + 1 write)
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

static double sink;

static void k_read(const double *a, size_t n) {
    __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
    for (size_t i = 0; i + 32 <= n; i += 32) {
        s0 = _mm256_add_pd(s0, _mm256_load_pd(a + i));
        s1 = _mm256_add_pd(s1, _mm256_load_pd(a + i + 4));
        s2 = _mm256_add_pd(s2, _mm256_load_pd(a + i + 8));
        s3 = _mm256_add_pd(s3, _mm256_load_pd(a + i + 12));
        s4 = _mm256_add_pd(s4, _mm256_load_pd(a + i + 16));
        s5 = _mm256_add_pd(s5, _mm256_load_pd(a + i + 20));
        s6 = _mm256_add_pd(s6, _mm256_load_pd(a + i + 24));
        s7 = _mm256_add_pd(s7, _mm256_load_pd(a + i + 28));
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)),
                              _mm256_add_pd(_mm256_add_pd(s4, s5), _mm256_add_pd(s6, s7)));
    double r[4];
    _mm256_storeu_pd(r, s);
    sink += r[0] + r[1] + r[2] + r[3];
}

static void k_write(double *a, size_t n) {
    const __m256d v = _mm256_set1_pd(1.5);
    for (size_t i = 0; i + 4 <= n; i += 4) _mm256_store_pd(a + i, v);
}

static void k_copy(double *b, const double *a, size_t n) {
    for (size_t i = 0; i + 4 <= n; i += 4) _mm256_store_pd(b + i, _mm256_load_pd(a + i));
}

static void k_triad(double *c, const double *a, const double *b, size_t n) {
    const __m256d s = _mm256_set1_pd(0.5);
    for (size_t i = 0; i + 4 <= n; i += 4)
        _mm256_store_pd(c + i, _mm256_fmadd_pd(s, _mm256_load_pd(b + i), _mm256_load_pd(a + i)));
}

int main(int argc, char **argv) {
    size_t maxbytes = (argc > 1) ? (size_t) atoll(argv[1]) : ((size_t) 2 << 30);
    size_t minbytes = (argc > 2) ? (size_t) atoll(argv[2]) : 8192;
    size_t nmax = maxbytes / sizeof(double);
    double *a = aligned_alloc(64, nmax * sizeof(double));
    double *b = aligned_alloc(64, nmax * sizeof(double));
    double *c = aligned_alloc(64, nmax * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (size_t i = 0; i < nmax; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    printf("%12s %10s %10s %10s %10s   (GB/s, one core; size = bytes per array)\n",
           "size", "read", "write", "copy", "triad");
    for (size_t bytes = minbytes; bytes <= maxbytes; bytes *= 2) {
        size_t n = bytes / sizeof(double);
        double best[4] = {0, 0, 0, 0};
        for (int k = 0; k < 4; k++) {
            double moved = (k == 0 || k == 1) ? bytes : (k == 2 ? 2.0 * bytes : 3.0 * bytes);
            double tstart = now();
            int reps = 0;
            while (now() - tstart < 0.2 || reps < 3) {
                double t0 = now();
                switch (k) {
                case 0: k_read(a, n); break;
                case 1: k_write(c, n); break;
                case 2: k_copy(c, a, n); break;
                case 3: k_triad(c, a, b, n); break;
                }
                double dt = now() - t0;
                double gbs = moved / dt / 1e9;
                if (gbs > best[k]) best[k] = gbs;
                reps++;
            }
        }
        char lab[32];
        if (bytes >= (1u << 20)) snprintf(lab, sizeof lab, "%zu MiB", bytes >> 20);
        else snprintf(lab, sizeof lab, "%zu KiB", bytes >> 10);
        printf("%12s %10.1f %10.1f %10.1f %10.1f\n", lab, best[0], best[1], best[2], best[3]);
        fflush(stdout);
    }
    printf("(sink %g)\n", sink);
    return 0;
}
