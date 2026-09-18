/*
 * Standalone (no project build system, no Python, no dependencies)
 * correctness + rough-speed probe for a candidate ARM NEON mask+
 * threshold collect kernel, BEFORE it's wired into
 * bslz4_collect_simd.hpp. Completely unverified -- zero iterations on
 * real hardware yet, unlike the POWER9 version this is modeled on
 * (tools/bslz4_power9_collect_probe.c, which took three rounds on real
 * hardware to get right: v1 vectorized the compare but paid an
 * unconditional per-lane extraction cost and came back SLOWER than
 * scalar; v2 added a "skip if nothing matched" gate but the probe's own
 * test data was a poor match for real sparse detector data so the gate
 * never got to prove itself; v3 fixed that and came back 3.75-8.2x
 * faster). This file starts from that ending point instead of
 * repeating the journey: the any-match gate and realistic sparse test
 * data are both here from the start.
 *
 * Targets AArch64 (64-bit ARM, ARMv8-A+) specifically -- matches this
 * project's actual CI target (cibuildwheel builds aarch64, not 32-bit
 * armv7), and vmaxvq_u16/u32 (the reduction this kernel's gate relies
 * on) are AArch64-only NEON intrinsics; 32-bit ARM would need a
 * different (slower, manual pairwise) reduction.
 *
 * Compile and run directly on the ARM box:
 *
 *   gcc -O2 -march=armv8-a -o neonprobe bslz4_neon_collect_probe.c
 *   ./neonprobe
 *
 * (clang works too if gcc isn't available: same flags.) Prints PASS/FAIL
 * per test, and on an all-PASS run, a rough timing ratio at the end.
 * Paste back whatever it prints -- especially any FAIL lines and the
 * exact compiler error text if it doesn't build at all.
 *
 * This file is a throwaway probe, not part of the package build -- it's
 * in the repo (not /tmp) so it's reachable from a shared-filesystem ARM
 * box the same way the POWER9 probe was. Not referenced by setup.py/CI.
 */
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ---- scalar reference: identical logic to bslz4_core.hpp's inline
 * collect loop (the thing every tier, x86/POWER/ARM, must match). ---- */

static int collect_scalar_u16(const uint16_t *block, const uint8_t *mask,
                               size_t n, uint16_t cut,
                               uint16_t *out_vals, uint32_t *out_adr) {
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if ((mask[j] > 0) && (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) j;
            npx++;
        }
    }
    return npx;
}

static int collect_scalar_u32(const uint32_t *block, const uint8_t *mask,
                               size_t n, uint32_t cut,
                               uint32_t *out_vals, uint32_t *out_adr) {
    int npx = 0;
    for (size_t j = 0; j < n; j++) {
        if ((mask[j] > 0) && (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) j;
            npx++;
        }
    }
    return npx;
}

/* ---- candidate NEON kernels (the thing under test) ----
 *
 * vcgtq_u16/u32 is a native unsigned compare-greater (no sign-bias
 * trick needed, like POWER, unlike x86 SSE2/AVX2). vmaxvq_u16/u32
 * (AArch64-only) reduces a vector to its max lane in one instruction --
 * if the max of a compare result (each lane 0xFFFF/0 or
 * 0xFFFFFFFF/0) is 0, no lane matched, and mask can't change that
 * (mask>0 & value>cut needs value>cut regardless), so it's provably
 * safe to skip the whole per-lane extraction for that chunk.
 *
 * vgetq_lane_u16/u32 need a compile-time-constant lane index (a real
 * ACLE restriction, unlike POWER's vec_extract which tolerates a
 * runtime index) -- so the fallback path spills the compare result and
 * the value vector to small stack arrays via vst1q_u16/u32 and then
 * uses ordinary scalar indexing, avoiding that restriction rather than
 * fighting it with a manually unrolled constant-index ladder.
 */

static int collect_neon_u16(const uint16_t *block, const uint8_t *mask,
                             size_t n, uint16_t cut,
                             uint16_t *out_vals, uint32_t *out_adr) {
    int npx = 0;
    size_t j = 0;
    uint16x8_t vcut = vdupq_n_u16(cut);
    for (; j + 8 <= n; j += 8) {
        uint16x8_t v = vld1q_u16(&block[j]);
        uint16x8_t gt = vcgtq_u16(v, vcut);
        if (vmaxvq_u16(gt) == 0) continue; /* no lane's value > cut: safe to skip */
        uint16_t vbuf[8], gbuf[8];
        vst1q_u16(vbuf, v);
        vst1q_u16(gbuf, gt);
        for (int lane = 0; lane < 8; lane++) {
            if (mask[j + lane] > 0 && gbuf[lane]) {
                out_vals[npx] = vbuf[lane];
                out_adr[npx] = (uint32_t) (j + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if ((mask[j] > 0) && (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) j;
            npx++;
        }
    }
    return npx;
}

static int collect_neon_u32(const uint32_t *block, const uint8_t *mask,
                             size_t n, uint32_t cut,
                             uint32_t *out_vals, uint32_t *out_adr) {
    int npx = 0;
    size_t j = 0;
    uint32x4_t vcut = vdupq_n_u32(cut);
    for (; j + 4 <= n; j += 4) {
        uint32x4_t v = vld1q_u32(&block[j]);
        uint32x4_t gt = vcgtq_u32(v, vcut);
        if (vmaxvq_u32(gt) == 0) continue;
        uint32_t vbuf[4], gbuf[4];
        vst1q_u32(vbuf, v);
        vst1q_u32(gbuf, gt);
        for (int lane = 0; lane < 4; lane++) {
            if (mask[j + lane] > 0 && gbuf[lane]) {
                out_vals[npx] = vbuf[lane];
                out_adr[npx] = (uint32_t) (j + lane);
                npx++;
            }
        }
    }
    for (; j < n; j++) {
        if ((mask[j] > 0) && (block[j] > cut)) {
            out_vals[npx] = block[j];
            out_adr[npx] = (uint32_t) j;
            npx++;
        }
    }
    return npx;
}

/* ---- deterministic PRNG, so runs are reproducible across machines ---- */
static uint64_t rng_state = 88172645463325252ull;
static uint64_t xorshift64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* Crude Poisson(~0.01)-like generator for the TIMING section: mostly
 * exact zero, rare small nonzero hits -- what real sparse detector data
 * actually looks like (matches this project's own test fixtures,
 * np.random.poisson(0.01, ...)). The correctness section below uses
 * full-range uniform noise deliberately instead, as a more thorough
 * stress test across the whole value range. */
static uint32_t sparse_value(void) {
    if (xorshift64() % 1000 < 10) /* ~1% nonzero */
        return (uint32_t) (xorshift64() % 20) + 1;
    return 0;
}

#define MAXN 8192

static uint8_t mask[MAXN];
static uint16_t block16[MAXN];
static uint32_t block32[MAXN];
static uint16_t ref_vals16[MAXN], cand_vals16[MAXN];
static uint32_t ref_vals32[MAXN], cand_vals32[MAXN];
static uint32_t ref_adr[MAXN], cand_adr[MAXN];

static int all_pass = 1;

static void check_u16(size_t n, uint16_t cut, double density, const char *label) {
    for (size_t j = 0; j < n; j++) {
        mask[j] = (xorshift64() % 100 < density * 100) ? 1 : 0;
        block16[j] = (uint16_t) (xorshift64() & 0xFFFF);
    }
    int nref = collect_scalar_u16(block16, mask, n, cut, ref_vals16, ref_adr);
    int ncand = collect_neon_u16(block16, mask, n, cut, cand_vals16, cand_adr);
    int ok = (nref == ncand) &&
             memcmp(ref_vals16, cand_vals16, (size_t) nref * sizeof(uint16_t)) == 0 &&
             memcmp(ref_adr, cand_adr, (size_t) nref * sizeof(uint32_t)) == 0;
    printf("%s u16 n=%zu cut=%u density=%.2f: scalar_npx=%d neon_npx=%d -> %s\n",
           label, n, cut, density, nref, ncand, ok ? "PASS" : "FAIL");
    if (!ok) all_pass = 0;
}

static void check_u32(size_t n, uint32_t cut, double density, const char *label) {
    for (size_t j = 0; j < n; j++) {
        mask[j] = (xorshift64() % 100 < density * 100) ? 1 : 0;
        block32[j] = (uint32_t) xorshift64();
    }
    int nref = collect_scalar_u32(block32, mask, n, cut, ref_vals32, ref_adr);
    int ncand = collect_neon_u32(block32, mask, n, cut, cand_vals32, cand_adr);
    int ok = (nref == ncand) &&
             memcmp(ref_vals32, cand_vals32, (size_t) nref * sizeof(uint32_t)) == 0 &&
             memcmp(ref_adr, cand_adr, (size_t) nref * sizeof(uint32_t)) == 0;
    printf("%s u32 n=%zu cut=%u density=%.2f: scalar_npx=%d neon_npx=%d -> %s\n",
           label, n, cut, density, nref, ncand, ok ? "PASS" : "FAIL");
    if (!ok) all_pass = 0;
}

int main(void) {
    size_t sizes[] = {0, 1, 3, 7, 8, 9, 15, 16, 17, 100, 4095, 4096, 4097, 8192};
    double densities[] = {0.001, 0.01, 0.1, 0.5, 1.0};
    uint16_t cuts16[] = {0, 1, 100, 65535};
    uint32_t cuts32[] = {0, 1, 100, 4294967295u};

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++)
        for (size_t di = 0; di < sizeof(densities) / sizeof(densities[0]); di++)
            for (size_t ci = 0; ci < sizeof(cuts16) / sizeof(cuts16[0]); ci++)
                check_u16(sizes[si], cuts16[ci], densities[di], "correctness");

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++)
        for (size_t di = 0; di < sizeof(densities) / sizeof(densities[0]); di++)
            for (size_t ci = 0; ci < sizeof(cuts32) / sizeof(cuts32[0]); ci++)
                check_u32(sizes[si], cuts32[ci], densities[di], "correctness");

    printf("\n%s\n\n", all_pass ? "ALL CORRECTNESS CHECKS PASSED" : "SOME CHECKS FAILED -- see above");

    if (!all_pass) {
        printf("Stopping before timing -- fix correctness first.\n");
        return 1;
    }

    /* Rough timing: ~1%% nonzero pixels, ~90%% mask density, at a
     * realistic block size (4096 u16 elements / 2048 u32 elements,
     * matching bslz4's default 8KB block). */
    size_t n16 = 4096, n32 = 2048;
    int reps = 20000;

    for (size_t j = 0; j < n16; j++) {
        mask[j] = (xorshift64() % 100 < 90) ? 1 : 0;
        block16[j] = (uint16_t) sparse_value();
    }
    clock_t t0 = clock();
    for (int r = 0; r < reps; r++) collect_scalar_u16(block16, mask, n16, 0, ref_vals16, ref_adr);
    clock_t t1 = clock();
    for (int r = 0; r < reps; r++) collect_neon_u16(block16, mask, n16, 0, cand_vals16, cand_adr);
    clock_t t2 = clock();
    double t_scalar16 = (double) (t1 - t0) / CLOCKS_PER_SEC;
    double t_neon16 = (double) (t2 - t1) / CLOCKS_PER_SEC;
    printf("u16 n=4096 ~1%% nonzero pixels: scalar=%.4fs neon=%.4fs speedup=%.2fx\n",
           t_scalar16, t_neon16, t_scalar16 / t_neon16);

    for (size_t j = 0; j < n32; j++) {
        mask[j] = (xorshift64() % 100 < 90) ? 1 : 0;
        block32[j] = sparse_value();
    }
    t0 = clock();
    for (int r = 0; r < reps; r++) collect_scalar_u32(block32, mask, n32, 0, ref_vals32, ref_adr);
    t1 = clock();
    for (int r = 0; r < reps; r++) collect_neon_u32(block32, mask, n32, 0, cand_vals32, cand_adr);
    t2 = clock();
    double t_scalar32 = (double) (t1 - t0) / CLOCKS_PER_SEC;
    double t_neon32 = (double) (t2 - t1) / CLOCKS_PER_SEC;
    printf("u32 n=2048 ~1%% nonzero pixels: scalar=%.4fs neon=%.4fs speedup=%.2fx\n",
           t_scalar32, t_neon32, t_scalar32 / t_neon32);

    return 0;
}
