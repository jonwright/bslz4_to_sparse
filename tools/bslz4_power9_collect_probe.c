/*
 * Standalone (no project build system, no Python, no dependencies)
 * correctness + rough-speed probe for a candidate POWER9/VSX
 * mask+threshold collect kernel, BEFORE it's wired into
 * bslz4_collect_simd.hpp. This is a first draft, unverified: I have no
 * POWER9 hardware or compiler to check it against, so treat any part
 * of the VSX kernel below with suspicion -- especially the mask
 * extraction, which uses vec_extract (scalar per-lane reads) rather
 * than a bit-permute movemask emulation specifically because I'm not
 * confident I'd get the vbpermq bit-numbering right for ppc64le without
 * a way to check it, and a wrong movemask would silently produce wrong
 * detector data. This version trades some peak throughput for being
 * much more likely to just be *correct* -- vec_cmpgt itself is real,
 * well-documented, native unsigned SIMD compare (POWER doesn't need
 * x86's sign-bias trick for unsigned compares at all), so even this
 * conservative version should show whether vectorizing the compare
 * step alone is worth anything here before investing more.
 *
 * Compile and run directly on the POWER9 box:
 *
 *   gcc -O2 -mcpu=power9 -maltivec -mvsx -o p9probe bslz4_power9_collect_probe.c
 *   ./p9probe
 *
 * (clang works too if gcc isn't available: same flags.) Prints PASS/FAIL
 * per test, and on an all-PASS run, a rough timing ratio at the end.
 * Paste back whatever it prints -- especially any FAIL lines and the
 * exact compiler error text if it doesn't build at all.
 *
 * This file is a throwaway probe, not part of the package build -- it's
 * only here (rather than /tmp) because this repo's filesystem is
 * visible from the POWER9 box too. Not referenced by setup.py/CI.
 *
 * v2: v1 (plain vec_cmpgt + a vec_extract loop over every lane, every
 * chunk) came back from real hardware CORRECT but SLOWER than scalar
 * (0.53x u16, 0.40x u32) -- vec_extract with a runtime lane index isn't
 * cheap on POWER9, and v1 paid that cost unconditionally. v2 adds one
 * guard before the extraction loop: vec_any_gt(v, vcut) -- if no lane's
 * value exceeds cut, mask can't change that (mask>0 & value>cut needs
 * value>cut regardless), so it's provably safe to skip the whole
 * per-lane extraction for that chunk. vec_any_gt is a plain, standard,
 * endianness-unambiguous AltiVec reduction (unlike vec_vbpermq's
 * movemask emulation, which is still being deliberately avoided here)
 * -- much lower risk than v1's alternative would have been. For sparse
 * data (most values near/at zero) most chunks should skip the loop
 * entirely; dense data (density=1.0 in the correctness sweep below)
 * exercises the worst case, one extra cheap comparison per chunk with
 * no skip.
 *
 * v3: v2 came back correct but STILL just as slow (0.52x u16, 0.41x
 * u32) -- turned out to be a bug in this probe, not the kernel: the
 * TIMING section generated pixel VALUES as xorshift64()&0xFFFF, uniform
 * over the full range, so with cut=0 the odds of hitting an exact zero
 * were 1/65536 -- vec_any_gt(v,0) was true for nearly every chunk
 * regardless of the "density" sweep value, which only ever controlled
 * the mask array, never the pixel values. v3 fixes the timing section
 * to use sparse_value() (mostly exact zero, rare small nonzero hits --
 * what real detector data actually looks like) so the skip path
 * actually gets exercised. The correctness section is untouched (still
 * uniform noise there deliberately, a more thorough range stress test).
 */
#include <altivec.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ---- scalar reference: identical logic to bslz4_core.hpp's inline
 * collect loop (the thing every tier, x86 or POWER, must match). ---- */

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

/* ---- candidate VSX kernels (the thing under test) ----
 *
 * vec_cmpgt on unsigned vector types is a real, native unsigned compare
 * (no sign-bias trick needed, unlike x86 SSE2/AVX2). Unaligned loads use
 * a plain memcpy into a vector-typed local -- a portable idiom that
 * modern GCC/Clang optimize into an efficient unaligned vector load on
 * any target, avoiding having to pick exactly the right load intrinsic
 * name (vec_vsx_ld vs vec_xl vs ...) for this compiler/target pairing.
 * Per-lane mask extraction is vec_extract in a small fixed-trip loop
 * (8 iterations for u16, 4 for u32) -- see the file comment above for
 * why this avoids the riskier bit-permute movemask emulation route.
 */

static int collect_vsx_u16(const uint16_t *block, const uint8_t *mask,
                            size_t n, uint16_t cut,
                            uint16_t *out_vals, uint32_t *out_adr) {
    int npx = 0;
    size_t j = 0;
    vector unsigned short vcut = vec_splats(cut);
    for (; j + 8 <= n; j += 8) {
        vector unsigned short v;
        memcpy(&v, &block[j], sizeof(v));
        if (!vec_any_gt(v, vcut)) continue; /* no lane's value > cut: mask can't
                                              * change that, safe to skip entirely */
        vector bool short gt = vec_cmpgt(v, vcut);
        for (int lane = 0; lane < 8; lane++) {
            if (mask[j + lane] > 0 && vec_extract((vector unsigned short) gt, lane)) {
                out_vals[npx] = vec_extract(v, lane);
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

static int collect_vsx_u32(const uint32_t *block, const uint8_t *mask,
                            size_t n, uint32_t cut,
                            uint32_t *out_vals, uint32_t *out_adr) {
    int npx = 0;
    size_t j = 0;
    vector unsigned int vcut = vec_splats(cut);
    for (; j + 4 <= n; j += 4) {
        vector unsigned int v;
        memcpy(&v, &block[j], sizeof(v));
        if (!vec_any_gt(v, vcut)) continue; /* see the u16 kernel's comment above */
        vector bool int gt = vec_cmpgt(v, vcut);
        for (int lane = 0; lane < 4; lane++) {
            if (mask[j + lane] > 0 && vec_extract((vector unsigned int) gt, lane)) {
                out_vals[npx] = vec_extract(v, lane);
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

/* Crude Poisson(~0.01)-like generator for the TIMING section only: mostly
 * exact zero, rare small nonzero hits -- v1/v2's timing runs used
 * xorshift64()&0xFFFF (uniform over the full range), so with cut=0 almost
 * every value was nonzero regardless of the "density" sweep (which only
 * ever controlled the mask, not the pixel values) -- vec_any_gt's skip
 * path never had a real chance to fire. This is what real sparse
 * detector data actually looks like (matches this project's own test
 * fixtures, np.random.poisson(0.01, ...)); the correctness section below
 * still uses full-range uniform noise deliberately, as a more thorough
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
static uint16_t ref_vals16[MAXN], vsx_vals16[MAXN];
static uint32_t ref_vals32[MAXN], vsx_vals32[MAXN];
static uint32_t ref_adr[MAXN], vsx_adr[MAXN];

static int all_pass = 1;

static void check_u16(size_t n, uint16_t cut, double density, const char *label) {
    for (size_t j = 0; j < n; j++) {
        mask[j] = (xorshift64() % 100 < density * 100) ? 1 : 0;
        block16[j] = (uint16_t) (xorshift64() & 0xFFFF);
    }
    int nref = collect_scalar_u16(block16, mask, n, cut, ref_vals16, ref_adr);
    int nvsx = collect_vsx_u16(block16, mask, n, cut, vsx_vals16, vsx_adr);
    int ok = (nref == nvsx) &&
             memcmp(ref_vals16, vsx_vals16, (size_t) nref * sizeof(uint16_t)) == 0 &&
             memcmp(ref_adr, vsx_adr, (size_t) nref * sizeof(uint32_t)) == 0;
    printf("%s u16 n=%zu cut=%u density=%.2f: scalar_npx=%d vsx_npx=%d -> %s\n",
           label, n, cut, density, nref, nvsx, ok ? "PASS" : "FAIL");
    if (!ok) all_pass = 0;
}

static void check_u32(size_t n, uint32_t cut, double density, const char *label) {
    for (size_t j = 0; j < n; j++) {
        mask[j] = (xorshift64() % 100 < density * 100) ? 1 : 0;
        block32[j] = (uint32_t) xorshift64();
    }
    int nref = collect_scalar_u32(block32, mask, n, cut, ref_vals32, ref_adr);
    int nvsx = collect_vsx_u32(block32, mask, n, cut, vsx_vals32, vsx_adr);
    int ok = (nref == nvsx) &&
             memcmp(ref_vals32, vsx_vals32, (size_t) nref * sizeof(uint32_t)) == 0 &&
             memcmp(ref_adr, vsx_adr, (size_t) nref * sizeof(uint32_t)) == 0;
    printf("%s u32 n=%zu cut=%u density=%.2f: scalar_npx=%d vsx_npx=%d -> %s\n",
           label, n, cut, density, nref, nvsx, ok ? "PASS" : "FAIL");
    if (!ok) all_pass = 0;
}

int main(void) {
    /* Sizes deliberately include non-multiples of 8/4 to exercise the
     * scalar tail, and both very sparse and fairly dense data. */
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

    /* Rough timing: sparse (density 0.01) at a realistic block size
     * (4096 u16 elements / 2048 u32 elements, matching bslz4's default
     * 8KB block). */
    size_t n16 = 4096, n32 = 2048;
    int reps = 20000;

    for (size_t j = 0; j < n16; j++) {
        mask[j] = (xorshift64() % 100 < 90) ? 1 : 0; /* most pixels active/unmasked */
        block16[j] = (uint16_t) sparse_value();
    }
    clock_t t0 = clock();
    for (int r = 0; r < reps; r++) collect_scalar_u16(block16, mask, n16, 0, ref_vals16, ref_adr);
    clock_t t1 = clock();
    for (int r = 0; r < reps; r++) collect_vsx_u16(block16, mask, n16, 0, vsx_vals16, vsx_adr);
    clock_t t2 = clock();
    double t_scalar16 = (double) (t1 - t0) / CLOCKS_PER_SEC;
    double t_vsx16 = (double) (t2 - t1) / CLOCKS_PER_SEC;
    printf("u16 n=4096 ~1%% nonzero pixels: scalar=%.4fs vsx=%.4fs speedup=%.2fx\n",
           t_scalar16, t_vsx16, t_scalar16 / t_vsx16);

    for (size_t j = 0; j < n32; j++) {
        mask[j] = (xorshift64() % 100 < 90) ? 1 : 0; /* most pixels active/unmasked */
        block32[j] = sparse_value();
    }
    t0 = clock();
    for (int r = 0; r < reps; r++) collect_scalar_u32(block32, mask, n32, 0, ref_vals32, ref_adr);
    t1 = clock();
    for (int r = 0; r < reps; r++) collect_vsx_u32(block32, mask, n32, 0, vsx_vals32, vsx_adr);
    t2 = clock();
    double t_scalar32 = (double) (t1 - t0) / CLOCKS_PER_SEC;
    double t_vsx32 = (double) (t2 - t1) / CLOCKS_PER_SEC;
    printf("u32 n=2048 ~1%% nonzero pixels: scalar=%.4fs vsx=%.4fs speedup=%.2fx\n",
           t_scalar32, t_vsx32, t_scalar32 / t_vsx32);

    return 0;
}
