/* c2py_amd64.h - x86_64 CPU feature flags
 *
 * Include this header in your .c2py 'headers:' field or your own C code
 * to access CPU feature globals. These are populated by c2py_runtime_init()
 * via the cpuid instruction at module load time.
 *
 * Naming mirrors GCC/Clang __builtin_cpu_supports() names.
 *
 * A flag is 1 only if the instructions are usable: the CPU has them, the
 * OS saves the register state they need (XCR0), and the features they build
 * on are also present (e.g. avx512vbmi2 implies avx512bw implies avx512f).
 * Each flag is one hardware feature; there are no tiers.  Check every
 * feature a function is compiled for, e.g. an avx2+fma function needs
 * c2py_amd64_avx2 && c2py_amd64_fma.
 */

#ifndef C2PY_AMD64_H
#define C2PY_AMD64_H

#ifdef __cplusplus
extern "C" {
#endif

/* Baseline features (always 1 on x86_64, check here for completeness) */
extern int c2py_amd64_mmx;
extern int c2py_amd64_sse;
extern int c2py_amd64_sse2;

/* SSE family */
extern int c2py_amd64_sse3;
extern int c2py_amd64_ssse3;
extern int c2py_amd64_sse4_1;
extern int c2py_amd64_sse4_2;

/* AVX family */
extern int c2py_amd64_avx;
extern int c2py_amd64_avx2;
extern int c2py_amd64_fma;
extern int c2py_amd64_f16c;
extern int c2py_amd64_avx_vnni;

/* AVX-512 family */
extern int c2py_amd64_avx512f;
extern int c2py_amd64_avx512bw;
extern int c2py_amd64_avx512dq;
extern int c2py_amd64_avx512vl;
extern int c2py_amd64_avx512cd;
extern int c2py_amd64_avx512ifma;
extern int c2py_amd64_avx512vbmi;
extern int c2py_amd64_avx512vbmi2;
extern int c2py_amd64_avx512vnni;
extern int c2py_amd64_avx512bitalg;
extern int c2py_amd64_avx512vpopcntdq;

/* Crypto / carry-less / Galois field.  gfni is the 128-bit form; its
 * 256-bit form also needs avx and its 512-bit form avx512f.  vaes and
 * vpclmulqdq imply avx. */
extern int c2py_amd64_aes;
extern int c2py_amd64_pclmulqdq;
extern int c2py_amd64_sha_ni;
extern int c2py_amd64_gfni;
extern int c2py_amd64_vaes;
extern int c2py_amd64_vpclmulqdq;

/* Vendor (from cpuid leaf 0).  Both 0 for other vendors or non-x86. */
extern int c2py_amd64_vendor_intel;
extern int c2py_amd64_vendor_amd;

/* BMI / bit manipulation */
extern int c2py_amd64_bmi1;
extern int c2py_amd64_bmi2;
extern int c2py_amd64_popcnt;
extern int c2py_amd64_lzcnt;
extern int c2py_amd64_adx;
extern int c2py_amd64_movbe;

#ifdef __cplusplus
}
#endif

#endif /* C2PY_AMD64_H */
