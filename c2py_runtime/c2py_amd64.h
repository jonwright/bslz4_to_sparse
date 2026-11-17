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

/* Runtime globals and functions must never be exported from the module:
 * under RTLD_GLOBAL a second c2py23 module would bind to the first one's
 * tables.  Independent of the user's compiler flags.  Defined identically
 * (and guarded) in every runtime header so any include order works. */
#ifndef C2PY_HIDDEN
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#define C2PY_HIDDEN __attribute__((visibility("hidden")))
#else
#define C2PY_HIDDEN
#endif
#endif


#ifdef __cplusplus
extern "C" {
#endif

/* Baseline features (always 1 on x86_64, check here for completeness) */
extern C2PY_HIDDEN int c2py_amd64_mmx;
extern C2PY_HIDDEN int c2py_amd64_sse;
extern C2PY_HIDDEN int c2py_amd64_sse2;

/* SSE family */
extern C2PY_HIDDEN int c2py_amd64_sse3;
extern C2PY_HIDDEN int c2py_amd64_ssse3;
extern C2PY_HIDDEN int c2py_amd64_sse4_1;
extern C2PY_HIDDEN int c2py_amd64_sse4_2;

/* AVX family */
extern C2PY_HIDDEN int c2py_amd64_avx;
extern C2PY_HIDDEN int c2py_amd64_avx2;
extern C2PY_HIDDEN int c2py_amd64_fma;
extern C2PY_HIDDEN int c2py_amd64_f16c;
extern C2PY_HIDDEN int c2py_amd64_avx_vnni;

/* AVX-512 family */
extern C2PY_HIDDEN int c2py_amd64_avx512f;
extern C2PY_HIDDEN int c2py_amd64_avx512bw;
extern C2PY_HIDDEN int c2py_amd64_avx512dq;
extern C2PY_HIDDEN int c2py_amd64_avx512vl;
extern C2PY_HIDDEN int c2py_amd64_avx512cd;
extern C2PY_HIDDEN int c2py_amd64_avx512ifma;
extern C2PY_HIDDEN int c2py_amd64_avx512vbmi;
extern C2PY_HIDDEN int c2py_amd64_avx512vbmi2;
extern C2PY_HIDDEN int c2py_amd64_avx512vnni;
extern C2PY_HIDDEN int c2py_amd64_avx512bitalg;
extern C2PY_HIDDEN int c2py_amd64_avx512vpopcntdq;

/* Crypto / carry-less / Galois field.  gfni is the 128-bit form; its
 * 256-bit form also needs avx and its 512-bit form avx512f.  vaes and
 * vpclmulqdq imply avx. */
extern C2PY_HIDDEN int c2py_amd64_aes;
extern C2PY_HIDDEN int c2py_amd64_pclmulqdq;
extern C2PY_HIDDEN int c2py_amd64_sha_ni;
extern C2PY_HIDDEN int c2py_amd64_gfni;
extern C2PY_HIDDEN int c2py_amd64_vaes;
extern C2PY_HIDDEN int c2py_amd64_vpclmulqdq;

/* Vendor (from cpuid leaf 0).  Both 0 for other vendors or non-x86. */
extern C2PY_HIDDEN int c2py_amd64_vendor_intel;
extern C2PY_HIDDEN int c2py_amd64_vendor_amd;

/* BMI / bit manipulation */
extern C2PY_HIDDEN int c2py_amd64_bmi1;
extern C2PY_HIDDEN int c2py_amd64_bmi2;
extern C2PY_HIDDEN int c2py_amd64_popcnt;
extern C2PY_HIDDEN int c2py_amd64_lzcnt;
extern C2PY_HIDDEN int c2py_amd64_adx;
extern C2PY_HIDDEN int c2py_amd64_movbe;

#ifdef __cplusplus
}
#endif

#endif /* C2PY_AMD64_H */
