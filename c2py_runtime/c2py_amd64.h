/* c2py_amd64.h - x86_64 CPU feature flags
 *
 * Include this header in your .c2py 'headers:' field or your own C code
 * to access CPU feature globals. These are populated by c2py_runtime_init()
 * via the cpuid instruction at module load time.
 *
 * Naming mirrors GCC/Clang __builtin_cpu_supports() names.
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

/* AVX-512 family */
extern int c2py_amd64_avx512f;
extern int c2py_amd64_avx512bw;
extern int c2py_amd64_avx512dq;
extern int c2py_amd64_avx512vl;

/* BMI / bit manipulation */
extern int c2py_amd64_bmi1;
extern int c2py_amd64_bmi2;
extern int c2py_amd64_popcnt;
extern int c2py_amd64_lzcnt;

#ifdef __cplusplus
}
#endif

#endif /* C2PY_AMD64_H */
