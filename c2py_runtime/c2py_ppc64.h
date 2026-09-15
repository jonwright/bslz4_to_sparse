/* c2py_ppc64.h - POWER CPU feature flags
 *
 * Include this header in your .c2py 'headers:' field or your own C code
 * to access CPU feature globals. These are populated by c2py_runtime_init()
 * via getauxval(AT_HWCAP) at module load time.
 */

#ifndef C2PY_PPC64_H
#define C2PY_PPC64_H

#ifdef __cplusplus
extern "C" {
#endif

/* SIMD / Vector */
extern int c2py_ppc64_altivec;
extern int c2py_ppc64_vsx;

/* ISA levels (minimum for each Power ISA version) */
extern int c2py_ppc64_power8;
extern int c2py_ppc64_power9;

#ifdef __cplusplus
}
#endif

#endif /* C2PY_PPC64_H */
