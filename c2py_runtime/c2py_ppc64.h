/* c2py_ppc64.h - POWER CPU feature flags
 *
 * Include this header in your .c2py 'headers:' field or your own C code
 * to access CPU feature globals. These are populated by c2py_runtime_init()
 * via getauxval(AT_HWCAP) at module load time.
 */

#ifndef C2PY_PPC64_H
#define C2PY_PPC64_H

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

/* SIMD / Vector */
extern C2PY_HIDDEN int c2py_ppc64_altivec;
extern C2PY_HIDDEN int c2py_ppc64_vsx;

/* ISA levels (minimum for each Power ISA version) */
extern C2PY_HIDDEN int c2py_ppc64_power8;
extern C2PY_HIDDEN int c2py_ppc64_power9;

#ifdef __cplusplus
}
#endif

#endif /* C2PY_PPC64_H */
