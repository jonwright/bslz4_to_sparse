/* c2py_arm64.h - AArch64 CPU feature flags
 *
 * Include this header in your .c2py 'headers:' field or your own C code
 * to access CPU feature globals. These are populated by c2py_runtime_init()
 * via getauxval(AT_HWCAP) at module load time.
 */

#ifndef C2PY_ARM64_H
#define C2PY_ARM64_H

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

/* Baseline (always 1 on AArch64) */
extern C2PY_HIDDEN int c2py_arm64_fp;
extern C2PY_HIDDEN int c2py_arm64_asimd;

/* Crypto extensions */
extern C2PY_HIDDEN int c2py_arm64_aes;
extern C2PY_HIDDEN int c2py_arm64_pmull;
extern C2PY_HIDDEN int c2py_arm64_sha1;
extern C2PY_HIDDEN int c2py_arm64_sha2;
extern C2PY_HIDDEN int c2py_arm64_crc32;

/* Optional extensions (Linux HWCAP bits; 0 on Apple and Windows for now) */
extern C2PY_HIDDEN int c2py_arm64_atomics;   /* LSE */
extern C2PY_HIDDEN int c2py_arm64_fphp;
extern C2PY_HIDDEN int c2py_arm64_asimdhp;
extern C2PY_HIDDEN int c2py_arm64_asimddp;   /* dot product */
extern C2PY_HIDDEN int c2py_arm64_i8mm;
extern C2PY_HIDDEN int c2py_arm64_bf16;

/* Scalable Vector Extension */
extern C2PY_HIDDEN int c2py_arm64_sve;
extern C2PY_HIDDEN int c2py_arm64_sve2;

#ifdef __cplusplus
}
#endif

#endif /* C2PY_ARM64_H */
