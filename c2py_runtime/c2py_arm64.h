/* c2py_arm64.h - AArch64 CPU feature flags
 *
 * Include this header in your .c2py 'headers:' field or your own C code
 * to access CPU feature globals. These are populated by c2py_runtime_init()
 * via getauxval(AT_HWCAP) at module load time.
 */

#ifndef C2PY_ARM64_H
#define C2PY_ARM64_H

#ifdef __cplusplus
extern "C" {
#endif

/* Baseline (always 1 on AArch64) */
extern int c2py_arm64_fp;
extern int c2py_arm64_asimd;

/* Crypto extensions */
extern int c2py_arm64_aes;
extern int c2py_arm64_pmull;
extern int c2py_arm64_sha1;
extern int c2py_arm64_sha2;
extern int c2py_arm64_crc32;

/* Scalable Vector Extension */
extern int c2py_arm64_sve;
extern int c2py_arm64_sve2;

#ifdef __cplusplus
}
#endif

#endif /* C2PY_ARM64_H */
