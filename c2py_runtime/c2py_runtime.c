/* c2py_runtime.c - CPython C-extension runtime loader
 *
 * Uses dlopen(NULL, ...) + dlsym() to resolve all CPython C API
 * function pointers at module load time (nimpy trick).  This eliminates
 * the need to link against -lpython, allowing one .so to work across
 * Python 2.7 through 3.15.
 *
 * In --pythonh mode (C2PY_USE_PYTHON_H defined), includes <Python.h>
 * directly.  No cross-version portability; one .so per Python version.
 *
 * Compile: gcc -c c2py_runtime.c -o c2py_runtime.o
 * Link:    gcc -shared ... c2py_runtime.o -ldl -o module.so
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <assert.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <sys/auxv.h> /* Linux: getauxval for CPU feature detection on ARM64/POWER */
#include <pthread.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#endif

#include "c2py_runtime.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

/* Cross-platform symbol resolution */
#ifdef _WIN32
#define C2PY_RESOLVE(handle, name) GetProcAddress((HMODULE)(handle), (name))
#else
#define C2PY_RESOLVE(handle, name) dlsym((handle), (name))
#endif

/* ---- Shared globals (visible to both backends) ---- */

/* CPU feature flags  --  always defined so .c2py files can reference
 * any arch features without link errors.  Non-matching arches stay 0. */

int c2py_amd64_mmx = 0;
int c2py_amd64_sse = 0;
int c2py_amd64_sse2 = 0;
int c2py_amd64_sse3 = 0;
int c2py_amd64_ssse3 = 0;
int c2py_amd64_sse4_1 = 0;
int c2py_amd64_sse4_2 = 0;
int c2py_amd64_avx = 0;
int c2py_amd64_avx2 = 0;
int c2py_amd64_fma = 0;
int c2py_amd64_avx512f = 0;
int c2py_amd64_avx512bw = 0;
int c2py_amd64_avx512dq = 0;
int c2py_amd64_avx512vl = 0;
int c2py_amd64_bmi1 = 0;
int c2py_amd64_bmi2 = 0;
int c2py_amd64_popcnt = 0;
int c2py_amd64_lzcnt = 0;

int c2py_arm64_fp = 0;
int c2py_arm64_asimd = 0;
int c2py_arm64_aes = 0;
int c2py_arm64_pmull = 0;
int c2py_arm64_sha1 = 0;
int c2py_arm64_sha2 = 0;
int c2py_arm64_crc32 = 0;
int c2py_arm64_sve = 0;
int c2py_arm64_sve2 = 0;

int c2py_ppc64_altivec = 0;
int c2py_ppc64_vsx = 0;
int c2py_ppc64_power8 = 0;
int c2py_ppc64_power9 = 0;

/* Tick source: 0 = wall clock (default), 1 = CPU cycle counter */
int _c2py_use_cycle_counter = 0;

/* Detected cycle counter frequency, or 0 if unknown (probed at init). */
uint64_t c2py_cycle_counter_frequency_hz = 0;

/* Active tick source frequency in Hz. */
uint64_t c2py_tick_frequency_hz = 0;

/* Runtime-discovered numpy ndarray layout (zero = not yet probed).
 * Shared by both backends (extern in c2py_runtime.h). */
c2py_ndarray_layout_t C2PY_NDARRAY = {0};

/* ---- Shared CPU feature probing (called by both backends) ---- */

static void _c2py_probe_cpu_features(void)
{
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    unsigned int eax1, ebx1, ecx1, edx1;
    unsigned int eax7, ebx7, ecx7, edx7;
    unsigned int eax81, ebx81, ecx81, edx81;
    unsigned int max_std;

#if defined(_MSC_VER)
    {
        int info[4];
        __cpuidex(info, 0, 0);
        max_std = (unsigned int)info[0];

        if (max_std >= 1) {
            __cpuidex(info, 1, 0);
            eax1 = (unsigned int)info[0];
            ebx1 = (unsigned int)info[1];
            ecx1 = (unsigned int)info[2];
            edx1 = (unsigned int)info[3];
            c2py_amd64_mmx    = (edx1 >> 23) & 1;
            c2py_amd64_sse    = (edx1 >> 25) & 1;
            c2py_amd64_sse2   = (edx1 >> 26) & 1;
            c2py_amd64_sse3   = (ecx1 >>  0) & 1;
            c2py_amd64_ssse3  = (ecx1 >>  9) & 1;
            c2py_amd64_sse4_1 = (ecx1 >> 19) & 1;
            c2py_amd64_sse4_2 = (ecx1 >> 20) & 1;
            c2py_amd64_avx    = (ecx1 >> 28) & 1;
            c2py_amd64_fma    = (ecx1 >> 12) & 1;
            c2py_amd64_popcnt = (ecx1 >> 23) & 1;
        }
        if (max_std >= 7) {
            __cpuidex(info, 7, 0);
            ebx7 = (unsigned int)info[1];
            ecx7 = (unsigned int)info[2];
            edx7 = (unsigned int)info[3];
            c2py_amd64_bmi1    = (ebx7 >>  3) & 1;
            c2py_amd64_avx2    = (ebx7 >>  5) & 1;
            c2py_amd64_bmi2    = (ebx7 >>  8) & 1;
            c2py_amd64_avx512f = (ebx7 >> 16) & 1;
            c2py_amd64_avx512dq = (ebx7 >> 17) & 1;
            c2py_amd64_avx512bw = (ebx7 >> 30) & 1;
            c2py_amd64_avx512vl = (ebx7 >> 31) & 1;
        }
        __cpuidex(info, 0x80000000, 0);
        if ((unsigned int)info[0] >= 0x80000001) {
            __cpuidex(info, 0x80000001, 0);
            ecx81 = (unsigned int)info[2];
            c2py_amd64_lzcnt = (ecx81 >> 5) & 1;
        }
    }
#elif defined(__GNUC__) || defined(__clang__)
    /* Determine max standard leaf */
    __asm__ __volatile__("cpuid"
        : "=a"(eax1) : "a"(0) : "ebx", "ecx", "edx");
    max_std = eax1;

    /* Leaf 1: baseline features */
    if (max_std >= 1) {
        __asm__ __volatile__("cpuid"
            : "=a"(eax1), "=b"(ebx1), "=c"(ecx1), "=d"(edx1)
            : "a"(1) : );
        c2py_amd64_mmx    = (edx1 >> 23) & 1;
        c2py_amd64_sse    = (edx1 >> 25) & 1;
        c2py_amd64_sse2   = (edx1 >> 26) & 1;
        c2py_amd64_sse3   = (ecx1 >>  0) & 1;
        c2py_amd64_ssse3  = (ecx1 >>  9) & 1;
        c2py_amd64_sse4_1 = (ecx1 >> 19) & 1;
        c2py_amd64_sse4_2 = (ecx1 >> 20) & 1;
        c2py_amd64_avx    = (ecx1 >> 28) & 1;
        c2py_amd64_fma    = (ecx1 >> 12) & 1;
        c2py_amd64_popcnt = (ecx1 >> 23) & 1;
    }

    /* Leaf 7, subleaf 0: extended features */
    if (max_std >= 7) {
        __asm__ __volatile__("cpuid"
            : "=a"(eax7), "=b"(ebx7), "=c"(ecx7), "=d"(edx7)
            : "a"(7), "c"(0));
        c2py_amd64_bmi1    = (ebx7 >>  3) & 1;
        c2py_amd64_avx2    = (ebx7 >>  5) & 1;
        c2py_amd64_bmi2    = (ebx7 >>  8) & 1;
        c2py_amd64_avx512f = (ebx7 >> 16) & 1;
        c2py_amd64_avx512dq = (ebx7 >> 17) & 1;
        c2py_amd64_avx512bw = (ebx7 >> 30) & 1;
        c2py_amd64_avx512vl = (ebx7 >> 31) & 1;
    }

    /* Leaf 0x80000001: extended feature bits (LZCNT) */
    /* Check extended leaf max first */
    __asm__ __volatile__("cpuid"
        : "=a"(eax81) : "a"(0x80000000) : "ebx", "ecx", "edx");
    if (eax81 >= 0x80000001) {
        __asm__ __volatile__("cpuid"
            : "=a"(eax81), "=b"(ebx81), "=c"(ecx81), "=d"(edx81)
            : "a"(0x80000001));
        c2py_amd64_lzcnt = (ecx81 >> 5) & 1;
    }
#else
    /* no inline assembly  --  CPU features stay 0 */
#endif /* _MSC_VER / __GNUC__ */
#endif

#if (defined(__aarch64__) || defined(__arm64__)) && !defined(_WIN32) && !defined(__APPLE__)
    {
        /* getauxval is Linux-specific; macOS ARM64 uses a separate path below. */
        unsigned long hwcap = getauxval(AT_HWCAP);
        unsigned long hwcap2 = getauxval(AT_HWCAP2);

        /* ARM64 HWCAP bits (stable kernel ABI) */
        c2py_arm64_fp    = (hwcap >> 0) & 1;
        c2py_arm64_asimd = (hwcap >> 1) & 1;
        c2py_arm64_aes   = (hwcap >> 3) & 1;
        c2py_arm64_pmull = (hwcap >> 4) & 1;
        c2py_arm64_sha1  = (hwcap >> 5) & 1;
        c2py_arm64_sha2  = (hwcap >> 6) & 1;
        c2py_arm64_crc32 = (hwcap >> 7) & 1;
        c2py_arm64_sve   = (hwcap >> 22) & 1;
        c2py_arm64_sve2  = (hwcap2 >> 1) & 1;
    }
#endif
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    {
        /* Apple Silicon: NEON (asimd) and FP are baseline on all chips.
         * sysctlbyname() can query optional features (AES, SHA, etc.),
         * but for now we set the baseline flags only. */
        c2py_arm64_fp    = 1;
        c2py_arm64_asimd = 1;
    }
#endif

#if (defined(__powerpc64__) || defined(__powerpc__)) && !defined(_WIN32) && !defined(__APPLE__)
    {
        unsigned long hwcap = getauxval(AT_HWCAP);
        unsigned long hwcap2 = getauxval(AT_HWCAP2);

        c2py_ppc64_altivec = (hwcap >> 28) & 1;        /* PPC_FEATURE_HAS_ALTIVEC = 0x10000000 */
        c2py_ppc64_vsx     = (hwcap >>  7) & 1;         /* PPC_FEATURE_HAS_VSX     = 0x00000080 */
        c2py_ppc64_power8  = (hwcap2 >> 31) & 1;        /* PPC_FEATURE2_ARCH_2_07  = 0x80000000 */
        c2py_ppc64_power9  = (hwcap2 >> 23) & 1;        /* PPC_FEATURE2_ARCH_3_00  = 0x00800000 */
    }
#endif

/* ---- Tick source frequency detection ---- */

    /* Default: wall clock returns nanoseconds, frequency is 1e9 Hz.
     * Cycle counter frequency is probed separately below and stored
     * in c2py_cycle_counter_frequency_hz; it is only used when the
     * caller toggles _c2py_use_cycle_counter to 1. */
    c2py_tick_frequency_hz = 1000000000ULL;

    /* Always attempt to detect CPU cycle counter frequency.
     * On platforms without a readable cycle counter this remains 0. */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    {
#if defined(_MSC_VER)
        int info[4];
        int got_freq = 0;
        __cpuidex(info, 0, 0);
        if ((unsigned int)info[0] >= 0x15) {
            __cpuidex(info, 0x15, 0);
            unsigned int eax = (unsigned int)info[0];
            unsigned int ebx = (unsigned int)info[1];
            unsigned int ecx = (unsigned int)info[2];
            if (ebx != 0 && eax != 0 && ecx != 0) {
                c2py_cycle_counter_frequency_hz =
                    (uint64_t)ecx * (uint64_t)ebx / (uint64_t)eax;
                got_freq = 1;
            }
        }
        if (!got_freq) {
            LARGE_INTEGER qpf;
            if (QueryPerformanceFrequency(&qpf) && qpf.QuadPart > 0) {
                c2py_cycle_counter_frequency_hz = (uint64_t)qpf.QuadPart;
            }
        }
#elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax, ebx, ecx, edx;
        unsigned int max_std;
        int got_freq = 0;
        __asm__ __volatile__("cpuid" : "=a"(max_std) : "a"(0) : "ebx", "ecx", "edx");
        if (max_std >= 0x15) {
            __asm__ __volatile__("cpuid"
                : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                : "a"(0x15));
            if (ebx != 0 && eax != 0 && ecx != 0) {
                c2py_cycle_counter_frequency_hz =
                    (uint64_t)ecx * (uint64_t)ebx / (uint64_t)eax;
                got_freq = 1;
            }
        }
        if (!got_freq) {
            FILE *f = fopen("/proc/cpuinfo", "r");
            if (f) {
                char line[256];
                while (fgets(line, sizeof(line), f)) {
                    double mhz;
                    if (sscanf(line, "cpu MHz : %lf", &mhz) == 1) {
                        c2py_cycle_counter_frequency_hz =
                            (uint64_t)(mhz * 1000000.0 + 0.5);
                        break;
                    }
                }
                fclose(f);
            }
        }
#else
        /* no inline assembly  --  cycle counter frequency stays 0 */
#endif /* _MSC_VER / __GNUC__ */
    }
#elif (defined(__aarch64__) || defined(__arm64__)) && !defined(_MSC_VER) \
      && (defined(__GNUC__) || defined(__clang__))
    {
        uint64_t freq;
        __asm__ __volatile__("mrs %0, CNTFRQ_EL0" : "=r"(freq));
        if (freq != 0) {
            c2py_cycle_counter_frequency_hz = freq;
        }
    }
#elif defined(__powerpc64__) || defined(__powerpc__)
    {
#ifdef _WIN32
        c2py_cycle_counter_frequency_hz = 0;
#else
        FILE *f = fopen("/proc/device-tree/cpus/timebase-frequency", "r");
        if (f) {
            unsigned char buf[8] = {0};
            size_t r = fread(buf, 1, 8, f);
            fclose(f);
            if (r == 4) {
                c2py_cycle_counter_frequency_hz =
                    ((uint64_t)buf[0] << 24) |
                    ((uint64_t)buf[1] << 16) |
                    ((uint64_t)buf[2] << 8)  |
                     (uint64_t)buf[3];
            } else if (r >= 8) {
                c2py_cycle_counter_frequency_hz =
                    ((uint64_t)buf[0] << 56) |
                    ((uint64_t)buf[1] << 48) |
                    ((uint64_t)buf[2] << 40) |
                    ((uint64_t)buf[3] << 32) |
                    ((uint64_t)buf[4] << 24) |
                    ((uint64_t)buf[5] << 16) |
                    ((uint64_t)buf[6] << 8)  |
                     (uint64_t)buf[7];
            }
        }
#endif /* _WIN32 */
    }
#endif
    /* If detection failed, c2py_cycle_counter_frequency_hz remains 0. */
}

/* ---- Backend selection ---- */

#ifdef C2PY_USE_PYTHON_H
#include "c2py_pythonh.c"
#else

/* ---- Nimpy/dlsym backend ---- */

/* Global API table */
c2py_api_t C2PY = {0};

static volatile int _c2py_runtime_initialized = 0;
static int _c2py_init_result = 0;

#ifndef _WIN32
static pthread_once_t _c2py_init_once = PTHREAD_ONCE_INIT;
#else
static CRITICAL_SECTION _c2py_init_cs;
static BOOL _c2py_init_cs_ready = FALSE;
#endif

#include "c2py_dlsym.c"

/* ---- Public init wrapper ---- */

int c2py_runtime_init(void)
{
#ifndef _WIN32
    pthread_once(&_c2py_init_once, _c2py_runtime_init_once);
    return _c2py_init_result;
#else
    if (!_c2py_runtime_initialized) {
        if (!_c2py_init_cs_ready) {
            InitializeCriticalSection(&_c2py_init_cs);
            _c2py_init_cs_ready = TRUE;
        }
        EnterCriticalSection(&_c2py_init_cs);
        if (!_c2py_runtime_initialized) {
            _c2py_runtime_init_once();
        }
        LeaveCriticalSection(&_c2py_init_cs);
    }
    return _c2py_init_result;
#endif
}

#endif /* C2PY_USE_PYTHON_H */
