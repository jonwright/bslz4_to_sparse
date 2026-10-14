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
int c2py_amd64_vendor_intel = 0;
int c2py_amd64_vendor_amd = 0;
int c2py_amd64_f16c = 0;
int c2py_amd64_movbe = 0;
int c2py_amd64_adx = 0;
int c2py_amd64_aes = 0;
int c2py_amd64_pclmulqdq = 0;
int c2py_amd64_sha_ni = 0;
int c2py_amd64_gfni = 0;
int c2py_amd64_vaes = 0;
int c2py_amd64_vpclmulqdq = 0;
int c2py_amd64_avx_vnni = 0;
int c2py_amd64_avx512cd = 0;
int c2py_amd64_avx512ifma = 0;
int c2py_amd64_avx512vbmi = 0;
int c2py_amd64_avx512vbmi2 = 0;
int c2py_amd64_avx512vnni = 0;
int c2py_amd64_avx512bitalg = 0;
int c2py_amd64_avx512vpopcntdq = 0;

int c2py_arm64_fp = 0;
int c2py_arm64_asimd = 0;
int c2py_arm64_aes = 0;
int c2py_arm64_pmull = 0;
int c2py_arm64_sha1 = 0;
int c2py_arm64_sha2 = 0;
int c2py_arm64_crc32 = 0;
int c2py_arm64_sve = 0;
int c2py_arm64_sve2 = 0;
int c2py_arm64_atomics = 0;
int c2py_arm64_fphp = 0;
int c2py_arm64_asimdhp = 0;
int c2py_arm64_asimddp = 0;
int c2py_arm64_i8mm = 0;
int c2py_arm64_bf16 = 0;

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

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
/* Returns 1 and fills r[0..3] = eax,ebx,ecx,edx, or 0 if no cpuid. */
static int _c2py_cpuid(unsigned int leaf, unsigned int sub, unsigned int *r)
{
#if defined(_MSC_VER)
    int info[4];
    __cpuidex(info, (int)leaf, (int)sub);
    r[0] = (unsigned int)info[0];
    r[1] = (unsigned int)info[1];
    r[2] = (unsigned int)info[2];
    r[3] = (unsigned int)info[3];
    return 1;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int a, b, c, d;
    /* ebx is the PIC register on i386, so let the compiler save it */
    __asm__ __volatile__(
#if defined(__i386__) && defined(__PIC__)
        "xchgl %%ebx, %1\n\tcpuid\n\txchgl %%ebx, %1"
        : "=a"(a), "=&r"(b), "=c"(c), "=d"(d)
#else
        "cpuid"
        : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
#endif
        : "a"(leaf), "c"(sub));
    r[0] = a; r[1] = b; r[2] = c; r[3] = d;
    return 1;
#else
    (void)leaf; (void)sub; (void)r;
    return 0; /* no inline assembly: CPU features stay 0 */
#endif
}

/* XCR0 (which register state the OS saves).  Only call when the OSXSAVE
 * bit (cpuid 1, ecx bit 27) is set.  Returns 0 if it cannot be read. */
static unsigned long long _c2py_xgetbv0(void)
{
#if defined(_MSC_VER)
#if _MSC_FULL_VER >= 160040219 /* VS2010 SP1: _xgetbv exists */
    return (unsigned long long)_xgetbv(0);
#elif defined(_WIN32)
    /* Older cl (e.g. the Python 2.7 toolchain) has no _xgetbv and no
     * inline asm on x64.  Windows 7 SP1 and later report the enabled
     * state mask (same bits as XCR0) through GetEnabledXStateFeatures. */
    {
        typedef unsigned __int64 (WINAPI *xstate_fn)(void);
        HMODULE k32 = GetModuleHandleA("kernel32.dll");
        xstate_fn fn = k32 ? (xstate_fn)GetProcAddress(k32, "GetEnabledXStateFeatures") : 0;
        return fn ? (unsigned long long)fn() : 0;
    }
#else
    return 0;
#endif
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int lo, hi;
    /* raw bytes so that no -mxsave is needed to assemble it */
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0"
        : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#else
    return 0;
#endif
}
#endif

static void _c2py_probe_cpu_features(void)
{
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    {
        unsigned int r[4];
        unsigned int max_std = 0, max_ext = 0;
        unsigned int ecx1 = 0, ebx7 = 0, ecx7 = 0, eax71 = 0, ecx81 = 0;
        unsigned long long xcr0 = 0;
        int ymm_ok = 0, zmm_ok = 0;
        int have_avx, have_avx512f, have_avx512bw;

        if (!_c2py_cpuid(0, 0, r)) {
            goto x86_done; /* no usable cpuid: all flags stay 0 */
        }
        max_std = r[0];
        /* vendor string is in ebx, edx, ecx of leaf 0 */
        c2py_amd64_vendor_intel = (r[1] == 0x756e6547u && r[3] == 0x49656e69u
                                   && r[2] == 0x6c65746eu); /* GenuineIntel */
        c2py_amd64_vendor_amd   = (r[1] == 0x68747541u && r[3] == 0x69746e65u
                                   && r[2] == 0x444d4163u); /* AuthenticAMD */
        if (max_std >= 1) {
            unsigned int edx1;
            _c2py_cpuid(1, 0, r);
            ecx1 = r[2];
            edx1 = r[3];
            c2py_amd64_mmx    = (edx1 >> 23) & 1;
            c2py_amd64_sse    = (edx1 >> 25) & 1;
            c2py_amd64_sse2   = (edx1 >> 26) & 1;
            c2py_amd64_sse3   = (ecx1 >>  0) & 1;
            c2py_amd64_pclmulqdq = (ecx1 >> 1) & 1;
            c2py_amd64_ssse3  = (ecx1 >>  9) & 1;
            c2py_amd64_sse4_1 = (ecx1 >> 19) & 1;
            c2py_amd64_sse4_2 = (ecx1 >> 20) & 1;
            c2py_amd64_movbe  = (ecx1 >> 22) & 1;
            c2py_amd64_popcnt = (ecx1 >> 23) & 1;
            c2py_amd64_aes    = (ecx1 >> 25) & 1;
        }
        if (max_std >= 7) {
            _c2py_cpuid(7, 0, r);
            ebx7 = r[1];
            ecx7 = r[2];
            if (r[0] >= 1) {
                _c2py_cpuid(7, 1, r);
                eax71 = r[0];
            }
        }
        _c2py_cpuid(0x80000000u, 0, r);
        max_ext = r[0];
        if (max_ext >= 0x80000001u) {
            _c2py_cpuid(0x80000001u, 0, r);
            ecx81 = r[2];
        }

        /* Vector register state: the CPUID bits only say the CPU has the
         * instructions.  The OS must also save YMM (XCR0 bits 1,2) and
         * ZMM/opmask (bits 5,6,7) state, or the instructions fault.
         * Without a readable XCR0 the AVX families are reported absent. */
        if ((ecx1 >> 27) & 1) {
            xcr0 = _c2py_xgetbv0();
            ymm_ok = (xcr0 & 0x06) == 0x06;
            zmm_ok = ymm_ok && (xcr0 & 0xE0) == 0xE0;
        }

        have_avx = ymm_ok && ((ecx1 >> 28) & 1);
        c2py_amd64_avx    = have_avx;
        c2py_amd64_fma    = have_avx && ((ecx1 >> 12) & 1);
        c2py_amd64_f16c   = have_avx && ((ecx1 >> 29) & 1);
        c2py_amd64_avx2   = have_avx && ((ebx7 >> 5) & 1);
        c2py_amd64_avx_vnni = have_avx && ((eax71 >> 4) & 1);
        c2py_amd64_vaes   = have_avx && ((ecx7 >> 9) & 1);
        c2py_amd64_vpclmulqdq = have_avx && ((ecx7 >> 10) & 1);

        /* VEX-encoded but touch no vector state: no XCR0 gate. */
        c2py_amd64_bmi1   = (ebx7 >>  3) & 1;
        c2py_amd64_bmi2   = (ebx7 >>  8) & 1;
        c2py_amd64_adx    = (ebx7 >> 19) & 1;
        c2py_amd64_sha_ni = (ebx7 >> 29) & 1;
        c2py_amd64_lzcnt  = (ecx81 >> 5) & 1;
        /* 128-bit (SSE) form only; the 256/512-bit forms also need avx
         * or avx512f respectively. */
        c2py_amd64_gfni   = (ecx7 >> 8) & 1;

        have_avx512f  = zmm_ok && ((ebx7 >> 16) & 1);
        have_avx512bw = have_avx512f && ((ebx7 >> 30) & 1);
        c2py_amd64_avx512f  = have_avx512f;
        c2py_amd64_avx512dq = have_avx512f && ((ebx7 >> 17) & 1);
        c2py_amd64_avx512ifma = have_avx512f && ((ebx7 >> 21) & 1);
        c2py_amd64_avx512cd = have_avx512f && ((ebx7 >> 28) & 1);
        c2py_amd64_avx512bw = have_avx512bw;
        c2py_amd64_avx512vl = have_avx512f && ((ebx7 >> 31) & 1);
        c2py_amd64_avx512vnni = have_avx512f && ((ecx7 >> 11) & 1);
        c2py_amd64_avx512vpopcntdq = have_avx512f && ((ecx7 >> 14) & 1);
        c2py_amd64_avx512vbmi  = have_avx512bw && ((ecx7 >> 1) & 1);
        c2py_amd64_avx512vbmi2 = have_avx512bw && ((ecx7 >> 6) & 1);
        c2py_amd64_avx512bitalg = have_avx512bw && ((ecx7 >> 12) & 1);
    }
x86_done:
    ;
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
        c2py_arm64_atomics = (hwcap >> 8) & 1;
        c2py_arm64_fphp  = (hwcap >> 9) & 1;
        c2py_arm64_asimdhp = (hwcap >> 10) & 1;
        c2py_arm64_asimddp = (hwcap >> 20) & 1;
        c2py_arm64_i8mm  = (hwcap2 >> 13) & 1;
        c2py_arm64_bf16  = (hwcap2 >> 14) & 1;
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
        unsigned int eax, ebx, ecx;
        __cpuidex(info, 0, 0);
        if ((unsigned int)info[0] >= 0x15) {
            __cpuidex(info, 0x15, 0);
            eax = (unsigned int)info[0];
            ebx = (unsigned int)info[1];
            ecx = (unsigned int)info[2];
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
