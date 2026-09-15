/* c2py_dlsym.h - nimpy-style dlsym backend
 *
 * Hand-declared CPython type definitions and system includes.
 * All CPython API is resolved at runtime via dlopen(NULL) + dlsym().
 * One .so works on Python 2.7 through 3.15 without recompilation.
 *
 * This header is NOT included directly.  Include c2py_runtime.h instead.
 */

/* ---- System includes ---- */

#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/* Symbol resolution helper -- dlsym on Unix, GetProcAddress on Windows */
#ifdef _WIN32
#define C2PY_RESOLVE(handle, name) GetProcAddress((HMODULE)(handle), (name))
#else
#define C2PY_RESOLVE(handle, name) dlsym((handle), (name))
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* MSVC C mode does not recognise the inline keyword (C++ only).
 * __inline is the MSVC equivalent (also recognised by MinGW). */
#ifdef _MSC_VER
#define inline __inline
#endif

/* DLL export attribute for module init functions.
 * On Windows the PyInit_<name> symbol must be in the .pyd export
 * table or Python cannot load the module. */
#ifdef _WIN32
#define C2PY_EXPORT __declspec(dllexport)
#else
#define C2PY_EXPORT
#endif

/* ------------------------------------------------------------------ */
/* Py_ssize_t - must be defined before any struct using it            */
/* ------------------------------------------------------------------ */

#if defined(__LP64__) || defined(_WIN64)
typedef long long Py_ssize_t;
#else
typedef long Py_ssize_t;
#endif

/* sizeof(Py_buffer) differs across CPython versions:
 * < 3.12: includes smalltable[2]  (96 bytes LP64, 52 bytes ILP32)
 * >= 3.12: smalltable removed for PEP 697 stable ABI (80 / 44)
 */
#if defined(__LP64__) || defined(_WIN64)
#define C2PY_PYBUFFER_SZ_PRE312   96
#define C2PY_PYBUFFER_SZ_POST312  80
#else
#define C2PY_PYBUFFER_SZ_PRE312   52
#define C2PY_PYBUFFER_SZ_POST312  44
#endif

/* ------------------------------------------------------------------ */
/* CPython type definitions (stable layouts across versions)          */
/* ------------------------------------------------------------------ */

/* PyObject layout: differs between GIL-enabled and free-threaded builds.
 *
 * GIL-enabled (CPython 2.7 - 3.14 standard):
 *   LP64:    16 bytes.  ob_refcnt at 0, ob_type at 8.
 *   ILP32:    8 bytes.  ob_refcnt at 0, ob_type at 4.
 *
 * Free-threaded (CPython 3.13t+ --disable-gil, LP64 only):
 *   32 bytes.   ob_ref_shared at 16, ob_type at 24.
 *   FT does not exist on ILP32.
 *
 * We define both layouts.  Generated code uses macros (C2PY_SET_MNAME,
 * C2PY_SET_MDOC, etc.) that work with either layout via C2PY offsets.
 */

/* GIL-enabled PyObject layout (standard CPython) */
typedef struct _c2py_object {
    Py_ssize_t ob_refcnt;
#ifdef C2PY_TARGET_PYPY
    void *ob_pypy_link;  /* PyPy cpyext inserts this field */
#endif
    void *ob_type;
} PyObject;

/* Free-threaded PyObject layout (CPython --disable-gil) */
/* PyMutex: per-object lock, uint8_t with two-bit state (private) */
typedef struct { uint8_t _bits; } PyMutex;

typedef struct _c2py_object_ft {
    uintptr_t ob_tid;
    uint16_t ob_flags;
    PyMutex ob_mutex;
    uint8_t ob_gc_bits;
    uint32_t ob_ref_local;
    Py_ssize_t ob_ref_shared;
    void *ob_type;
} PyObject_FT;

/* Shorthand for embedding PyObject at the head of a struct (GIL layout) */
#define PyObject_HEAD \
    Py_ssize_t ob_refcnt; \
    void *ob_type;

typedef void *(*PyCFunction)(PyObject*, PyObject*);

/* Py_buffer: stable since Python 2.6 (PEP 3118).
 * NOTE: includes smalltable[2] field present in CPython 2.7-3.11.
 * In CPython 3.12+ this was removed (PEP 697 stable ABI);
 * we use C2PY.pybuffer_size (set at init) for correct sizeof.
 */
typedef struct {
    void *buf;
    PyObject *obj;
    Py_ssize_t len;
    Py_ssize_t itemsize;
    int readonly;
    int ndim;
    char *format;
    Py_ssize_t *shape;
    Py_ssize_t *strides;
    Py_ssize_t *suboffsets;
    Py_ssize_t smalltable[2];  /* present on CPython 2.7-3.11 */
    void *internal;
} Py_buffer;

/* PyMethodDef: stable layout across all Python versions */
typedef struct {
    const char *ml_name;
    PyCFunction ml_meth;
    int ml_flags;
    const char *ml_doc;
} PyMethodDef;

/* PyModuleDef_Base: standard GIL-enabled layout (Python 3.0+) */
typedef struct PyModuleDef_Base {
    PyObject ob_base;
    PyObject *(*m_init)(void);
    Py_ssize_t m_index;
    PyObject *m_copy;
} PyModuleDef_Base;

/* PyModuleDef_Base for free-threaded builds (PyObject is 32 bytes) */
typedef struct PyModuleDef_Base_FT {
    PyObject_FT ob_base;
    PyObject *(*m_init)(void);
    Py_ssize_t m_index;
    PyObject *m_copy;
} PyModuleDef_Base_FT;

/* PyModuleDef for Python 3.x standard GIL layout */
typedef struct PyModuleDef {
    PyModuleDef_Base m_base;
    const char *m_name;
    const char *m_doc;
    Py_ssize_t m_size;
    PyMethodDef *m_methods;
    void *m_slots;
    void *m_traverse;
    void *m_clear;
    void *m_free;
} PyModuleDef;

/* PyModuleDef for free-threaded builds (PyModuleDef_Base is 56 bytes) */
typedef struct PyModuleDef_FT {
    PyModuleDef_Base_FT m_base;
    const char *m_name;
    const char *m_doc;
    Py_ssize_t m_size;
    PyMethodDef *m_methods;
    void *m_slots;
    void *m_traverse;
    void *m_clear;
    void *m_free;
} PyModuleDef_FT;

/* PyModuleDef_Slot: free-threading/extension slots (PEP 384, PEP 489) */
typedef struct {
    int slot;
    void *value;
} PyModuleDef_Slot;

/* Py_MOD_GIL_NOT_USED: signal that this module supports free-threading.
 * Value is (void*)1 (stable across CPython 3.13+).
 * c2py23 uses PyUnstable_Module_SetGIL() (resolved at runtime) to
 * declare free-threading support on the module object, rather than
 * the older PyModuleDef_Slot / Py_mod_gil slot-number mechanism
 * (whose slot number changed from 4 in 3.13-3.14 to 87 in 3.15+). */
#define Py_MOD_GIL_NOT_USED  ((void*)1)

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

/* Py_buffer flags */
#define PyBUF_SIMPLE   0
#define PyBUF_WRITABLE 0x0001
#define PyBUF_FORMAT   0x0004
#define PyBUF_ND       0x0008
#define PyBUF_STRIDES  (0x0010 | PyBUF_ND)
#define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

/* PyMethodDef flags */
#define METH_VARARGS   0x0001
#define METH_KEYWORDS  0x0002
#define METH_NOARGS    0x0004
#define METH_O         0x0008
#define METH_CLASS     0x0010
#define METH_STATIC    0x0020
#define METH_FASTCALL  0x0080

/* Module init macro - initializes the PyModuleDef_Base embedded in PyModuleDef. */
#ifdef C2PY_TARGET_PYPY
#define PyModuleDef_HEAD_INIT { {1, NULL, NULL}, NULL, 0, NULL }
#else
#define PyModuleDef_HEAD_INIT { {1, NULL}, NULL, 0, NULL }
#endif

/* Module init macro for free-threaded builds (PyObject is 32 bytes).
 * ob_type = NULL, m_init = NULL, m_index = 0, m_copy = NULL.
 * ob_mutex is zeroed via {0} (PyMutex is struct { uint8_t _bits; }).
 * ob_flags = _Py_STATICALLY_ALLOCATED_FLAG (4), ob_ref_local =
 * _Py_IMMORTAL_REFCNT_LOCAL (UINT32_MAX), ob_ref_shared = 0.
 * These match the actual CPython 3.14t/3.15t PyModuleDef_HEAD_INIT(NULL)
 * expansion.  Earlier versions used 0/0/1 which 3.14t tolerated but 3.15t
 * hard-rejects. */
#define _Py_STATICALLY_ALLOCATED_FLAG (1 << 2)
#define _Py_IMMORTAL_REFCNT_LOCAL     0xFFFFFFFFU
#define PyModuleDef_HEAD_INIT_FT \
    { {0, _Py_STATICALLY_ALLOCATED_FLAG, {0}, 0, \
       _Py_IMMORTAL_REFCNT_LOCAL, 0, NULL}, NULL, 0, NULL}
