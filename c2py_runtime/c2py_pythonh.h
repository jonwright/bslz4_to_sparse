/* c2py_pythonh.h - pythonh backend (#include <Python.h> directly)
 *
 * Standard CPython extension.  No dlsym trick, no cross-version
 * portability.  The .so/.pyd is tied to one Python version.
 * Used for GraalPy, debugging, static builds.
 *
 * This header is NOT included directly.  Include c2py_runtime.h instead.
 */

#undef _GNU_SOURCE   /* runtime.c defines it; PyPy/GraalPy Python.h may redefine */
#undef _POSIX_C_SOURCE  /* Python 2.7 pyconfig.h may redefine */
#undef _XOPEN_SOURCE
#include <Python.h>
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* ---- Python 2.7 stubs (not available in Python.h) ---- */

#if PY_MAJOR_VERSION == 2
typedef struct {
    PyObject_HEAD
    void *m_init;
    Py_ssize_t m_index;
    void *m_copy;
    const char *m_name;
    const char *m_doc;
    Py_ssize_t m_size;
    PyMethodDef *m_methods;
    void *m_slots;
    void *m_traverse;
    void *m_clear;
    void *m_free;
} PyModuleDef;
#define PyModuleDef_HEAD_INIT 1, NULL, NULL, 0, NULL
#endif

#ifndef METH_FASTCALL
#define METH_FASTCALL 0x0080
#endif

/* ---- Common definitions ---- */

#ifdef _WIN32
#define C2PY_EXPORT __declspec(dllexport)
#else
#define C2PY_EXPORT
#endif

/* Buffer flags -- pythonh aliases for CPython constants */
#define C2PY_BUF_READ   PyBUF_READ
#define C2PY_BUF_WRITE  PyBUF_WRITE

/* Thin wrappers that call the real CPython API directly.
 * The same names are defined as inline functions in the dlsym backend;
 * in pythonh mode, these macros take precedence. */
#define c2py_acquire_buffer(o,b,w) \
    PyObject_GetBuffer((PyObject*)(o),(b), \
        ((w) ? (PyBUF_FORMAT|PyBUF_C_CONTIGUOUS|PyBUF_STRIDES|PyBUF_ND|PyBUF_WRITABLE) \
              : (PyBUF_FORMAT|PyBUF_C_CONTIGUOUS|PyBUF_STRIDES|PyBUF_ND)))
#define c2py_release_buffer(b)  PyBuffer_Release(b)

/* FT module defs are just standard ones in pythonh mode */
#define PyModuleDef_FT            PyModuleDef
#define PyModuleDef_HEAD_INIT_FT  PyModuleDef_HEAD_INIT

/* c2py_set_module_attr -- simple wrapper for pythonh mode */
#define c2py_set_module_attr(mod, name, val) \
    PyObject_SetAttrString((PyObject*)(mod), (name), (PyObject*)(val))

int c2py_runtime_init(void);
