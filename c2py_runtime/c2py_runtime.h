/* c2py_runtime.h - CPython C-extension wrapper runtime
 *
 * Two backends:
 *   Default (nimpy):  dlopen(NULL) + dlsym(), one .so per Python 2.7-3.15
 *   --pythonh mode:   #include <Python.h> direct, one .so per version
 *
 * Include this header in generated wrapper .c files.  Never include
 * the backend headers (c2py_dlsym.h / c2py_pythonh.h) directly.
 */

#ifndef C2PY_RUNTIME_H
#define C2PY_RUNTIME_H

/* ---- Pick one backend ---- */

#ifdef C2PY_USE_PYTHON_H
#include "c2py_pythonh.h"
#else
#include "c2py_dlsym.h"
#endif

/* ------------------------------------------------------------------ */
/* Function pointer table - populated by c2py_runtime_init()          */
/* Identical for both backends.                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    void *dl_handle;
    int version_major;
    int version_minor;

    int use_fastcall;               /* 1 = use METH_FASTCALL wrappers (Python >= 3.12) */
    int is_free_threaded;           /* 1 = Python built with --disable-gil */
    int is_pypy;                    /* 1 = running on PyPy (cpyext, PyPy_* symbols) */
    Py_ssize_t pybuffer_size;      /* actual sizeof(Py_buffer) for this Python version */
    Py_ssize_t pyobject_size;      /* actual sizeof(PyObject) for this Python version */
    Py_ssize_t pyobject_size_ft;   /* sizeof(PyObject) for free-threaded builds (32 LP64) */
    Py_ssize_t pymoduledef_max_size; /* max(sizeof(PyModuleDef), sizeof(PyModuleDef_FT)) */
    ptrdiff_t ob_refcnt_offset;    /* offset of ob_refcnt (or ob_ref_shared on FT) in PyObject */
    ptrdiff_t ob_type_offset;      /* offset of ob_type field in PyObject */

    /* Buffer protocol */
    int (*GetBuffer)(PyObject*, Py_buffer*, int);
    void (*ReleaseBuffer)(Py_buffer*);

    /* Old buffer protocol (Python 2.x only, NULL on Python 3) */
    int (*AsReadBuffer)(PyObject*, const void**, Py_ssize_t*);
    int (*AsWriteBuffer)(PyObject*, void**, Py_ssize_t*);
    void (*Err_Clear)(void);
    int buffer_api_is_pep3118;  /* 0 = old API only, 1 = PEP 3118 available */

    /* Argument parsing (positional-only, no keyword support) */
    int (*ParseTuple)(PyObject*, const char*, ...);

    /* Value construction */
    PyObject* (*Long_FromLong)(long);
    PyObject* (*Long_FromLongLong)(long long);
    PyObject* (*Long_FromUnsignedLongLong)(unsigned long long);
    PyObject* (*Float_FromDouble)(double);

    /* Tuple construction */
    PyObject* (*Tuple_New)(Py_ssize_t);
    int (*Tuple_SetItem)(PyObject*, Py_ssize_t, PyObject*);

    /* String construction (ASCII bytes only, no unicode/encodings) */
    PyObject* (*Bytes_FromStringAndSize)(const char*, Py_ssize_t);

    /* Scalar conversion from objects */
    long (*Long_AsLong)(PyObject*);
    long long (*Long_AsLongLong)(PyObject*);
    double (*Float_AsDouble)(PyObject*);

    /* Exception objects (pointers to the actual exception types) */
    void *exc_TypeError;
    void *exc_ValueError;
    void *exc_RuntimeError;
    void *exc_MemoryError;
    void *exc_BufferError;    /* PyExc_BufferError (optional, may be NULL) */
    void (*Err_SetString)(PyObject*, const char*);
    PyObject* (*Err_Occurred)(void);
    PyObject* (*Err_Format)(PyObject*, const char*, ...);

    /* None singleton (immortal on CPython 3.12+, INCREF/DECREF unnecessary) */
    PyObject *none_obj;
    int none_immortal;  /* 1 if IncRef on None is a no-op (3.12+ immortal None) */

    /* Module creation */
    PyObject* (*Module_Create2)(PyModuleDef*, int);
    PyObject* (*InitModule_2_7)(const char*, PyMethodDef*);

    /* Reference counting */
    void (*IncRef)(PyObject*);
    void (*DecRef)(PyObject*);

    /* Object attribute access */
    int (*SetAttrString)(PyObject*, const char*, PyObject*);
    PyObject* (*GetAttrString)(PyObject*, const char*);
    PyObject* (*Module_GetDict)(PyObject*);    /* for perf attr registration */
    void *_ds_set_item_string;  /* PyDict_SetItemString, resolved at init */
    int _ds_pypy_workaround;   /* 1 if PyPy 2.7-style dict-based attr set */

    /* General call support (optional, used by DLPack) */
    PyObject* (*CallObject)(PyObject*, PyObject*);

    /* Capsule API (optional, used by DLPack) */
    void* (*Capsule_GetPointer)(PyObject*, const char*);

    /* Pointer-to-int conversion (for exposing perf struct addresses) */
    PyObject* (*Long_FromVoidPtr)(void*);

    /* GIL management */
    void* (*SaveThread)(void);
    void (*RestoreThread)(void*);
    /* Free-threading: set Py_MOD_GIL_NOT_USED on module (optional, may be NULL) */
    void (*Unstable_Module_SetGIL)(PyObject*, void*);

} c2py_api_t;

/* The global API table */
extern c2py_api_t C2PY;

/* ------------------------------------------------------------------ */
/* Nimpy-only convenience macros  (Python.h provides these in pythonh mode) */
/* ------------------------------------------------------------------ */

#ifndef C2PY_USE_PYTHON_H

#define PyObject_GetBuffer(o, b, f)    C2PY.GetBuffer((PyObject*)(o), (b), (f))
#define PyBuffer_Release(b)            C2PY.ReleaseBuffer(b)

/* MSVC traditional preprocessor auto-removes the comma before an empty
 * __VA_ARGS__; GCC/Clang require the ## token-paste to do so.
 * MSVC 2022+ (conformant preprocessor) behaves like GCC and also needs ##. */
#ifdef _MSC_VER
#define PyArg_ParseTuple(a, f, ...)    C2PY.ParseTuple((PyObject*)(a), (f), ##__VA_ARGS__)
#define PyErr_Format(e, f, ...)        C2PY.Err_Format((PyObject*)(e), (f), ##__VA_ARGS__)
#else
#define PyArg_ParseTuple(a, f, ...)    C2PY.ParseTuple((PyObject*)(a), (f), ##__VA_ARGS__)
#define PyErr_Format(e, f, ...)        C2PY.Err_Format((PyObject*)(e), (f), ##__VA_ARGS__)
#endif

#define PyLong_FromLong(v)             C2PY.Long_FromLong(v)
#define PyLong_FromLongLong(v)         C2PY.Long_FromLongLong(v)
#define PyLong_FromUnsignedLongLong(v) C2PY.Long_FromUnsignedLongLong(v)
#define PyFloat_FromDouble(v)          C2PY.Float_FromDouble(v)
#define PyBytes_FromStringAndSize(s, n) C2PY.Bytes_FromStringAndSize((s), (n))
#define PyLong_AsLong(o)               C2PY.Long_AsLong((PyObject*)(o))
#define PyLong_AsLongLong(o)           C2PY.Long_AsLongLong((PyObject*)(o))
#define PyFloat_AsDouble(o)            C2PY.Float_AsDouble((PyObject*)(o))
#define PyErr_SetString(e, m)          C2PY.Err_SetString((PyObject*)(e), (m))
#define PyErr_Clear()                  C2PY.Err_Clear()
#define PyErr_Occurred()               C2PY.Err_Occurred()
#define Py_RETURN_NONE                 do { \
    if (C2PY.none_immortal) { return C2PY.none_obj; } \
    C2PY.IncRef(C2PY.none_obj); return C2PY.none_obj; \
} while(0)
#define Py_INCREF(o)                   C2PY.IncRef((PyObject*)(o))
#define Py_DECREF(o)                   C2PY.DecRef((PyObject*)(o))
#define PyObject_SetAttrString(o, n, v) C2PY.SetAttrString((PyObject*)(o), (n), (PyObject*)(v))
#define PyObject_GetAttrString(o, n)   C2PY.GetAttrString((PyObject*)(o), (n))
#define PyModule_GetDict(m)            C2PY.Module_GetDict((PyObject*)(m))

/* Set a module attribute.  PyObject_SetAttrString silently fails
 * on PyPy 2.7 cpyext modules; C2PY._ds_pypy_workaround gates the
 * dict-based fallback (resolved once at init, zero runtime dlsym). */
#define c2py_set_module_attr(m, name, val) do { \
    if (C2PY._ds_pypy_workaround) { \
        PyObject *_d = C2PY.Module_GetDict((PyObject*)(m)); \
        if (_d && C2PY._ds_set_item_string) { \
            typedef int (*_ds_fn)(PyObject*, const char*, PyObject*); \
            ((_ds_fn)C2PY._ds_set_item_string)(_d, (name), (PyObject*)(val)); \
        } \
    } else { \
        C2PY.SetAttrString((PyObject*)(m), (name), (PyObject*)(val)); \
    } \
} while(0)
#define PyLong_FromVoidPtr(p)          C2PY.Long_FromVoidPtr((void*)(p))
#define PyTuple_New(s)                 C2PY.Tuple_New(s)
#define PyTuple_SetItem(t, i, o)       C2PY.Tuple_SetItem((PyObject*)(t), (i), (PyObject*)(o))
#define PyEval_SaveThread()            C2PY.SaveThread()
#define PyEval_RestoreThread(s)        C2PY.RestoreThread((void*)(s))

#define PyExc_TypeError                ((PyObject*)C2PY.exc_TypeError)
#define PyExc_ValueError               ((PyObject*)C2PY.exc_ValueError)
#define PyExc_RuntimeError             ((PyObject*)C2PY.exc_RuntimeError)
#define PyExc_MemoryError              ((PyObject*)C2PY.exc_MemoryError)
#define PyExc_BufferError              ((PyObject*)C2PY.exc_BufferError)
#define PyObject_CallObject(c, a)      C2PY.CallObject((PyObject*)(c), (PyObject*)(a))
#define PyCapsule_GetPointer(c, n)     C2PY.Capsule_GetPointer((PyObject*)(c), (n))
#define Py_XDECREF(o)                  do { if (o) C2PY.DecRef((PyObject*)(o)); } while(0)

/* ------------------------------------------------------------------ */
/* Reference counting fallbacks (for CPython < 3.12 where Py_IncRef   */
/* is not an exported symbol)                                         */
/* ------------------------------------------------------------------ */

/* Manual refcount increment - accesses ob_refcnt via the correct offset
 * for the Python build (ob_refcnt on GIL, ob_ref_shared on free-threaded).
 * Safe on GIL-enabled builds where refcount is a simple Py_ssize_t field.
 * On free-threaded builds this fallback is UNSAFE (ob_ref_shared requires
 * atomic operations); always prefer Py_IncRef / Py_DecRef on 3.12+. */
static inline void _c2py_inc_ref_manual(PyObject *op)
{
    Py_ssize_t *refcnt = (Py_ssize_t*)((char*)op + C2PY.ob_refcnt_offset);
    ++(*refcnt);
}

/*
 * _c2py_dec_ref_manual: Fallback dec-ref for pre-3.12 Python where
 * Py_DecRef is not resolvable via dlsym. On those versions CPython uses
 * shared refcounts where zero-is-special does not apply -- refcounts
 * never reach zero through normal INCREF/DECREF alone.
 *
 * WARNING: This is a diagnostic-only fallback. When refcount hits zero
 * it prints a warning but does NOT call tp_dealloc destructor.
 * This is acceptable because the fallback is only active on Python < 3.12
 * where the zero-refcount invariant holds. If porting to a platform where
 * this invariant does not hold, implement proper deallocation.
 */
static inline void _c2py_dec_ref_manual(PyObject *op)
{
    Py_ssize_t *refcnt = (Py_ssize_t*)((char*)op + C2PY.ob_refcnt_offset);
    --(*refcnt);
    if (*refcnt == 0) {
        fprintf(stderr, "c2py_runtime: _c2py_dec_ref_manual reached zero "
                "refcount for %p -- possible leak\n", (void*)op);
    }
}

/* ------------------------------------------------------------------ */
/* Buffer acquisition helper with old-API fallback for Python 2.7     */
/* ------------------------------------------------------------------ */

/* Flags for c2py_acquire_buffer */
#define C2PY_BUF_READ   0
#define C2PY_BUF_WRITE  1

/* Returns 0 on success, -1 on failure (with Python exception set) */
static inline int
c2py_acquire_buffer(PyObject *obj, Py_buffer *buf, int want_writable)
{
    int flags = PyBUF_STRIDES | PyBUF_FORMAT;
    if (want_writable) flags |= PyBUF_WRITABLE;

    memset(buf, 0, C2PY.pybuffer_size);

    if (C2PY.buffer_api_is_pep3118) {
        return PyObject_GetBuffer(obj, buf, flags);
    }

    /* Python 2.7: try PEP 3118 first, fall back to old API */
    if (PyObject_GetBuffer(obj, buf, flags) == 0)
        return 0;

    PyErr_Clear();

    if (want_writable) {
        if (C2PY.AsWriteBuffer &&
            C2PY.AsWriteBuffer(obj, (void**)&buf->buf, &buf->len) == 0) {
            buf->readonly = 0;
        } else {
            return -1;
        }
    } else {
        if (C2PY.AsReadBuffer &&
            C2PY.AsReadBuffer(obj, (const void**)&buf->buf, &buf->len) == 0) {
            buf->readonly = 1;
        } else {
            return -1;
        }
    }

    buf->ndim = 1;
    buf->itemsize = 1;
    buf->format = NULL;
    buf->shape = NULL;
    buf->strides = NULL;
    return 0;
}

/* Release a buffer acquired by c2py_acquire_buffer */
static inline void
c2py_release_buffer(Py_buffer *buf)
{
    if (buf->obj != NULL) {
        PyBuffer_Release(buf);
    }
     /* Old buffer API (PyObject_AsRead/WriteBuffer) needs no release */
}
#endif /* !C2PY_USE_PYTHON_H */

/* ------------------------------------------------------------------ */
/* Unified buffer info struct -- the common interface for all         */
/* acquisition backends (PEP 3118, ndarray struct cast, DLPack).     */
/* Generated wrappers operate on c2py_ptr_info; the underlying       */
/* acquisition mechanism is hidden behind pin/unpin.                  */
/* ------------------------------------------------------------------ */

/* Maximum supported dimensions for DLPack stride computation.
 * Matches numpy's practical limit; arrays with >16D are vanishingly rare. */
#define C2PY_MAX_NDIM 32

/* c2py_ptr_info mirrors the leading fields of Py_buffer exactly so that
 * c2py_pin_buffer can memcpy the acquired Py_buffer straight into it.
 * Field order MUST match PEP 3118 Py_buffer layout (offset, type):
 *   0: buf     (void*)       -> ptr
 *   8: obj     (PyObject*)   -> _pin_buf_obj (unused by info, reserved)
 *  16: len     (Py_ssize_t)
 *  24: itemsize (Py_ssize_t)
 *  32: readonly (int)        -> _ro (unused)
 *  36: ndim    (int)
 *  40: format  (char*)
 *  48: shape   (Py_ssize_t*)
 *  56: strides (Py_ssize_t*)
 * Add reserved padding if the struct grows beyond what we memcpy. */
typedef struct {
    void *ptr;                /* data pointer (matches Py_buffer.buf) */
    PyObject *_pin_buf_obj;   /* reserved: maps to Py_buffer.obj, never accessed */
    Py_ssize_t len;           /* total length in bytes */
    Py_ssize_t itemsize;      /* size of one element */
    int _ro;                  /* reserved: maps to Py_buffer.readonly */
    int ndim;                 /* number of dimensions */
    char *format;             /* PEP 3118 format string (may be NULL) */
    Py_ssize_t *shape;        /* per-dimension sizes (may be NULL for 1D) */
    Py_ssize_t *strides;      /* per-dimension strides (may be NULL) */
} c2py_ptr_info;

/* Backend tags for c2py_buf_pin.kind -- tells c2py_unpin_buffer
 * which release path to use.  Zero-init = C2PY_PIN_NONE = no-op. */
#define C2PY_PIN_NONE     0
#define C2PY_PIN_PEP3118  1   /* pin->buf is valid; call PyBuffer_Release */
#define C2PY_PIN_NDARRAY  2   /* struct-cast; pin->buf.obj is the INCREF'd ndarray */
#define C2PY_PIN_DLPACK   3   /* DLPack; pin->ctx is the DLManagedTensor* */

typedef struct {
    Py_buffer buf;          /* opaque Py_buffer, valid when kind == PEP3118 */
    char _buf_pad[1200];    /* extra space: PyPy's Py_buffer is up to 1112 bytes
                               (PyBUF_MAX_NDIM=64 with inline arrays), but we
                               compiled with CPython's sizeof(~80) */
    int kind;               /* C2PY_PIN_* tag */
    void *ctx;              /* back-end context (DLManagedTensor* for DLPack) */
    char format_buf[8];     /* stack-local format string storage (non-PEP3118) */
    Py_ssize_t shape_buf[C2PY_MAX_NDIM];   /* DLPack: int64 shape copied to Py_ssize_t */
    Py_ssize_t stride_buf[C2PY_MAX_NDIM]; /* stride buffer */
} c2py_buf_pin;

/* ------------------------------------------------------------------ */
/* NumPy ndarray struct-cast fast path (lazy-probed, no import)       */
/* ------------------------------------------------------------------ */

/* Minimal PyTypeObject overlay to read tp_name without importing numpy.
 * Layout (GIL-only LP64): ob_refcnt(8) + ob_type(8) + ob_size(8) +
 * tp_name(8).  Free-threaded builds skip the fast path.
 * On PyPy, ob_pypy_link sits between ob_refcnt and ob_type. */
typedef struct {
    Py_ssize_t ob_refcnt;
#ifdef C2PY_TARGET_PYPY
    void     *ob_pypy_link;
#endif
    void     *ob_type;
    Py_ssize_t ob_size;
    const char *tp_name;
} c2py_type_min_t;

/* Runtime-discovered ndarray layout cache.
 * data_off is the offset of the data pointer from the PyObject base.
 * The remaining fields live at fixed relative offsets from data_off
 * (stable since numpy 1.0 through 2.x):
 *   data_off + 8:  nd          (int)
 *   data_off + 16: dimensions  (npy_intp*)
 *   data_off + 24: strides     (npy_intp*)
 *   data_off + 40: descr       (PyArray_Descr*)
 *   data_off + 48: flags       (int)      */
typedef struct {
    void *ndarray_type;     /* cached ob_type of numpy.ndarray, NULL if unknown */
    int data_off;           /* offset of data ptr from PyObject base */
    int probed;             /* 1 = layout known, 0 = probing deferred */
} c2py_ndarray_layout_t;

extern c2py_ndarray_layout_t C2PY_NDARRAY;

/* NPY_ARRAY_WRITEABLE = 0x0400 (stable since numpy 1.0) */
#define C2PY_NPY_WRITEABLE  0x0400

/* Map a PEP 3118 type character to itemsize.
 * Excludes 'l'/'L' (platform-sized -- callers use sizeof(long) at
 * the expression level).  Used by both ndarray and DLPack backends. */
static inline int
c2py_format_itemsize(char type_char)
{
    switch (type_char) {
    case 'b': case 'B': case '?': return 1;
    case 'h': case 'H':          return 2;
    case 'i': case 'I':          return 4;
    case 'l': case 'L':          return (int)sizeof(long);
    case 'q': case 'Q':          return 8;
    case 'f':                    return 4;
    case 'd':                    return 8;
    case 'g':                    return (int)sizeof(long double);
    default:                     return 1;
    }
}

/* ------------------------------------------------------------------ */
/* DLPack struct definitions (hard-coded, no external headers)        */
/* ------------------------------------------------------------------ */

#define C2PY_DLCPU  1
#define C2PY_DLCUDA 2

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} c2py_dl_dtype_t;

typedef struct {
    int32_t device_type;
    int32_t device_id;
} c2py_dl_device_t;

typedef struct {
    void *data;
    c2py_dl_device_t device;
    int32_t ndim;
    c2py_dl_dtype_t dtype;
    int64_t *shape;
    int64_t *strides;
    int64_t byte_offset;
} c2py_dl_tensor_t;

typedef struct c2py_dl_managed_tensor {
    c2py_dl_tensor_t dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct c2py_dl_managed_tensor *);
} c2py_dl_managed_tensor;

/* DLPack type codes */
#define C2PY_DL_INT    0
#define C2PY_DL_UINT   1
#define C2PY_DL_FLOAT  2
#define C2PY_DL_BFLOAT 4
#define C2PY_DL_COMPLEX 5
#define C2PY_DL_BOOL   6

/* Map a DLPack dtype to PEP 3118 format character.
 * Returns 0 for unsupported types. */
static inline char
c2py_dl_format_char(c2py_dl_dtype_t *dt)
{
    int bits = dt->bits;
    if (dt->lanes != 1) return 0;
    switch (dt->code) {
    case C2PY_DL_INT:
        if (bits == 8)  return 'b';
        if (bits == 16) return 'h';
        if (bits == 32) return 'i';
        if (bits == 64) return 'q';
        return 0;
    case C2PY_DL_UINT:
        if (bits == 8)  return 'B';
        if (bits == 16) return 'H';
        if (bits == 32) return 'I';
        if (bits == 64) return 'Q';
        return 0;
    case C2PY_DL_FLOAT:
        if (bits == 16) return 0; /* no PEP 3118 half-float char */
        if (bits == 32) return 'f';
        if (bits == 64) return 'd';
        return 0;
    case C2PY_DL_BOOL:
        if (bits == 8) return '?';
        return 0;
    default: return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Acquisition functions                                              */
/* ------------------------------------------------------------------ */

/* Acquire via the standard PEP 3118 path. */
static inline int
c2py_pin_buffer(PyObject *obj, c2py_buf_pin *pin, c2py_ptr_info *info,
                int want_writable)
{
    if (c2py_acquire_buffer(obj, &pin->buf, want_writable) == -1)
        return -1;

    memcpy(info, &pin->buf, sizeof(c2py_ptr_info));

    pin->kind = C2PY_PIN_PEP3118;
    return 0;
}

/* Field offsets within PyArrayObject, relative to the data pointer.
 * Empirically stable since numpy 1.0 through 2.x, but the offsets
 * differ between LP64 and ILP32: numpy's int nd field (plus padding)
 * and the following pointers pack differently, and PyArray_Descr.type
 * sits at a different byte offset.  Verified by tests/check_numpy_abi.c
 * on LP64 (REL_OFF +8/+16/+24/+40/+48, type char at 25) and by the
 * ILP32 probe in CI (REL_OFF +4/+8/+12/+20/+24, type char at 13).
 * ABI3T (limited-API) numpy builds do not affect this: Py_TARGET_ABI3T
 * only hides the struct from compile-time consumers, so the runtime
 * allocation and layout (and therefore these offsets) are unchanged. */
#if defined(__LP64__) || defined(_WIN64)
#define C2PY_NDARR_OFF_ND        8
#define C2PY_NDARR_OFF_DIMS      16
#define C2PY_NDARR_OFF_STRIDES   24
#define C2PY_NDARR_OFF_DESCR     40
#define C2PY_NDARR_OFF_FLAGS     48
#define C2PY_DESCR_TYPE_CHAR_OFF 25
#else
#define C2PY_NDARR_OFF_ND        4
#define C2PY_NDARR_OFF_DIMS      8
#define C2PY_NDARR_OFF_STRIDES   12
#define C2PY_NDARR_OFF_DESCR     20
#define C2PY_NDARR_OFF_FLAGS     24
#define C2PY_DESCR_TYPE_CHAR_OFF 13
#endif

/* Acquire via numpy ndarray struct-cast (no PyObject_GetBuffer).
 * Returns 0 on success, -1 to signal "try next backend" or real failure.
 * On first encounter of a numpy.ndarray, probes the data-pointer
 * offset by acquiring a buffer then scanning object memory.  All
 * subsequent calls use a ~1 ns type-pointer comparison. */
static inline int
c2py_pin_ndarray(PyObject *obj, c2py_buf_pin *pin, c2py_ptr_info *info,
                 int want_writable)
{
    c2py_ndarray_layout_t *L = &C2PY_NDARRAY;
    void *tp;
    char *base;
    void *dptr, *descr;
    int nd, flags;
    Py_ssize_t nelem, i;
    char type_char;

    /* Free-threaded Python and PyPy: type-object layout differs from
     * standard GIL CPython.  PyPy cpyext wraps ndarray objects in
     * proxies -- the data pointer is not at a fixed offset from id().
     * Skip the fast path and fall through to buffer protocol. */
    if (C2PY.is_free_threaded || C2PY.is_pypy)
        return -1;

    tp = *(void**)((char*)obj + C2PY.ob_type_offset);

    if (L->ndarray_type && tp == L->ndarray_type) {
        goto fill;
    }

    if (!L->probed && tp) {
        const char *name = ((c2py_type_min_t*)tp)->tp_name;
        if (name && strcmp(name, "numpy.ndarray") == 0) {
            /* First encounter: validate via buffer protocol, then
             * scan object memory to locate the data pointer at runtime.
             * Note: PyObject_GetBuffer COPIES shape/strides to temporary
             * arrays for numpy, so we cannot verify against buf.shape. */
            if (c2py_acquire_buffer(obj, &pin->buf, want_writable) != 0)
                return -1;

            base = (char*)obj;
            {
                int off;
                dptr = pin->buf.buf;
                for (off = (int)C2PY.pyobject_size;
                     off < (int)C2PY.pyobject_size + 80;
                     off += (int)sizeof(void*)) {
                    if (*(void**)(base + off) == dptr) {
                        L->data_off = off;
                        break;
                    }
                }
            }

            /* Verify the layout against known numpy struct:
             * data_off+OFF_ND      -> nd   (int, 0 <= nd <= 32)
             * data_off+OFF_DIMS    -> shape ptr (reasonable pointer)
             * data_off+OFF_DESCR   -> descr (non-NULL pointer) */
            nd    = *(int*)(base + L->data_off + C2PY_NDARR_OFF_ND);
            descr = *(void**)(base + L->data_off + C2PY_NDARR_OFF_DESCR);
            if (nd < 0 || nd > C2PY_MAX_NDIM || descr == NULL)
                goto probe_fail;
            /* Sanity: shape pointer should be non-NULL if nd>0 */
            if (nd > 0 && *(void**)(base + L->data_off + C2PY_NDARR_OFF_DIMS) == NULL)
                goto probe_fail;

            L->ndarray_type = tp;
            L->probed = 1;


            /* First call: info already filled via c2py_acquire_buffer.
             * Return via pep3118 path so unpin releases the buffer. */
            memcpy(info, &pin->buf, sizeof(c2py_ptr_info));
            pin->kind = C2PY_PIN_PEP3118;
            return 0;

        probe_fail:
            c2py_release_buffer(&pin->buf);
            return -1;
        }
    }

    return -1;

fill:
    base = (char*)obj;

    dptr  = *(void**)(base + L->data_off);
    nd    = *(int*)(base + L->data_off + C2PY_NDARR_OFF_ND);
    flags = *(int*)(base + L->data_off + C2PY_NDARR_OFF_FLAGS);

    if (want_writable && !(flags & C2PY_NPY_WRITEABLE)) {
        PyErr_SetString(PyExc_TypeError,
                        "numpy array is not writeable");
        return -1;
    }

    descr = *(void**)(base + L->data_off + C2PY_NDARR_OFF_DESCR);
    if (descr) {
        /* type char at a fixed byte offset within PyArray_Descr
         * (stable since numpy 1.0 through 2.x on GIL builds) */
        type_char = ((char*)descr)[C2PY_DESCR_TYPE_CHAR_OFF];
        pin->format_buf[0] = type_char;
        pin->format_buf[1] = '\0';
        info->format = pin->format_buf;
        info->itemsize = c2py_format_itemsize(type_char);
    } else {
        info->format = NULL;
        info->itemsize = 1;
    }

    info->ptr     = dptr;
    info->ndim    = nd;
    info->shape   = *(Py_ssize_t**)(base + L->data_off + C2PY_NDARR_OFF_DIMS);
    info->strides = *(Py_ssize_t**)(base + L->data_off + C2PY_NDARR_OFF_STRIDES);

    nelem = 1;
    if (info->shape) {
        for (i = 0; i < nd && info->shape[i] >= 0; i++)
            nelem *= info->shape[i];
    }
    info->len = nelem * (Py_ssize_t)(info->itemsize > 0 ? info->itemsize : 1);

    /* Hold a reference on the ndarray: we skipped PyObject_GetBuffer
     * which normally INCREFs the exporter.  Store it in buf.obj so
     * unpin can DECREF it. */
    C2PY.IncRef(obj);
    pin->buf.obj = obj;
    pin->kind = C2PY_PIN_NDARRAY;
    return 0;
}

/* Acquire via DLPack capsule extraction.
 * Calls obj.__dlpack__() to get the capsule, then reads the DLTensor
 * struct fields directly.  CPU-only; GPU tensors are rejected.
 * All DLPack API symbols (CallObject, Capsule_GetPointer)
 * are optional: if any is NULL, this function returns -1 immediately,
 * falling through to the next backend.
 * Returns 0 on success, -1 to fall through to next backend. */
static inline int
c2py_pin_dlpack(PyObject *obj, c2py_buf_pin *pin, c2py_ptr_info *info,
                int want_writable)
{
    PyObject *dl_method = NULL;
    PyObject *dl_args = NULL;
    PyObject *capsule = NULL;
    c2py_dl_managed_tensor *managed;
    c2py_dl_tensor_t *t;
    char format_char;
    Py_ssize_t nelem;
    int i;

    (void)want_writable;

    /* DLPack symbols are optional; if not resolved, skip this backend. */
    if (!C2PY.CallObject || !C2PY.Capsule_GetPointer)
        return -1;

    dl_method = PyObject_GetAttrString(obj, "__dlpack__");
    if (!dl_method) { PyErr_Clear(); return -1; }

    dl_args = PyTuple_New(0);
    if (!dl_args) goto fail;

    capsule = PyObject_CallObject(dl_method, dl_args);
    if (!capsule) { PyErr_Clear(); goto fail; }

    managed = (c2py_dl_managed_tensor*)PyCapsule_GetPointer(capsule, "dltensor");
    if (!managed) {
        PyErr_Clear();
        managed = (c2py_dl_managed_tensor*)PyCapsule_GetPointer(capsule, "dltensor_versioned");
    }
    if (!managed) { PyErr_Clear(); goto fail; }

    t = &managed->dl_tensor;

    if (t->device.device_type != C2PY_DLCPU) {
        PyErr_SetString(PyExc_TypeError,
            "DLPack: unsupported device (expected CPU)");
        goto fail;
    }

    format_char = c2py_dl_format_char(&t->dtype);
    if (!format_char) {
        PyErr_SetString(PyExc_TypeError,
            "DLPack: unsupported dtype");
        goto fail;
    }

    pin->format_buf[0] = format_char;
    pin->format_buf[1] = '\0';
    info->format   = pin->format_buf;
    info->itemsize = (t->dtype.bits / 8) * (t->dtype.lanes > 0 ? t->dtype.lanes : 1);
    info->ptr      = (char*)t->data + t->byte_offset;
    info->ndim     = t->ndim;

    nelem = 1;
    if (t->shape && t->ndim > 0) {
        int d;
        /* DLPack shapes/strides are int64_t, but Py_ssize_t is only 32-bit
         * on ILP32.  Copy into the pinned buffers instead of aliasing the
         * int64 array, otherwise multi-dimension shapes read wrong values. */
        if (t->ndim > C2PY_MAX_NDIM) {
            PyErr_SetString(PyExc_ValueError,
                "DLPack: too many dimensions (exceeds C2PY_MAX_NDIM)");
            goto fail;
        }
        for (d = 0; d < t->ndim; d++)
            pin->shape_buf[d] = (Py_ssize_t)t->shape[d];
        info->shape = pin->shape_buf;
        if (t->strides) {
            /* DLPack strides are in elements; convert to bytes. */
            for (d = 0; d < t->ndim; d++) {
                pin->stride_buf[d] = (Py_ssize_t)t->strides[d] * info->itemsize;
            }
            info->strides = pin->stride_buf;
        } else if (t->ndim <= C2PY_MAX_NDIM) {
            /* Implied C-contiguous strides: last dim = itemsize,
             * each preceding dim = product of later dims * itemsize */
            Py_ssize_t st = info->itemsize;
            for (d = t->ndim - 1; d >= 0; d--) {
                pin->stride_buf[d] = st;
                st *= (Py_ssize_t)t->shape[d];
            }
            info->strides = pin->stride_buf;
        } else {
            /* >8D with NULL strides: reject (would need dynamic alloc) */
            PyErr_SetString(PyExc_ValueError,
                "DLPack: >8 dimensions without explicit strides");
            goto fail;
        }
        for (i = 0; i < t->ndim; i++)
            nelem *= (Py_ssize_t)t->shape[i];
    } else {
        info->shape   = NULL;
        info->strides = NULL;
    }
    info->len = nelem * info->itemsize;

    pin->ctx  = managed;
    pin->kind = C2PY_PIN_DLPACK;

    /* Keep the capsule alive: decref'ing it would invoke the
     * capsule destructor, which might call managed->deleter
     * prematurely.  Store it in buf.obj for later release. */
    pin->buf.obj = capsule;
    capsule = NULL;  /* ownership transferred to pin */

    Py_DECREF(dl_method);
    Py_DECREF(dl_args);
    /* capsule NOT decref'd -- stored in pin->buf.obj */
    return 0;

fail:
    Py_XDECREF(dl_method);
    Py_XDECREF(dl_args);
    Py_XDECREF(capsule);
    pin->kind = C2PY_PIN_NONE;
    return -1;
}

/* Release any backend acquired via c2py_pin_buffer/ndarray/dlpack.
 * No-op if kind == C2PY_PIN_NONE. */
static inline void
c2py_unpin_buffer(c2py_buf_pin *pin)
{
    switch (pin->kind) {
    case C2PY_PIN_PEP3118:
        c2py_release_buffer(&pin->buf);
        break;
    case C2PY_PIN_NDARRAY:
        /* We INCREF'd the ndarray in c2py_pin_ndarray;
         * release that reference now. */
        if (pin->buf.obj) {
            C2PY.DecRef(pin->buf.obj);
            pin->buf.obj = NULL;
        }
        break;
    case C2PY_PIN_DLPACK:
        /* Release the capsule we kept alive -- its destructor
         * called via Py_DECREF will invoke DLManagedTensor.deleter. */
        if (pin->buf.obj) {
            C2PY.DecRef(pin->buf.obj);
            pin->buf.obj = NULL;
        }
        pin->ctx = NULL;
        break;
    default:
        break;
    }
    pin->kind = C2PY_PIN_NONE;
}

/* Multi-source acquisition: tries each backend in order.
 * src_order is a uint8_t[] emitted by the generator from the
 * acquire: key in the .c2py file.  0 on success, -1 on failure
 * (Python exception set). */
static inline int
c2py_pin(PyObject *obj, c2py_buf_pin *pin, c2py_ptr_info *info,
         int want_writable, const uint8_t *src_order, int n_src)
{
    int i;
    for (i = 0; i < n_src; i++) {
        switch (src_order[i]) {
        case C2PY_PIN_NDARRAY:
            if (!C2PY.is_pypy) {
                if (c2py_pin_ndarray(obj, pin, info, want_writable) == 0)
                    return 0;
            }
            break;
        case C2PY_PIN_DLPACK:
            if (c2py_pin_dlpack(obj, pin, info, want_writable) == 0)
                return 0;
            break;
        case C2PY_PIN_PEP3118:
            if (c2py_pin_buffer(obj, pin, info, want_writable) == 0)
                return 0;
            break;
        default:
            break;
        }
    }
    PyErr_SetString(PyExc_TypeError,
                    "buffer acquisition failed (all backends exhausted)");
    return -1;
}

/* ------------------------------------------------------------------ */
/* Performance timing (optional, enabled via timing: true in .c2py)   */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t call_count;

    uint64_t t_enter;          /* last call: wrapper entry */
    uint64_t t_pre_c;          /* last call: just before C code */
    uint64_t t_post_c;         /* last call: just after C returns */
    uint64_t t_exit;           /* last call: before return to Python */

    uint64_t t_c_min;          /* min C-call wall time (ns) */
    uint64_t t_c_max;          /* max C-call wall time (ns) */
    uint64_t t_c_total;        /* accumulated C wall time */

    uint64_t t_wrap_min;       /* min wrapper overhead (ns) */
    uint64_t t_wrap_max;       /* max wrapper overhead (ns) */
    uint64_t t_wrap_total;     /* accumulated wrapper overhead */

    int variant;               /* active variant index (0-based), -1 if unset */
    int group_idx;             /* active outer group index, -1 if flat */
    const char *variant_name;  /* points to static string, NULL if unset */
} c2py_perf_t;

/* Tick source: wall clock by default, cycle counter on request.
 *
 * Default mode returns nanoseconds via clock_gettime() (Unix) or
 * QueryPerformanceCounter (Windows).  Set _c2py_use_cycle_counter=1
 * at runtime to switch to the CPU cycle counter (rdtsc on x86,
 * CNTVCT_EL0 on aarch64, timebase on POWER).
 *
 * The cycle counter returns raw platform-dependent ticks.  Convert
 * deltas to nanoseconds with c2py_ticks_to_ns().
 *
 * All tick calls are guarded by _c2py_do_time so there is zero cost
 * when timing is not enabled in the .c2py interface.
 */

/* Runtime tick source selection.
 * 0 = wall clock (default): nanosecond ticks, freq = 1e9
 * 1 = CPU cycle counter: raw cycles, converted to ns via c2py_ticks_to_ns()
 *     Requires c2py_cycle_counter_frequency_hz > 0. */
extern int _c2py_use_cycle_counter;

/* Detected CPU cycle counter frequency in Hz, probed once at init.
 * 0 if the platform does not support cycle counters or detection failed.
 * When the cycle counter is selected (use_cycle_counter=1), the active
 * frequency is copied to c2py_tick_frequency_hz. */
extern uint64_t c2py_cycle_counter_frequency_hz;

/* Helper: raw cycle counter read (used internally by c2py_ticks when
 * _c2py_use_cycle_counter is set). */
static inline uint64_t c2py_cycle_counter_ticks(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __rdtsc();
#elif (defined(__x86_64__) || defined(__i386__)) \
      && (defined(__GNUC__) || defined(__clang__))
    {
        unsigned int lo, hi;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    {
        uint64_t cnt;
        __asm__ __volatile__("mrs %0, CNTVCT_EL0" : "=r"(cnt));
        return cnt;
    }
#elif defined(__powerpc64__) || defined(__powerpc__)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ppc_get_timebase();
#else
    return 0;
#endif
#else
    return 0;
#endif
}

/* Monotonic timestamp, always returns nanoseconds.
 *
 * When _c2py_use_cycle_counter == 0 (default):
 *   Returns ns from clock_gettime() on Unix or QueryPerformanceCounter
 *   on Windows.  c2py_tick_frequency_hz == 1e9.
 *
 * When _c2py_use_cycle_counter == 1:
 *   Returns raw cycles from rdtsc/CNTVCT_EL0/timebase.  Convert to ns
 *   via c2py_ticks_to_ns(delta, c2py_tick_frequency_hz) where
 *   c2py_tick_frequency_hz is set to c2py_cycle_counter_frequency_hz.
 *
 * If the cycle counter is requested but the platform does not support
 * it (c2py_cycle_counter_ticks() returns 0), silently falls back to
 * wall-clock nanoseconds. */
static inline uint64_t c2py_ticks(void) {
    if (_c2py_use_cycle_counter) {
        uint64_t c = c2py_cycle_counter_ticks();
        if (c != 0) return c;
    }
#ifdef _WIN32
    {
        LARGE_INTEGER freq, counter;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&counter);
        return (uint64_t)(counter.QuadPart * 1000000000ULL / freq.QuadPart);
    }
#elif !defined(__STRICT_ANSI__)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
#else
    /* no clock_gettime -- fall back to 0 */
    return 0;
#endif
}

/* Convert cycle counter ticks to nanoseconds given the counter frequency
 * in Hz.  Returns ticks * 1e9 / freq_hz.
 * Safe against freq_hz == 0 (returns 0).  May overflow if ticks exceeds
 * ~1.8e10 (18 seconds at 1 GHz), which is well beyond any per-call timing.
 */
static inline uint64_t c2py_ticks_to_ns(uint64_t ticks, uint64_t freq_hz) {
    if (freq_hz == 0) return 0;
    return ticks * 1000000000ULL / freq_hz;
}

/* Active tick source frequency in Hz, set at init and updated when
 * _c2py_use_cycle_counter is toggled via __c2py_set_tick_source().
 * Default (wall clock): 1,000,000,000 (ns ticks).
 * Cycle counter mode: detected cycle counter frequency, or 0 if
 * the platform cannot be probed. */
extern uint64_t c2py_tick_frequency_hz;

/* Returns the active tick source frequency in Hz.  Generated wrapper
 * code uses this to convert tick deltas to nanoseconds. */
static inline uint64_t c2py_tick_frequency(void) {
    return c2py_tick_frequency_hz;
}

/* Update a perf record with one call's tick measurements.
 * t_enter:  wrapper entry
 * t_pre_c:  just before C call
 * t_post_c: just after C returns
 * t_exit:   just before returning to Python
 */
static inline void c2py_perf_record(c2py_perf_t *p,
    uint64_t t_enter, uint64_t t_pre_c, uint64_t t_post_c, uint64_t t_exit)
{
    uint64_t c_dur = t_post_c - t_pre_c;
    uint64_t w_dur = (t_pre_c - t_enter) + (t_exit - t_post_c);

    p->call_count++;
    p->t_enter  = t_enter;
    p->t_pre_c  = t_pre_c;
    p->t_post_c = t_post_c;
    p->t_exit   = t_exit;

    p->t_c_total += c_dur;
    p->t_wrap_total += w_dur;

    if (p->call_count == 1) {
        p->t_c_min = c_dur;
        p->t_c_max = c_dur;
        p->t_wrap_min = w_dur;
        p->t_wrap_max = w_dur;
    } else {
        if (c_dur < p->t_c_min) p->t_c_min = c_dur;
        if (c_dur > p->t_c_max) p->t_c_max = c_dur;
        if (w_dur < p->t_wrap_min) p->t_wrap_min = w_dur;
        if (w_dur > p->t_wrap_max) p->t_wrap_max = w_dur;
    }
}

/* Update a perf record for a single C call (no wrapper overhead).
 * Used for per-overload timing inside _impl functions. */
static inline void c2py_perf_record_call(c2py_perf_t *p,
    uint64_t t_pre, uint64_t t_post)
{
    uint64_t c_dur = t_post - t_pre;

    p->call_count++;
    p->t_pre_c  = t_pre;
    p->t_post_c = t_post;

    p->t_c_total += c_dur;

    if (p->call_count == 1) {
        p->t_c_min = c_dur;
        p->t_c_max = c_dur;
    } else {
        if (c_dur < p->t_c_min) p->t_c_min = c_dur;
        if (c_dur > p->t_c_max) p->t_c_max = c_dur;
    }
}

/* Zero a c2py_perf_t struct, resetting all counters to their
 * initial state.  Safe to call with timing enabled or disabled. */
static inline void c2py_perf_reset(c2py_perf_t *p)
{
    if (p != NULL) {
        memset(p, 0, sizeof(c2py_perf_t));
        p->variant = -1;       /* unset */
        p->group_idx = -1;     /* flat */
        p->t_c_min = UINT64_MAX;
        p->t_wrap_min = UINT64_MAX;
    }
}

/* Copy all uint64_t fields from a c2py_perf_t into a caller-provided
 * array.  buf must have room for 11 elements:
 *   [0] call_count,     [1] t_enter,        [2] t_pre_c,
 *   [3] t_post_c,       [4] t_exit,
 *   [5] t_c_min,        [6] t_c_max,        [7] t_c_total,
 *   [8] t_wrap_min,     [9] t_wrap_max,     [10] t_wrap_total
 *
 * The variant, group_idx, and variant_name fields are NOT included
 * here (they are not uint64_t).  Use c2py_perf_t members directly
 * for those, or a separate helper. */
static inline void c2py_perf_extract_u64(const c2py_perf_t *p,
                                         uint64_t *buf)
{
    buf[0] = p->call_count;
    buf[1] = p->t_enter;
    buf[2] = p->t_pre_c;
    buf[3] = p->t_post_c;
    buf[4] = p->t_exit;
    buf[5] = p->t_c_min;
    buf[6] = p->t_c_max;
    buf[7] = p->t_c_total;
    buf[8] = p->t_wrap_min;
    buf[9] = p->t_wrap_max;
    buf[10] = p->t_wrap_total;
}

/* ------------------------------------------------------------------ */
/* CPU feature probing (for user extensibility, callable from         */
/* __attribute__((constructor)) functions)                            */
/* ------------------------------------------------------------------ */

#ifdef __x86_64__
#if defined(_MSC_VER)
#include <intrin.h>
static inline unsigned int c2py_cpuid_reg(int leaf, int subleaf, int reg) {
    int info[4];
    __cpuidex(info, leaf, subleaf);
    switch (reg & 3) {
    case 0: return (unsigned int)info[0];
    case 1: return (unsigned int)info[1];
    case 2: return (unsigned int)info[2];
    default: return (unsigned int)info[3];
    }
}
#elif defined(__GNUC__) || defined(__clang__)
static inline unsigned int c2py_cpuid_reg(int leaf, int subleaf, int reg) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(leaf), "c"(subleaf));
    switch (reg & 3) {
    case 0: return eax;
    case 1: return ebx;
    case 2: return ecx;
    default: return edx;
    }
}
#else
static inline unsigned int c2py_cpuid_reg(int leaf, int subleaf, int reg) {
    (void)leaf; (void)subleaf; (void)reg;
    return 0;
}
#endif

static inline int c2py_cpuid_bit(int leaf, int subleaf, int reg, int bit) {
    return (c2py_cpuid_reg(leaf, subleaf, reg) >> bit) & 1;
}

/* Do NOT call cpu_supports from constructors; it may malloc internally.
 * Use c2py_cpuid_bit() instead for custom feature probes. */
#endif

/* ------------------------------------------------------------------ */
/* Init function                                                      */
/* ------------------------------------------------------------------ */

int c2py_runtime_init(void);

#ifdef __cplusplus
}
#endif

#endif /* C2PY_RUNTIME_H */
