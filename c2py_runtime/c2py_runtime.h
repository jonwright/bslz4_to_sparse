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


/* ---- Pick one backend ---- */

#ifdef C2PY_USE_PYTHON_H
#include "c2py_pythonh.h"
#else
#include "c2py_dlsym.h"
#endif

/* printf conversion for long long.  Older MSVC CRTs and msvcrt.dll (MinGW)
 * do not accept "%lld"; every Windows CRT accepts "%I64d". */
#if defined(_WIN32)
#define C2PY_LLD "%I64d"
#else
#define C2PY_LLD "%lld"
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
    Py_ssize_t pyobject_size;      /* actual sizeof(PyObject) for this Python version */
    Py_ssize_t pyobject_size_ft;   /* sizeof(PyObject_FT): 32 on LP64, 20 on ILP32 */
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

    /* Tuple construction and access (access used by the VARARGS shim) */
    PyObject* (*Tuple_New)(Py_ssize_t);
    int (*Tuple_SetItem)(PyObject*, Py_ssize_t, PyObject*);
    Py_ssize_t (*Tuple_Size)(PyObject*);
    PyObject* (*Tuple_GetItem)(PyObject*, Py_ssize_t);

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
    /* Keyword call, for __dlpack__(max_version=...) (optional) */
    PyObject* (*Call)(PyObject*, PyObject*, PyObject*);
    PyObject* (*Dict_New)(void);

    /* Capsule API (optional, used by DLPack) */
    void* (*Capsule_GetPointer)(PyObject*, const char*);

    /* Pointer-to-int conversion (for exposing perf struct addresses),
     * and int-to-pointer (for opaque void* parameters). */
    PyObject* (*Long_FromVoidPtr)(void*);
    void* (*Long_AsVoidPtr)(PyObject*);
    /* operator.index(): classifies a failed int-to-pointer conversion */
    PyObject* (*Number_Index)(PyObject*);

    /* GIL management */
    void* (*SaveThread)(void);
    void (*RestoreThread)(void*);
    /* Free-threading: set Py_MOD_GIL_NOT_USED on module (optional, may be NULL) */
    void (*Unstable_Module_SetGIL)(PyObject*, void*);

} c2py_api_t;

/* The global API table */
extern C2PY_HIDDEN c2py_api_t C2PY;

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
#define PyLong_AsVoidPtr(o)            C2PY.Long_AsVoidPtr((PyObject*)(o))
#define PyNumber_Index(o)              C2PY.Number_Index((PyObject*)(o))
#define PyTuple_New(s)                 C2PY.Tuple_New(s)
#define PyTuple_SetItem(t, i, o)       C2PY.Tuple_SetItem((PyObject*)(t), (i), (PyObject*)(o))
#define PyTuple_Size(t)                C2PY.Tuple_Size((PyObject*)(t))
#define PyTuple_GetItem(t, i)          C2PY.Tuple_GetItem((PyObject*)(t), (i))
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
/* Buffer acquisition helper with old-API fallback for Python 2.7     */
/* ------------------------------------------------------------------ */

/* Flags for c2py_acquire_buffer */
#define C2PY_BUF_READ   0
#define C2PY_BUF_WRITE  1

/* Returns 0 on success, -1 on failure (with Python exception set) */
static inline int
c2py_acquire_buffer(PyObject *obj, Py_buffer *buf, int want_writable)
{
    Py_buffer zero = {0};
    int flags = PyBUF_STRIDES | PyBUF_FORMAT;
    if (want_writable) flags |= PyBUF_WRITABLE;

    *buf = zero;  /* sizeof(Py_buffer); see c2py_pin_init */

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

/* Largest Py_ssize_t, without relying on PY_SSIZE_T_MAX from Python.h. */
#define C2PY_SSIZE_T_MAX ((Py_ssize_t)(((size_t)-1) >> 1))

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

/* Overflow-checked long long arithmetic for integer check:, when: and map:
 * expressions.  Buffer sizes are Py_ssize_t, which is 32-bit on ILP32, so
 * a.n * 8 or a.n + b.n can wrap there; evaluating in long long fixes that,
 * and these helpers also catch what long long cannot hold (a.n * b.n for
 * two 4 GiB buffers on a 64-bit build) and division by zero.  On failure
 * they set *ovf and return 0; the wrapper then raises ValueError instead
 * of acting on a wrapped value.  No compiler builtins: plain C99. */
static inline long long
c2py_add_ll(long long a, long long b, int *ovf)
{
    if ((b > 0 && a > LLONG_MAX - b) || (b < 0 && a < LLONG_MIN - b)) { *ovf = 1; return 0; }
    return a + b;
}

static inline long long
c2py_sub_ll(long long a, long long b, int *ovf)
{
    if ((b < 0 && a > LLONG_MAX + b) || (b > 0 && a < LLONG_MIN + b)) { *ovf = 1; return 0; }
    return a - b;
}

static inline long long
c2py_mul_ll(long long a, long long b, int *ovf)
{
    if (a == 0 || b == 0) return 0;
    if (a > 0 ? (b > 0 ? a > LLONG_MAX / b : b < LLONG_MIN / a)
              : (b > 0 ? a < LLONG_MIN / b : a < LLONG_MAX / b)) { *ovf = 1; return 0; }
    return a * b;
}

static inline long long
c2py_div_ll(long long a, long long b, int *ovf)
{
    if (b == 0 || (a == LLONG_MIN && b == -1)) { *ovf = 1; return 0; }
    return a / b;
}

static inline long long
c2py_mod_ll(long long a, long long b, int *ovf)
{
    if (b == 0 || (a == LLONG_MIN && b == -1)) { *ovf = 1; return 0; }
    return a % b;
}

/* <buf>.ptr % n in a when:/checks: expression with a non-literal n.  The
 * address is reduced in uintptr_t.  n <= 0 yields 1 (an alignment test
 * never matches) instead of dividing by zero. */
static inline uintptr_t
c2py_ptr_mod(const void *p, long long n)
{
    if (n <= 0) return 1;
    return (uintptr_t)p % (uintptr_t)n;
}

static inline long long
c2py_neg_ll(long long a, int *ovf)
{
    if (a == LLONG_MIN) { *ovf = 1; return 0; }
    return -a;
}

/* Python int -> C pointer for an opaque void* parameter.  Returns 0 and
 * stores the pointer, or returns -1 with an exception set.
 *
 * PyLong_AsVoidPtr is the pointer-width conversion: it takes the unsigned
 * address (what ctypes.addressof() and id() give) and also a negative int,
 * which wraps.  Its failure mode is not uniform, though: for an object
 * that is not an int it raises SystemError on Python 2.7 and TypeError on
 * 3.x, and it never tries __index__.  So on failure the argument is put
 * through PyNumber_Index, which raises TypeError for a non-integer on
 * every version, and the conversion is repeated on the result.
 *
 * PyPy's PyLong_AsVoidPtr is unsigned only: a negative int raises
 * ValueError there.  So when the repeat fails too, the int is read as a
 * signed long long and a negative value that fits a pointer is wrapped
 * here, which gives the same accepted range on every interpreter.  An int
 * outside that range raises OverflowError.  The first call succeeds for
 * every in-range int on CPython, so none of this costs anything on the
 * normal path. */
static inline int
c2py_as_voidptr(PyObject *o, void **out)
{
    PyObject *idx;
    long long v;
    void *p = PyLong_AsVoidPtr(o);
    if (p == NULL && PyErr_Occurred()) {
        PyErr_Clear();
        idx = PyNumber_Index(o);
        if (idx == NULL) return -1;
        p = PyLong_AsVoidPtr(idx);
        if (p == NULL && PyErr_Occurred()) {
            PyErr_Clear();
            /* OverflowError here when the int does not fit a long long */
            v = PyLong_AsLongLong(idx);
            if (v == -1 && PyErr_Occurred()) { Py_DECREF(idx); return -1; }
            if (v < 0 && (long long)(intptr_t)v == v) {
                p = (void *)(intptr_t)v;
            } else {
                /* Fits a long long but not a pointer (32-bit only).
                 * Repeat a conversion that is certain to overflow, to
                 * get the interpreter's own OverflowError. */
                if (v < 0) (void)PyLong_AsLong(idx);
                else (void)PyLong_AsVoidPtr(idx);
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_ValueError,
                                    "int does not fit a pointer");
                Py_DECREF(idx);
                return -1;
            }
        }
        Py_DECREF(idx);
    }
    *out = p;
    return 0;
}

/* Runtime enforcement of the no-alias contract generated wrappers rely
 * on when passing buffer pointers to C functions as restrict.
 *
 * The first `n_writable` entries of `infos` are writable buffers and the
 * rest are read-only.  Any pair with at least one writable endpoint must
 * not overlap in memory; read-only/read-only overlap is allowed.  Returns
 * 0 when clean, or -1 with ValueError set when a writable buffer overlaps
 * another buffer.
 *
 * `infos` is an array of pointers to the wrapper's c2py_ptr_info locals,
 * ordered writables-first by the generator.  This centralizes the O(n^2)
 * check in one out-of-line helper instead of inlining a block per pair
 * (which exploded generated wrapper size for many-buffer functions). */
static inline int
c2py_check_no_overlap(c2py_ptr_info *const *infos, int n_writable, int n)
{
    int i, j;
    uintptr_t a, b;
    for (i = 0; i < n_writable; i++) {
        /* An empty buffer occupies no bytes: nothing can be written
         * through it, so it overlaps nothing (a[5:5] inside a). */
        if (infos[i]->len <= 0) continue;
        a = (uintptr_t)infos[i]->ptr;
        for (j = 0; j < n; j++) {
            if (i == j || infos[j]->len <= 0) continue;
            b = (uintptr_t)infos[j]->ptr;
            /* Integer compare: relational comparison of pointers into
             * different objects is undefined in C. */
            if (a < b + (uintptr_t)infos[j]->len && b < a + (uintptr_t)infos[i]->len) {
                PyErr_SetString(PyExc_ValueError, "buffer aliasing forbidden");
                return -1;
            }
        }
    }
    return 0;
}

/* Bounds-safe shape[k] / strides[k] for generated map:, when: and checks:
 * expressions.  Out-of-range k (k < 0 or k >= ndim) yields -1 instead of
 * reading past the Py_buffer arrays.  A NULL shape (1-D, and every Python 2.7
 * old-buffer object) is the flat length; a NULL strides (PEP 3118: C order)
 * is itemsize for k == 0 and the C-order step for k > 0. */
static inline Py_ssize_t
c2py_shape_at(const c2py_ptr_info *info, int k)
{
    if (k < 0 || k >= info->ndim) return -1;
    if (info->shape == NULL) {
        if (k != 0 || info->itemsize <= 0) return -1;
        return info->len / info->itemsize;
    }
    return info->shape[k];
}

static inline Py_ssize_t
c2py_stride_at(const c2py_ptr_info *info, int k)
{
    Py_ssize_t _s;
    int _d;

    if (k < 0 || k >= info->ndim) return -1;
    if (info->strides != NULL) return info->strides[k];
    if (k == 0 && info->shape == NULL) return info->itemsize;
    if (info->shape == NULL) return -1;
    _s = info->itemsize;
    for (_d = info->ndim - 1; _d > k; _d--) _s *= info->shape[_d];
    return _s;
}

/* Validate that a buffer is C-contiguous or Fortran-contiguous and report
 * the slowest/fastest varying axis indices through `slow_axis`/`fast_axis`
 * (used by `when:` dispatch on `.slow_axis`/`.fast_axis`).  The caller
 * must initialize both out-params; they are left untouched only for the
 * degenerate ndim<1-with-strides case, which the wrapper initializes to
 * -1.  Returns 0 when contiguous, or -1 with ValueError set otherwise.
 *
 * Centralizing this keeps the generated wrapper small: the check body is
 * ~40 lines per buffer and was previously inlined into every expanded
 * method.  `shape`/`strides` are read-only here.  `where` names the
 * argument in the error message ("f: arg 'a'"); NULL is allowed. */
static inline int
c2py_check_contiguity(const c2py_ptr_info *info, int *slow_axis, int *fast_axis,
                      const char *where)
{
    int _ok = 1;
    int _d;
    Py_ssize_t _expected;

    /* NULL strides mean C order (PEP 3118).  The ndarray and DLPack
     * backends give shape and strides whenever ndim >= 1, the 2.7 old
     * buffer API has ndim 1 with neither; only a PEP 3118 exporter that
     * ignores the PyBUF_STRIDES request leaves strides out otherwise. */
    if (info->strides == NULL && (info->ndim <= 1 || info->shape != NULL)) {
        *slow_axis = 0;
        *fast_axis = (int)(info->ndim - 1);
        return 0;
    }
    if (info->len == 0) {
        *slow_axis = 0;
        *fast_axis = (int)(info->ndim - 1);
        return 0;
    }
    /* A non-conforming exporter with no shape (and ndim >= 1) has no
     * layout to check: rejected below, shape[] is never indexed. */
    _ok = info->ndim < 1 || info->shape != NULL;
    /* An axis of length 1 is never stepped along, so its stride is
     * irrelevant: numpy leaves it raw (0 for a[None, :], junk under
     * NPY_RELAXED_STRIDES_DEBUG) and only normalises it in its PEP 3118
     * export.  C order is tested first so that an array that is both
     * (e.g. (1, N) or (N, 1)) reports the same axes on every backend. */
    if (_ok && info->ndim >= 1) {
        /* check C-contiguous (row-major): last dim varies fastest */
        _expected = info->itemsize;
        for (_d = info->ndim - 1; _d >= 0; _d--) {
            if (info->shape[_d] == 1) continue;
            if (info->strides[_d] < 0) { _ok = 0; break; }
            if (info->strides[_d] != _expected) { _ok = 0; break; }
            _expected *= info->shape[_d];
        }
        if (_ok) {
            *slow_axis = 0;
            *fast_axis = (int)(info->ndim - 1);
            return 0;
        }
        /* check F-contiguous (column-major): first dim varies fastest */
        _ok = 1;
        _expected = info->itemsize;
        for (_d = 0; _d < info->ndim; _d++) {
            if (info->shape[_d] == 1) continue;
            if (info->strides[_d] < 0) { _ok = 0; break; }
            if (info->strides[_d] != _expected) { _ok = 0; break; }
            _expected *= info->shape[_d];
        }
        if (_ok) {
            *slow_axis = (int)(info->ndim - 1);
            *fast_axis = 0;
            return 0;
        }
    }
    if (!_ok) {
        char _msg[320];
        snprintf(_msg, sizeof(_msg),
            "%s: buffer not contiguous (C or Fortran contiguous required);"
            " pass a contiguous copy, e.g. numpy.ascontiguousarray(x)",
            where ? where : "c2py23");
        PyErr_SetString(PyExc_ValueError, _msg);
        return -1;
    }
    return 0;
}

/* Reject a non-empty buffer whose data pointer is not aligned for the C
 * parameter it is passed to (e.g. np.frombuffer(..., dtype="f8", offset=1)
 * handed to a double*).  `align` is sizeof the C pointee type and `ctype`
 * its name: the requirement comes from the C signature, not from the
 * buffer's own itemsize, so a uint64 buffer at an odd address may be
 * passed to a char* or void* parameter.  Only alignments that are powers
 * of two from 2 to 16 are checked.  `where` names the argument in the
 * error message ("f: arg 'a'"); NULL is allowed.  Returns 0, or -1 with
 * ValueError set. */
static inline int
c2py_check_alignment(const c2py_ptr_info *info, size_t align,
                     const char *ctype, const char *where)
{
    uintptr_t off;

    if (info->len <= 0 || align < 2 || align > 16
        || (align & (align - 1)) != 0)
        return 0;
    off = (uintptr_t)info->ptr & (uintptr_t)(align - 1);
    if (off != 0) {
        char _msg[400];
        snprintf(_msg, sizeof(_msg),
            "%s: buffer not aligned for the C parameter type %s *"
            " (alignment " C2PY_LLD ", address %% " C2PY_LLD " == " C2PY_LLD
            "); pass an aligned copy,"
            " e.g. numpy.array(x), or set check_alignment: False in the .c2py"
            " if the C code accepts unaligned data",
            where ? where : "c2py23", ctype ? ctype : "?",
            (long long)align, (long long)align, (long long)off);
        PyErr_SetString(PyExc_ValueError, _msg);
        return -1;
    }
    return 0;
}

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

/* Reset a pin before acquisition: zero the Py_buffer (sizeof covers
 * the largest CPython layout; the _buf_pad tail is left untouched,
 * it is only scratch for PyPy's larger Py_buffer) and mark it empty
 * so c2py_unpin_buffer is a no-op until a backend succeeds.
 * Struct assignment rather than memset: gcc -O2 emits rep stos for a
 * fixed 96-byte memset, which costs more than a few vector stores. */
static inline void
c2py_pin_init(c2py_buf_pin *p)
{
    Py_buffer zero = {0};
    p->buf = zero;
    p->kind = C2PY_PIN_NONE;
    p->ctx = NULL;
}

/* ------------------------------------------------------------------ */
/* NumPy ndarray struct-cast fast path (lazy-probed, no import)       */
/* ------------------------------------------------------------------ */

/* PyTypeObject.tp_name follows the object header and ob_size, so its
 * offset is C2PY.pyobject_size + sizeof(Py_ssize_t) on every build:
 * 24 on GIL LP64, 40 on free-threaded LP64, 12 / 24 on ILP32. */
#define C2PY_TP_NAME_OFF (C2PY.pyobject_size + (Py_ssize_t)sizeof(Py_ssize_t))

/* Publishing the probed ndarray layout.  On free-threaded Python two
 * threads can meet the first ndarray at once, and a reader must never
 * see ndarray_type set before data_off: the layout is published with a
 * release store and read with an acquire load.  GCC/Clang (and MinGW)
 * use __atomic builtins, MSVC the Interlocked functions (full
 * barriers).  Without either, C2PY_HAVE_ATOMICS is 0 and the fast path
 * stays off on free-threaded builds (GIL builds need no barrier). */
#if defined(__clang__) || \
    (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)))
#define C2PY_HAVE_ATOMICS 1
#define c2py_load_ptr_acq(p)     __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define c2py_store_ptr_rel(p, v) __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#define c2py_load_int(p)         __atomic_load_n((p), __ATOMIC_RELAXED)
#define c2py_store_int(p, v)     __atomic_store_n((p), (v), __ATOMIC_RELAXED)
#elif defined(_MSC_VER)
#define C2PY_HAVE_ATOMICS 1
#define c2py_load_ptr_acq(p) \
    InterlockedCompareExchangePointer((PVOID volatile *)(p), NULL, NULL)
#define c2py_store_ptr_rel(p, v) \
    ((void)InterlockedExchangePointer((PVOID volatile *)(p), (PVOID)(v)))
#define c2py_load_int(p) \
    ((int)InterlockedCompareExchange((LONG volatile *)(p), 0, 0))
#define c2py_store_int(p, v) \
    ((void)InterlockedExchange((LONG volatile *)(p), (LONG)(v)))
#else
#define C2PY_HAVE_ATOMICS 0
#define c2py_load_ptr_acq(p)     (*(p))
#define c2py_store_ptr_rel(p, v) ((void)(*(p) = (v)))
#define c2py_load_int(p)         (*(p))
#define c2py_store_int(p, v)     ((void)(*(p) = (v)))
#endif

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
    /* Accessed only through c2py_load_* / c2py_store_* (see above):
     * data_off is stored before ndarray_type is published. */
    void *ndarray_type;     /* cached ob_type of numpy.ndarray, NULL if unknown */
    int data_off;           /* offset of data ptr from PyObject base */
    int probed;             /* 1 = layout known, 0 = probing deferred */
} c2py_ndarray_layout_t;

extern C2PY_HIDDEN c2py_ndarray_layout_t C2PY_NDARRAY;

/* NPY_ARRAY_WRITEABLE = 0x0400 (stable since numpy 1.0) */
#define C2PY_NPY_WRITEABLE  0x0400

/* Host byte order, resolved at compile time: a compiled .so is built
 * for one architecture, so this is a fixed build property, never a
 * runtime probe.  __BYTE_ORDER__ is provided by GCC and Clang on every
 * target (including big-endian ones, e.g. a future ppc64be); MSVC only
 * ever targets little-endian architectures (x86/x64/ARM), so it is
 * safe to default to little-endian when the macro is unavailable. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define C2PY_BIG_ENDIAN 1
#else
#define C2PY_BIG_ENDIAN 0
#endif

/* c2py23 only supports native-byte-order buffers -- a buffer whose PEP
 * 3118 format string carries an explicit non-native prefix ('<', '>',
 * or '!') would otherwise be read with the wrong byte order, silently
 * producing wrong values instead of an error.  Returns 1 (native, or
 * unspecified/NULL) when the format is safe to read as-is, 0 when it
 * explicitly requests the other byte order. */
static inline int
c2py_format_is_native(const char *fmt)
{
    size_t len;
    char prefix;

    if (!fmt) return 1;
    len = strlen(fmt);
    if (len < 2) return 1;

    prefix = fmt[len - 2];
#if C2PY_BIG_ENDIAN
    return prefix != '<';
#else
    return prefix != '>' && prefix != '!';
#endif
}

/* Type character of a single-element, native-order PEP 3118 format, or
 * '\0' for anything else.  Accepted: one type char with an optional
 * byte-order prefix ("d", "<d", "=d", "@d"), the prefix being non-native
 * only for 1-byte types.  Rejected: complex ("Zd": no complex type, see
 * docs/design.md #53), repeat counts ("2d"), structs ("T{...}") and
 * non-native order.  Comparing only the last character let all of these
 * through as plain 'd'.
 *
 * A NULL format is 'B' (unsigned bytes), as PEP 3118 defines it.  It is
 * also exactly what the Python 2.7 old buffer API fallback hands over
 * (itemsize 1, .n counts bytes), so a 'd' check must not accept it:
 * array.array('d') on 2.7 has only the old API, and treating NULL as
 * "matches anything" let .n be 8x the element count. */
static inline char
c2py_format_type(const char *fmt)
{
    char c;

    if (fmt == NULL) return 'B';
    if (fmt[0] == '@' || fmt[0] == '=' || fmt[0] == '<' || fmt[0] == '>' || fmt[0] == '!') {
        if (!c2py_format_is_native(fmt) && fmt[1] != 'b' && fmt[1] != 'B'
            && fmt[1] != 'c' && fmt[1] != '?')
            return '\0';
        fmt++;
    }
    c = fmt[0];
    if (c == '\0' || fmt[1] != '\0') return '\0';
    if ((c >= '0' && c <= '9') || c == 'Z' || c == 'T' || c == '{' || c == '(') return '\0';
    return c;
}

/* Map a numpy type character to itemsize for the ndarray fast path, or
 * 0 when the size is not known here: complex ('F', 'D', 'G'), datetime,
 * string, void and object dtypes, and 'g' (long double is 8, 12 or 16
 * bytes depending on the compiler numpy was built with, which need not
 * be ours, e.g. MinGW against an MSVC numpy).  On 0 the fast path
 * declines and the PEP 3118 backend, which reports numpy's own
 * itemsize, takes over.  'l'/'L' are sizeof(long) on both sides. */
static inline int
c2py_format_itemsize(char type_char)
{
    switch (type_char) {
    case 'b': case 'B': case '?': return 1;
    case 'h': case 'H': case 'e': return 2;
    case 'i': case 'I':          return 4;
    case 'l': case 'L':          return (int)sizeof(long);
    case 'q': case 'Q':          return 8;
    case 'f':                    return 4;
    case 'd':                    return 8;
    default:                     return 0;
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

/* DLPack >= 1.0 ("dltensor_versioned" capsule): the tensor comes after
 * the version, context, deleter and flags, and the flags say whether
 * the producer allows writes.  The legacy struct above has no such
 * flag. */
typedef struct {
    uint32_t major;
    uint32_t minor;
} c2py_dl_version_t;

typedef struct c2py_dl_managed_tensor_versioned {
    c2py_dl_version_t version;
    void *manager_ctx;
    void (*deleter)(struct c2py_dl_managed_tensor_versioned *);
    uint64_t flags;
    c2py_dl_tensor_t dl_tensor;
} c2py_dl_managed_tensor_versioned;

#define C2PY_DL_FLAG_READ_ONLY ((uint64_t)1)

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
#else
#define C2PY_NDARR_OFF_ND        4
#define C2PY_NDARR_OFF_DIMS      8
#define C2PY_NDARR_OFF_STRIDES   12
#define C2PY_NDARR_OFF_DESCR     20
#define C2PY_NDARR_OFF_FLAGS     24
#endif

/* PyArray_Descr is an object header, then PyTypeObject *typeobj, then
 * char kind, type, byteorder ('<','>','=','|').  The header is larger
 * on free-threaded builds, so the offsets follow C2PY.pyobject_size:
 * type char at 25 on GIL LP64, 41 on free-threaded LP64, 13 on ILP32. */
#define C2PY_DESCR_TYPE_CHAR_OFF (C2PY.pyobject_size + (Py_ssize_t)sizeof(void*) + 1)
#define C2PY_DESCR_BYTEORDER_OFF (C2PY_DESCR_TYPE_CHAR_OFF + 1)

/* Keeps cold code (the ndarray probe, which runs until the first
 * suitable array and then never again) out of the hot acquire path.
 * Not 'inline' (GCC rejects inline + noinline); 'unused' because a
 * translation unit may never call it. */
#if defined(__GNUC__) || defined(__clang__)
#define C2PY_NOINLINE __attribute__((noinline, unused))
#elif defined(_MSC_VER)
#define C2PY_NOINLINE __declspec(noinline)
#else
#define C2PY_NOINLINE
#endif

/* First encounter of a numpy.ndarray (type tp): acquire via the buffer
 * protocol, scan object memory to locate the data pointer, then check
 * the struct reading against the buffer's answer before publishing the
 * layout.  Returns 0 with pin->buf holding the PEP 3118 answer (the
 * caller copies it to info: info must not escape the inlined caller, or
 * the compiler keeps it in memory on every call), or -1 (with the
 * exception from the buffer protocol, if any). */
static C2PY_NOINLINE int
c2py_probe_ndarray(PyObject *obj, void *tp, c2py_buf_pin *pin,
                   int want_writable)
{
    c2py_ndarray_layout_t *L = &C2PY_NDARRAY;
    char *base;
    void *dptr, *descr;
    int nd, data_off = -1;
    Py_ssize_t nelem, i;
    char type_char;

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
                data_off = off;
                break;
            }
        }
    }
    /* Data pointer not found (e.g. NULL for an empty array), or
     * an answer that cannot be compared with the struct (0-d,
     * more dims than C2PY_MAX_NDIM, no shape, or the Python 2.7
     * old buffer API, which has buf.obj == NULL): nothing
     * learned, retry on a later array. */
    if (data_off < 0 || pin->buf.obj == NULL || pin->buf.ndim < 1 ||
        pin->buf.ndim > C2PY_MAX_NDIM || pin->buf.shape == NULL)
        goto fail;

    /* Validate the hard-coded offsets against the PEP 3118 answer
     * (finding A7): nd, itemsize and type char of the descr, every
     * dimension, and every stride numpy does not normalise in its
     * export (length-1 axes and empty arrays).  On disagreement
     * the layout is not numpy's: mark the probe done with no type
     * published, so the fast path stays off in this process, and
     * serve this call from the PEP 3118 answer. */
    {
        Py_ssize_t *dims, *strides;
        int isz, ok = 1;

        nd      = *(int*)(base + data_off + C2PY_NDARR_OFF_ND);
        dims    = *(Py_ssize_t**)(base + data_off + C2PY_NDARR_OFF_DIMS);
        strides = *(Py_ssize_t**)(base + data_off + C2PY_NDARR_OFF_STRIDES);
        descr   = *(void**)(base + data_off + C2PY_NDARR_OFF_DESCR);
        if (nd != pin->buf.ndim || dims == NULL || strides == NULL ||
            descr == NULL) {
            ok = 0;
        } else {
            type_char = ((char*)descr)[C2PY_DESCR_TYPE_CHAR_OFF];
            isz = c2py_format_itemsize(type_char);
            if (isz != 0) {
                char ft = c2py_format_type(pin->buf.format);
                if (isz != pin->buf.itemsize ||
                    (ft != '\0' && ft != type_char))
                    ok = 0;
            }
            nelem = 1;
            for (i = 0; ok && i < nd; i++) {
                if (dims[i] != pin->buf.shape[i])
                    ok = 0;
                nelem *= (dims[i] != 0);
            }
            if (ok && nelem && pin->buf.strides != NULL) {
                for (i = 0; i < nd; i++) {
                    if (dims[i] > 1 && strides[i] != pin->buf.strides[i])
                        ok = 0;
                }
            }
        }
        if (!ok) {
            c2py_store_int(&L->probed, 1);
            goto use_pep3118;
        }
    }

    /* Publish: data_off first, then the type (release), so a
     * thread that sees the type also sees data_off.  A racing
     * prober stores the same values. */
    c2py_store_int(&L->data_off, data_off);
    c2py_store_int(&L->probed, 1);
    c2py_store_ptr_rel(&L->ndarray_type, tp);

use_pep3118:
    /* First call: served from the buffer acquired above, as the
     * pep3118 backend, so unpin releases it. */
    pin->kind = C2PY_PIN_PEP3118;
    return 0;

fail:
    c2py_release_buffer(&pin->buf);
    return -1;
}

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

    /* PyPy cpyext wraps ndarray objects in proxies -- the data pointer
     * is not at a fixed offset from id() -- so there is no struct to
     * cast: fall through to the buffer protocol.  Free-threaded builds
     * use the fast path when the layout can be published safely. */
    if (C2PY.is_pypy)
        return -1;
#if !C2PY_HAVE_ATOMICS
    if (C2PY.is_free_threaded)
        return -1;
#endif

    tp = *(void**)((char*)obj + C2PY.ob_type_offset);

    if (tp && tp == c2py_load_ptr_acq(&L->ndarray_type)) {
        goto fill;
    }

    if (!c2py_load_int(&L->probed) && tp) {
        const char *name = *(const char**)((char*)tp + C2PY_TP_NAME_OFF);
        if (name && strcmp(name, "numpy.ndarray") == 0) {
            if (c2py_probe_ndarray(obj, tp, pin, want_writable) != 0)
                return -1;
            memcpy(info, &pin->buf, sizeof(c2py_ptr_info));
            return 0;
        }
    }

    return -1;

fill:
    base = (char*)obj + c2py_load_int(&L->data_off);

    dptr  = *(void**)base;
    nd    = *(volatile int*)(base + C2PY_NDARR_OFF_ND);
    flags = *(int*)(base + C2PY_NDARR_OFF_FLAGS);

    if (want_writable && !(flags & C2PY_NPY_WRITEABLE)) {
        /* ValueError, as numpy raises through PEP 3118, so the error is
         * the same whichever backend (or free-threaded build) ran. */
        PyErr_SetString(PyExc_ValueError,
                        "numpy array is read-only");
        return -1;
    }

    descr = *(void**)(base + C2PY_NDARR_OFF_DESCR);
    if (descr) {
        /* Explicit non-native byte order: this fast path only ever
         * synthesizes a bare type char (no PEP 3118 prefix), so it
         * cannot be checked downstream -- fall through to the buffer
         * protocol backend instead, whose format string does carry
         * the prefix and gets checked by c2py_format_is_native(). */
        char byteorder = ((char*)descr)[C2PY_DESCR_BYTEORDER_OFF];
#if C2PY_BIG_ENDIAN
        if (byteorder == '<') return -1;
#else
        if (byteorder == '>') return -1;
#endif
        /* type char at a fixed byte offset within PyArray_Descr
         * (stable since numpy 1.0 through 2.x on GIL builds) */
        type_char = ((char*)descr)[C2PY_DESCR_TYPE_CHAR_OFF];
        pin->format_buf[0] = type_char;
        pin->format_buf[1] = '\0';
        info->format = pin->format_buf;
        info->itemsize = c2py_format_itemsize(type_char);
        /* Unknown size: fall through to the buffer protocol backend. */
        if (info->itemsize == 0) return -1;
    } else {
        return -1;
    }

    /* Copy shape and strides into the pin (finding A6), so the wrapper
     * holds no pointer into the array other than the data pointer.  nd
     * and the two array pointers are each read exactly once (volatile:
     * the compiler may not reload them), and nd is bounded before it is
     * used, so whatever another thread does to the array (an in-place
     * "a.shape = ..." on a free-threaded build) the copy never runs past
     * shape_buf / stride_buf.  More than C2PY_MAX_NDIM dimensions is a
     * real array: decline, PEP 3118 takes it.
     *
     * A torn read can still give a wrong shape.  A wrong but plausible
     * one goes to the function's checks: like any other shape.  An
     * impossible one (missing arrays, a negative dimension, an element
     * count that overflows) raises ValueError instead of declining:
     * numpy's own PEP 3118 export is not safe against a concurrent
     * reshape, and a real array never gets here.  numpy frees and
     * rewrites dims/strides during such a reshape, so the copy itself
     * may read stale numpy memory; only numpy can close that. */
    if ((unsigned)nd > (unsigned)C2PY_MAX_NDIM)
        return -1;
    info->ptr  = dptr;
    info->ndim = nd;
    if (nd > 0) {
        const Py_ssize_t *dims, *strides;
        Py_ssize_t d, neg = 0;
        size_t prod = 1;

        dims    = *(Py_ssize_t *volatile *)(base + C2PY_NDARR_OFF_DIMS);
        strides = *(Py_ssize_t *volatile *)(base + C2PY_NDARR_OFF_STRIDES);
        if ((dims == NULL) | (strides == NULL))
            goto torn;
        for (i = 0; i < nd; i++) {
            d = dims[i];
            pin->shape_buf[i]  = d;
            pin->stride_buf[i] = strides[i];
            neg |= d;
            prod *= (size_t)d;
        }
        /* itemsize <= 8 here (c2py_format_itemsize), so len fits. */
        if ((neg < 0) | (prod > ((size_t)C2PY_SSIZE_T_MAX >> 3)))
            goto torn;
        nelem = (Py_ssize_t)prod;
        info->shape   = pin->shape_buf;
        info->strides = pin->stride_buf;
    } else {
        nelem = 1;
        info->shape   = NULL;
        info->strides = NULL;
    }
    info->len = nelem * (Py_ssize_t)info->itemsize;

    /* Hold a reference on the ndarray: we skipped PyObject_GetBuffer
     * which normally INCREFs the exporter.  Store it in buf.obj so
     * unpin can DECREF it. */
    C2PY.IncRef(obj);
    pin->buf.obj = obj;
    pin->kind = C2PY_PIN_NDARRAY;
    return 0;

torn:
    PyErr_SetString(PyExc_ValueError,
                    "numpy array shape changed while being read");
    return -1;
}

/* Acquire via DLPack capsule extraction.
 * Calls obj.__dlpack__(max_version=(1, 0)) -- falling back to a plain
 * obj.__dlpack__() for producers that predate DLPack 1.0 -- and reads
 * the DLTensor fields directly.  CPU-only.  Every field is validated
 * before use: a producer is foreign code and its tensor may be
 * malformed.
 *
 * Writability: a versioned capsule carries a read-only flag, which is
 * honoured.  A legacy capsule carries none, so it is never written
 * through: a writable request declines it and the next backend
 * decides.  This is what numpy does as a consumer (np.from_dlpack of a
 * legacy capsule is a read-only array); producers such as JAX export
 * immutable buffers through legacy DLPack.
 *
 * Returns 0 on success.  Returns -1 with no exception set to decline
 * (next backend), or -1 with an exception set for a hard failure. */
static inline int
c2py_pin_dlpack(PyObject *obj, c2py_buf_pin *pin, c2py_ptr_info *info,
                int want_writable)
{
    PyObject *dl_method = NULL;
    PyObject *dl_args = NULL;
    PyObject *capsule = NULL;
    c2py_dl_tensor_t *t;
    void *ctx;
    int read_only;  /* 1 read-only, 0 writable, -1 unknown (legacy) */
    char format_char;
    Py_ssize_t nelem;

    /* DLPack symbols are optional; if not resolved, skip this backend. */
    if (!C2PY.CallObject || !C2PY.Capsule_GetPointer)
        return -1;

    dl_method = PyObject_GetAttrString(obj, "__dlpack__");
    if (!dl_method) { PyErr_Clear(); return -1; }

    dl_args = PyTuple_New(0);
    if (!dl_args) goto fail;

    if (C2PY.Call && C2PY.Dict_New && C2PY._ds_set_item_string) {
        typedef int (*dset_fn)(PyObject *, const char *, PyObject *);
        PyObject *kw = C2PY.Dict_New();
        PyObject *ver = PyTuple_New(2);
        if (kw && ver) {
            PyObject *v0 = PyLong_FromLong(1);
            PyObject *v1 = v0 ? PyLong_FromLong(0) : NULL;
            if (!v1) {
                /* Out of memory: leave the exception set (hard failure,
                 * not a decline). */
                Py_XDECREF(v0);
                Py_DECREF(kw);
                Py_DECREF(ver);
                goto fail;
            }
            PyTuple_SetItem(ver, 0, v0);
            PyTuple_SetItem(ver, 1, v1);
            if (((dset_fn)C2PY._ds_set_item_string)(kw, "max_version", ver) == 0)
                capsule = C2PY.Call(dl_method, dl_args, kw);
        } else {
            /* Allocation failed: the MemoryError stays set. */
            Py_XDECREF(kw);
            Py_XDECREF(ver);
            goto fail;
        }
        if (kw) Py_DECREF(kw);
        if (ver) Py_DECREF(ver);
        /* TypeError from a producer without max_version: retry plainly */
        if (!capsule) PyErr_Clear();
    }
    if (!capsule) {
        capsule = PyObject_CallObject(dl_method, dl_args);
        if (!capsule) { PyErr_Clear(); goto decline; }
    }

    {
        c2py_dl_managed_tensor_versioned *vm = (c2py_dl_managed_tensor_versioned*)
            PyCapsule_GetPointer(capsule, "dltensor_versioned");
        if (vm) {
            /* Only the major version fixes the layout. */
            if (vm->version.major != 1) goto decline;
            t = &vm->dl_tensor;
            ctx = vm;
            read_only = (vm->flags & C2PY_DL_FLAG_READ_ONLY) ? 1 : 0;
        } else {
            c2py_dl_managed_tensor *m;
            PyErr_Clear();
            m = (c2py_dl_managed_tensor*)PyCapsule_GetPointer(capsule, "dltensor");
            if (!m) { PyErr_Clear(); goto decline; }
            t = &m->dl_tensor;
            ctx = m;
            read_only = -1;
        }
    }

    if (want_writable && read_only == 1) {
        PyErr_SetString(PyExc_ValueError, "DLPack tensor is read-only");
        goto fail;
    }
    if (want_writable && read_only == -1)
        goto decline;

    if (t->device.device_type != C2PY_DLCPU) {
        PyErr_SetString(PyExc_TypeError,
            "DLPack: unsupported device (expected CPU)");
        goto fail;
    }

    /* Unsupported dtype (e.g. float16): decline, PEP 3118 may know it. */
    format_char = c2py_dl_format_char(&t->dtype);
    if (!format_char)
        goto decline;

    if (t->ndim < 0 || t->ndim > C2PY_MAX_NDIM) {
        PyErr_SetString(PyExc_ValueError,
            "DLPack: ndim out of range (0 to C2PY_MAX_NDIM)");
        goto fail;
    }
    if (t->ndim > 0 && t->shape == NULL) {
        PyErr_SetString(PyExc_ValueError, "DLPack: ndim > 0 but shape is NULL");
        goto fail;
    }
    /* byte_offset is int64: on ILP32 it must fit Py_ssize_t, and the
     * offset pointer must not wrap. */
    if (t->byte_offset < 0 || t->byte_offset > (int64_t)C2PY_SSIZE_T_MAX ||
        (uintptr_t)t->data > (uintptr_t)-1 - (uintptr_t)t->byte_offset)
        goto too_large;

    pin->format_buf[0] = format_char;
    pin->format_buf[1] = '\0';
    info->format   = pin->format_buf;
    info->itemsize = t->dtype.bits / 8;  /* lanes == 1, bits in 8..64 */
    info->ptr      = (char*)t->data + t->byte_offset;
    info->ndim     = t->ndim;

    nelem = 1;
    if (t->ndim > 0) {
        int d;
        /* DLPack shapes/strides are int64_t, but Py_ssize_t is only 32-bit
         * on ILP32.  Copy into the pinned buffers instead of aliasing the
         * int64 array, otherwise multi-dimension shapes read wrong values.
         * Range-check in int64 before narrowing: on ILP32 a shape, stride
         * or byte count above 2 GiB would otherwise wrap silently.  A
         * zero-length dimension makes nelem 0 whatever the other dims. */
        for (d = 0; d < t->ndim; d++) {
            if (t->shape[d] < 0 || t->shape[d] > (int64_t)C2PY_SSIZE_T_MAX)
                goto too_large;
            if (t->shape[d] == 0)
                nelem = 0;
            pin->shape_buf[d] = (Py_ssize_t)t->shape[d];
        }
        if (nelem != 0) {
            for (d = 0; d < t->ndim; d++) {
                if (nelem > C2PY_SSIZE_T_MAX / pin->shape_buf[d])
                    goto too_large;
                nelem *= pin->shape_buf[d];
            }
            if (nelem > C2PY_SSIZE_T_MAX / info->itemsize)
                goto too_large;
        }
        info->shape = pin->shape_buf;
        if (t->strides) {
            /* DLPack strides are in elements; convert to bytes. */
            int64_t lim = (int64_t)C2PY_SSIZE_T_MAX / info->itemsize;
            for (d = 0; d < t->ndim; d++) {
                if (t->strides[d] > lim || t->strides[d] < -lim)
                    goto too_large;
                pin->stride_buf[d] = (Py_ssize_t)t->strides[d] * info->itemsize;
            }
        } else {
            /* Implied C-contiguous strides: last dim = itemsize,
             * each preceding dim = product of later dims * itemsize.
             * With a zero-length dim the product can still overflow. */
            Py_ssize_t st = info->itemsize;
            for (d = t->ndim - 1; d >= 0; d--) {
                pin->stride_buf[d] = st;
                if (pin->shape_buf[d] != 0 &&
                    st > C2PY_SSIZE_T_MAX / pin->shape_buf[d])
                    goto too_large;
                st *= pin->shape_buf[d];
            }
        }
        info->strides = pin->stride_buf;
    } else {
        info->shape   = NULL;
        info->strides = NULL;
    }
    info->len = nelem * info->itemsize;
    if (info->len > 0 && t->data == NULL) {
        PyErr_SetString(PyExc_ValueError, "DLPack: non-empty tensor with NULL data");
        goto fail;
    }

    pin->ctx  = ctx;
    pin->kind = C2PY_PIN_DLPACK;

    /* Keep the capsule alive: decref'ing it would invoke the
     * capsule destructor, which might call the deleter prematurely.
     * Store it in buf.obj for later release. */
    pin->buf.obj = capsule;
    capsule = NULL;  /* ownership transferred to pin */

    Py_DECREF(dl_method);
    Py_DECREF(dl_args);
    /* capsule NOT decref'd -- stored in pin->buf.obj */
    return 0;

too_large:
    PyErr_SetString(PyExc_ValueError,
        "DLPack: shape, stride, offset or size does not fit in Py_ssize_t");
    goto fail;
decline:
    PyErr_Clear();
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
    pin->kind = C2PY_PIN_NONE;  /* defined even if every backend fails */
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
        /* A backend that declines leaves no exception; one that set an
         * exception found a real problem (read-only, malformed tensor).
         * Report it rather than call the next backend with an exception
         * pending (CPython debug builds assert on that). */
        if (C2PY.Err_Occurred())
            return -1;
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
extern C2PY_HIDDEN int _c2py_use_cycle_counter;

/* Detected CPU cycle counter frequency in Hz, probed once at init.
 * 0 if the platform does not support cycle counters or detection failed.
 * When the cycle counter is selected (use_cycle_counter=1), the active
 * frequency is copied to c2py_tick_frequency_hz. */
extern C2PY_HIDDEN uint64_t c2py_cycle_counter_frequency_hz;

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

/* Convert counter ticks to nanoseconds given the counter frequency in Hz.
 * Returns ticks * 1e9 / freq_hz, split into whole seconds and remainder so
 * that absolute counter values (e.g. QueryPerformanceCounter since boot)
 * do not overflow; exact for any freq_hz below ~1.8e10 Hz.
 * Safe against freq_hz == 0 (returns 0).
 */
static inline uint64_t c2py_ticks_to_ns(uint64_t ticks, uint64_t freq_hz) {
    if (freq_hz == 0) return 0;
    return (ticks / freq_hz) * 1000000000ULL +
           (ticks % freq_hz) * 1000000000ULL / freq_hz;
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
        return c2py_ticks_to_ns((uint64_t)counter.QuadPart, (uint64_t)freq.QuadPart);
    }
#elif !defined(__STRICT_ANSI__)
    {
        struct timespec ts;
#if defined(__APPLE__) && defined(CLOCK_MONOTONIC_RAW)
        /* Darwin's CLOCK_MONOTONIC ticks in microseconds, so a short C
         * call measures as zero; the RAW clock has the full resolution. */
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
        clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
#else
    /* no clock_gettime -- fall back to 0 */
    return 0;
#endif
}


/* Active tick source frequency in Hz, set at init and updated when
 * _c2py_use_cycle_counter is toggled via __c2py_set_tick_source().
 * Default (wall clock): 1,000,000,000 (ns ticks).
 * Cycle counter mode: detected cycle counter frequency, or 0 if
 * the platform cannot be probed. */
extern C2PY_HIDDEN uint64_t c2py_tick_frequency_hz;

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
        p->t_c_min = ~(uint64_t)0;
        p->t_wrap_min = ~(uint64_t)0;
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

C2PY_HIDDEN int c2py_runtime_init(void);
/* Raise ImportError(reason) after c2py_runtime_init() failed. */
C2PY_HIDDEN void c2py_runtime_init_failed(const char *modname);

#ifdef __cplusplus
}
#endif

#endif /* C2PY_RUNTIME_H */
