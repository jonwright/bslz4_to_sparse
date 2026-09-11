/* c2py_dlsym.c - nimpy-style dlsym backend (runtime init)
 *
 * This file is #included by c2py_runtime.c when C2PY_USE_PYTHON_H is
 * NOT defined.  Do NOT compile this file directly.
 *
 * Contains: symbol resolution helpers, CPU feature probe, module init
 * helpers, and the main _c2py_runtime_init_once() function.
 */

/* ---- Private helpers ---- */

/* On Windows GetProcAddress returns a function-pointer type;
 * casting it to void* and back is inherent to the nimpy trick. */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4152)
#endif
static int _resolve(void **ptr, const char *name)
{
    *ptr = C2PY_RESOLVE(C2PY.dl_handle, name);
    if (*ptr == NULL) {
        /* Try PyPy symbol prefix: "Py_Xxx" -> "PyPy_Xxx",
         *                           "_Py_Xxx" -> "_PyPy_Xxx".
         * PyPy's cpyext exports all CPython API symbols under
         * the PyPy_* prefix from libpypy3.X-c.so.  This
         * fallback lets a single .so work on both runtimes. */
        char pypy_name[256];
        int prefix_len = (name[0] == '_') ? 3 : 2;
        snprintf(pypy_name, sizeof(pypy_name), "%.*sPy%s",
                 prefix_len, name, name + prefix_len);
        *ptr = C2PY_RESOLVE(C2PY.dl_handle, pypy_name);
        if (*ptr == NULL) {
            return -1;
        }
    }
    return 0;
}

/* Same as _resolve but returns the pointer directly (for raw C2PY_RESOLVE
 * call sites that need the PyPy fallback). */
static void *_resolve_raw(const char *name)
{
    void *p = C2PY_RESOLVE(C2PY.dl_handle, name);
    if (p) return p;
    char pypy_name[256];
    int prefix_len = (name[0] == '_') ? 3 : 2;
    snprintf(pypy_name, sizeof(pypy_name), "%.*sPy%s",
             prefix_len, name, name + prefix_len);
    return C2PY_RESOLVE(C2PY.dl_handle, pypy_name);
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define RESOLVE(ptr, name) _resolve((void**)&(ptr), name)
#define RESOLVE_REQ(ptr, name) do { \
    if (_resolve((void**)&(ptr), name) != 0) { \
        fprintf(stderr, "c2py_runtime: FATAL - missing symbol: %s\n", name); \
        return; \
    } \
} while(0)

/* Python 2.7 module init helper */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4152)
#endif
static PyObject*
_init_module_2_7(const char *name, PyMethodDef *methods)
{
    /* Try Py_InitModule4 first (Python 2.7 preferred) */
    typedef PyObject* (*init4_fn)(const char*, PyMethodDef*, const char*,
                                   PyObject*, int);
    init4_fn fn4 = (init4_fn)_resolve_raw("Py_InitModule4_64");
    if (fn4 == NULL) fn4 = (init4_fn)_resolve_raw("Py_InitModule4");
    if (fn4 != NULL) {
        return fn4(name, methods, NULL, NULL, 1013 /* PYTHON_API_VERSION */);
    }

    /* Fallback: Py_InitModule3 */
    typedef PyObject* (*init3_fn)(const char*, PyMethodDef*, const char*);
    init3_fn fn3 = (init3_fn)_resolve_raw("Py_InitModule3");
    if (fn3 != NULL) {
        return fn3(name, methods, NULL);
    }

    /* PyPy 2.7 cpyext: no Py_InitModule*, use PyModule_New + manual
     * method registration via PyModule_GetDict + PyDict_SetItemString.
     * Resolve these symbols on demand (only needed for this fallback). */
    {
        typedef PyObject* (*mod_new_fn)(const char*);
        typedef PyObject* (*mod_dict_fn)(PyObject*);
        typedef PyObject* (*cfn_new_fn)(PyMethodDef*, PyObject*, PyObject*);
        typedef PyObject* (*imp_add_fn)(const char*);

        imp_add_fn imp_add = (imp_add_fn)_resolve_raw("PyImport_AddModule");
        mod_new_fn mnew = imp_add
            ? NULL : (mod_new_fn)_resolve_raw("PyModule_New");
        mod_dict_fn mdict = (mod_dict_fn)_resolve_raw("PyModule_GetDict");
        cfn_new_fn cfn_new = (cfn_new_fn)_resolve_raw("PyCFunction_NewEx");

        /* Prefer PyImport_AddModule: it registers in sys.modules
         * and returns a module that supports attribute setting.
         * On PyPy 2.7 cpyext, PyModule_New creates a module that
         * cannot have attributes set via SetAttrString. */
        PyObject *module = imp_add ? imp_add(name) : (mnew ? mnew(name) : NULL);
        if (module && mdict && cfn_new) {
            PyObject *dict = mdict(module);
            if (dict) {
                while (methods && methods->ml_name) {
                PyObject *fn = cfn_new(methods, NULL, NULL);
                if (fn) {
                    if (C2PY._ds_set_item_string) {
                        typedef int (*dset_fn)(PyObject*, const char*, PyObject*);
                        ((dset_fn)C2PY._ds_set_item_string)(dict, methods->ml_name, fn);
                    }
                    Py_DECREF(fn);
                }
                    methods++;
                }
                return module;
            }
        }
    }

    fprintf(stderr, "c2py_runtime: could not find module init function\n");
    return NULL;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif




/* ---- Main init function ---- */

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4152)  /* GetProcAddress fn/data ptr cast */
#endif
static void _c2py_runtime_init_once(void)
{
    _c2py_init_result = -1;  /* assume failure until full success */

    /* CPU feature probing runs first -- does not depend on Python */
    _c2py_probe_cpu_features();

#ifdef _WIN32
    {
        static const char *candidates[] = {
            "python315.dll", "python314.dll", "python313.dll",
            "python312.dll", "python311.dll", "python310.dll",
            "python39.dll", "python38.dll", "python37.dll",
            "python36.dll", "python27.dll",
            "python3.dll",
            /* Free-threaded builds may use 't' suffix DLL names */
            "python315t.dll", "python314t.dll", "python313t.dll",
            "python3t.dll",
            /* PyPy on Windows exports cpyext symbols from libpypy3.X-c.dll */
            "libpypy3.11-c.dll", "libpypy3.10-c.dll",
            "libpypy3.9-c.dll",  "libpypy3-c.dll",
            "libpypy2.7-c.dll",  "libpypy-c.dll",
            NULL
        };
        int i;
        for (i = 0; candidates[i]; i++) {
            C2PY.dl_handle = (void*)GetModuleHandleA(candidates[i]);
            if (C2PY.dl_handle) break;
        }
        /* On free-threaded and some embedded builds, Python symbols
         * may be exported from the .exe itself rather than a separate
         * DLL.  GetModuleHandleA(NULL) returns the .exe handle. */
        if (C2PY.dl_handle == NULL) {
            C2PY.dl_handle = (void*)GetModuleHandleA(NULL);
        }
        /* Last resort: try to load python3.dll or python3XX.dll
         * explicitly via LoadLibraryA.  This works when the DLL exists
         * on disk but was not loaded as a dependency of the main
         * executable (e.g. free-threaded Python 3.14t/3.15t on Windows
         * where the executable may have a non-standard import table). */
        if (C2PY.dl_handle == NULL) {
            static const char *load_candidates[] = {
                "python3.dll",
                "python315.dll", "python314.dll",
                "python3t.dll",
                "python315t.dll", "python314t.dll",
                NULL
            };
            for (i = 0; load_candidates[i]; i++) {
                C2PY.dl_handle = (void*)LoadLibraryA(load_candidates[i]);
                if (C2PY.dl_handle) break;
            }
        }
        if (C2PY.dl_handle == NULL) {
            fprintf(stderr, "c2py_runtime: GetModuleHandle / LoadLibrary failed "
                    "(python3.dll not found). "
                    "GetLastError=%lu\n", GetLastError());
            fprintf(stderr, "c2py_runtime: interpreter may be statically "
                    "linked or embedded in an unusual host.\n");
            return;
        }
    }
#else
    C2PY.dl_handle = dlopen(NULL, RTLD_LAZY | RTLD_GLOBAL);
    if (C2PY.dl_handle == NULL) {
        fprintf(stderr, "c2py_runtime: dlopen(NULL) failed: %s\n", dlerror());
        fprintf(stderr, "c2py_runtime: interpreter may be statically linked "
                "(requires --enable-shared or export-dynamic).\n");
        return;
    }
    /* If no CPython symbols found, try PyPy's separate C library.
     * PyPy exports cpyext symbols with a PyPy_* prefix from
     * libpypy3.X-c.so rather than the main executable. */
    {
        void *test = dlsym(C2PY.dl_handle, "Py_GetVersion");
        test = test ? test : dlsym(C2PY.dl_handle, "PyPy_GetVersion");
        if (test == NULL) {
            static const char *pypy_libs[] = {
                "libpypy3.11-c.so", "libpypy3.10-c.so",
                "libpypy3.9-c.so",  "libpypy3-c.so",
                "libpypy2.7-c.so",  "libpypy-c.so", NULL
            };
            int i;
            for (i = 0; pypy_libs[i]; i++) {
                void *h = dlopen(pypy_libs[i], RTLD_LAZY | RTLD_GLOBAL);
                if (h) {
                    C2PY.dl_handle = h;
                    break;
                }
            }
        }
    }
#endif

    void *dl = C2PY.dl_handle;

    /* --- Detect Python version --- */
    {
        typedef const char* (*ver_fn)(void);
        ver_fn getver = (ver_fn)_resolve_raw("Py_GetVersion");
        if (getver) {
            const char *v = getver();
#ifdef _MSC_VER
            if (v) sscanf_s(v, "%d.%d", &C2PY.version_major, &C2PY.version_minor);
#else
            if (v) sscanf(v, "%d.%d", &C2PY.version_major, &C2PY.version_minor);
#endif
        }
        if (C2PY.version_major == 0) {
            /* Fallback: check for Py3-only symbol */
            if (_resolve_raw("PyModule_Create2")) {
                C2PY.version_major = 3;
            } else {
                C2PY.version_major = 2;
            }
            C2PY.version_minor = 0;
        }
    }

    /* --- Detect PyPy cpyext ---
     * PyPy's cpyext emulates the CPython C API but with different struct
     * layouts: PyObject is 24 bytes (ob_pypy_link at +8, ob_type at +16),
     * and manual ob_refcnt manipulation is unsafe (must use PyPy_IncRef). */
    {
        void *test = C2PY_RESOLVE(dl, "PyPy_IncRef");
        C2PY.is_pypy = (test != NULL) ? 1 : 0;
    }
    if (C2PY.is_pypy) {
        C2PY.pyobject_size = 24;
        C2PY.ob_refcnt_offset = 0;
        C2PY.ob_type_offset = 16;  /* ob_pypy_link at 8 pushes ob_type to 16 */
    }

    /* --- Reject unsupported Python versions --- */
#ifdef _WIN32
    if (C2PY.version_major >= 3 && C2PY.version_minor > 15) {
        fprintf(stderr,
                "c2py_runtime: Python %d.%d on Windows is not yet supported.\n"
                "Supported versions: 2.7, 3.6-3.15.\n"
                "To add Windows support for a new Python version, audit the CPython\n"
                "headers for ABI changes and update checks in c2py_runtime.c.\n",
                C2PY.version_major, C2PY.version_minor);
        return;
    }
#else
    if (C2PY.version_major >= 3 && C2PY.version_minor > 15) {
        fprintf(stderr,
                "c2py_runtime: Python %d.%d is not supported.\n"
                "Supported versions: 2.7, 3.6-3.15.\n"
                "To add support for a new Python version, audit the CPython\n"
                "headers for ABI changes and update checks in c2py_runtime.c.\n",
                C2PY.version_major, C2PY.version_minor);
        return;
    }
#endif

    /* --- Detect free-threaded build --- */
    if (C2PY.is_pypy) goto ft_detection_done;
    {
        int found_ft = 0;
        const char *force_ft = getenv("C2PY_FORCE_FT");
        if (force_ft != NULL) {
            if (force_ft[0] == '1') {
                found_ft = 1;
            }
            C2PY.is_free_threaded = found_ft;
            goto ft_detection_done;
        }
        {
            typedef const char* (*ver_fn)(void);
            ver_fn getver = (ver_fn)C2PY_RESOLVE(dl, "Py_GetVersion");
            const char *vstr = getver ? getver() : "";
            if (vstr && strstr(vstr, "free-threading") != NULL)
                found_ft = 1;
        }
        if (!found_ft) {
            typedef int (*gil_check_fn)(void);
            gil_check_fn gilchk = (gil_check_fn)C2PY_RESOLVE(dl, "_Py_IsGILEnabled");
            if (gilchk && gilchk() == 0)
                found_ft = 1;
        }
        C2PY.is_free_threaded = found_ft;
    }
    ft_detection_done:
        /* Nothing to do here */

    /* --- Set ABI layout --- */
    if (C2PY.is_pypy) {
        /* already set above */
    } else if (C2PY.is_free_threaded) {
        C2PY.pyobject_size = 32;
        C2PY.ob_refcnt_offset = 16;
        C2PY.ob_type_offset = 24;
    } else {
        C2PY.pyobject_size = sizeof(PyObject);
        C2PY.ob_refcnt_offset = 0;
        C2PY.ob_type_offset = sizeof(Py_ssize_t);
    }
    C2PY.pyobject_size_ft = 32;

    assert(C2PY.pyobject_size > 0);
    assert(C2PY.ob_refcnt_offset >= 0);
    assert(C2PY.ob_type_offset >= 0);
    assert(C2PY.ob_type_offset + (Py_ssize_t)sizeof(void*) <= C2PY.pyobject_size);

    {
        Py_ssize_t sz_gil = sizeof(PyModuleDef);
        Py_ssize_t sz_ft  = sizeof(PyModuleDef_FT);
        C2PY.pymoduledef_max_size = (sz_gil > sz_ft) ? sz_gil : sz_ft;
    }

    /* --- Buffer protocol (required) --- */
    RESOLVE_REQ(C2PY.GetBuffer, "PyObject_GetBuffer");
    RESOLVE_REQ(C2PY.ReleaseBuffer, "PyBuffer_Release");
    if (C2PY.GetBuffer == NULL || C2PY.ReleaseBuffer == NULL) return;

    C2PY.AsReadBuffer = (int (*)(PyObject*, const void**, Py_ssize_t*))
        _resolve_raw("PyObject_AsReadBuffer");
    C2PY.AsWriteBuffer = (int (*)(PyObject*, void**, Py_ssize_t*))
        _resolve_raw("PyObject_AsWriteBuffer");
    C2PY.Err_Clear = (void (*)(void))_resolve_raw("PyErr_Clear");
    RESOLVE_REQ(C2PY.Err_Clear, "PyErr_Clear");
    if (C2PY.Err_Clear == NULL) return;
    C2PY.buffer_api_is_pep3118 = (C2PY.version_major >= 3);

    C2PY.pybuffer_size = (C2PY.version_major >= 3)
        ? C2PY_PYBUFFER_SZ_POST312 : C2PY_PYBUFFER_SZ_PRE312;

    C2PY.use_fastcall = (C2PY.version_major >= 3 && C2PY.version_minor >= 12);

    RESOLVE_REQ(C2PY.ParseTuple, "PyArg_ParseTuple");
    if (C2PY.ParseTuple == NULL) return;

    RESOLVE_REQ(C2PY.Err_Occurred, "PyErr_Occurred");
    if (C2PY.Err_Occurred == NULL) return;

    RESOLVE_REQ(C2PY.Long_FromLong, "PyLong_FromLong");
    RESOLVE_REQ(C2PY.Long_FromLongLong, "PyLong_FromLongLong");
    RESOLVE_REQ(C2PY.Long_FromUnsignedLongLong, "PyLong_FromUnsignedLongLong");
    RESOLVE_REQ(C2PY.Float_FromDouble, "PyFloat_FromDouble");
    if (C2PY.Long_FromLong == NULL || C2PY.Float_FromDouble == NULL) return;

    /* Runtime PyObject layout probe */
    {
        PyObject *tmp = C2PY.Long_FromLong(1);
        if (tmp) {
            void *p_type = *(void**)((char*)tmp + sizeof(Py_ssize_t));
            if (p_type != NULL && (uintptr_t)p_type >= 0x100000) {
                /* GIL layout confirmed */
            } else {
                C2PY.is_free_threaded = 1;
                C2PY.pyobject_size = 32;
                C2PY.ob_refcnt_offset = 16;
                C2PY.ob_type_offset = 24;
            }
            {
                typedef void (*dref_fn)(PyObject*);
                dref_fn dref = (dref_fn)_resolve_raw("Py_DecRef");
                if (!dref) dref = (dref_fn)_resolve_raw("_Py_DecRef");
                if (dref) dref(tmp);
            }
        }
    }

    RESOLVE_REQ(C2PY.Tuple_New, "PyTuple_New");
    RESOLVE_REQ(C2PY.Tuple_SetItem, "PyTuple_SetItem");
    if (C2PY.Tuple_New == NULL || C2PY.Tuple_SetItem == NULL) return;

    RESOLVE(C2PY.Bytes_FromStringAndSize, "PyBytes_FromStringAndSize");

    RESOLVE_REQ(C2PY.Long_AsLong, "PyLong_AsLong");
    RESOLVE_REQ(C2PY.Long_AsLongLong, "PyLong_AsLongLong");
    RESOLVE_REQ(C2PY.Float_AsDouble, "PyFloat_AsDouble");
    if (C2PY.Long_AsLong == NULL || C2PY.Long_AsLongLong == NULL ||
        C2PY.Float_AsDouble == NULL) return;

    RESOLVE_REQ(C2PY.exc_TypeError, "PyExc_TypeError");
    RESOLVE_REQ(C2PY.exc_ValueError, "PyExc_ValueError");
    RESOLVE_REQ(C2PY.exc_RuntimeError, "PyExc_RuntimeError");
    RESOLVE_REQ(C2PY.exc_MemoryError, "PyExc_MemoryError");
    RESOLVE_REQ(C2PY.Err_SetString, "PyErr_SetString");
    RESOLVE_REQ(C2PY.Err_Format, "PyErr_Format");
    if (C2PY.exc_TypeError   == NULL || C2PY.exc_ValueError  == NULL ||
        C2PY.exc_RuntimeError == NULL || C2PY.exc_MemoryError == NULL ||
        C2PY.Err_SetString    == NULL || C2PY.Err_Format      == NULL) return;

    C2PY.exc_TypeError = *(void **)C2PY.exc_TypeError;
    C2PY.exc_ValueError = *(void **)C2PY.exc_ValueError;
    C2PY.exc_RuntimeError = *(void **)C2PY.exc_RuntimeError;
    C2PY.exc_MemoryError = *(void **)C2PY.exc_MemoryError;
    C2PY.exc_BufferError = C2PY.exc_BufferError ? *(void **)C2PY.exc_BufferError : NULL;

    {
        void *mc = _resolve_raw("PyModule_Create2");
        C2PY.Module_Create2 = (PyObject* (*)(PyModuleDef*, int))mc;
    }
    C2PY.InitModule_2_7 = _init_module_2_7;

    {
        if (_resolve((void**)&C2PY.IncRef, "Py_IncRef") != 0)
            _resolve((void**)&C2PY.IncRef, "_Py_IncRef");
        if (_resolve((void**)&C2PY.DecRef, "Py_DecRef") != 0)
            _resolve((void**)&C2PY.DecRef, "_Py_DecRef");

        if (C2PY.is_free_threaded) {
            if (C2PY.IncRef == NULL || C2PY.DecRef == NULL) {
                fprintf(stderr, "c2py_runtime: FATAL - free-threaded build "
                        "requires Py_IncRef / Py_DecRef symbols\n");
                return;
            }
        } else {
            if (C2PY.IncRef == NULL)
                C2PY.IncRef = _c2py_inc_ref_manual;
            if (C2PY.DecRef == NULL)
                C2PY.DecRef = _c2py_dec_ref_manual;
        }
    }

    RESOLVE_REQ(C2PY.SetAttrString, "PyObject_SetAttrString");
    if (C2PY.SetAttrString == NULL) return;
    RESOLVE_REQ(C2PY.GetAttrString, "PyObject_GetAttrString");
    if (C2PY.GetAttrString == NULL) return;
    RESOLVE(C2PY.Module_GetDict, "PyModule_GetDict");

    {
        void *p = C2PY_RESOLVE(dl, "PyPyDict_SetItemString");
        if (!p) p = C2PY_RESOLVE(dl, "PyDict_SetItemString");
        C2PY._ds_set_item_string = p;
    }
    C2PY._ds_pypy_workaround = (C2PY._ds_set_item_string != NULL
                                && C2PY.version_major < 3) ? 1 : 0;

    C2PY.CallObject = (PyObject*(*)(PyObject*, PyObject*))
        _resolve_raw("PyObject_CallObject");
    C2PY.Capsule_GetPointer = (void*(*)(PyObject*, const char*))
        _resolve_raw("PyCapsule_GetPointer");
    C2PY.exc_BufferError = (void*)_resolve_raw("PyExc_BufferError");

    RESOLVE_REQ(C2PY.Long_FromVoidPtr, "PyLong_FromVoidPtr");
    if (C2PY.Long_FromVoidPtr == NULL) return;

    RESOLVE_REQ(C2PY.SaveThread, "PyEval_SaveThread");
    RESOLVE_REQ(C2PY.RestoreThread, "PyEval_RestoreThread");
    if (C2PY.SaveThread == NULL || C2PY.RestoreThread == NULL) return;
    RESOLVE(C2PY.Unstable_Module_SetGIL, "PyUnstable_Module_SetGIL");

    {
        void *none = _resolve_raw("_Py_NoneStruct");
        if (none == NULL) {
            void **pnone = (void**)_resolve_raw("Py_None");
            if (pnone) none = *pnone;
        }
        C2PY.none_obj = (PyObject*)none;
        if (C2PY.none_obj == NULL) {
            fprintf(stderr, "c2py_runtime: could not resolve Py_None\n");
            return;
        }
        {
            Py_ssize_t before, after;
            before = *(Py_ssize_t*)((char*)C2PY.none_obj + C2PY.ob_refcnt_offset);
            C2PY.IncRef(C2PY.none_obj);
            after = *(Py_ssize_t*)((char*)C2PY.none_obj + C2PY.ob_refcnt_offset);
            C2PY.none_immortal = (before == after) ? 1 : 0;
            if (!C2PY.none_immortal) {
                C2PY.DecRef(C2PY.none_obj);
            }
        }
    }

    /* Runtime Py_buffer size probe */
    if (!C2PY.is_pypy) {
        unsigned char probe[96];
        Py_buffer *pb = (Py_buffer*)probe;
        typedef PyObject* (*bytes_fn)(const char*, Py_ssize_t);
        bytes_fn mkbytes = (bytes_fn)_resolve_raw("PyBytes_FromStringAndSize");
        if (!mkbytes)
            mkbytes = (bytes_fn)_resolve_raw("PyString_FromStringAndSize");
        PyObject *by = mkbytes ? mkbytes("x", 1) : NULL;
        if (by) {
            memset(probe, 0xAA, sizeof(probe));
            if (C2PY.GetBuffer(by, pb, PyBUF_STRIDES | PyBUF_FORMAT) == 0) {
                {
                    Py_ssize_t internal_off;
#if defined(__LP64__) || defined(_WIN64)
                    internal_off = 72;
#else
                    internal_off = 40;
#endif
                    if (*((char*)pb + internal_off) == 0)
                        C2PY.pybuffer_size = C2PY_PYBUFFER_SZ_POST312;
                    else
                        C2PY.pybuffer_size = C2PY_PYBUFFER_SZ_PRE312;
                }
                C2PY.ReleaseBuffer(pb);
            }
            C2PY.DecRef(by);
        }
    }

    _c2py_init_result = 0;
    _c2py_runtime_initialized = 1;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif
