/* c2py_pythonh.c - pythonh backend (runtime init)
 *
 * This file is #included by c2py_runtime.c when C2PY_USE_PYTHON_H is
 * defined.  Do NOT compile this file directly.
 */

/* ---- C2PY global ---- */

c2py_api_t C2PY = {0};

/* ---- Python 2.7 module init helper ---- */

#if PY_MAJOR_VERSION == 2
static PyObject*
_c2py_pythonh_init_module(const char *name, PyMethodDef *methods)
{
    PyObject *m = Py_InitModule3(name, methods, "");
    if (m != NULL) {
        C2PY._ds_set_item_string = (void *)PyDict_SetItemString;
        C2PY._ds_pypy_workaround = 1;
    }
    return m;
}
#endif

/* ---- c2py_runtime_init ---- */

int c2py_runtime_init(void)
{
    /* CPU feature probing (shared with dlsym backend) */
    _c2py_probe_cpu_features();

    /* ---- Both Python 2 and 3 (common) ---- */

    C2PY.GetBuffer     = PyObject_GetBuffer;
    C2PY.ReleaseBuffer = PyBuffer_Release;
    C2PY.Err_Clear     = PyErr_Clear;
    C2PY.buffer_api_is_pep3118 = 1;
    C2PY.ParseTuple    = (int (*)(PyObject*, const char*, ...))PyArg_ParseTuple;
    C2PY.Long_FromLong      = PyLong_FromLong;
    C2PY.Float_FromDouble   = PyFloat_FromDouble;
    C2PY.Tuple_New          = PyTuple_New;
    C2PY.Tuple_SetItem      = PyTuple_SetItem;
    C2PY.Bytes_FromStringAndSize = PyBytes_FromStringAndSize;
    C2PY.Long_AsLong        = PyLong_AsLong;
    C2PY.Float_AsDouble     = PyFloat_AsDouble;
    C2PY.Err_SetString      = PyErr_SetString;
    C2PY.Err_Occurred       = PyErr_Occurred;
    C2PY.Err_Format         = (PyObject*(*)(PyObject*,const char*,...))PyErr_Format;
    C2PY.none_obj           = Py_None;
    C2PY.SetAttrString      = PyObject_SetAttrString;
    C2PY.GetAttrString      = PyObject_GetAttrString;
    C2PY.Module_GetDict     = PyModule_GetDict;
    C2PY._ds_set_item_string = (void *)PyDict_SetItemString;
    C2PY.CallObject         = PyObject_CallObject;
    C2PY.Long_FromVoidPtr   = PyLong_FromVoidPtr;
    C2PY.SaveThread         = (void*(*)(void))PyEval_SaveThread;
    C2PY.RestoreThread      = (void(*)(void*))PyEval_RestoreThread;
    C2PY.exc_TypeError      = (void *)PyExc_TypeError;
    C2PY.exc_ValueError     = (void *)PyExc_ValueError;
    C2PY.exc_RuntimeError   = (void *)PyExc_RuntimeError;
    C2PY.exc_MemoryError    = (void *)PyExc_MemoryError;

    C2PY.version_major      = PY_MAJOR_VERSION;
    C2PY.version_minor      = PY_MINOR_VERSION;
    C2PY.pybuffer_size      = sizeof(Py_buffer);
    C2PY.pyobject_size      = sizeof(PyObject);
    C2PY.pyobject_size_ft   = sizeof(PyObject);

#if defined(Py_GIL_DISABLED) && Py_GIL_DISABLED
    C2PY.ob_refcnt_offset   = offsetof(PyObject, ob_ref_shared);
#else
    C2PY.ob_refcnt_offset   = offsetof(PyObject, ob_refcnt);
#endif
    C2PY.ob_type_offset     = offsetof(PyObject, ob_type);

    c2py_tick_frequency_hz          = 1000000000ULL;

    /* ---- Python 2 defaults (overridden below for Python 3) ---- */

    C2PY.Long_FromLongLong         = NULL;
    C2PY.Long_FromUnsignedLongLong = NULL;
    C2PY.Long_AsLongLong           = NULL;
    C2PY.exc_BufferError           = NULL;
    C2PY.none_immortal             = 0;
    C2PY.Module_Create2            = NULL;
    C2PY.use_fastcall              = 0;
    C2PY.is_free_threaded          = 0;
    C2PY.Capsule_GetPointer        = NULL;
    C2PY.is_pypy                   = 0;
    C2PY.Unstable_Module_SetGIL    = NULL;
    C2PY._ds_pypy_workaround       = 0;
    C2PY.pymoduledef_max_size      = 0;
    c2py_cycle_counter_frequency_hz = 0;

#if PY_MAJOR_VERSION == 2
    C2PY.InitModule_2_7            = _c2py_pythonh_init_module;
    C2PY.IncRef                    = NULL;   /* manual refcount on 2.7 */
    C2PY.DecRef                    = NULL;
#else
    C2PY.InitModule_2_7            = NULL;
#endif

    /* ---- Python 3 overrides ---- */

#if PY_MAJOR_VERSION >= 3
    C2PY.Long_FromLongLong         = (PyObject*(*)(long long))PyLong_FromLongLong;
    C2PY.Long_FromUnsignedLongLong = (PyObject*(*)(unsigned long long))PyLong_FromUnsignedLongLong;
    C2PY.Long_AsLongLong           = (long long(*)(PyObject*))PyLong_AsLongLong;
    C2PY.Module_Create2            = (PyObject*(*)(PyModuleDef*,int))PyModule_Create2;
    C2PY.Capsule_GetPointer        = PyCapsule_GetPointer;
    C2PY.pymoduledef_max_size      = sizeof(PyModuleDef);

    if (PY_MINOR_VERSION >= 3) {
        C2PY.exc_BufferError       = (void *)PyExc_BufferError;
    }
    if (PY_MINOR_VERSION >= 7) {
        C2PY.use_fastcall          = 1;
    }
    if (PY_MINOR_VERSION >= 12) {
        C2PY.none_immortal         = 1;
#if defined(Py_IncRef)
        C2PY.IncRef                = Py_IncRef;
        C2PY.DecRef                = Py_DecRef;
#endif
    }

# if defined(Py_GIL_DISABLED)
    C2PY.is_free_threaded          = Py_GIL_DISABLED;
# endif
#endif /* PY_MAJOR_VERSION >= 3 */

    return 0;
}
