"""c2py_loader - Load a c2py23-native .so by explicit filename.

Convention: <module>.c2py23-<os>_<arch>.so

Example files:
    _mymodule.c2py23-linux_x86_64.so
    _mymodule.c2py23-linux_ppc64le.so
    _mymodule.c2py23-linux_aarch64.so
    _mymodule.c2py23-win_amd64.pyd
    _mymodule.c2py23-darwin_arm64.so

No monkeypatching of EXTENSION_SUFFIXES.  No sys.path hacking.
Loads the .so by full path via ExtensionFileLoader (Python 3.x) or
imp.load_dynamic (Python 2.7).

Usage in your package's __init__.py:

    from c2py23.c2py_loader import load_native
    import os as _os
    _mod = load_native(_os.path.dirname(_os.path.abspath(__file__)),
                       '_mymodule')
    # Re-export public names (skip dunders, keep single-underscore API)
    for _k, _v in _mod.__dict__.items():
        if _k.startswith('__') and _k.endswith('__'):
            continue
        globals()[_k] = _v

Users who do not want a c2py23 runtime dependency can copy the
function body into their own __init__.py.  The code is intentionally
small and self-contained.
"""

from __future__ import print_function

import os
import sys
import platform as _platform


def _platform_key(target="cpython"):
    """Return 'linux_x86_64', 'win_amd64', 'wasm_wasm32', etc."""
    _os = sys.platform
    if _os == "linux2":
        _os = "linux"
    elif _os == "win32":
        _os = "win"
    elif _os == "emscripten":
        _os = "wasm"
    _arch = _platform.machine()
    if _arch == "AMD64":
        if _os == "win":
            _arch = "amd64"
        else:
            _arch = "x86_64"
    elif _arch in ("wasm32", "wasm"):
        _arch = "wasm32"
    if target == "pypy":
        return "pypy-%s_%s" % (_os, _arch)
    return "%s_%s" % (_os, _arch)


def load_native(package_dir, module_name="_native", tag="c2py23"):
    """Load <module_name>.<tag>-<platform_key>.so from package_dir.

    Args:
    package_dir: Absolute path to the package directory containing
                 the .so files.
    module_name: Base module name (default '_native').
                 Must match the c2py23 'module:' field.
                 Use a unique name per package (e.g. '_mymodule')
                 to avoid collisions in sys.modules.
        tag:         Tag string inserted in the filename (default 'c2py23').

    Returns:
        The loaded module object.  The caller should re-export public
        names from it.

    Example:
        _mod = load_native(os.path.dirname(__file__), '_mymodule')
    """
    _key = _platform_key()
    if os.name == "nt":
        _ext = ".pyd"
    else:
        _ext = ".so"
    _filename = "%s.%s-%s%s" % (module_name, tag, _key, _ext)
    _path = os.path.join(package_dir, _filename)

    _trace = os.environ.get("C2PY_TRACE")
    if _trace:
        print("[c2py_loader] platform=%s file=%s" % (_key, _filename), file=sys.stderr)

    if not os.path.isfile(_path):
        _alternatives = [f for f in os.listdir(package_dir) if f.startswith(module_name + ".")]
        _hint = ", ".join(sorted(_alternatives)) if _alternatives else "none"
        raise ImportError(
            "c2py23: native module not found for platform '%s'\n"
            "  Expected: %s\n"
            "  Available: %s\n"
            "  Build with: gcc -shared -fPIC ... -o %s" % (_key, _filename, _hint, _path)
        )

    if _trace:
        print("[c2py_loader] loading %s -> %s" % (module_name, _path), file=sys.stderr)

    if sys.version_info[0] >= 3:
        import importlib.machinery
        import importlib.util

        loader = importlib.machinery.ExtensionFileLoader(module_name, _path)
        spec = importlib.util.spec_from_file_location(module_name, _path, loader=loader)
        # Warn if overwriting an existing module in sys.modules.
        # Using distinct module names per package (e.g. '_mymodule'
        # instead of '_native') avoids collisions.
        #
        # LOCAL PATCH (not upstream v0.5.4): sample sys.modules before
        # module_from_spec(), which itself registers single-phase-init
        # extensions -- what c2py23 generates -- so sampling after it
        # always finds a module and never detects anything.
        _existing = sys.modules.get(module_name)
        mod = importlib.util.module_from_spec(spec)
        if _existing is not None and _existing is not mod:
            import warnings as _w

            _w.warn(
                "c2py_loader: overwriting existing module '%s' "
                "in sys.modules. Use a unique module name." % module_name
            )
        sys.modules[module_name] = mod
        loader.exec_module(mod)
        return mod
    else:
        import imp

        return imp.load_dynamic(module_name, _path)
