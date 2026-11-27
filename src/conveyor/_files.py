"""File helpers the conveyor's readers and writers share, in one spelling for Python 2.7 and 3."""

from __future__ import absolute_import, division, print_function

import contextlib
import errno
import os
import shutil
import tempfile


def makedirs(d):
    """os.makedirs(d, exist_ok=True)"""
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST or not os.path.isdir(d):
            raise


# Atomically put a file in place of another.  Python 2 has no os.replace; its os.rename does the
# same on POSIX, where the conveyor runs.
replace = getattr(os, "replace", os.rename)


def not_found(e):
    """True when e is an IOError/OSError for a missing file (Python 3's FileNotFoundError)."""
    return getattr(e, "errno", None) == errno.ENOENT


@contextlib.contextmanager
def temporary_directory():
    """tempfile.TemporaryDirectory(), as a context manager yielding the path."""
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
