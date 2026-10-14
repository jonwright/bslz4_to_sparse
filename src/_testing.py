"""Test instrumentation (not public API): per block counters of which step values ran
(src/pipeline/registry.c).  They are off in a fresh process, because decoding threads would
share their cache lines; reset_counters() zeroes and starts them, stop_counters() stops them,
and counting() does both around a block."""

import contextlib

import numpy as np

from . import _ext
from ._pipeline import STEPS, NAMES

VALUE_SLOTS = 16  # BSLZ4_VALUE_SLOTS (src/pipeline/pipeline.h)


def reset_counters():
    """Zero the counters and start counting."""
    _ext.reset_counters()


def stop_counters():
    _ext.set_counters(0)


@contextlib.contextmanager
def counting():
    """Count the blocks run inside the with block (zeroed on entry, off on exit)."""
    reset_counters()
    try:
        yield
    finally:
        stop_counters()


def counters():
    """{step: {value name: blocks}} for the values that ran since the reset."""
    buf = np.zeros(len(STEPS) * VALUE_SLOTS, np.uint64)
    _ext.read_counters(buf)
    buf = buf.reshape(len(STEPS), VALUE_SLOTS)
    return {
        s: {NAMES[s][v]: int(buf[i, v]) for v in range(1, len(NAMES[s])) if buf[i, v]}
        for i, s in enumerate(STEPS)
    }
