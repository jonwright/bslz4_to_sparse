"""AGENTS.md 6: every module of the shared Python 2.7/3 subset starts the same way, so that
division, print and imports mean the same on both."""

from __future__ import absolute_import, division, print_function

import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE = "from __future__ import absolute_import, division, print_function"
SHARED = ["src/*.py", "src/*/*.py", "test/*.py", "examples/*.py", "tools/build_extension.py",
          "tools/index_chunks.py"]


def test_every_shared_module_has_the_future_line():
    paths = sorted(p for g in SHARED for p in glob.glob(os.path.join(REPO, g)))
    assert len(paths) > 50
    missing = []
    for p in paths:
        with open(p) as f:
            if LINE not in [line.strip() for line in f]:
                missing.append(os.path.relpath(p, REPO))
    assert not missing, missing
