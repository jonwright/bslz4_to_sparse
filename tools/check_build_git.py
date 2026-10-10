#!/usr/bin/env python3
"""Fail unless a built extension carries a clean git description.

The "git" field of build_info() is how a wheel says which commit it was
built from.  A CI build that cannot run git (a container refusing a
checkout owned by another user, or a shallow clone without tags) embeds
None or a bare hash, and nothing else notices: 0.0.21a1 went to PyPI that
way.  Run after each CI build:

    python tools/check_build_git.py <.so/.pyd> [expected]

It reads the description from the file (without loading it) and fails if
it is missing or "-dirty"; with `expected` (the tag of a release build),
the description must equal it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_extension import embedded_git  # noqa: E402


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    path = sys.argv[1]
    expected = sys.argv[2] if len(sys.argv) == 3 and sys.argv[2] else None
    got = embedded_git(path)
    print("%s: git %r%s" % (os.path.basename(path), got,
                            "" if expected is None else ", expected %r" % expected))
    if not got:
        raise SystemExit("no git description embedded: git did not run in the build "
                         "(container ownership, or no .git?)")
    if got.endswith("-dirty"):
        raise SystemExit("the build is -dirty: a compiled source differs from the commit")
    if expected is not None and got != expected:
        raise SystemExit("the build describes itself as %r, not the release tag %r "
                         "(a shallow checkout without tags?)" % (got, expected))


if __name__ == "__main__":
    main()
