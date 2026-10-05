"""The third-party licence texts in licenses/ must match the submodules'
own files (a submodule bump must not leave a stale copy), and every
component compiled into the extension must have one."""

import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPIES = {
    "LZ4-LICENSE": "lz4/lib/LICENSE",
    "ZSTD-LICENSE": "zstd/LICENSE",
    "BITSHUFFLE-LICENSE": "bitshuffle/LICENSE",
    "KCB-LICENSE-MIT": "kcb/LICENSE-MIT",
    "KCB-LICENSE-APACHE": "kcb/LICENSE-APACHE",
}


def test_every_licence_present():
    for name in list(COPIES) + ["C2PY23-LICENSE", "README.md"]:
        assert os.path.getsize(os.path.join(REPO, "licenses", name)) > 100, name


@pytest.mark.parametrize("name,src", sorted(COPIES.items()))
def test_copy_matches_submodule(name, src):
    path = os.path.join(REPO, src)
    if not os.path.exists(path):          # sdist without the submodule's licence file
        pytest.skip("%s not present" % src)
    with open(path, "rb") as a, open(os.path.join(REPO, "licenses", name), "rb") as b:
        assert a.read() == b.read(), "licenses/%s differs from %s" % (name, src)
