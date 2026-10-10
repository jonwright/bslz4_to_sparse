#!/usr/bin/env python3
"""Check the documentation rules of docs/AGENTS.md (fast: runs no examples).

  1. nothing under docs/gen/ is tracked by git, and docs/gen/ is ignored
  2. docs/index.md holds no code blocks and no measured numbers
  3. every examples/ex*.py has a title docstring and is in the mkdocs nav,
     and every example page in the nav has its script
  4. every public function, class and method has a docstring
  5. docs/bench/real_data.json covers every case and operation of
     tools/bench_docs.py, with every figure present

Exits non-zero listing every problem found.

    BSLZ4_TO_SPARSE_PATH=lib python3 tools/verify_docs.py
"""
import ast
import glob
import inspect
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)
sys.path.insert(0, os.path.join(REPO, "tools"))

import generate_docs as gd  # noqa: E402  (imports bslz4_to_sparse as gd.b)

problems = []


def check_gen_untracked():
    tracked = subprocess.run(["git", "ls-files", "docs/gen"], cwd=REPO, stdout=subprocess.PIPE,
                             universal_newlines=True).stdout.split()
    if tracked:
        problems.append("generated pages are tracked by git: %s" % tracked)
    r = subprocess.run(["git", "check-ignore", "-q", "docs/gen/api.md"], cwd=REPO)
    if r.returncode != 0:
        problems.append("docs/gen/ is not git-ignored")


# measured-looking numbers: a value with a speed, time or size unit
_MEASURED = re.compile(r"\d\s*(frames/s|fps|[kMG]B/s|ms\b|us\b|µs|[kMG]B\b|x faster|%)", re.I)


def check_index():
    with open(os.path.join(REPO, "docs", "index.md")) as fh:
        text = fh.read()
    if "```" in text or re.search(r"^( {4}|\t)\S", text, re.M):
        problems.append("docs/index.md has a code block: code belongs in examples/ex*.py")
    for m in _MEASURED.finditer(text):
        problems.append("docs/index.md has a measured-looking number %r: numbers come "
                        "from docs/bench/*.json" % m.group(0))


def check_examples():
    with open(os.path.join(REPO, "docs", "mkdocs.yml")) as fh:
        nav = set(re.findall(r"gen/examples/(\w+)\.md", fh.read()))
    scripts = {os.path.splitext(os.path.basename(p))[0]: p for p in gd.examples()}
    for name, path in sorted(scripts.items()):
        with open(path) as fh:
            mod = ast.parse(fh.read())
        doc = ast.get_docstring(mod)
        if not doc or not doc.splitlines()[0].strip():
            problems.append("%s has no title docstring" % os.path.relpath(path, REPO))
        if name not in nav:
            problems.append("%s is not in the nav of docs/mkdocs.yml" % name)
    for name in sorted(nav - set(scripts)):
        problems.append("docs/mkdocs.yml lists gen/examples/%s.md but examples/%s.py "
                        "does not exist" % (name, name))


def check_docstrings():
    b = gd.b
    names = [(n, getattr(b, n)) for n in b.__all__ if callable(getattr(b, n))]
    for cname in gd.PUBLIC_CLASSES:
        cls = getattr(b, cname)
        names += [("%s.%s" % (cname, m), getattr(cls, m)) for m in gd.PUBLIC_METHODS]
    for name, obj in names:
        if not inspect.getdoc(obj):
            problems.append("%s has no docstring" % name)


def _bench_spec():
    """CASES keys and OPS of tools/bench_docs.py, read without importing it."""
    with open(os.path.join(REPO, "tools", "bench_docs.py")) as fh:
        mod = ast.parse(fh.read())
    spec = {}
    for node in mod.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("CASES", "OPS"):
                spec[node.targets[0].id] = ast.literal_eval(node.value)
    return [c["key"] for c in spec["CASES"]], list(spec["OPS"])


def check_bench():
    try:
        with open(gd.BENCH) as fh:
            bench = json.load(fh)
    except (OSError, ValueError) as e:
        problems.append("docs/bench/real_data.json: %s" % e)
        return
    keys, ops = _bench_spec()
    have = {(r["case"], r["op"]) for r in bench.get("results", [])}
    for k in keys:
        for op in ops:
            if (k, op) not in have:
                problems.append("docs/bench/real_data.json has no %s / %s" % (k, op))
    for r in bench.get("results", []):
        for f in ("fps", "compressed_gbs", "pixel_gbs"):
            if not isinstance(r.get(f), (int, float)) or r[f] <= 0:
                problems.append("docs/bench/real_data.json %s / %s: bad %s"
                                % (r.get("case"), r.get("op"), f))
    for f in ("cpu", "cores_usable", "git", "src_sha256", "py_sha256"):
        if f not in bench.get("machine", {}):
            problems.append("docs/bench/real_data.json machine has no %s" % f)


def main():
    check_gen_untracked()
    check_index()
    check_examples()
    check_docstrings()
    check_bench()
    for p in problems:
        print("FAIL:", p)
    print("%d problem(s)" % len(problems) if problems else "docs OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
