#!/usr/bin/env python3
"""Notebooks from the examples: each examples/ex*.py as a Jupyter notebook, run, with its figures.

    python3 tools/make_notebooks.py [--run] [--out DIR] [examples/ex1_sparse_pixels.py ...]

An example is a plain script: a docstring (title, then what it shows) and code.  The notebook is
made from it mechanically, so the script stays the one source:

  * the docstring becomes a markdown cell (its first line the heading);
  * the code is cut between top-level statements at blank lines, one code cell for each block;
  * a block of full-line comments (# ...) that leads a block becomes a markdown cell above it;
  * a figure is whatever matplotlib draws: the notebook keeps what the kernel displays.

--run executes the notebook (nbclient, a python3 kernel) in the directory holding the files the
example is written for, and saves the outputs in the notebook.  tools/generate_docs.py uses
notebook_to_markdown for the documentation pages, so output and pictures are shown there too.
Needs nbformat, and with --run nbclient and ipykernel; matplotlib for the figures.
"""

import argparse
import ast
import base64
import inspect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def split_example(source):
    """(title, body, blocks) of an example's source: its docstring, and the code as a list of
    ("markdown", text) and ("code", text) blocks in order."""
    mod = ast.parse(source)
    doc = inspect.cleandoc(mod.body[0].value.value)
    title, _, body = doc.partition("\n")
    lines = source.splitlines()
    stmts = mod.body[1:]
    # where each top-level statement starts, taking in the comment lines directly above it
    starts = []
    for node in stmts:
        first = node.lineno - 1
        if getattr(node, "decorator_list", None):
            first = min(first, min(d.lineno for d in node.decorator_list) - 1)
        while (
            first > 0
            and lines[first - 1].lstrip().startswith("#")
            and not lines[first - 1].startswith(" ")
        ):
            first -= 1
        starts.append(first)
    ends = [n.end_lineno for n in stmts]  # 1-based inclusive
    # a new block where a blank line lies between the previous statement and this one's start
    imports = (ast.Import, ast.ImportFrom)
    groups, cur = [], []
    for i, node in enumerate(stmts):
        blank = any(not lines[k].strip() for k in range(ends[i - 1], starts[i])) if i else False
        both_imports = i and isinstance(node, imports) and isinstance(stmts[i - 1], imports)
        if cur and blank and not both_imports:  # a run of imports stays one block
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    blocks = []
    for g in groups:
        text = "\n".join(lines[starts[g[0]] : ends[g[-1]]])
        # leading full-line comments are prose
        head = []
        rest = text.splitlines()
        while rest and rest[0].startswith("#"):
            head.append(re.sub(r"^#\s?", "", rest.pop(0)))
        if head:
            blocks.append(("markdown", "\n".join(head)))
        if rest:
            blocks.append(("code", "\n".join(rest)))
    return title.strip().rstrip("."), body.strip(), blocks


def build_notebook(path):
    """The (not yet run) nbformat notebook of an example script."""
    import nbformat
    from nbformat import v4

    with open(path) as fh:
        title, body, blocks = split_example(fh.read())
    cells = [v4.new_markdown_cell("# %s\n\n%s" % (title, body))]
    for kind, text in blocks:
        cells.append(v4.new_markdown_cell(text) if kind == "markdown" else v4.new_code_cell(text))
    nb = v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["source_script"] = os.path.relpath(path, REPO)
    nbformat.validate(nb)
    return nb


def run_notebook(nb, cwd, timeout=900):
    """Execute the notebook in cwd (nbclient); the outputs stay in it."""
    from nbclient import NotebookClient

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    NotebookClient(
        nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": cwd}}
    ).execute()
    return nb


def notebook_to_markdown(nb, figdir, figrel, skip_first=False):
    """Markdown of a run notebook: prose as it is, code in fences, printed output in text fences,
    figures saved in figdir and linked as figrel/figN.png."""
    out, nfig = [], 0
    for cell in nb.cells[1 if skip_first else 0 :]:
        src = cell.source.rstrip()
        if cell.cell_type == "markdown":
            out.append(src)
            continue
        out.append("```python\n%s\n```" % src)
        text = []
        for o in cell.get("outputs", []):
            if o.output_type == "stream":
                text.append(o.text)
            elif o.output_type == "error":
                text.append("\n".join(o.traceback))
            elif o.output_type in ("display_data", "execute_result"):
                data = o.get("data", {})
                if "image/png" in data:
                    nfig += 1
                    os.makedirs(figdir, exist_ok=True)
                    name = "fig%d.png" % nfig
                    with open(os.path.join(figdir, name), "wb") as fh:
                        fh.write(base64.b64decode(data["image/png"]))
                    if text:
                        out.append("```text\n%s\n```" % "".join(text).rstrip())
                        text = []
                    out.append("![figure %d](%s/%s)" % (nfig, figrel, name))
                elif "text/plain" in data:
                    text.append(data["text/plain"] + "\n")
        if text:
            out.append("```text\n%s\n```" % "".join(text).rstrip())
    return "\n\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples", nargs="*")
    ap.add_argument("--run", action="store_true", help="execute them, with the stand-in files")
    ap.add_argument("--out", default=os.path.join(REPO, "notebooks"))
    a = ap.parse_args()
    import glob
    import nbformat

    paths = a.examples or sorted(glob.glob(os.path.join(REPO, "examples", "ex*.py")))
    os.makedirs(a.out, exist_ok=True)
    folder = None
    for p in paths:
        nb = build_notebook(p)
        if a.run:
            if folder is None:
                import tempfile

                sys.path.insert(0, HERE)
                import standin

                folder = tempfile.mkdtemp(prefix="bslz4_notebooks_")
                standin.make_standin(folder)
            run_notebook(nb, folder)
        name = os.path.splitext(os.path.basename(p))[0] + ".ipynb"
        nbformat.write(nb, os.path.join(a.out, name))
        print("wrote", os.path.join(a.out, name))


if __name__ == "__main__":
    main()
