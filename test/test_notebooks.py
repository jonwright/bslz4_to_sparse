"""tools/make_notebooks.py: an example script becomes a notebook of the right cells."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
)
nbformat = pytest.importorskip("nbformat")

import make_notebooks as mn  # noqa: E402

SOURCE = '''"""Title of it.

What it shows,
in two lines.
"""

import os
import sys

# A comment block that
# is prose.
x = 1
if x:
    y = 2

    z = 3  # a blank line inside a block does not cut it
print(x)

# one more paragraph
def f():
    return 1


f()
'''


def test_split_example_cuts_at_top_level_blank_lines_only():
    title, body, blocks = mn.split_example(SOURCE)
    assert title == "Title of it" and body == "What it shows,\nin two lines."
    kinds = [k for k, _ in blocks]
    assert kinds == ["code", "markdown", "code", "markdown", "code", "code"]
    assert blocks[0][1] == "import os\nimport sys"  # a run of imports is one block
    assert blocks[1][1] == "A comment block that\nis prose."
    assert "    z = 3" in blocks[2][1] and blocks[2][1].endswith(
        "print(x)"
    )  # no cut inside a block
    assert blocks[3][1] == "one more paragraph"
    assert blocks[4][1].startswith("def f():") and blocks[5][1] == "f()"


def test_every_example_makes_a_valid_notebook():
    import glob

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in sorted(glob.glob(os.path.join(repo, "examples", "ex*.py"))):
        nb = mn.build_notebook(path)
        assert nb.cells[0].cell_type == "markdown" and nb.cells[0].source.startswith("# ")
        # the code of the cells, joined, is the script's code
        code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        compile(code, path, "exec")


def test_markdown_of_a_run_notebook_has_output_and_figures(tmp_path):
    import base64

    from nbformat import v4

    png = base64.b64encode(b"\x89PNG fake").decode()
    nb = v4.new_notebook(
        cells=[
            v4.new_markdown_cell("# T\n\nbody"),
            v4.new_code_cell("print(1)"),
        ]
    )
    nb.cells[1].outputs = [
        v4.new_output("stream", name="stdout", text="1\n"),
        v4.new_output("display_data", data={"image/png": png}),
    ]
    md = mn.notebook_to_markdown(nb, str(tmp_path / "f"), "f", skip_first=True)
    assert (
        "```python\nprint(1)\n```" in md
        and "```text\n1\n```" in md
        and "![figure 1](f/fig1.png)" in md
    )
    assert (tmp_path / "f" / "fig1.png").read_bytes().startswith(b"\x89PNG")
    assert "# T" not in md
