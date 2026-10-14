#!/usr/bin/env python3
"""The example pages run on real data, built here by hand and committed to docs/real/.

    BSLZ4_TO_SPARSE_PATH=<dir holding a built bslz4_to_sparse> python3 tools/real_data_pages.py

CI cannot run these: it has no copy of the data.  So, unlike docs/gen/, the pages, their
notebooks and figures are committed, and tools/generate_docs.py leaves these examples out.  Each
example in PAGES runs, as in tools/generate_docs.py, in a folder where its h5filename is the
download of tools/example_data.py (silicon_u16_zstd2_256k.h5, ...), beside mask.h5 and
tools/calib.poni.  Run it again, and commit what changed, whenever the example or the code it
calls changes.
"""

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import example_data  # noqa: E402
import generate_docs as gd  # noqa: E402  (imports bslz4_to_sparse as gd.b)
import standin  # noqa: E402

PAGES = {"ex13_peaks_out_of_powder": "silicon"}  # example -> key of example_data.FILES
DOI = "https://doi.org/10.5281/zenodo.23361659"


def build(name, key):
    src = example_data.example_dir()
    folder = tempfile.mkdtemp(prefix="bslz4_real_")
    try:
        os.symlink(example_data.path(key, "zstd", src), os.path.join(folder, standin.H5NAME % 0))
        os.symlink(os.path.join(src, example_data.MASK), os.path.join(folder, standin.MASK))
        shutil.copy(example_data.PONI, os.path.join(folder, standin.PONI))
        note = (
            "Output, from running it on real data when this page was made: `%s` of the example "
            "data ([%s](%s)), with its `mask.h5` and `tools/calib.poni`.  This page is committed, "
            "made by `tools/real_data_pages.py`; CI does not run it on this data."
            % (example_data.FILES[key], DOI, DOI)
        )
        path = os.path.join(gd.REPO, "examples", name + ".py")
        _title, page = gd.example_page(path, folder, note, gd.REAL)
    finally:
        shutil.rmtree(folder)
    page = page.replace("tools/generate_docs.py", "tools/real_data_pages.py")
    with open(os.path.join(gd.REAL, name + ".md"), "w") as fh:
        fh.write(page)
    print("wrote", os.path.relpath(os.path.join(gd.REAL, name + ".md"), gd.REPO))


def main():
    print("bslz4_to_sparse", gd.b.version, gd.b.__file__)
    for name, key in sorted(PAGES.items()):
        build(name, key)


if __name__ == "__main__":
    main()
