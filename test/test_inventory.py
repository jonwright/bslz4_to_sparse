"""The kernel inventory (src/_inventory.py) against the code: every row's claims are checked by
running the kernel, so that the table cannot drift from what the kernels do.

  - the rows are the step values of _pipeline.VALUES, in the same order
  - a row without a cpu requirement is usable on every machine
  - dtypes: the values refuse exactly the pixel dtypes the row leaves out
  - matrix: each dot builds for exactly the matrices its requirement admits
  - contract and routes: each dot gives the powder its contract says, on both routes, equal to a
    numpy product (exact for the integer contracts)
"""
from __future__ import print_function

import os
import sys

import pytest
import numpy as np

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import bslz4_to_sparse as b
from bslz4_to_sparse import _pipeline, _matrix, _inventory

ALL_DTYPES = ("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64")
NP = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64, "i8": np.int8,
      "i16": np.int16, "i32": np.int32, "i64": np.int64, "f32": np.float32, "f64": np.float64}


def _usable(step, name):
    return _pipeline.available(step, _pipeline.NAMES[step].index(name)) == 1


def test_rows_are_the_step_values_in_order():
    for step in _pipeline.STEPS:
        assert [k["name"] for k in _inventory.rows(step)] == list(_pipeline.NAMES[step][1:]), step
    assert len(_inventory.KERNELS) == sum(len(_pipeline.NAMES[s]) - 1 for s in _pipeline.STEPS)


def test_portable_rows_are_usable():
    for k in _inventory.KERNELS:
        if k["cpu"] is None:
            assert _usable(k["step"], k["name"]), (k["step"], k["name"])


def test_matrix_and_contract_names_are_known():
    for k in _inventory.rows("dot"):
        assert k["matrix"] in _inventory.MATRIX, k["name"]
        assert k["contract"] in _inventory.CONTRACTS, k["name"]
        assert k["routes"] == ("dense", "sparse"), k["name"]
    for k in _inventory.KERNELS:
        if k["step"] != "dot":
            assert k["matrix"] is None and k["contract"] is None and k["reads"] is None


@pytest.mark.parametrize("step", ["untranspose", "collect"])
def test_step_dtypes(step):
    mask = np.ones((4, 8), np.uint8)
    for k in _inventory.rows(step):
        if not _usable(step, k["name"]):
            continue
        for d in ALL_DTYPES:
            want = k["dtypes"] is None or d in k["dtypes"]
            try:
                b.chunk2sparse(mask, NP[d], pipeline={step: k["name"]})
                got = True
            except (ValueError, TypeError):
                got = False
            assert got == want, (step, k["name"], d)


# ---- matrices: each kind, and the requirement names it satisfies --------------------------

NI, NJ = 4, 8
NPIX = NI * NJ


def _csc(cols, nbins):
    """A scipy-like CSC from a list (per pixel) of [(bin, weight), ...]."""
    indptr = np.zeros(NPIX + 1, np.uint32)
    indices, data = [], []
    for p, entries in enumerate(cols):
        for bin_, w in entries:
            indices.append(bin_)
            data.append(w)
        indptr[p + 1] = len(indices)
    return type("_CSC", (), {"indptr": indptr, "indices": np.array(indices, np.uint32),
                             "data": np.array(data, np.float32), "shape": (nbins, NPIX)})()


def _kinds():
    rng = np.random.RandomState(7)
    w = lambda: float(rng.uniform(0.1, 1.0))
    general = [[(p % 5, w()), ((p % 5) + 2, w())] for p in range(NPIX)]       # a gap in the bins
    runs3 = [[((p % 5) + j, w()) for j in range(1 + p % 3)] for p in range(NPIX)]
    runs70 = [[(j, w()) for j in range(70)] if p == 3 else [(p % 5, w())] for p in range(NPIX)]
    nosplit = [[(p % 6, 1.0)] for p in range(NPIX)]                              # bins repeat
    perm = [[((p * 7) % NPIX, 1.0)] for p in range(NPIX)]                        # each bin once
    any_ = {"any"}
    runs = any_ | {"runs", "runs<=64"}
    return {
        "general": (_csc(general, 8), any_),
        "runs3": (_csc(runs3, 8), runs),
        "runs70": (_csc(runs70, 72), any_ | {"runs"}),
        "nosplit": (_csc(nosplit, 6), runs | {"nosplit"}),
        "perm": (_csc(perm, NPIX), runs | {"nosplit", "permutation"}),
    }


def _matrices():
    """(label, matrix to pass, requirement names it satisfies, dense numpy matrix)."""
    mask = np.ones((NI, NJ), np.uint8)
    out = []
    for label, (csc, sat) in sorted(_kinds().items()):
        dense = np.zeros(csc.shape)
        for p in range(NPIX):
            for k in range(csc.indptr[p], csc.indptr[p + 1]):
                dense[csc.indices[k], p] += csc.data[k]
        out.append((label, csc, sat, dense))
        for wbits in (32, 16):
            try:
                fx = b.fixed_point_matrix(csc, mask, np.uint16, wbits=wbits)
            except ValueError:
                continue
            fsat = {"fixed%d" % wbits}
            if "runs" in sat:
                fsat.add("fixed%d-runs" % wbits)
            out.append(("%s fixed%d" % (label, wbits), fx, fsat, dense))
    return out


def _dots():
    return [k for k in _inventory.rows("dot") if _usable("dot", k["name"])]


def test_dot_matrix_requirements():
    mask = np.ones((NI, NJ), np.uint8)
    for k in _dots():
        for label, m, sat, _ in _matrices():
            want = k["matrix"] in sat
            try:
                b.chunk2sparseCSC(mask, m, np.uint16, pipeline={"dot": k["name"]})
                got = True
            except ValueError:
                got = False
            assert got == want, (k["name"], label, k["matrix"])


def test_dot_dtypes():
    mask = np.ones((NI, NJ), np.uint8)
    for k in _dots():
        label, m, _, _ = next(x for x in _matrices() if k["matrix"] in x[2])
        for d in ALL_DTYPES:
            want = k["dtypes"] is None or d in k["dtypes"]
            try:
                if isinstance(m, _matrix.FixedPointMatrix) and m.dtype != np.dtype(NP[d]):
                    if np.dtype(NP[d]).kind not in "ui":
                        raise ValueError("not an integer dtype")
                    csc = next(x for x in _matrices() if x[0] == label.split()[0])[1]
                    big = np.dtype(NP[d]).itemsize == 8      # 64-bit pixels need a maxval
                    mm = b.fixed_point_matrix(csc, mask, NP[d], wbits=m.data.dtype.itemsize * 8,
                                              maxval=2 ** 40 if big else None)
                else:
                    mm = m
                c = b.chunk2sparseCSC(mask, mm, NP[d], pipeline={"dot": k["name"]})
                c.from_frames(np.ones((1, NPIX), NP[d]), 0)
                got = True
            except (ValueError, TypeError):
                got = False
            assert got == want, (k["name"], d)


def test_dot_contracts_and_routes():
    mask = np.ones((NI, NJ), np.uint8)
    frame = (np.arange(NPIX) % 7).astype(np.uint16).reshape(1, NPIX)
    frame[0, ::5] = 0                                   # some zeros for the sparse route
    for k in _dots():
        label, m, _, dense = next(x for x in _matrices() if k["matrix"] in x[2])
        for route in k["routes"]:
            c = b.chunk2sparseCSC(mask, m, np.uint16, pipeline={"dot": k["name"], "route": route})
            npx, _, powder = c.from_frames(frame, 0)
            powder = np.asarray(powder)[0]
            if k["contract"] == "pixels":
                assert powder.dtype == np.uint16
                want = np.zeros(dense.shape[0], np.uint16)
                want[np.argmax(dense, axis=0)] = frame[0]
                assert np.array_equal(powder, want), (k["name"], route)
                continue
            ref = dense.dot(frame[0].astype(np.float64))
            if k["contract"] == "int64-scaled":
                assert powder.dtype == np.int64 and c.powder_scale, (k["name"], route)
                q = m.to_scipy().toarray().astype(np.int64)
                assert np.array_equal(powder, q.dot(frame[0].astype(np.int64))), (k["name"], route)
                np.testing.assert_allclose(powder / float(c.powder_scale), ref, rtol=1e-6)
            else:
                assert powder.dtype == np.float64 and c.powder_scale is None, (k["name"], route)
                if k["contract"] == "float64-of-int64":
                    assert np.array_equal(powder, ref), (k["name"], route)
                np.testing.assert_allclose(powder, ref, rtol=1e-6, err_msg=k["name"] + " " + route)


def test_matrix_bytes():
    nm = _matrix._fold_mask(_matrix.normalise_matrix(_kinds()["runs3"][0], npix=NPIX),
                            np.ones(NPIX, np.uint8))
    s = _inventory.sizes(nm)
    for k in _inventory.rows("dot"):
        n = k["matrix_bytes"](s)
        assert isinstance(n, int) and n > 0, k["name"]
