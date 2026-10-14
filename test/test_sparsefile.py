"""ImageD11SparseSink: one sparse file per input file in ImageD11's layout, read back by h5py and by
ImageD11 itself, with the exact powder beside it."""

import numpy as np
import pytest
import scipy.sparse as sp

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from test_conveyor import DSET, SHAPE, direct, make_files  # noqa: E402

from bslz4_to_sparse import conveyor as cv  # noqa: E402


def matrix(nbins=12):
    npix = SHAPE[0] * SHAPE[1]
    rows = np.arange(npix) % nbins
    return sp.csc_matrix((np.ones(npix, np.float32), (rows, np.arange(npix))), (nbins, npix)), rows


@pytest.mark.parametrize("compression", ["bslz4", "bszstd", "lzf", None])
def test_files_match_a_direct_decode(tmp_path, compression):
    paths = make_files(tmp_path)
    units, shape, dtype = cv.make_work_units(paths, DSET)  # a unit is a file
    mask = np.ones(SHAPE, np.uint8)
    out = str(tmp_path / "sparse")
    m, rows = matrix()
    r = cv.run(
        units,
        cv.CpuBackend(mask, dtype, csc=m, exact=True),
        workers=2,
        sink=lambda k: cv.ImageD11SparseSink(out, DSET, compression=compression, integrator="i.h5"),
    )
    assert r["frames"] == 3 * 37
    want = direct(paths, mask)
    for fi, p in enumerate(paths):
        with h5py.File("%s/%s" % (out, cv.imaged11_name(p)), "r") as h:
            g = h[DSET.strip("/")]
            assert g.attrs["nframes"] == 37 and g.attrs["shape1"] == SHAPE[1]
            assert g.attrs["itype"] == "uint16" and g["row"].dtype == np.uint16
            nnz = g["nnz"][:]
            ptr = np.concatenate([[0], np.cumsum(nnz, dtype=np.int64)])
            assert g.attrs["npx"] == ptr[-1]
            row, col, inten = g["row"][:], g["col"][:], g["intensity"][:]
            pw = g["powder"]
            assert pw.dtype == np.int64 and pw.attrs["integrator"] == "i.h5"
            for i in range(37):
                s = slice(ptr[i], ptr[i + 1])
                flat = row[s].astype(np.int64) * SHAPE[1] + col[s]
                assert (int(nnz[i]), int(inten[s].sum(dtype=np.int64)), int(flat.sum())) == want[
                    (fi, i)
                ]
            # the powder: exact photon sums per bin
            with h5py.File(p, "r") as src:
                frames = src[DSET][:]
            ref = np.stack([np.bincount(rows, weights=f.ravel(), minlength=12) for f in frames])
            assert np.array_equal(pw[:].astype(np.float64) * (1.0 / pw.attrs["scale"]), ref)


def test_float_powder_and_whole_files_only(tmp_path):
    paths = make_files(tmp_path, nfiles=1)
    mask = np.ones(SHAPE, np.uint8)
    units, shape, dtype = cv.make_work_units(paths, DSET)
    cv.run(
        units,
        cv.CpuBackend(mask, dtype, csc=matrix()[0], exact=True),
        workers=1,
        sink=lambda k: cv.ImageD11SparseSink(str(tmp_path / "o"), DSET, powder_dtype=np.float32),
    )
    with h5py.File(str(tmp_path / "o" / cv.imaged11_name(paths[0])), "r") as h:
        g = h[DSET.strip("/")]
        assert g["intensity"].dtype == np.uint16  # always the input dtype
        assert g["powder"].dtype == np.float32 and "scale" not in g["powder"].attrs
        with h5py.File(paths[0], "r") as src:
            frames = src[DSET][:]
        want = np.stack([np.bincount(matrix()[1], weights=f.ravel(), minlength=12) for f in frames])
        assert np.array_equal(g["powder"][:], want.astype(np.float32))
    small, _, _ = cv.make_work_units(paths, DSET, max_unit_bytes=3000)
    with pytest.raises(RuntimeError, match="written per input file"):
        cv.run(
            small,
            cv.CpuBackend(mask, dtype),
            workers=1,
            sink=lambda k: cv.ImageD11SparseSink(str(tmp_path / "o2"), DSET),
        )


def test_imageD11_reads_it(tmp_path):
    sparseframe = pytest.importorskip("ImageD11.sparseframe")
    paths = make_files(tmp_path, nfiles=1)
    mask = np.ones(SHAPE, np.uint8)
    units, shape, dtype = cv.make_work_units(paths, DSET)
    out = str(tmp_path / "o")
    cv.run(
        units,
        cv.CpuBackend(mask, dtype),
        workers=1,
        sink=lambda k: cv.ImageD11SparseSink(out, DSET),
    )
    s = sparseframe.SparseScan("%s/%s" % (out, cv.imaged11_name(paths[0])), DSET.strip("/"))
    with h5py.File(paths[0], "r") as src:
        frames = src[DSET][:]
    for i in (0, 5, 36):
        assert np.array_equal(s.getframe(i).to_dense("intensity"), frames[i])


def test_small_objects_are_dropped_with_min_pixels(tmp_path):
    pytest.importorskip("ImageD11.sparseframe")
    import hdf5plugin

    shape = (40, 50)
    frames = np.zeros((3,) + shape, np.uint16)
    for f in frames:
        f[5:7, 5:7] = 9  # a 2 x 2 object: 4 pixels
        f[20, 10:13] = 7  # a line of 3 pixels
        f[30, 40] = 5  # single pixels: noise
        f[10, 30] = 6
        f[35, 3] = 4
    p = str(tmp_path / "o.h5")
    with h5py.File(p, "w") as h:
        h.create_dataset(
            DSET, data=frames, chunks=(1,) + shape, **hdf5plugin.Bitshuffle(nelems=0, cname="lz4")
        )
    cv.write_index(cv.harvest(p, DSET), p)
    units, _, dtype = cv.make_work_units([p], DSET)
    mask = np.ones(shape, np.uint8)
    got = {}
    for n in (None, 2, 4):
        out = str(tmp_path / ("o%s" % n))
        cv.run(
            units,
            cv.CpuBackend(mask, dtype),
            workers=1,
            sink=lambda k: cv.ImageD11SparseSink(out, DSET, min_pixels=n),
        )
        with h5py.File("%s/%s" % (out, cv.imaged11_name(p)), "r") as h:
            g = h[DSET.strip("/")]
            got[n] = (
                g["nnz"][:].tolist(),
                g["row"][:],
                g["col"][:],
                g["intensity"][:],
                dict(g.attrs),
            )
    assert got[None][0] == [10, 10, 10]  # every pixel: 4 + 3 + three singles
    assert got[2][0] == [7, 7, 7] and got[4][0] == [4, 4, 4]  # without singles; only the 2 x 2
    assert "min_pixels_in_object" not in got[None][4] and got[2][4]["min_pixels_in_object"] == 2
    # what is left is the objects, exactly
    r, c, v = got[4][1][:4], got[4][2][:4], got[4][3][:4]
    assert sorted(zip(r.tolist(), c.tolist(), v.tolist())) == [
        (5, 5, 9),
        (5, 6, 9),
        (6, 5, 9),
        (6, 6, 9),
    ]
