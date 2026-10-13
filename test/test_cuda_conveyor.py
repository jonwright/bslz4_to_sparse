"""The conveyor with the GPU backend: the exact int64 powder equals the CPU backend's.  Skipped
without cupy and a CUDA GPU."""

import threading

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
sp = pytest.importorskip("scipy.sparse")
cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device", allow_module_level=True)
except Exception:  # noqa: BLE001
    pytest.skip("no CUDA device", allow_module_level=True)

from test_conveyor import DSET, SHAPE, make_files  # noqa: E402

from bslz4_to_sparse import conveyor as cv  # noqa: E402


class Powder(cv.Sink):
    out = {}
    lock = threading.Lock()

    def write(self, file, frame0, npx, values, indices, powder):
        with self.lock:
            for j in range(len(npx)):
                self.out[(file, frame0 + j)] = powder[j].copy()


def split_matrix(nbins=12):
    """every pixel shared between two bins, weights summing to 1 (like pixel splitting)"""
    npix = SHAPE[0] * SHAPE[1]
    rng = np.random.default_rng(8)
    r0 = rng.integers(0, nbins, npix)
    r1 = (r0 + 1 + rng.integers(0, nbins - 1, npix)) % nbins
    w = rng.random(npix).astype(np.float32) * 0.8 + 0.1
    cols = np.arange(npix)
    return sp.csc_matrix(
        (np.concatenate([w, 1 - w]), (np.concatenate([r0, r1]), np.concatenate([cols, cols]))),
        (nbins, npix),
    )


@pytest.mark.parametrize("weights", ["unit", "split"])
def test_gpu_exact_equals_cpu_exact(tmp_path, weights):
    paths = make_files(tmp_path)
    units, shape, dtype = cv.make_work_units(paths, DSET)
    mask = np.ones(SHAPE, np.uint8)
    npix = SHAPE[0] * SHAPE[1]
    if weights == "unit":
        m = sp.csc_matrix(
            (np.ones(npix, np.float32), (np.arange(npix) % 12, np.arange(npix))), (12, npix)
        )
    else:
        m = split_matrix()
    res = {}
    for name, be in (
        ("cpu", cv.CpuBackend(mask, dtype, csc=m, exact=True)),
        ("gpu", cv.GpuBackend(mask, dtype, csc=m, exact=True)),
    ):
        Powder.out = {}
        scales = []

        class Sink(Powder):
            def begin(self, meta):
                scales.append(meta["powder_scale"])

        cv.run(units, be, workers=2, sink=lambda k: Sink(), batch=8)
        res[name] = (Powder.out, scales[0])
    cpu, gpu = res["cpu"], res["gpu"]
    assert cpu[1] == gpu[1] and cpu[1] is not None
    assert len(cpu[0]) == len(gpu[0]) == 3 * 37
    for k in cpu[0]:
        assert cpu[0][k].dtype == np.int64 and gpu[0][k].dtype == np.int64
        assert np.array_equal(cpu[0][k], gpu[0][k]), k
