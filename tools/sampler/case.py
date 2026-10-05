"""One benchmark case, for profiling: lib dataset case secs.
case: 'sparsify:<cut>' or '<dot>:<matrix>' (matrix npz name in the shared data).
Data: $BSLZ4_SUITE_DATA (as tools/compiler_suite.py).  Pinned to $CPU
(default: the last core this process may use)."""
import os, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np, h5py, scipy.sparse as sp, bslz4_to_sparse as b
D = os.environ.get("BSLZ4_SUITE_DATA", "/tmp_14_days/wright/bslz4_bench_shared")
ds, case, secs = sys.argv[2], sys.argv[3], float(sys.argv[4])
if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {int(os.environ.get("CPU", max(os.sched_getaffinity(0))))})
else:  # Windows
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    k32.SetProcessAffinityMask(k32.GetCurrentProcess(), 1 << int(os.environ.get("CPU", os.cpu_count() - 1)))
if ds.startswith("WAu"):
    fn = os.path.join(D, "WAu5um_eiger%s_f400-599_maskzero.h5" % ds[3:])
    mask = np.load(os.path.join(D, "WAu5um_eiger%s_mask.npy" % ds[3:]))
else:
    fn = os.path.join(D, "frames_4M_mid%s.h5" % ("" if ds == "mid" else "_" + ds))
    mask = (np.load(os.path.join(D, "detmask.npy")) == 0).astype(np.uint8)
mask = np.ascontiguousarray(mask, np.uint8)
with h5py.File(fn, "r") as h:
    ch = [h["data"].id.read_direct_chunk((k, 0, 0))[1] for k in range(100)]
a, c = case.split(":")
if a == "sparsify":
    integ, cut = b.chunk2sparseMulti(mask, dtype=np.uint16), int(c)
else:
    z = np.load(os.path.join(D, c + ".npz"))
    M = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    integ, cut = b.chunk2sparseCSCmulti(mask, M, dtype=np.uint16, dot=a), 0
for s in range(0, 100, 25): integ(ch[s:s + 25], cut)
t0 = time.perf_counter(); n = 0
while time.perf_counter() - t0 < secs:
    for s in range(0, 100, 25): integ(ch[s:s + 25], cut)
    n += 100
print("%s %s %.3f ms/frame" % (ds, case, (time.perf_counter() - t0) * 1e3 / n))
