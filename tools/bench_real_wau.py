"""Real-data timing sweep (WAu5um eiger_0008/0012, frames 400-599, original
and mask=0 copies) plus the synthetic 4M sets: old pipeline vs current
defaults vs defaults + mask planes.  Needs the copies made in /tmp/$USER/bslz4_bench
(see notes_dot_layouts.md, "Real data: WAu5um")."""
import os, sys, time
sys.path.insert(0, 'tools')
import bench_suite as bs, bench_perf as bp
import numpy as np, h5py, hdf5plugin, bslz4_to_sparse as b
os.sched_setaffinity(0, {7})
U = os.environ["USER"]
def best(fn, chunks, n=5):
    fn(chunks[:25]); t = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        for s in range(0, len(chunks), 25): fn(chunks[s:s + 25])
        t = min(t, time.perf_counter() - t0)
    return t * 1e3 / len(chunks)
sets = []
for num in ("0008", "0012"):
    SRC = "/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_%s.h5" % num
    with h5py.File(SRC, "r") as h:
        ds = h["/entry_0000/ESRF-ID11/eiger/data"]
        m = ds[0] == 65535            # this scan's stand-in for the fixed mask (special case)
        orig = [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(400, 600)]
    with h5py.File("/tmp/%s/bslz4_bench/WAu5um_eiger%s_f400-599_maskzero.h5" % (U, num), "r") as h:
        zero = [h["data"].id.read_direct_chunk((k, 0, 0))[1] for k in range(200)]
    v = np.ascontiguousarray((~m).astype(np.uint8))
    sets += [("WAu%s m=65535" % num, v, orig, (0, 2)), ("WAu%s m=0" % num, v, zero, (0, 2))]
ai = bs.make_ai("4M", "mid")
v4 = np.ascontiguousarray((1 - ai.detector.mask).astype(np.uint8))
for kind in ("sparse", "medium", "dense"):
    ch, _ = bs.load_chunks(bp.frames_for("4M", "mid", kind, ai))
    sets.append(("synth %s" % kind, v4, ch[:100], (0, bp.CUT_ON[kind])))
old = b.pack_pipeline(collect=1, options=0)
print("%-17s %-6s %8s %8s %8s %8s | %7s %7s" % ("data", "cut", "before", "now", "now+mp", "before", "now", "now+mp"))
for name, v, ch, cuts in sets:
    for cut in sorted(set(cuts)):
        r = []
        for cfg in ("old", "now", "mp", "old"):
            b.set_mask_planes(cfg == "mp")
            integ = b.chunk2sparseMulti(v, dtype=np.uint16, pipeline=old if cfg == "old" else None)
            r.append(best(lambda c: integ(c, cut), ch))
        base = (r[0] + r[3]) / 2
        print("%-17s %-6s %8.3f %8.3f %8.3f %8.3f | %+6.1f%% %+6.1f%%" % (name, "cut %d" % cut, r[0], r[1], r[2], r[3],
              100 * (r[1] / base - 1), 100 * (r[2] / base - 1)), flush=True)
b.set_mask_planes(False)
