"""Every pipeline this machine can run gives the right answer.

The frames are encoded four ways (bitshuffle LZ4 and zstd level 2, each
with 8 kB and 256 kB blocks), with masked pixels holding 0 (as our Eigers
write them), 65535 (as others do), or no masked pixels at all (so the mask
step 'none' runs too).  For each:

  * every decode x mask x untranspose x collect value gives the sparse
    output numpy finds in the decompressed frame;
  * every dot x route value gives the powder of the matrix product, on a
    split radial matrix, a one-bin-per-pixel histogram, a histogram with a
    q moment and a permutation, so that every dot has a matrix it accepts;
  * the automatic mask step settles on 'planes' only for masked pixels at
    65535 (and blocks the planes take).

Only the documented refusals are accepted: mask 'none' with masked pixels,
a dot that cannot represent the matrix, and bsb-csr's uint16 in-block index
(blocks of at most 65536 pixels).  Anything else that is refused fails.
"""
import itertools
import os
import sys

_path = os.environ.get("BSLZ4_TO_SPARSE_PATH")
if _path:
    sys.path.insert(0, _path)

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")
sp = pytest.importorskip("scipy.sparse")

import bslz4_to_sparse as b
from bslz4_to_sparse import _pipeline as P

SHAPE = (384, 512)        # 196608 px: 48 blocks of 8 kB, 1.5 of 256 kB (a tail)
NPIX = SHAPE[0] * SHAPE[1]
VARIANTS = [("lz4", None, 8192), ("lz4", None, 262144), ("zstd", 2, 8192), ("zstd", 2, 262144)]
VARIANT_IDS = ["%s%s-%dk" % (c, l or "", k // 1024) for c, l, k in VARIANTS]
FILLS = ("zero", "max", "valid")


def usable(step):
    return [n for i, n in enumerate(P.NAMES[step]) if i and P.available(step, i) == 1]


def make_mask(fill):
    m = np.ones(SHAPE, np.uint8)
    if fill != "valid":
        m[180:192, :] = 0                    # a module gap
        m[:, 300:303] = 0
        m[np.random.default_rng(5).random(SHAPE) < 0.01] = 0     # bad pixels
    return m


def make_frames(mask, fill):
    """Two frames: sparse counts everywhere, a dense band of values < 256 and
    a few spots above 255 (so the u16 low-planes paths take some blocks and
    fall back on others)."""
    rng = np.random.default_rng(11)
    f = rng.poisson(0.05, (2,) + SHAPE)
    f[:, 40:80] += rng.poisson(20, (2, 40, SHAPE[1]))
    for k in range(2):
        f[k, rng.integers(0, SHAPE[0], 30), rng.integers(0, SHAPE[1], 30)] = rng.integers(256, 5000, 30)
    f = np.minimum(f, 65534).astype(np.uint16)
    f[:, mask == 0] = 65535 if fill == "max" else 0
    return f


def encode(frames, cname, clevel, block):
    kw = hdf5plugin.Bitshuffle(nelems=block // 2, cname=cname,
                               **({"clevel": clevel} if clevel else {}))
    with h5py.File("t.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("d", data=frames, chunks=(1,) + SHAPE, **kw)
        return [ds.id.read_direct_chunk((i, 0, 0))[1] for i in range(len(frames))]


def matrices():
    yy, xx = np.indices(SHAPE)
    r = np.hypot(yy - 150.3, xx - 260.7).ravel() / 4
    b0 = np.floor(r).astype(np.int64)
    frac = (r - b0).astype(np.float32)
    nb = int(b0.max()) + 2
    pix = np.arange(NPIX)
    split = sp.csc_matrix((np.concatenate([1 - frac, frac]),
                           (np.concatenate([b0, b0 + 1]), np.concatenate([pix, pix]))),
                          shape=(nb, NPIX))
    hist = sp.csc_matrix((np.ones(NPIX, np.float32), (b0, pix)), shape=(nb, NPIX))
    q = (r / r.max()).astype(np.float32)               # a per-pixel "q", interleaved [I, qI]
    moment = sp.csc_matrix((np.concatenate([np.ones(NPIX, np.float32), q]),
                            (np.concatenate([2 * b0, 2 * b0 + 1]), np.concatenate([pix, pix]))),
                           shape=(2 * nb, NPIX))
    perm = sp.csc_matrix((np.ones(NPIX, np.float32), (pix[::-1].copy(), pix)), shape=(NPIX, NPIX))
    return {"split": split, "hist": hist, "moment": moment, "permute": perm}


def accepted_refusal(msg, fill):
    """The documented refusals: mask 'none' with masked pixels, a dot that
    cannot represent the matrix, bsb-csr's 65536-pixel block limit."""
    if "pipeline['mask']='none'" in msg:
        return fill != "valid"
    return "cannot represent this matrix" in msg or "block_elems" in msg


@pytest.fixture(scope="module")
def data():
    out = {}
    for fill in FILLS:
        mask = make_mask(fill)
        frames = make_frames(mask, fill)
        for v in VARIANTS:
            out[(fill, v)] = (mask, frames, encode(frames, *v))
    return out


@pytest.mark.parametrize("fill", FILLS)
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_every_sparsify_pipeline(data, variant, fill):
    mask, frames, chunks = data[(fill, variant)]
    codec = b.CODEC_ZSTD if variant[0] == "zstd" else b.CODEC_LZ4
    ref = [np.flatnonzero((f.ravel() > 0) & (mask.ravel() > 0)) for f in frames]
    decodes = [d for d in usable("decode") if (d == "zstd") == (variant[0] == "zstd")]
    ran, refused, wrong = 0, [], []
    for combo in itertools.product(decodes, usable("mask"), usable("untranspose"), usable("collect")):
        p = dict(zip(("decode", "mask", "untranspose", "collect"), combo))
        try:
            c = b.chunk2sparse(mask, dtype=np.uint16, codec=codec, pipeline=p)
        except ValueError as e:
            refused.append((p, str(e)))
            continue
        npx, (v, i) = c.multi(chunks, 0)
        for k, f in enumerate(frames):
            if not (np.array_equal(i[k, :npx[k]], ref[k])
                    and np.array_equal(v[k, :npx[k]], f.ravel()[ref[k]])):
                wrong.append(p)
                break
        ran += 1
    assert not wrong, wrong
    bad = [(p, m) for p, m in refused if not accepted_refusal(m, fill)]
    assert not bad, bad
    # everything ran except mask 'none' on a masked frame
    n_none = sum(1 for p, _ in refused if p["mask"] == "none")
    assert len(refused) == n_none
    assert n_none == (0 if fill == "valid" else
                      len(decodes) * len(usable("untranspose")) * len(usable("collect")))
    assert ran > 0


@pytest.mark.parametrize("fill", ("max", "valid"))
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_every_dot_and_route(data, variant, fill):
    mask, frames, chunks = data[(fill, variant)]
    codec = b.CODEC_ZSTD if variant[0] == "zstd" else b.CODEC_LZ4
    masked_frames = frames.reshape(len(frames), -1).astype(np.float64) * (mask.ravel() > 0)
    ran_dots, refused, wrong = set(), [], []
    for mname, M in matrices().items():
        Mf = M.astype(np.float64)
        ref = np.stack([Mf @ fr for fr in masked_frames])
        for dot, route in itertools.product(usable("dot"), usable("route")):
            p = {"dot": dot, "route": route}
            try:
                c = b.chunk2sparseCSC(mask, M, dtype=np.uint16, codec=codec, pipeline=p)
                npx, (v, i), powder = c.multi(chunks, 0)
            except ValueError as e:
                refused.append((mname, p, str(e)))
                continue
            got = np.asarray(powder, np.float64).reshape(ref.shape)
            tol = 1e-3 if dot.endswith(("int", "int16")) else 1e-6
            if np.abs(got - ref).max() > tol * max(1.0, np.abs(ref).max()):
                wrong.append((mname, p, float(np.abs(got - ref).max())))
            ran_dots.add(dot)
    assert not wrong, wrong
    bad = [(m, p, e) for m, p, e in refused if not accepted_refusal(e, fill)]
    assert not bad, bad
    # every dot ran on some matrix (bsb-csr's index limits it to 65536-pixel blocks)
    expect = set(usable("dot"))
    if variant[2] // 2 > 65536:
        expect -= {"bsb-csr", "bsb-csr-nosplit"}
    assert expect <= ran_dots, sorted(expect - ran_dots)


@pytest.mark.parametrize("fill", FILLS)
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_automatic_mask_from_first_frame(data, variant, fill):
    mask, frames, chunks = data[(fill, variant)]
    codec = b.CODEC_ZSTD if variant[0] == "zstd" else b.CODEC_LZ4
    c = b.chunk2sparse(mask, dtype=np.uint16, codec=codec)
    c.multi(chunks, 0)
    got = b.describe(c.pipeline)["mask"]
    if fill == "valid":
        assert got == "none"
    elif fill == "max" and variant[2] // 2 <= 8192:
        assert got == "planes"
    else:
        assert got == "pixel"
    # an explicit mask step is never changed
    c = b.chunk2sparse(mask, dtype=np.uint16, codec=codec,
                       pipeline={"mask": "pixel" if fill != "valid" else "none"})
    c.multi(chunks, 0)
    assert b.describe(c.pipeline)["mask"] == ("pixel" if fill != "valid" else "none")
