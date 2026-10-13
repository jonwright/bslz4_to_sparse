"""Clipping afterwards: ImageD11's wtmax, applied to an exact powder from the sparse pixels.

The dot with fixed-point integer weights is linear in the pixel values, so clipping every pixel to
wtmax before the sum is the same as subtracting, from the unclipped integer sums, the weights times
the excess (value - wtmax) of each pixel above wtmax.  Those pixels are in the sparse output when
the sparse path keeps every pixel above wtmax: the cut is at most wtmax, and no frame was cut back
(howmany) so far that a pixel above wtmax was dropped.  The result is bit for bit what summing the
clipped frames gives, in integer arithmetic, with the same weights.  The normalisation is untouched
(a clipped pixel is still a pixel), unlike a pixel that is dropped.
"""

import numpy as np


def fixed_weights(csc, mask, dtype):
    """(W, scale): the integer weights chunk2sparseCSC(exact=True) uses for this matrix, mask and
    pixel dtype (the "csc-int" dot), as a scipy CSC matrix of int64, and the scale of its sums."""
    import scipy.sparse as sp

    from bslz4_to_sparse import _matrix

    mask = np.asarray(mask)
    nm = _matrix._fold_mask(_matrix.normalise_matrix(csc, npix=mask.size), mask.ravel())
    b = _matrix._fixed_point_bits(nm, np.dtype(dtype), 32)
    q = _matrix._quantise(nm, b, np.uint32)
    W = sp.csc_matrix(
        (q.astype(np.int64), nm.indices.astype(np.int64), nm.indptr.astype(np.int64)),
        shape=nm.shape,
    )
    return W, 2.0**-b


def clip_exact_powder(powder, npx, values, indices, wtmax, weights, in_place=False):
    """The int64 powder (nframes, nbins) as if every pixel had been clipped to wtmax before the
    sum.  npx, values, indices: the sparse output of the same frames (rows of values and indices
    hold the pixels of one frame in their first npx[i] entries); weights: W of fixed_weights.
    Every pixel above wtmax must be in the sparse output (see the module docstring)."""
    out = powder if in_place else powder.copy()
    for i in range(len(npx)):
        k = int(npx[i])
        v = values[i, :k].astype(np.int64)
        hot = v > wtmax
        if hot.any():
            out[i] -= weights[:, indices[i, :k][hot].astype(np.int64)].dot(v[hot] - int(wtmax))
    return out
