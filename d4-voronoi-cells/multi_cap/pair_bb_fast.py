#!/usr/bin/env python3
"""
pair_bb_fast.py -- the branch and bound of radial_count_check.py for the pair
inequality K(d, d', u) <= Pi(d/2, d'/2, u), processed in batches.

The mathematics is that of radial_count_check.py, step 2: the tensor
Bernstein coefficients of K on the root box are computed exactly and rounded
once, subdivided at midpoints in floating point with the rounding error bounded
a priori, and on a box the largest coefficient plus that allowance bounds K.
A box is closed when that bound is at most 0, or at most a rigorous lower bound
of Pi at the corner (d_hi, d'_hi, u_lo) computed in Arb (lem:pair-closed: Pi
decreases in d and d' and increases in u).  It is dropped when it lies in
d > d' (exact integer test) or wholly beyond u = a(d, d') (an exact rational
test, made only when a floating-point test is not decisive).  Boxes are handled
level by level in numpy arrays instead of one at a time; the Arb values are
memoised by corner.
"""
import time
from fractions import Fraction as Fr

import numpy as np

from radial_count_check import bernstein, cheb_T, cheb_U, Pi_lower
from truncated_search import pair as pair_float


def root_coefficients(A, D, r, c1, c2, dmax):
    alpha, beta = (4 - c1) / c2, 2 * (dmax - 2) / c2
    umax = (2 * dmax ** 2 - 4) / (2 * dmax ** 2)
    Ts = [bernstein(p, r) for p in cheb_T(alpha, beta, r)]
    Us = [bernstein([x / (k + 1) for x in p], D) for k, p in enumerate(cheb_U(Fr(-1), umax + 1, D))]
    B0 = [[[Fr(0)] * (D + 1) for _ in range(r + 1)] for _ in range(r + 1)]
    for k in range(D + 1):
        for a in range(r + 1):
            for b in range(r + 1):
                if A[k][a][b] == 0:
                    continue
                for i in range(r + 1):
                    ci = A[k][a][b] * Ts[a][i]
                    if ci == 0:
                        continue
                    for j in range(r + 1):
                        cij = ci * Ts[b][j]
                        if cij == 0:
                            continue
                        row = B0[i][j]
                        for l in range(D + 1):
                            row[l] += cij * Us[k][l]
    Cmax = max(abs(x) for plane in B0 for row in plane for x in row)
    Bf = np.array([[[float(x) for x in row] for row in plane] for plane in B0])
    return Bf, Cmax, umax


def split_batch(B, axis):
    """de Casteljau at the midpoint along axis (1, 2 or 3) for a batch B of shape (n, ...)."""
    X = np.moveaxis(B, axis, 0)
    n = X.shape[0] - 1
    left, right = [X[0]], [X[n]]
    cur = X
    for _ in range(n):
        cur = (cur[:-1] + cur[1:]) * 0.5
        left.append(cur[0]); right.append(cur[-1])
    return np.moveaxis(np.array(left), 0, axis), np.moveaxis(np.array(right[::-1]), 0, axis)


def pair_check_fast(A, D, r, c1, c2, dmax, depth_max=80, batch=5000):
    t0 = time.time()
    Bf, Cmax, umax = root_coefficients(A, D, r, c1, c2, dmax)
    err = float(Cmax) * 2.0 ** -53 * (1 + depth_max * max(r, D)) * 1.01
    fd, fdm, fum = 2.0, float(dmax - 2), float(umax + 1)
    # boxes: integer cells (i, level) per axis
    I = np.zeros((1, 3), dtype=np.int64); L = np.zeros((1, 3), dtype=np.int64)
    Bs = Bf[None]
    depth = 0
    closed = dropped = 0
    memo = {}
    arbcalls = 0
    pending = [(I, L, Bs)]
    while pending:
        I, L, Bs = pending.pop()
        if len(I) == 0:
            continue
        if len(I) > batch:
            pending.append((I[batch:], L[batch:], Bs[batch:]))
            I, L, Bs = I[:batch], L[:batch], Bs[:batch]
        if L.max() > depth_max:
            return False, 'depth limit reached'
        scale = 2.0 ** (-L)
        dlo = fd + fdm * I[:, 0] * scale[:, 0]; dhi = fd + fdm * (I[:, 0] + 1) * scale[:, 0]
        elo = fd + fdm * I[:, 1] * scale[:, 1]; ehi = fd + fdm * (I[:, 1] + 1) * scale[:, 1]
        ulo = -1.0 + fum * I[:, 2] * scale[:, 2]
        # d > d': exact on the integer cells (i1 / 2^n1 > (i2 + 1) / 2^n2)
        m = np.maximum(L[:, 0], L[:, 1])
        if m.max() <= 60:
            drop = (I[:, 0] << (m - L[:, 0])) > ((I[:, 1] + 1) << (m - L[:, 1]))
        else:
            drop = np.array([int(a) * 2 ** (int(mm) - int(la)) > (int(b) + 1) * 2 ** (int(mm) - int(lb))
                             for a, b, la, lb, mm in zip(I[:, 0], I[:, 1], L[:, 0], L[:, 1], m)], dtype=bool)
        # beyond u = a(d_hi, d'_hi): floating point with a margin, exact where undecided
        am = (dhi * dhi + ehi * ehi - 4) / (2 * dhi * ehi)
        beyond = ulo > am + 1e-12
        unsure = (~beyond) & (ulo > am - 1e-12) & ~drop
        for k in np.nonzero(unsure)[0]:
            dh = 2 + (dmax - 2) * Fr(int(I[k, 0]) + 1, 2 ** int(L[k, 0]))
            eh = 2 + (dmax - 2) * Fr(int(I[k, 1]) + 1, 2 ** int(L[k, 1]))
            ul = -1 + (umax + 1) * Fr(int(I[k, 2]), 2 ** int(L[k, 2]))
            if ul > (dh * dh + eh * eh - 4) / (2 * dh * eh):
                beyond[k] = True
        drop |= beyond
        dropped += int(drop.sum())
        keep = ~drop
        I, L, Bs, dhi, ehi, ulo = I[keep], L[keep], Bs[keep], dhi[keep], ehi[keep], ulo[keep]
        if len(I) == 0:
            continue
        kub = np.nextafter(Bs.reshape(len(Bs), -1).max(1) + err, np.inf)
        ok = kub <= 0
        # the corner value of Pi, first in floating point, then rigorously where it may close the box
        cand = (~ok) & (kub <= pair_float(dhi / 2, ehi / 2, ulo) - 1e-9)
        for k in np.nonzero(cand)[0]:
            key = (int(I[k, 0]), int(L[k, 0]), int(I[k, 1]), int(L[k, 1]), int(I[k, 2]), int(L[k, 2]))
            if key not in memo:
                dh = 2 + (dmax - 2) * Fr(key[0] + 1, 2 ** key[1])
                eh = 2 + (dmax - 2) * Fr(key[2] + 1, 2 ** key[3])
                ul = -1 + (umax + 1) * Fr(key[4], 2 ** key[5])
                memo[key] = Pi_lower(dh, eh, ul)
                arbcalls += 1
            if kub[k] <= memo[key]:
                ok[k] = True
        closed += int(ok.sum())
        I, L, Bs = I[~ok], L[~ok], Bs[~ok]
        if len(I) == 0:
            continue
        # split along the widest axis, as radial_count_check.py does
        w = np.stack([2.0 ** (-L[:, 0]), 2.0 ** (-L[:, 1]), 2.0 ** (-L[:, 2]) * fum / fdm / 3], 1)
        ax = np.argmax(w, 1)
        for a in range(3):
            sel = ax == a
            if not sel.any():
                continue
            Ia, La, Ba = I[sel], L[sel], Bs[sel]
            left, right = split_batch(Ba, a + 1)
            Il, Ir = Ia.copy(), Ia.copy()
            Il[:, a] = 2 * Ia[:, a]; Ir[:, a] = 2 * Ia[:, a] + 1
            La2 = La.copy(); La2[:, a] += 1
            pending.append((np.r_[Il, Ir], np.r_[La2, La2], np.concatenate([left, right])))
    return True, '%d boxes closed, %d dropped, %d values of Pi in Arb [%.0f s]' % (closed, dropped, arbcalls, time.time() - t0)
