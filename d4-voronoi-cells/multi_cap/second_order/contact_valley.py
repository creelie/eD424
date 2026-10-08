#!/usr/bin/env python3
"""
contact_valley.py -- the contact-cell volume vol{x : <x, w_i> <= 1} near the
root system, over all sets of 24 unit directions (floating point, qhull;
exploration).

tilt_block.py proves that the second variation under pure tilts is positive
semidefinite and vanishes, beyond the six rotations, on a nine-dimensional
space F.  Along straight lines t k, k in F, the volume grows at fourth order,
(vol - 8)/t^4 between 0.05 and 0.23 for random k.  This script lets the other
57 tilt directions relax: it minimises (vol - 8)/t^4 over c = t k + t^2 y,
k a unit vector of F and y in the complement of the kernel, jointly (L-BFGS,
central differences, 4 starts), for t = 0.05, 0.1, 0.2, 0.3, 0.45, and
describes the minimiser: vol - 8, the unrelaxed value along the same k, the
deviation of its sorted Gram rows from those of the root system (zero for any
rotation and relabelling of it), the number of distinct vertices of its
contact cell, the range of the facet 3-volumes (4/3 at the root system), the
largest inner product, and how far antipodal roots are from staying antipodal.
At the point found for t = 0.3 it also prints the smallest eigenvalues of the
finite-difference Hessian in the 66 coordinates that are rotation-free at the
root system.  The curve it finds is the one of prop:hexagon-loop: one A2
hexagon of roots turns in its plane and the other eighteen tilt towards the
orthogonal plane (hexagon_loop.py gives it in closed form).
Usage: python3 contact_valley.py
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import HalfspaceIntersection, ConvexHull
import tilt_quartic as TQ


def vol(c):
    hs = np.hstack([TQ.dirs(c), -np.ones((24, 1))])
    P = HalfspaceIntersection(hs, np.zeros(4)).intersections
    for opt in ('Qt', 'Qt Q12', 'QJ'):
        try:
            return ConvexHull(P, qhull_options=opt).volume
        except Exception:
            pass
    raise RuntimeError('qhull')


def describe(c):
    w = TQ.dirs(c)
    G = w @ w.T; G0 = TQ.U @ TQ.U.T
    dev = np.abs(np.sort(G, axis=1) - np.sort(G0, axis=1)).max()
    hs = np.hstack([w, -np.ones((24, 1))])
    V = np.unique(np.round(HalfspaceIntersection(hs, np.zeros(4)).intersections, 8), axis=0)
    fv = []
    for i in range(24):
        P = V[np.abs(V @ w[i] - 1) < 1e-7]
        B = np.linalg.svd(np.eye(4) - np.outer(w[i], w[i]))[0][:, :3]
        fv.append(ConvexHull((P - w[i]) @ B, qhull_options='QJ').volume)
    anti = max(np.min(np.linalg.norm(w + w[i], axis=1)) for i in range(24))
    off = G[~np.eye(24, dtype=bool)]
    return dev, len(V), min(fv), max(fv), off.max(), anti


def main():
    rng = np.random.default_rng(2026)
    print('t     vol - 8     (vol-8)/t^4  unrelaxed/t^4  Gram dev  vertices  facet 3-volumes     max <w_i,w_j>  antipodal  [time]')
    keep = {}
    for t in (0.05, 0.1, 0.2, 0.3, 0.45):
        t0 = time.time(); best = None
        for s in range(4):
            z0 = np.concatenate([rng.normal(size=9), np.zeros(57)])
            r = minimize(TQ.objective, z0, args=(t,), jac=TQ.grad, method='L-BFGS-B',
                         options={'maxiter': 300, 'gtol': 1e-10})
            if best is None or r.fun < best.fun:
                best = r
        v = best.x[:9] / np.linalg.norm(best.x[:9])
        c = t * (v @ TQ.FLAT) + t * t * (best.x[9:] @ TQ.TRANS)
        keep[t] = c
        unrel = (vol(t * (v @ TQ.FLAT)) - 8) / t ** 4
        dev, nv, fmin, fmax, mx, anti = describe(c)
        print('%.2f  %9.2e   %10.2e   %10.4f    %.4f    %4d     [%.5f, %.5f]   %.4f        %.1e   [%.0fs]'
              % (t, vol(c) - 8, (vol(c) - 8) / t ** 4, unrel, dev, nv, fmin, fmax, mx, anti, time.time() - t0), flush=True)
    c0 = keep[0.3]; basis = np.vstack([TQ.FLAT, TQ.TRANS]); h = 2e-4; n = 66; f0 = vol(c0)
    fp = [vol(c0 + h * basis[i]) for i in range(n)]
    Hm = np.zeros((n, n))
    for i in range(n):
        Hm[i, i] = (fp[i] - 2 * f0 + vol(c0 - h * basis[i])) / h ** 2
        for j in range(i + 1, n):
            Hm[i, j] = Hm[j, i] = (vol(c0 + h * (basis[i] + basis[j])) - fp[i] - fp[j] + f0) / h ** 2
    ev = np.sort(np.linalg.eigvalsh(Hm))
    print('Hessian at the point for t = 0.3, smallest eigenvalues:', ' '.join('%.4f' % x for x in ev[:10]))
    print('(one zero: the curve of volume 8 through the point; the next six come from the rotations, whose')
    print(' tangents at the point are not quite the ones removed at the root system)')


if __name__ == '__main__':
    main()
