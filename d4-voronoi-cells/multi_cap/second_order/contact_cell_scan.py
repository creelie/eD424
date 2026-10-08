#!/usr/bin/env python3
"""
contact_cell_scan.py -- the least contact-cell volume vol{x : <x, w_i> <= 1}
over all sets of 24 unit directions, with no packing constraint (floating
point, qhull; exploration).

Near the root system the contact-cell volume is at least 8 on every set
tested, with equality along closed curves (prop:hexagon-loop,
hexagon_loop.py).  Far from it the
packing constraint (pairwise angles at least 60 degrees) is what forces the
root system, by thm:m24.  Without the constraint, the question is the
least volume of a polytope with 24 facets circumscribed about the unit ball.
The script minimises the volume over 24 unit vectors from random starts
(L-BFGS on the unnormalised vectors, central differences) and prints the
least volume found and the largest inner product of the minimiser, which
says how far it is from satisfying the packing constraint.
Usage: python3 contact_cell_scan.py [starts]
"""
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import HalfspaceIntersection, ConvexHull


def vol_dirs(z):
    w = z.reshape(24, 4)
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    try:
        hs = np.hstack([w, -np.ones((24, 1))])
        P = HalfspaceIntersection(hs, np.zeros(4)).intersections
        if not np.all(np.isfinite(P)) or np.abs(P).max() > 1e3:
            return 1e3
        return ConvexHull(P, qhull_options='QJ').volume
    except Exception:
        return 1e3            # unbounded or degenerate


def grad(z, h=1e-6):
    g = np.zeros_like(z)
    for j in range(len(z)):
        e = np.zeros_like(z); e[j] = h
        g[j] = (vol_dirs(z + e) - vol_dirs(z - e)) / (2 * h)
    return g


def facet_check(w):
    hs = np.hstack([w, -np.ones((24, 1))])
    V = np.unique(np.round(HalfspaceIntersection(hs, np.zeros(4)).intersections, 9), axis=0)
    area, nf = 0.0, 0
    for i in range(24):
        P = V[np.abs(V @ w[i] - 1) < 1e-7]
        if len(P) < 4:
            continue
        B = np.linalg.svd(np.eye(4) - np.outer(w[i], w[i]))[0][:, :3]
        try:
            a = ConvexHull((P - w[i]) @ B, qhull_options='QJ').volume
        except Exception:
            continue
        if a > 1e-12:
            area += a; nf += 1
    return ConvexHull(V, qhull_options='Qt').volume, area / 4, nf, len(V)


def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rng = np.random.default_rng(11)
    best = (np.inf, None); t0 = time.time()
    for s in range(starts):
        z0 = rng.normal(size=96)
        while vol_dirs(z0) >= 1e3:
            z0 = rng.normal(size=96)
        r = minimize(vol_dirs, z0, jac=grad, method='L-BFGS-B', options={'maxiter': 400})
        w = r.x.reshape(24, 4); w /= np.linalg.norm(w, axis=1, keepdims=True)
        G = w @ w.T; mx = G[~np.eye(24, dtype=bool)].max()
        if r.fun < best[0]:
            best = (r.fun, mx); wbest = w.copy()
        print('start %2d: volume %.6f, largest inner product %.4f   [%.0fs]' % (s, r.fun, mx, time.time() - t0), flush=True)
    print('least contact-cell volume found over 24 unit directions, no packing constraint: %.6f '
          '(largest inner product %.4f); the root system gives 8' % best)
    v1, v2, nf, nv = facet_check(wbest)
    print('best minimiser: volume %.8f without joggling, %.8f as a quarter of the surface area; %d facets, %d vertices'
          % (v1, v2, nf, nv))
    np.save('contact_cell_min.npy', wbest)


if __name__ == '__main__':
    main()
