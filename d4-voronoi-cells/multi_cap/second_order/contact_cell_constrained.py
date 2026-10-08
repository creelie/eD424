#!/usr/bin/env python3
"""
contact_cell_constrained.py -- the least contact-cell volume
vol{x : <x, w_i> <= 1} over 24 unit directions with pairwise inner products
at most 1/2 + s (floating point, qhull; exploration).

With prop:push-integrated, a bound vol(Q_w) >= 8 on this set would give the first
inequality of statement (ii) for every set of 24 centres whose directions
have slack at most s, near the root system or not.  Without the constraint
the volume drops to 7.96553 (contact_cell_scan.py); along the curves of
prop:hexagon-loop it is exactly 8 at slack 4C^2/(4C^2+3) - 1/2.  This script
minimises the volume on the constrained set from random starts, which are
first spread into codes of slack s by a penalty, and from perturbations of
the root system, with the exact gradient: moving the facet normal w_i by dw
changes the volume by -A_i <c_i, dw>, A_i the 3-volume and c_i the centroid
of the facet.  It prints, for each slack, the least volume found, how far
the minimiser is from the root system (the deviation of its sorted Gram rows),
its largest inner product, and how many starts reached a code of that slack.
Usage: python3 contact_cell_constrained.py [starts] [slacks...]
"""
import sys
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.spatial import HalfspaceIntersection, ConvexHull

IU = np.triu_indices(24, 1)
ROOTS = np.array([np.eye(4)[i] * a + np.eye(4)[j] * b for i in range(4) for j in range(i + 1, 4)
                  for a in (1, -1) for b in (1, -1)]) / np.sqrt(2)
G0 = np.sort(ROOTS @ ROOTS.T, axis=1)


def facet_data(w):
    """Volume of the contact cell and, for each facet, its 3-volume and centroid."""
    hs = np.hstack([w, -np.ones((24, 1))])
    V = HalfspaceIntersection(hs, np.zeros(4)).intersections
    V = np.unique(np.round(V, 12), axis=0)
    A = np.zeros(24); C = np.zeros((24, 4))
    for i in range(24):
        P = V[np.abs(V @ w[i] - 1) < 1e-8]
        if len(P) < 4:
            continue
        B = np.linalg.svd(np.eye(4) - np.outer(w[i], w[i]))[0][:, :3]
        X = (P - w[i]) @ B
        try:
            hull = ConvexHull(X, qhull_options='Qt')
        except Exception:
            hull = ConvexHull(X, qhull_options='QJ')
        o = X.mean(axis=0)
        vol = 0.0; cen = np.zeros(3)
        for tri in hull.simplices:
            T = X[tri]
            v = abs(np.linalg.det(T - o)) / 6
            vol += v; cen += v * (T.sum(axis=0) + o) / 4
        if vol > 0:
            A[i] = vol; C[i] = w[i] + B @ (cen / vol)
    return A.sum() / 4, A, C


def unit(z):
    Z = z.reshape(24, 4)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True), np.linalg.norm(Z, axis=1)


def f_and_grad(z):
    w, nz = unit(z)
    try:
        v, A, C = facet_data(w)
    except Exception:
        return 1e3, np.zeros(96)
    if not np.isfinite(v) or v > 1e3:
        return 1e3, np.zeros(96)
    g = np.zeros((24, 4))
    for i in range(24):
        gw = -A[i] * C[i]
        g[i] = (gw - (gw @ w[i]) * w[i]) / nz[i]
    return v, g.ravel()


def cons(z, s):
    w, _ = unit(z)
    return 0.5 + s - (w @ w.T)[IU]


def cons_jac(z, s):
    w, nz = unit(z)
    J = np.zeros((len(IU[0]), 96))
    for k, (i, j) in enumerate(zip(*IU)):
        gi = -(w[j] - (w[j] @ w[i]) * w[i]) / nz[i]
        gj = -(w[i] - (w[i] @ w[j]) * w[j]) / nz[j]
        J[k, 4 * i:4 * i + 4] = gi; J[k, 4 * j:4 * j + 4] = gj
    return J


def spread(z, s):
    """Push a random configuration towards a code of slack s."""
    for k in (0.0, 0.3, 0.45, 0.5):
        pen = lambda z: np.sum(np.maximum(-cons(z, s - 0.5 + k if k < 0.5 else s), 0) ** 2)
        def pg(z):
            c = cons(z, s - 0.5 + k if k < 0.5 else s)
            return 2 * (-np.maximum(-c, 0)) @ cons_jac(z, 0) * 1.0
        z = minimize(pen, z, jac=pg, method='L-BFGS-B', options={'maxiter': 3000}).x
    return z


def one(args):
    seed, s, near = args
    rng = np.random.default_rng(seed)
    if near:
        z = spread((ROOTS + near * rng.normal(size=(24, 4))).ravel(), s)
    else:
        z = spread(rng.normal(size=96), s)
    feas0 = cons(z, s).min()
    if feas0 < -1e-3:
        return (seed, s, near, None, feas0, None, None)
    r = minimize(f_and_grad, z, jac=True, method='SLSQP',
                 constraints=[{'type': 'ineq', 'fun': cons, 'jac': cons_jac, 'args': (s,)}],
                 options={'maxiter': 400, 'ftol': 1e-12})
    w, _ = unit(r.x)
    G = w @ w.T
    dev = np.abs(np.sort(G, axis=1) - G0).max()
    return (seed, s, near, r.fun, cons(r.x, s).min(), dev, G[IU].max())


def check_gradient():
    rng = np.random.default_rng(1)
    z = (ROOTS + 0.05 * rng.normal(size=(24, 4))).ravel()
    v, g = f_and_grad(z)
    e = rng.normal(size=96); h = 1e-6
    fd = (f_and_grad(z + h * e)[0] - f_and_grad(z - h * e)[0]) / (2 * h)
    print('gradient check: exact %.8f, central difference %.8f' % (g @ e, fd), flush=True)


def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    slacks = [float(x) for x in sys.argv[2:]] or [0.005, 0.01, 0.02, 0.03]
    check_gradient()
    t0 = time.time()
    for s in slacks:
        jobs = [(1000 * k + int(1e4 * s), s, 0.0) for k in range(starts)]
        jobs += [(5000 + k, s, 0.02 + 0.02 * (k % 4)) for k in range(8)]
        with Pool(4) as pool:
            res = pool.map(one, jobs)
        ok = [r for r in res if r[3] is not None and r[4] > -1e-7]
        print('slack %.3f: %d of %d starts reached a code of this slack' % (s, sum(1 for r in res if r[3] is not None), len(res)))
        for r in sorted(ok, key=lambda r: r[3])[:6]:
            print('   volume %.8f  (vol - 8 = %+.2e)  Gram deviation from D4 %.4f  largest inner product %.5f  %s'
                  % (r[3], r[3] - 8, r[5], r[6], 'near D4 start' if r[2] else 'random start'))
        far = [r for r in ok if r[5] > 0.05]
        if far:
            b = min(far, key=lambda r: r[3])
            print('   least volume among minimisers away from D4 (Gram deviation > 0.05): %.8f (deviation %.4f)' % (b[3], b[5]))
        print('   [%.0fs]' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
