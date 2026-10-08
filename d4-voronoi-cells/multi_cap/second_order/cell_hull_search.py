#!/usr/bin/env python3
"""
cell_hull_search.py -- the least value of vol(V(Y) ∩ K(Y)) over sets Y of 24
centres within sqrt6 of c = 0 (floating point, qhull; exploration).

By prop:inversion-hull (the inversion hull), V_c contains V(Y) ∩ K(Y) for every
set Y of centres, whatever the rest of the packing does; with Y the centres
within sqrt6, the case of conj:shell with exactly 24 of them follows from

    (G)  vol(V(Y) ∩ K(Y)) >= 8 for every packing set Y of 24 centres with
         2 <= |y| <= sqrt6, with equality only at the root system.

Globally (G) is false: with centres pushed out towards sqrt6 the inversion
hull shrinks, and the unrestricted minimisation reaches 4.75 on a packing
(runs/cell_hull_search_unrestricted.log; the configuration is saved in
cell_hull_free_example.npy).  There the pair terms of lem:no-triple
settle the case, since their bound T(Y) (truncated_search.py) exceeds 8.  The
statement the case needs is (G) on the gap {T(Y) <= 8}, and with "gap" as the
third argument the script adds that constraint: it pulls each start into the
gap by a penalty and minimises vol(V(Y) ∩ K(Y)) subject to the packing
constraints and T(Y) <= 8.  It minimises from random starts (directions and
distances spread into a packing by a penalty) and from perturbations of the
root system, with the packing constraints |y_i| >= 2, |y_i| <= sqrt6,
|y_i - y_j| >= 2 (SLSQP, forward differences), and prints the least values
found, the distances of the minimiser from c, the deviation of its
directions from the root system, and T(Y).
Usage: python3 cell_hull_search.py [random starts] [near starts] [gap]
"""
import sys
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize, linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
sys.path.insert(0, '..')
from truncated_search import T as pair_bound

GAP = len(sys.argv) > 3 and sys.argv[3] == 'gap'

IU = np.triu_indices(24, 1)
ROOTS = np.array([np.eye(4)[i] * a + np.eye(4)[j] * b for i in range(4) for j in range(i + 1, 4)
                  for a in (1, -1) for b in (1, -1)])          # norm sqrt2; centres at 2 are sqrt2 * ROOTS
G0 = np.sort((ROOTS @ ROOTS.T) / 2, axis=1)
R6 = np.sqrt(6.0)


def cell_volume(Y):
    """vol(V(Y) ∩ K(Y)), V(Y) = {<x,y> <= |y|^2/2}, K(Y) = conv{0, 4y/|y|^2}."""
    n2 = np.sum(Y * Y, axis=1)
    hsV = np.hstack([Y, -(n2 / 2)[:, None]])
    Q = np.vstack([np.zeros(4), 4 * Y / n2[:, None]])
    try:
        hK = ConvexHull(Q)
    except Exception:
        return 1e2
    hs = np.vstack([hsV, hK.equations])
    # interior point: Chebyshev centre
    A = hs[:, :4]; b = -hs[:, 4]; nrm = np.linalg.norm(A, axis=1)
    lp = linprog(np.r_[np.zeros(4), -1.0], A_ub=np.hstack([A, nrm[:, None]]), b_ub=b,
                 bounds=[(None, None)] * 4 + [(0, None)], method='highs')
    if lp.status != 0 or lp.x[4] < 1e-9:
        return 1e2
    try:
        P = HalfspaceIntersection(hs, lp.x[:4]).intersections
        return ConvexHull(P).volume
    except Exception:
        try:
            return ConvexHull(P, qhull_options='QJ').volume
        except Exception:
            return 1e2


def fgrad(z, h=1e-6):
    Y = z.reshape(24, 4); f0 = cell_volume(Y)
    g = np.zeros(96)
    for k in range(96):
        e = np.zeros(96); e[k] = h
        g[k] = (cell_volume((z + e).reshape(24, 4)) - f0) / h
    return f0, g


def cons(z):
    Y = z.reshape(24, 4); n2 = np.sum(Y * Y, axis=1)
    D = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)[IU]
    return np.concatenate([n2 - 4, 6 - n2, D - 4])


def gap(z):
    return np.array([8.0 - pair_bound(z.reshape(24, 4))])


def gap_jac(z, h=1e-7):
    g0 = gap(z)[0]; J = np.zeros((1, 96))
    for k in range(96):
        e = np.zeros(96); e[k] = h
        J[0, k] = (gap(z + e)[0] - g0) / h
    return J


def cons_jac(z):
    Y = z.reshape(24, 4)
    J = np.zeros((48 + len(IU[0]), 96))
    for i in range(24):
        J[i, 4 * i:4 * i + 4] = 2 * Y[i]
        J[24 + i, 4 * i:4 * i + 4] = -2 * Y[i]
    for k, (i, j) in enumerate(zip(*IU)):
        d = 2 * (Y[i] - Y[j])
        J[48 + k, 4 * i:4 * i + 4] = d; J[48 + k, 4 * j:4 * j + 4] = -d
    return J


def spread(z):
    pen = lambda z: np.sum(np.minimum(cons(z), 0) ** 2)
    pg = lambda z: 2 * np.minimum(cons(z), 0) @ cons_jac(z)
    return minimize(pen, z, jac=pg, method='L-BFGS-B', options={'maxiter': 5000}).x


def one(args):
    seed, near = args
    rng = np.random.default_rng(seed)
    if near:
        # inside the gap: tilts of size `near`, and a few centres pushed out, one of them up to delta = 0.19
        push = np.zeros(24)
        k = rng.integers(0, 5)
        idx = rng.choice(24, size=k, replace=False)
        push[idx] = rng.uniform(0, 0.19, size=k) / 2
        Y = np.sqrt(2) * ROOTS * (1 + push[:, None]) + near * rng.normal(size=(24, 4))
    else:
        W = rng.normal(size=(24, 4)); W /= np.linalg.norm(W, axis=1, keepdims=True)
        Y = W * rng.uniform(2.0, 2.35, size=(24, 1))
    z = spread(Y.ravel())
    constraints = [{'type': 'ineq', 'fun': cons, 'jac': cons_jac}]
    if GAP:
        pen = lambda z: np.sum(np.minimum(cons(z), 0) ** 2) + np.minimum(gap(z)[0], 0) ** 2
        z = minimize(pen, z, method='L-BFGS-B', options={'maxiter': 3000}).x
        if gap(z)[0] < -1e-6:
            return seed, near, None, None, None, None, None, None
        constraints.append({'type': 'ineq', 'fun': gap, 'jac': gap_jac})
    if cons(z).min() < -1e-6:
        return seed, near, None, None, None, None, None, None
    r = minimize(fgrad, z, jac=True, method='SLSQP', constraints=constraints,
                 bounds=[(-R6, R6)] * 96, options={'maxiter': 300, 'ftol': 1e-11})
    Y = r.x.reshape(24, 4); d = np.linalg.norm(Y, axis=1); W = Y / d[:, None]
    dev = np.abs(np.sort(W @ W.T, axis=1) - G0).max()
    feas = min(cons(r.x).min(), gap(r.x)[0] if GAP else 0.0)
    return seed, near, cell_volume(Y), feas, np.sort(d), dev, pair_bound(Y), Y


def main():
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    print('root system: vol(V(Y) ∩ K(Y)) = %.10f' % cell_volume(np.sqrt(2) * ROOTS), flush=True)
    scales = [0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3]
    jobs = [(k, 0.0) for k in range(nr)] + [(10000 + k, scales[k % len(scales)]) for k in range(nn)]
    res = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(one, jobs):
            res.append(r)
            if r[2] is not None:
                print('   start %5d: vol %.8f  feasibility %.1e  distances %.4f..%.4f  direction deviation %.4f  T %.5f  [%.0fs]'
                      % (r[0], r[2], r[3], r[4][0], r[4][-1], r[5], r[6], time.time() - t0), flush=True)
    ok = [r for r in res if r[2] is not None and r[3] > -1e-6]
    print('%d of %d starts ended at a packing%s  [%.0fs]' % (len(ok), len(jobs), ' in the gap T <= 8' if GAP else '', time.time() - t0))
    for r in sorted(ok, key=lambda r: r[2])[:12]:
        print('   vol %.8f (vol - 8 = %+.2e)  distances %.4f..%.4f, %2d within 2.02  direction deviation %.4f  T %.5f  %s'
              % (r[2], r[2] - 8, r[4][0], r[4][-1], int(np.sum(r[4] < 2.02)), r[5], r[6], 'near start' if r[1] else 'random start'))
    if ok:
        best = min(ok, key=lambda r: r[2])
        print('least value found: %.8f' % best[2])
        np.save('cell_hull_best_%s.npy' % ('gap' if GAP else 'free'), best[7])


if __name__ == '__main__':
    main()
