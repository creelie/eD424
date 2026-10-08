#!/usr/bin/env python3
"""
code24_second.py -- 24-point spherical codes in S^3 other than the root
system: the least slack s = (largest inner product) - 1/2 at which one exists
(floating point; exploration).

Statement (i) of "What is left" asks that every 24 unit vectors with pairwise
inner products at most 1/2 + s lie near the root system, for s of order
10^-2.  contact_cell_constrained.py finds 24-point codes far from the root
system at s = 0.02.  This script starts from random configurations, spreads
them into codes of slack 0.03, and then minimises the largest inner product
t (SLSQP on (z, t) with the 276 constraints <w_i, w_j> <= t).  It prints the
least t reached by minimisers far from the root system (sorted Gram rows at
least 0.05 from those of D4) and by those near it, and saves the best far
code to code24_second.npy.
Usage: python3 code24_second.py [starts]
"""
import sys
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from contact_cell_constrained import IU, ROOTS, G0, unit, cons, cons_jac, spread


def minimax(z):
    w, _ = unit(z)
    t0 = (w @ w.T)[IU].max()
    x0 = np.concatenate([z, [t0]])
    fun = lambda x: x[-1]
    jac = lambda x: np.concatenate([np.zeros(96), [1.0]])
    def c(x):
        return x[-1] - 0.5 + cons(x[:-1], 0)     # t - <w_i, w_j>
    def cj(x):
        J = -cons_jac(x[:-1], 0)
        return np.hstack([-J, np.ones((J.shape[0], 1))])
    r = minimize(fun, x0, jac=jac, method='SLSQP',
                 constraints=[{'type': 'ineq', 'fun': c, 'jac': cj}],
                 options={'maxiter': 1000, 'ftol': 1e-14})
    w, _ = unit(r.x[:-1])
    G = w @ w.T
    return G[IU].max(), np.abs(np.sort(G, axis=1) - G0).max(), w


def one(seed):
    rng = np.random.default_rng(seed)
    z = spread(rng.normal(size=96), 0.03)
    if cons(z, 0.03).min() < -1e-3:
        return seed, None, None, None
    t, dev, w = minimax(z)
    return seed, t, dev, w


def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    t0 = time.time()
    with Pool(4) as pool:
        res = [r for r in pool.map(one, range(starts)) if r[1] is not None]
    far = sorted([r for r in res if r[2] > 0.05], key=lambda r: r[1])
    near = sorted([r for r in res if r[2] <= 0.05], key=lambda r: r[1])
    print('%d of %d starts reached a code of slack 0.03  [%.0fs]' % (len(res), starts, time.time() - t0))
    print('near the root system: %d minimisers, least slack %s' % (len(near), '%.2e' % (near[0][1] - 0.5) if near else '-'))
    print('far from it: %d minimisers; least slacks:' % len(far))
    for r in far[:8]:
        print('   slack %.6f  (largest inner product %.6f, Gram deviation from D4 %.4f)' % (r[1] - 0.5, r[1], r[2]))
    if far:
        np.save('code24_second.npy', far[0][3])


if __name__ == '__main__':
    main()
