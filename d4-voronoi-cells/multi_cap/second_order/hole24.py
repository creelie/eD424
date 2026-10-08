#!/usr/bin/env python3
"""
hole24.py -- the deepest hole of a 24-point code of slack s in S^3
(floating point; exploration).

A further centre z within sqrt6 of c, beside 24 centres within 2 + delta, needs
<z/|z|, w_i> <= a(2 + delta, sqrt6) for every direction w_i, where
a(d, d') = (d^2 + d'^2 - 4)/(2 d d'); for delta = 0.0161 (thm:kissing-stable: at most
24 centres within 2.0161) that is 0.6141.  At the root system the least
possible value of max_i <theta, w_i> is 1/sqrt2 = 0.7071, at the 24 deep
holes.  This script minimises t = max_i <theta, w_i> jointly over theta and
over 24-point codes with inner products at most 1/2 + s, from the root system
with random perturbations and from random codes, and prints the least t found
for each slack, with the Gram deviation of the code from the root system.
Usage: python3 hole24.py [starts] [slacks...]
"""
import sys
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from contact_cell_constrained import IU, ROOTS, G0, unit, cons, cons_jac, spread

HOLE = np.array([1.0, 0, 0, 0])      # a vertex direction of the 24-cell: a deep hole


def solve(z0, th0, s):
    x0 = np.concatenate([z0, th0, [np.max(unit(z0)[0] @ (th0 / np.linalg.norm(th0)))]])
    def ang(x):
        th = x[96:100] / np.linalg.norm(x[96:100])
        return unit(x[:96])[0] @ th
    def c(x):
        return np.concatenate([cons(x[:96], s), x[-1] - ang(x)])
    def cj(x):
        z = x[:96]; w, nz = unit(z); tn = np.linalg.norm(x[96:100]); th = x[96:100] / tn
        J1 = np.hstack([cons_jac(z, s), np.zeros((len(IU[0]), 5))])
        J2 = np.zeros((24, 101))
        a = w @ th
        for i in range(24):
            J2[i, 4 * i:4 * i + 4] = -(th - a[i] * w[i]) / nz[i]
            J2[i, 96:100] -= (w[i] - a[i] * th) / tn
        J2[:, 100] = 1.0
        return np.vstack([J1, J2])
    r = minimize(lambda x: x[-1], x0, jac=lambda x: np.r_[np.zeros(100), 1.0], method='SLSQP',
                 constraints=[{'type': 'ineq', 'fun': c, 'jac': cj}], options={'maxiter': 1000, 'ftol': 1e-13})
    w, _ = unit(r.x[:96]); th = r.x[96:100] / np.linalg.norm(r.x[96:100])
    feas = cons(r.x[:96], s).min()
    return (w @ th).max(), feas, np.abs(np.sort(w @ w.T, axis=1) - G0).max()


def one(args):
    seed, s, near = args
    rng = np.random.default_rng(seed)
    if near:
        # the root system, a random orthogonal map applied, with the probe at a random deep hole
        Qm = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        z = (ROOTS @ Qm).ravel() + near * rng.normal(size=96)
        th = (np.eye(4)[rng.integers(4)] * rng.choice([-1, 1])) @ Qm + near * rng.normal(size=4)
    else:
        z = spread(rng.normal(size=96), s); th = rng.normal(size=4)
        if cons(z, s).min() < -1e-4:
            return seed, s, near, None, None, None
    t, feas, dev = solve(z, th, s)
    return seed, s, near, t, feas, dev


def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    slacks = [float(x) for x in sys.argv[2:]] or [0.0, 0.004, 0.008, 0.012, 0.017, 0.02]
    for s in slacks:
        jobs = [(k, s, 0.0) for k in range(starts)] + [(1000 + k, s, [0.01, 0.03, 0.06][k % 3]) for k in range(starts)]
        with Pool(4) as pool:
            res = [r for r in pool.map(one, jobs) if r[3] is not None and r[4] > -1e-7]
        best = min(res, key=lambda r: r[3])
        nearD4 = [r for r in res if r[5] < 0.05]
        bn = min(nearD4, key=lambda r: r[3]) if nearD4 else None
        print('slack %.3f: %d codes; least max <theta, w_i> = %.5f (Gram deviation %.4f); near the root system: %s'
              % (s, len(res), best[3], best[5], '%.5f' % bn[3] if bn else '-'), flush=True)


if __name__ == '__main__':
    main()
