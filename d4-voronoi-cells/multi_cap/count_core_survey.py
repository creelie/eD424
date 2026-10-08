#!/usr/bin/env python3
"""
count_core_survey.py -- the right side T of lem:no-triple for M centres within
sqrt 6 of which at least 23 lie within rho_M (floating point, exploration, not
proof; rem:what-c-needs).

A packing set of M centres, 25 <= M <= 30, with T <= 8 has at least 23 centres
within rho_M, the distance at which 22 S(2) + (M - 22) S(rho_M) equals
9 pi^2/8 - 8 (rho_25 = 2.2677, rho_30 = 2.3273): otherwise the union of the
caps is below that level (rem:what-c-needs).  The unrestricted minima of
count_survey.py do not satisfy this -- they keep only 22 centres that close --
so they say nothing about the case a proof still has to treat.  This script
minimises T over packings of M centres in the shell with the first K of them
(K = 23 or 24) held within rho_M, from starts near the root system and random
starts, with the functions and the exact constraint Jacobian of
count_survey.py.

Usage: python3 count_core_survey.py M K [starts] [seed]
"""
import math
import sys

import numpy as np
from scipy.optimize import brentq, minimize

from truncated_search import S, T, roots
from count_survey import gradT

R6 = 6.0


def rho(M):
    lvl = (9 * math.pi ** 2 / 8 - 8 - 22 * float(S(2.0))) / (M - 22)
    return brentq(lambda d: float(S(d)) - lvl, 2.0, math.sqrt(6) - 1e-9)


def run(M, K, nstart, rng):
    rM = rho(M)
    I, J = np.triu_indices(M, 1)
    ncon = 2 * M + len(I) + K

    def cons(x):
        Y = x.reshape(M, 4); n2 = np.sum(Y * Y, axis=1); D = Y[I] - Y[J]
        return np.concatenate([n2 - 4, R6 - n2, np.sum(D * D, axis=1) - 4, rM * rM - n2[:K]])

    def jac(x):
        Y = x.reshape(M, 4); Jm = np.zeros((ncon, M, 4)); a = np.arange(M)
        Jm[a, a] = 2 * Y
        Jm[M + a, a] = -2 * Y
        D = Y[I] - Y[J]; k = 2 * M + np.arange(len(I))
        Jm[k, I] = 2 * D
        Jm[k, J] = -2 * D
        b = np.arange(K)
        Jm[2 * M + len(I) + b, b] = -2 * Y[:K]
        return Jm.reshape(ncon, 4 * M)

    def pen(x):
        return np.sum(np.minimum(cons(x), 0) ** 2)

    def start(k):
        if k % 2 == 0:
            base = roots() * (1 + 0.04 * rng.random(24))[:, None] + 0.08 * rng.standard_normal((24, 4))
            Y = list(base[:K]) + list(base[K:24])
        else:
            Y = []
        while len(Y) < M:
            v = rng.standard_normal(4); v /= np.linalg.norm(v)
            Y.append(v * (2 + (rM - 2 if len(Y) < K else math.sqrt(6) - 2) * rng.random()))
        x = minimize(lambda x: 1e4 * pen(x), np.array(Y).ravel(), method='L-BFGS-B', options={'maxiter': 20000}).x
        x = minimize(pen, x, method='L-BFGS-B', options={'maxiter': 20000, 'ftol': 1e-30, 'gtol': 1e-16}).x
        return x if np.min(cons(x)) >= -1e-9 else None

    best = None
    for k in range(nstart):
        x0 = start(k)
        if x0 is None:
            print('  start %2d: no packing found' % k, flush=True)
            continue
        r = minimize(lambda x: T(x.reshape(M, 4)), x0, jac=lambda x: gradT(x.reshape(M, 4)).ravel(), method='SLSQP',
                     constraints=[{'type': 'ineq', 'fun': cons, 'jac': jac}], options={'maxiter': 3000, 'ftol': 1e-12})
        x = r.x
        if np.min(cons(x)) < -1e-8:
            x = minimize(pen, x, method='L-BFGS-B', options={'maxiter': 20000, 'ftol': 1e-30, 'gtol': 1e-16}).x
            if np.min(cons(x)) < -1e-8:
                print('  start %2d: ended outside the constraints (discarded)' % k, flush=True)
                continue
        Y = x.reshape(M, 4); d = np.sort(np.linalg.norm(Y, axis=1)); v = T(Y)
        print('  start %2d: T = %.6f; distances %s' % (k, v, ' '.join('%.3f' % t for t in d)), flush=True)
        if best is None or v < best[0]:
            best = (v, d)
    return rM, best


def main():
    M, K = int(sys.argv[1]), int(sys.argv[2])
    nstart = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    rM, best = run(M, K, nstart, rng)
    if best:
        print('M = %d, at least %d centres within rho_M = %.4f: least T found %.6f' % (M, K, rM, best[0]))
    print('(floating point and local optimisation: evidence, not a bound)')


if __name__ == '__main__':
    main()
