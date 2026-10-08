#!/usr/bin/env python3
"""
budget_far.py -- how far from a root system can twenty-four centres be while
T(Y) <= 8 + eta?  The localisation (L) that (G) and (C) both need asserts a
bound on this distance.  Floating point, local optimisation from many starts:
it measures the room a proof of (L) would have, it proves nothing.

Twenty-four centres y_i with 2 <= |y_i| <= sqrt6 and |y_i - y_j| >= 2.  The
distance to the root system is the root-sum-square
    dist(Y) = min_{sigma, labelling} (sum_i |y_i/|y_i| - sigma(a_i)/sqrt2|^2)^(1/2),
computed by Procrustes alignment with the labelling re-matched (Hungarian
algorithm) until it is stable.  The programme maximises the Procrustes distance
for the current labelling subject to T(Y) <= 8 + eta and the packing
conditions, from the root system pushed out by a random amount, and reports
the true distance of the result.

    python3 budget_far.py eta starts seed [rmax]

With rmax the 24 centres are also held within rmax of c (the radial counts of
prop:C-radial put the 24 closest within 2.25 at thirty centres, 2.3 at
twenty-nine, 2.35 at twenty-seven and twenty-eight, and lem:twentyfour-close
within 2.444 always).
"""
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'multi_cap'))
from truncated_search import T, roots  # noqa: E402

A = np.array(roots(), float)
A = A / np.linalg.norm(A, axis=1)[:, None]
n = len(A)
iu, ju = np.triu_indices(n, 1)


def procrustes(W, B):
    U_, s, Vt = np.linalg.svd(W.T @ B)
    return U_ @ Vt, 2 * n - 2 * s.sum()  # rotation taking B to W, squared distance


def true_dist(W):
    best = None
    for _ in range(30):
        perm = np.arange(n)
        for _ in range(50):
            R, _ = procrustes(W, A[perm])
            C = -W @ (A @ R.T).T
            newp = linear_sum_assignment(C)[1]
            if np.all(newp == perm):
                break
            perm = newp
        R, d2 = procrustes(W, A[perm])
        if best is None or d2 < best:
            best = d2
        # restart from a random rotation for the next round
        Q, _ = np.linalg.qr(np.random.standard_normal((4, 4)))
        perm = linear_sum_assignment(-W @ (A @ Q).T)[1]
    return math.sqrt(max(best, 0))


def run(eta, starts, seed, keep=None, rmax=math.sqrt(6)):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    out = []
    for s in range(starts):
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        Y0 = (2.0 + 0.02 * rng.random((n, 1))) * (A @ Q.T + 0.05 * rng.standard_normal((n, 4)))
        Y0 = Y0 / np.linalg.norm(Y0, axis=1)[:, None] * (2.0 + 0.03 * rng.random((n, 1)))
        target = A @ Q.T

        def negdist(v):
            Y = v.reshape(n, 4)
            W = Y / np.linalg.norm(Y, axis=1)[:, None]
            return -procrustes(W, target)[1]

        cons = [{'type': 'ineq', 'fun': lambda v: 8 + eta - T(v.reshape(n, 4))},
                {'type': 'ineq', 'fun': lambda v: np.sum(v.reshape(n, 4) ** 2, axis=1) - 4},
                {'type': 'ineq', 'fun': lambda v: rmax ** 2 - np.sum(v.reshape(n, 4) ** 2, axis=1)},
                {'type': 'ineq', 'fun': lambda v: np.sum((v.reshape(n, 4)[iu] - v.reshape(n, 4)[ju]) ** 2, axis=1) - 4}]
        # first get feasible: minimise T
        r0 = minimize(lambda v: T(v.reshape(n, 4)), Y0.ravel(), method='SLSQP', constraints=cons[1:],
                      options={'maxiter': 400, 'ftol': 1e-12})
        r = minimize(negdist, r0.x, method='SLSQP', constraints=cons, options={'maxiter': 600, 'ftol': 1e-12})
        Y = r.x.reshape(n, 4)
        d = np.linalg.norm(Y, axis=1)
        W = Y / d[:, None]
        gap = np.min(np.sum((Y[iu] - Y[ju]) ** 2, axis=1)) - 4
        feas = T(Y) <= 8 + eta + 1e-7 and gap >= -1e-7 and d.min() >= 2 - 1e-7 and d.max() <= rmax + 1e-7
        td = true_dist(W)
        out.append((td, T(Y), np.sum(d - 2), feas))
        if keep is not None and feas:
            keep.append(Y)
        print('start %2d: T = %.5f, sum of push-outs %.4f, distance to a root system %.4f, max inner product %.4f%s'
              % (s, T(Y), np.sum(d - 2), td, np.max(np.sum(W[iu] * W[ju], axis=1)), '' if feas else '  (infeasible)'),
              flush=True)
    good = [o for o in out if o[3]]
    if good:
        print('eta = %.5f, rmax = %.4f: largest distance found %.4f over %d feasible ends'
              % (eta, rmax, max(o[0] for o in good), len(good)))



def ends(eta, starts, seed):
    """the feasible end configurations of run(eta, starts, seed)"""
    keep = []
    run(eta, starts, seed, keep)
    return keep


if __name__ == '__main__':
    run(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
        rmax=float(sys.argv[4]) if len(sys.argv) > 4 else math.sqrt(6))
