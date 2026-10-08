#!/usr/bin/env python3
"""
hole_room.py -- the room for the hole form (H) of the budgeted localisation:
the least T over 25 centres when the 25th is held at distance r from c.
Statement (C) at 25 centres asks T > 8 for every r; the paper's search finds
8.264 as the least T over all 25-centre packings.  Local minimisation
(SLSQP) from root systems pushed out, with the 25th centre started in a hole
of largest depth (45 degrees from the nearest root direction).  Floating point,
exploration only.

    python3 hole_room.py starts seed r1 r2 ...
"""
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'multi_cap'))
from truncated_search import T, roots  # noqa: E402

A = np.array(roots(), float)
A = A / np.linalg.norm(A, axis=1)[:, None]
N = 25
iu, ju = np.triu_indices(N, 1)
HOLE = np.array([1.0, 0.0, 0.0, 0.0])  # at 45 degrees from (e0 +- e1)/sqrt2 etc.: a deepest hole of the roots


def best(r, starts, seed):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(starts):
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        W = A @ Q.T + 0.03 * rng.standard_normal((24, 4))
        W /= np.linalg.norm(W, axis=1)[:, None]
        Y0 = np.r_[W * (2.0 + 0.3 * rng.random((24, 1))), [r * (HOLE @ Q.T)]]
        cons = [{'type': 'ineq', 'fun': lambda v: np.sum(v.reshape(N, 4) ** 2, axis=1) - 4},
                {'type': 'ineq', 'fun': lambda v: 6 - np.sum(v.reshape(N, 4) ** 2, axis=1)},
                {'type': 'eq', 'fun': lambda v: np.sum(v.reshape(N, 4)[-1] ** 2) - r * r},
                {'type': 'ineq', 'fun': lambda v: np.sum((v.reshape(N, 4)[iu] - v.reshape(N, 4)[ju]) ** 2, axis=1) - 4}]
        # first spread to feasibility, then minimise T
        res = minimize(lambda v: 0.0 * T(v.reshape(N, 4)) + np.sum((v - Y0.ravel()) ** 2), Y0.ravel(), method='SLSQP',
                       constraints=cons, options={'maxiter': 300, 'ftol': 1e-10})
        res = minimize(lambda v: T(v.reshape(N, 4)), res.x, method='SLSQP', constraints=cons,
                       options={'maxiter': 800, 'ftol': 1e-12})
        Y = res.x.reshape(N, 4)
        d = np.linalg.norm(Y, axis=1)
        gap = np.min(np.sum((Y[iu] - Y[ju]) ** 2, axis=1)) - 4
        if gap >= -1e-7 and d.min() >= 2 - 1e-7 and d.max() <= math.sqrt(6) + 1e-7 and abs(d[-1] - r) < 1e-6:
            out.append(T(Y))
    return out


def main():
    starts, seed = int(sys.argv[1]), int(sys.argv[2])
    for r in map(float, sys.argv[3:]):
        vals = best(r, starts, seed)
        print('25th centre at %.4f: least T found %s over %d feasible ends' % (
            r, '%.4f' % min(vals) if vals else '--', len(vals)), flush=True)


if __name__ == '__main__':
    main()
