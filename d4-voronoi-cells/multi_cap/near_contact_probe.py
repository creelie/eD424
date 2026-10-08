#!/usr/bin/env python3
"""
near_contact_probe.py -- exploration, not proof: the numbers behind the
remarks of sec:closure and sec:remains about the inversion hull and about the conjecture near
the root system.  Floating point throughout (scipy's polytope routines and
SLSQP); nothing here is used in a proof.

  (1) the inversion-hull bound F(Y) = vol(V(Y) cap K(Y)) of prop:inversion-hull,
      K(Y) the convex hull of 0 and the points 4y/|y|^2, beside the true
      volume vol(V(Y)), on the root system, on the root system pushed out to
      distance 2 + delta, and on the deletion with one centre held at 2 + delta
      on the deleted root's axis (whose volume is 25/3 - (1/3)(1 - delta/2)^4);
  (2) near the root system: 24 centres around 0 are pushed a random amount
      out and turned a random amount, one of them is held at a prescribed
      distance 2 + s, and the volume of the cell is minimised under the
      packing constraints.  For each endpoint the script reports
        - ||eps|| / K, where eps is the displacement of the directions from
          the best-matching rotated root system and K the sum over the 96
          tight pairs of the slack kappa_ij that the distances allow; the
          proof of thm:near-contact shows ||eps|| <= 2.71 K once ||eps|| <= 1/48;
        - (vol - 8 - (2/3) sum delta) / (||eps||^2 + |delta|^2), which the
          proof bounds below by a constant; the on-axis family gives -1/2.

    python3 near_contact_probe.py [trials] [seed]
"""
import itertools
import math
import sys

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection

import shell_neighbour_search as S

R = S.R24                                   # the 24 unit roots
TIGHT = [(i, j) for i, j in itertools.combinations(range(24), 2) if abs(R[i] @ R[j] - 0.5) < 1e-12]
BOX = np.vstack([np.hstack([np.eye(4), -30 * np.ones((4, 1))]),
                 np.hstack([-np.eye(4), -30 * np.ones((4, 1))])])


def vol_hs(hs):
    hs = np.vstack([hs, BOX])
    A, b = hs[:, :4], -hs[:, 4]
    res = linprog(np.r_[np.zeros(4), -1], A_ub=np.c_[A, np.linalg.norm(A, axis=1)], b_ub=b,
                  bounds=[(None, None)] * 4 + [(0, None)])
    if res.x is None or res.x[4] < 1e-9:
        return 0.0
    return ConvexHull(HalfspaceIntersection(hs, res.x[:4]).intersections).volume


def cellV(Y):
    return np.hstack([Y, -0.5 * (Y * Y).sum(1)[:, None]])


def vol(Y):
    return vol_hs(cellV(Y))


def F(Y):
    Q = 4 * Y / (Y * Y).sum(1)[:, None]
    K = ConvexHull(np.vstack([np.zeros(4), Q])).equations
    return vol_hs(np.vstack([cellV(Y), K]))


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rng = np.random.default_rng(seed)

    print('(1) the inversion-hull bound F beside the volume')
    print('    root system:                   vol %.9f   F %.9f' % (vol(2 * R), F(2 * R)))
    for dl in (0.001, 0.01, 0.05):
        print('    all 24 at 2 + %.3f:            vol %.9f   F %.9f   (2/3) sum delta = %.4f'
              % (dl, vol((2 + dl) * R), F((2 + dl) * R), 16 * dl))
    for dl in (0.001, 0.01, 0.1, 0.4):
        Y = 2 * R.copy()
        Y[0] = (2 + dl) * R[0]
        print('    deletion, axis centre %.3f:    vol %.9f   F %.9f   exact %.9f'
              % (dl, vol(Y), F(Y), 25 / 3 - (1 / 3) * (1 - dl / 2) ** 4))

    print()
    print('(2) near the root system: %d volume minimisations with one centre held at 2 + s' % trials)
    worst_rig, worst_rem, n_used = 0.0, float('inf'), 0
    for _ in range(trials):
        s = 10 ** rng.uniform(-4, -1.5)
        Y0 = 2 * R @ np.linalg.qr(rng.normal(size=(4, 4)))[0]
        Y0 = Y0 * (1 + rng.uniform(0, s, size=(24, 1))) + rng.normal(size=(24, 4)) * s * 0.5
        k = int(rng.integers(0, 24))
        cons = [S.constraints(24),
                {'type': 'ineq', 'fun': lambda z, k=k: np.array([z[4 * k:4 * k + 4] @ z[4 * k:4 * k + 4] - (2 + s) ** 2])}]
        res = minimize(S.objective, Y0.ravel(), args=(24,), jac=True, method='SLSQP',
                       constraints=cons, options={'maxiter': 300, 'ftol': 1e-14})
        if S.constraints(24)['fun'](res.x).min() < -1e-10:
            continue
        Y = res.x.reshape(24, 4)
        d = np.linalg.norm(Y, axis=1)
        delta = d - 2
        Wd = Y / d[:, None]
        # best rotation onto the roots (the labels are kept: the start was a rotated root system)
        U, _, Vt = np.linalg.svd(Wd.T @ R)
        eps = Wd @ (U @ Vt) - R
        ne = np.linalg.norm(eps)
        K = sum((d[i] ** 2 + d[j] ** 2 - d[i] * d[j] - 4) / (2 * d[i] * d[j]) for i, j in TIGHT)
        if ne <= 1 / 48 and K > 0:
            worst_rig = max(worst_rig, ne / K)
        rem = (vol(Y) - 8 - (2 / 3) * delta.sum()) / (ne ** 2 + (delta ** 2).sum() + 1e-300)
        worst_rem = min(worst_rem, rem)
        n_used += 1
    print('    feasible endpoints: %d' % n_used)
    print('    largest ||eps|| / K:  %.6f   (the proof of thm:near-contact allows 2.71)' % worst_rig)
    print('    least (vol - 8 - (2/3) sum delta) / (||eps||^2 + |delta|^2):  %.4f' % worst_rem)
    print('RESULT: exploration, not proof')


if __name__ == '__main__':
    main()
