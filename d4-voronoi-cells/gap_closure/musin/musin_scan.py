#!/usr/bin/env python3
"""
musin_scan.py -- the distance-labelled two-point kernel of thm:count31 with
Musin's relaxation (Ann. of Math. 168 (2008)): the pair inequality K <= Pi is
dropped for nearly antipodal pairs, and the excess is bounded centre by centre
through the number of centres that fit near the antipode.  Floating point on
sampled constraints: it shows whether the method can reach the level
9 pi^2/8 - 8 for statement (C) at a given count, it proves nothing.

Notation as in multi_cap/radial_count_sdp.py.  Fix t0 in (0, 1) and call a
pair bad if <w, w'> < -t0 (its caps are disjoint, Pi = 0).  For a centre y let
E(y) be the sum of K(y, y') over the bad partners y' of y.  Summing the pair
inequality over the good ordered pairs and adding E,

    U(Y) <= sum_y [S(d) + K(y, y)/2 + E(y)/2 - z . p(d)] + t/2.

The bad partners of y lie in the cap of angular radius arccos(t0) about -w and
are pairwise at angle at least phi = arccos(2/3) (the least angle between two
centres of the shell, at |y| = |y'| = sqrt6).  So the j-th closest of them to
-w is at angle at least theta_j from it, theta_j being the least radius of a
cap of S^3 that holds j points pairwise at angle >= phi, and

    E(y) <= sum_j g_j(d),  g_j(d) >= max(0, K(y, y')) for -cos(theta_j) <= u < -t0.

theta_j is bounded below through the Tammes problem on the boundary sphere of
the cap (j = 2, 3, 4: diameter, triangle, tetrahedron; 5, 6: octahedron, ...).
The g_j are polynomials in p(d).  With t0 = 1 there is no bad pair and this is
the programme of radial_count_sdp.py.

    python3 musin_scan.py D r t0 M1 M2 ...
"""
import math
import os
import sys
import time

import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'multi_cap'))
from truncated_search import pair, S  # noqa: E402

R6 = math.sqrt(6)
TARGET = 9 * math.pi ** 2 / 8 - 8
C1, C2 = 2 + R6, R6 - 2
PHI = math.acos(2 / 3)
# cos of the Tammes angle for j points on S^2, j = 2..13
TAMMES = {2: -1.0, 3: -0.5, 4: -1 / 3, 5: 0.0, 6: 0.0, 7: math.cos(math.radians(77.8695)),
          8: math.cos(math.radians(74.8585)), 9: 1 / 3, 10: math.cos(math.radians(66.1468)),
          11: 1 / math.sqrt(5), 12: 1 / math.sqrt(5), 13: math.cos(math.radians(57.1367))}


def theta(j):
    """least radius of a cap of S^3 holding j points pairwise at angle >= PHI
    (points on the boundary sphere: cos PHI >= cos^2 th + sin^2 th cos psi_j)"""
    if j == 1:
        return 0.0
    c = TAMMES[j]
    return math.asin(math.sqrt((1 - math.cos(PHI)) / (1 - c)))


def pbasis(d, r):
    x = (2 * np.asarray(d, float) - C1) / C2
    out = [np.ones_like(x), x]
    for k in range(2, r + 1):
        out.append(2 * x * out[-1] - out[-2])
    return np.stack(out[:r + 1], -1)


def ubasis(u, D):
    u = np.asarray(u, float)
    out = [np.ones_like(u), 2 * u]
    for j in range(2, D + 1):
        out.append(2 * u * out[-1] - out[-2])
    return np.stack([out[j] / (j + 1) for j in range(D + 1)], -1)


def amax(d1, d2):
    return (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)


def cheb(lo, hi, n):
    return lo + (hi - lo) * (1 - np.cos(np.linspace(0, np.pi, n))) / 2


def samples(nd, nu, ulo):
    """pairs (d, d', u) with u in [ulo, a(d, d')], both orders of (d, d')"""
    ds = cheb(2, R6, nd)
    P, Q, W = [], [], []
    for d1 in ds:
        for d2 in ds:
            for u in cheb(ulo, amax(d1, d2), nu):
                P.append(d1); Q.append(d2); W.append(u)
    for d1, d2 in ((2.0, 2.0), (2.0, R6), (R6, 2.0), (R6, R6)):
        for u in cheb(ulo, amax(d1, d2), 400):
            P.append(d1); Q.append(d2); W.append(u)
    return map(np.array, (P, Q, W))


def band(nd, nu, lo, hi):
    ds = cheb(2, R6, nd)
    P, Q, W = [], [], []
    for d1 in ds:
        for d2 in ds:
            for u in cheb(lo, hi, nu):
                P.append(d1); Q.append(d2); W.append(u)
    return map(np.array, (P, Q, W))


def kern(A, D, r, P, Q, W):
    bp, bq, uk = pbasis(P, r), pbasis(Q, r), ubasis(W, D)
    return sum(cp.multiply(uk[:, k], cp.sum(cp.multiply(bp @ A[k], bq), axis=1)) for k in range(D + 1))


def solve(M, D, r, t0, nd=15, nu=36):
    ulo = -1.0 if t0 >= 1 else -t0
    P, Q, W = samples(nd, nu, ulo)
    pv = pair(P / 2, Q / 2, W)
    A = [cp.Variable((r + 1, r + 1), symmetric=True) for _ in range(D + 1)]
    z, t, m = cp.Variable(r + 1), cp.Variable(), cp.Variable()
    dd = cheb(2, R6, 400)
    bd = pbasis(dd, r)
    Kd = sum(cp.sum(cp.multiply(bd @ A[k], bd), axis=1) for k in range(D + 1))
    Z = cp.bmat([[A[0], cp.reshape(z, (r + 1, 1), order='C')],
                 [cp.reshape(z, (1, r + 1), order='C'), cp.reshape(t, (1, 1), order='C')]])
    cons = [A[k] >> 0 for k in range(1, D + 1)] + [Z >> 0, kern(A, D, r, P, Q, W) <= pv]
    excess = 0
    tiers = []
    if t0 < 1:
        th0 = math.acos(t0)
        j = 1
        while j <= 13 and theta(j) < th0:
            lo = -1.0 if j == 1 else -math.cos(theta(j))
            Pb, Qb, Wb = band(nd, 24, lo, -t0)
            g = cp.Variable(r + 1)
            cons += [kern(A, D, r, Pb, Qb, Wb) <= pbasis(Pb, r) @ g, bd @ g >= 0]
            excess = excess + bd @ g
            tiers.append((j, math.degrees(theta(j))))
            j += 1
        if j <= 13 and theta(j) < th0:
            raise SystemExit('cap too large for the Tammes table')
    cons.append(S(dd) + Kd / 2 + excess / 2 - bd @ z <= m)
    prob = cp.Problem(cp.Minimize(M * m + t / 2), cons)
    prob.solve(solver='CLARABEL')
    return prob.value, prob.status, tiers


def main():
    D, r, t0 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    for M in map(int, sys.argv[4:]):
        t1 = time.time()
        val, st, tiers = solve(M, D, r, t0)
        print('M = %2d, D = %d, r = %d, t0 = %.3f (%d tiers): %s, bound %.5f against %.5f (%s) [%.0f s]'
              % (M, D, r, t0, len(tiers), st, val, TARGET, 'below' if val < TARGET else 'above', time.time() - t1),
              flush=True)


if __name__ == '__main__':
    main()
