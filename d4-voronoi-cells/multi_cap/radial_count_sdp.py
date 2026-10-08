#!/usr/bin/env python3
"""
radial_count_sdp.py -- a two-point radial certificate that the pair terms of
prop:truncated exceed 8 whenever at least M centres lie within sqrt6 (thm:count31:
statement (C) of prop:reduction-G for every count from M = 31 on).  Finds the
certificate (floating point) and writes it in exact dyadic form;
radial_count_check.py proves it.

Let Y be the centres y with 2 <= |y| < sqrt6, d = |y|, w = y/d, and let
U(Y) = sum_y S(d) - sum_{pairs} Pi, the volume of the union of the caps of
B(sqrt(3/2)) (prop:truncated), so that T(Y) = 9 pi^2/8 - U(Y).  The kernel

    K(y, y') = sum_{k=0}^{D} U_k(<w, w'>)/(k+1) * p(d)^T A_k p(d'),

with p(d) = (T_0(x), ..., T_r(x)) Chebyshev polynomials of x = (2d - c1)/c2
(c1, c2 rationals close to 2 + sqrt6 and sqrt6 - 2) and U_k the Chebyshev
polynomials of the second kind, is positive definite on the shell whenever every
A_k is positive semidefinite (U_k is the zonal kernel of S^3).  So for any
finite Y, with v = sum_y p(|y|),

    sum_{y, y'} K(y, y') >= v^T A_0 v.

If K(y, y') <= Pi(|y|/2, |y'|/2, <w, w'>) for every pair allowed by the packing
(<w, w'> <= a(d, d') = (d^2 + d'^2 - 4)/(2 d d')), then
sum_{pairs} Pi >= (v^T A_0 v - sum_y K(y, y))/2, and for any vector z and
t >= z^T A_0^+ z (that is, [[A_0, z], [z^T, t]] psd),

    U(Y) <= sum_y [S(d) + K(y, y)/2 - z . p(d)] + t/2 <= |Y| m + t/2,

where m bounds the bracket on [2, sqrt6].  With m < 0 the bound falls as |Y|
grows, so one certificate with M m + t/2 < 9 pi^2/8 - 8 settles every count
from M on.

The programme minimises M m + t/2 over the A_k, z, t and m.  It imposes the
pair inequality with a margin eps at sampled (d, d', u): a grid of nd
distances each way and nu inner products for each pair of distances, and fine
grids in u at the corners (d, d') in {2, sqrt6}^2.  After each solution it
tests the pair inequality at two million further admissible pairs; if any has
K - Pi above -eps/2, the 3000 worst are added to the samples and the programme
is solved again.  The bracket is bounded at 600 sampled distances.

    python3 radial_count_sdp.py M D r [eps [nd nu]]

writes radial_certificates/radial_M.json, which radial_count_check.py proves;

    python3 radial_count_sdp.py scan D r M1 M2 ...

prints, for each count M, the least value of M m + t/2 the programme reaches
with no margin (the best two-point bound on the union, floating point).
"""
import json
import math
import os
import sys
import time
from fractions import Fraction as Fr

import numpy as np
import cvxpy as cp

from truncated_search import pair, S

R6 = math.sqrt(6)
TARGET = 9 * math.pi ** 2 / 8 - 8
C1 = Fr(2 + R6).limit_denominator(10 ** 12)
C2 = Fr(R6 - 2).limit_denominator(10 ** 12)


def pbasis(d, r):
    x = (2 * np.asarray(d, float) - float(C1)) / float(C2)
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


def dyadic(x, bits=48):
    return Fr(round(x * 2 ** bits), 2 ** bits)


def samples(nd, nu):
    ds = 2 + (R6 - 2) * (1 - np.cos(np.linspace(0, np.pi, nd))) / 2
    P, Q, W = [], [], []
    for i, d1 in enumerate(ds):
        for d2 in ds[i:]:
            for u in -1 + (amax(d1, d2) + 1) * (1 - np.cos(np.linspace(0, np.pi, nu))) / 2:
                P.append(d1); Q.append(d2); W.append(u)
    # fine grids in u at the corners (d, d') in {2, sqrt6}^2, where the check is tightest
    for d1, d2 in ((2.0, 2.0), (2.0, R6), (R6, R6)):
        for u in -1 + (amax(d1, d2) + 1) * (1 - np.cos(np.linspace(0, np.pi, 400))) / 2:
            P.append(d1); Q.append(d2); W.append(u)
    return map(np.array, (P, Q, W))


def solve(M, D, r, eps, P, Q, W):
    pv = pair(P / 2, Q / 2, W)
    A = [cp.Variable((r + 1, r + 1), symmetric=True) for _ in range(D + 1)]
    z, t, m = cp.Variable(r + 1), cp.Variable(), cp.Variable()
    bp, bq, uk = pbasis(P, r), pbasis(Q, r), ubasis(W, D)
    K = sum(cp.multiply(uk[:, k], cp.sum(cp.multiply(bp @ A[k], bq), axis=1)) for k in range(D + 1))
    dd = 2 + (R6 - 2) * (1 - np.cos(np.linspace(0, np.pi, 600))) / 2
    bd = pbasis(dd, r)
    Kd = sum(cp.sum(cp.multiply(bd @ A[k], bd), axis=1) for k in range(D + 1))
    Z = cp.bmat([[A[0], cp.reshape(z, (r + 1, 1), order='C')],
                 [cp.reshape(z, (1, r + 1), order='C'), cp.reshape(t, (1, 1), order='C')]])
    cons = [A[k] - 1e-9 * np.eye(r + 1) >> 0 for k in range(1, D + 1)]
    cons += [Z - 1e-9 * np.eye(r + 2) >> 0, K <= pv - eps, S(dd) + Kd / 2 - bd @ z <= m - eps]
    if eps > 0:
        cons.append(m <= -eps)
    prob = cp.Problem(cp.Minimize(M * m + t / 2), cons)
    prob.solve(solver='CLARABEL')
    return prob, A, z, t, m


def violation(A, D, r, rng):
    """the largest K - Pi at two million admissible pairs: random ones, a band along
    u = a(d, d'), and the edges d = 2, d' = sqrt6 and d = d' = 2."""
    Av = np.array([a.value for a in A])
    n = 2000000
    p_ = 2 + (R6 - 2) * rng.random(n); q_ = 2 + (R6 - 2) * rng.random(n)
    p_[: n // 8] = 2.0; q_[n // 8: n // 4] = R6
    p_[n // 2: n // 2 + n // 16] = 2.0; q_[n // 2: n // 2 + n // 16] = 2.0
    top = amax(p_, q_)
    u_ = -1 + (top + 1) * rng.random(n) ** 0.5
    u_[n // 4: n // 2] = top[n // 4: n // 2] - 2e-3 * rng.random(n // 4)
    viol = np.einsum('nk,na,kab,nb->n', ubasis(u_, D), pbasis(p_, r), Av, pbasis(q_, r)) - pair(p_ / 2, q_ / 2, u_)
    return viol, p_, q_, u_


def main():
    if sys.argv[1] == 'scan':
        D, r = int(sys.argv[2]), int(sys.argv[3])
        P, Q, W = samples(19, 40)
        for M in map(int, sys.argv[4:]):
            t0 = time.time()
            prob = solve(M, D, r, 0.0, P, Q, W)[0]
            print('M = %2d: best two-point bound on the union %.5f against %.5f (%s) [%.0f s]'
                  % (M, prob.value, TARGET, 'below' if prob.value < TARGET else 'above', time.time() - t0), flush=True)
        print('(floating point, sampled constraints: where the method can work, not a proof)')
        return
    M, D, r = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    eps = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-5
    nd = int(sys.argv[5]) if len(sys.argv) > 5 else 19
    nu = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    P, Q, W = samples(nd, nu)
    rng = np.random.default_rng(7)
    for rnd in range(6):
        t0 = time.time()
        prob, A, z, t, m = solve(M, D, r, eps, P, Q, W)
        viol, p_, q_, u_ = violation(A, D, r, rng)
        print('round %d: M %d, degree %d in the angle, %d in the distance, margin %.0e: %s; union <= %.5f against %.5f; '
              'largest K - Pi between the samples %.2e [%.0f s, %d pairs sampled]'
              % (rnd, M, D, r, eps, prob.status, prob.value, TARGET, viol.max(), time.time() - t0, len(P)), flush=True)
        if viol.max() <= -eps / 2:
            break
        worst = np.argsort(viol)[-3000:]
        P, Q, W = np.r_[P, p_[worst]], np.r_[Q, q_[worst]], np.r_[W, u_[worst]]
    # exact dyadic form: the A_k with 2^-30 added on the diagonal, so that they stay psd after rounding
    Ak = [np.array(a.value) + (2.0 ** -30) * np.eye(r + 1) * (k > 0) for k, a in enumerate(A)]
    Ak = [(a + a.T) / 2 for a in Ak]
    cert = {'M': M, 'D': D, 'r': r, 'c1': str(C1), 'c2': str(C2), 'dmax': '24494898/10000000',
            'A': [[[str(dyadic(a[i, j])) for j in range(r + 1)] for i in range(r + 1)] for a in Ak],
            'z': [str(dyadic(v)) for v in z.value], 'm': str(dyadic(m.value + 2e-5))}
    os.makedirs('radial_certificates', exist_ok=True)
    json.dump(cert, open('radial_certificates/radial_%d.json' % M, 'w'), indent=1)
    print('wrote radial_certificates/radial_%d.json (t is fixed exactly by the check)' % M)


if __name__ == '__main__':
    main()
