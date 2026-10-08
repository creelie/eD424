#!/usr/bin/env python3
"""
count_survey.py -- the right side T of prop:truncated for every count of centres
within sqrt 6 from 24 to the largest that fits (sec:remains of the paper;
floating point, exploration, not proof).

For each M the script minimises T over configurations of M centres in the shell
2 <= |y| <= sqrt 6, pairwise at least 2 apart, from several starts, with the
functions of truncated_search.py.  It differs from that script only in the local
solver: the constraints get their exact Jacobian, the solver runs longer, and a
point that ends a hair outside the constraints is pulled back by the penalty of
the starts before T is evaluated, so that far fewer runs are discarded.

A centre at distance sqrt 6 contributes nothing to T (its cap is empty), so a
count M can be met by fewer centres inside and the rest on the sphere of radius
sqrt 6.  For each local minimum the script reports T and how many centres sit
on that sphere.

Usage: python3 count_survey.py M_FROM M_TO [starts] [seed]
       python3 count_survey.py pack M_FROM M_TO [starts] [seed]
       python3 count_survey.py gradcheck
The second form only looks for packings of M centres in the shell, minimising
the squared violations of the three constraints with their exact gradient from
random starts, and reports the first packing found or the least violation.
"""
import math
import sys

import numpy as np
from scipy.optimize import minimize

from truncated_search import R, R2, S, T, pair, roots

R6 = 6.0
EPS = 1e-6


def gradT(Y):
    """the gradient of T in the coordinates of the centres: the cap S(d) has
    derivative -A(d/2)/2 with A the volume of the section of B(sqrt(3/2)); the
    pair terms are differentiated in their three arguments by central
    differences, one vectorised evaluation per argument and sign, and the
    chain rule does the rest"""
    M = len(Y)
    d = np.linalg.norm(Y, axis=1); h = d / 2; W = Y / d[:, None]
    inside = d < math.sqrt(R6)
    g = np.zeros_like(Y)
    A = 4 * np.pi / 3 * np.maximum(R2 - h * h, 0) ** 1.5
    g += ((A / 2) * inside)[:, None] * W          # -dS/dd along the radius
    I, J = np.triu_indices(M, 1)
    keep = inside[I] & inside[J]
    I, J = I[keep], J[keep]
    hi, hj = h[I], h[J]
    u = np.sum(W[I] * W[J], axis=1)
    dhi = (pair(hi + EPS, hj, u) - pair(hi - EPS, hj, u)) / (2 * EPS)
    dhj = (pair(hi, hj + EPS, u) - pair(hi, hj - EPS, u)) / (2 * EPS)
    up, um = np.minimum(u + EPS, 1), np.maximum(u - EPS, -1)
    du = (pair(hi, hj, up) - pair(hi, hj, um)) / (up - um)
    # dh/dy = W/2; du/dy_i = (W_j - u W_i)/d_i
    gi = (dhi / 2)[:, None] * W[I] + du[:, None] * (W[J] - u[:, None] * W[I]) / d[I][:, None]
    gj = (dhj / 2)[:, None] * W[J] + du[:, None] * (W[I] - u[:, None] * W[J]) / d[J][:, None]
    np.add.at(g, I, gi)
    np.add.at(g, J, gj)
    return g


def survey(M, nstart, rng, prev=None):
    I, J = np.triu_indices(M, 1)
    ncon = 2 * M + len(I)

    def cons(x):
        Y = x.reshape(M, 4); n2 = np.sum(Y * Y, axis=1); D = Y[I] - Y[J]
        return np.concatenate([n2 - 4, R6 - n2, np.sum(D * D, axis=1) - 4])

    def jac(x):
        Y = x.reshape(M, 4); Jm = np.zeros((ncon, M, 4))
        a = np.arange(M)
        Jm[a, a] = 2 * Y
        Jm[M + a, a] = -2 * Y
        D = Y[I] - Y[J]; k = 2 * M + np.arange(len(I))
        Jm[k, I] = 2 * D
        Jm[k, J] = -2 * D
        return Jm.reshape(ncon, 4 * M)

    def pen(x):
        Y = x.reshape(M, 4); n2 = np.sum(Y * Y, axis=1); D2 = np.sum((Y[I] - Y[J]) ** 2, axis=1)
        return (np.sum(np.maximum(0, 4.000001 - n2) ** 2) + np.sum(np.maximum(0, n2 - 5.999999) ** 2)
                + np.sum(np.maximum(0, 4.000001 - D2) ** 2))

    def feasible(x):
        if np.min(cons(x)) >= -1e-10:
            return x
        x = minimize(pen, x, method='L-BFGS-B', options={'maxiter': 20000, 'ftol': 1e-30, 'gtol': 1e-16}).x
        return x if np.min(cons(x)) >= -1e-10 else None

    def start(k):
        Y = []
        if k % 3 == 1 and prev is not None:
            # the best configuration of M - 1 centres, and one more where it fits
            Y = list(prev)
            for _ in range(20000):
                v = rng.standard_normal(4); v *= math.sqrt(R6) * (1 - 1e-9) / np.linalg.norm(v)
                if np.min(np.sum((np.array(Y) - v) ** 2, axis=1)) >= 4:
                    Y.append(v)
                    break
        elif k % 3 == 0:
            Y = list(roots() * (1 + 0.03 * rng.random(24))[:, None] + 0.05 * rng.standard_normal((24, 4)))[:M]
        while len(Y) < M:
            v = rng.standard_normal(4); v /= np.linalg.norm(v)
            Y.append(v * (2 + (math.sqrt(6) - 2) * rng.random()))
        spread = lambda x: pen(x) * 1e4
        x = minimize(spread, np.array(Y).ravel(), method='L-BFGS-B', options={'maxiter': 20000}).x
        return feasible(x)

    out = []
    for k in range(nstart):
        x0 = start(k)
        if x0 is None:
            print('  start %2d: no packing found' % k, flush=True)
            continue
        r = minimize(lambda x: T(x.reshape(M, 4)), x0, jac=lambda x: gradT(x.reshape(M, 4)).ravel(), method='SLSQP',
                     constraints=[{'type': 'ineq', 'fun': cons, 'jac': jac}],
                     options={'maxiter': 3000, 'ftol': 1e-12})
        x = feasible(r.x)
        if x is None:
            print('  start %2d: the local solver ended outside the constraints (discarded)' % k, flush=True)
            continue
        Y = x.reshape(M, 4); d = np.linalg.norm(Y, axis=1)
        v = T(Y); on = int(np.sum(d > math.sqrt(R6) - 1e-4))
        print('  start %2d: T = %.6f, %2d centres on the sphere of radius sqrt 6' % (k, v, on), flush=True)
        out.append((v, on, np.sort(d), Y))
    return out


def pack(M, nstart, rng):
    I, J = np.triu_indices(M, 1)

    def pen(x):
        Y = x.reshape(M, 4); n2 = np.sum(Y * Y, axis=1); D = Y[I] - Y[J]; D2 = np.sum(D * D, axis=1)
        a = np.maximum(0, 4 - n2); b = np.maximum(0, n2 - R6); c = np.maximum(0, 4 - D2)
        g = (-4 * a + 4 * b)[:, None] * Y
        gc = (-4 * c)[:, None] * D
        np.add.at(g, I, gc); np.add.at(g, J, -gc)
        return np.sum(a * a) + np.sum(b * b) + np.sum(c * c), g.ravel()

    least = np.inf
    for k in range(nstart):
        Y = rng.standard_normal((M, 4))
        Y *= ((2 + (math.sqrt(R6) - 2) * rng.random(M)) / np.linalg.norm(Y, axis=1))[:, None]
        r = minimize(pen, Y.ravel(), jac=True, method='L-BFGS-B',
                     options={'maxiter': 50000, 'ftol': 1e-30, 'gtol': 1e-14})
        least = min(least, r.fun)
        if r.fun < 1e-20:
            d = np.linalg.norm(r.x.reshape(M, 4), axis=1)
            print('M = %d: a packing, from start %d; %d of its centres within 1e-4 of the sphere of radius sqrt 6'
                  % (M, k, int(np.sum(d > math.sqrt(R6) - 1e-4))), flush=True)
            return
    print('M = %d: no packing from %d starts (least squared violation %.3e)' % (M, nstart, least), flush=True)


def gradcheck():
    """the gradient of T against central differences at random configurations"""
    rng = np.random.default_rng(3)
    for M in (24, 26, 30):
        Y = rng.standard_normal((M, 4))
        Y *= ((2 + 0.4 * rng.random(M)) / np.linalg.norm(Y, axis=1))[:, None]
        f = lambda x: T(x.reshape(M, 4))
        x = Y.ravel(); E = np.eye(4 * M)
        num = np.array([(f(x + 1e-6 * e) - f(x - 1e-6 * e)) / 2e-6 for e in E])
        print('M = %d: largest difference between gradT and central differences %.1e (gradient norm %.2f)'
              % (M, np.max(np.abs(gradT(Y).ravel() - num)), np.linalg.norm(num)))


def main():
    if sys.argv[1] == 'gradcheck':
        gradcheck()
        return
    if sys.argv[1] == 'pack':
        m0, m1 = int(sys.argv[2]), int(sys.argv[3])
        nstart = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 5)
        for M in range(m0, m1 + 1):
            pack(M, nstart, rng)
        print('(floating point: a packing found is a packing up to rounding; none found is not a proof that none exists)')
        return
    m0, m1 = int(sys.argv[1]), int(sys.argv[2])
    nstart = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    rng = np.random.default_rng(seed)
    print('T at the root system (24 centres at distance 2): %.6f' % T(roots()))
    prev = None
    for M in range(m0, m1 + 1):
        print('M = %d' % M, flush=True)
        res = survey(M, nstart, rng, prev)
        if not res:
            print('M = %d: no feasible configuration found from %d starts' % (M, nstart), flush=True)
            prev = None
            continue
        v, on, d, prev = min(res, key=lambda t: t[0])
        print('M = %d: least T over %d local minima (of %d starts) = %.6f, with %d centres on the sphere of radius sqrt 6; distances:'
              % (M, len(res), nstart, v, on), flush=True)
        print('  ' + ' '.join('%.4f' % t for t in d), flush=True)
    print('(floating point and local optimisation: evidence about where the truncated bound fails, not a bound)')


if __name__ == '__main__':
    main()
