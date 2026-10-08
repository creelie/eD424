#!/usr/bin/env python3
"""
truncated_search.py -- how low can the right side of prop:truncated go with M
centres within sqrt 6?  (sec:remains of the paper; floating point,
exploration, not proof.)

For a finite set Y of centres (|y| >= 2, pairwise at least 2 apart), prop:truncated
gives vol(V_c) >= T(Y) = 9 pi^2/8 - sum S(|y|) + sum_{pairs} Pair, with S the cap
of B(sqrt(3/2)) beyond |y|/2 and Pair the overlap of two such caps.  thm:m23
proves T > 8 whenever at most 23 centres lie within sqrt 6.  At the root system,
24 centres at distance 2, T = 7.906940..., so the pair terms cannot settle 24
centres.  This script minimises T over configurations of exactly M centres within
sqrt 6 by sequential quadratic programming from many starts, to see where the
truncated bound fails.

Pair has an elementary closed form: in the plane of the two normals (angle g),
the region beyond both hyperplanes is r > h_1 / cos(phi), r > h_2 / cos(phi - g),
r < R, and the orthogonal plane contributes the area pi (R^2 - r^2), so

    Pair = (pi/4) [ G(h_2, phi_c - g) - G(h_2, L - g) + G(h_1, U) - G(h_1, phi_c) ],
    G(h, x) = R^4 x - 2 R^2 h^2 tan x + h^4 (tan x + tan^3 x / 3),

with L = max(-a_1, g - a_2), U = min(a_1, g + a_2), a_k = arccos(h_k / R), and
phi_c the angle where the two hyperplanes cross, clipped to [L, U]; the script
checks it against numerical quadrature before it starts.

Starts: the root system pushed out a little with random extra centres, and
random points spread by a penalty until the packing conditions hold.

Usage: python3 truncated_search.py M [starts] [seed]
       python3 truncated_search.py rays
The second form prints where T reaches 8 along two families through the root
system: all 24 centres pushed out to 2 + delta, and one centre pushed out.
"""
import math
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

R2 = 1.5
R = math.sqrt(R2)
R4 = R2 * R2
VB = 9 * math.pi ** 2 / 8


def G(h, x):
    tx = np.tan(x)
    return R4 * x - 2 * R2 * h * h * tx + h ** 4 * (tx + tx ** 3 / 3)


def pair(h1, h2, u):
    h1, h2, u = np.broadcast_arrays(np.asarray(h1, float), np.asarray(h2, float), np.asarray(u, float))
    g = np.arccos(np.clip(u, -1, 1))
    a1 = np.arccos(np.clip(h1 / R, -1, 1)); a2 = np.arccos(np.clip(h2 / R, -1, 1))
    L = np.maximum(-a1, g - a2); U = np.minimum(a1, g + a2)
    pc = np.clip(np.arctan2(h2 - h1 * np.cos(g), h1 * np.maximum(np.sin(g), 1e-15)), L, U)
    val = G(h2, pc - g) - G(h2, L - g) + G(h1, U) - G(h1, pc)
    return np.where(L < U, np.pi / 4 * val, 0.0)


def pair_quad(h1, h2, u):
    g = math.acos(u)
    def f(phi):
        c1, c2 = math.cos(phi), math.cos(phi - g)
        if c1 <= 0 or c2 <= 0:
            return 0.0
        rm = max(h1 / c1, h2 / c2)
        return math.pi / 4 * (R2 - rm * rm) ** 2 if rm < R else 0.0
    lo, hi = max(-math.pi / 2, g - math.pi / 2), min(math.pi / 2, g + math.pi / 2)
    # break the range where the integrand has kinks: where each hyperplane leaves
    # the ball, and where the binding hyperplane changes
    pts = [lo, hi]
    for h, off in ((h1, 0.0), (h2, g)):
        if h < R:
            a = math.acos(h / R); pts += [off - a, off + a]
    pts.append(math.atan2(h2 - h1 * math.cos(g), h1 * math.sin(g)))
    pts = sorted(p for p in set(pts) if lo <= p <= hi)
    return sum(quad(f, a, b, limit=400, epsabs=1e-15, epsrel=1e-13)[0] for a, b in zip(pts[:-1], pts[1:]) if b > a)


def S(d):
    h = np.minimum(np.asarray(d, float) / 2, R)
    anti = h / 8 * (5 * R2 - 2 * h * h) * np.sqrt(np.maximum(R2 - h * h, 0)) + 3 * R4 / 8 * np.arcsin(h / R)
    return 4 * np.pi / 3 * (3 * R4 / 8 * np.pi / 2 - anti)


def T(Y):
    d = np.linalg.norm(Y, axis=1)
    Y, d = Y[d < math.sqrt(6)], d[d < math.sqrt(6)]
    W = Y / d[:, None]
    i, j = np.triu_indices(len(d), 1)
    return VB - np.sum(S(d)) + np.sum(pair(d[i] / 2, d[j] / 2, np.sum(W[i] * W[j], axis=1)))


def roots():
    out = []
    for a in range(4):
        for b in range(a + 1, 4):
            for sa in (1, -1):
                for sb in (1, -1):
                    v = np.zeros(4); v[a] = sa; v[b] = sb; out.append(v)
    return math.sqrt(2) * np.array(out)


def rays():
    from scipy.optimize import brentq
    U = roots() / 2
    print('T at the root system: %.6f' % T(2 * U))
    d1 = brentq(lambda d: T(U * (2 + d)) - 8, 1e-4, 0.05, xtol=1e-12)
    print('all 24 centres at distance 2 + delta: T = 8 at delta = %.5f, sum of the delta_i = %.4f' % (d1, 24 * d1))

    def one(d):
        Y = 2 * U.copy()
        Y[0] = U[0] * (2 + d)
        return T(Y) - 8
    d2 = brentq(one, 1e-3, 0.4, xtol=1e-12)
    print('one centre at distance 2 + delta, 23 at 2: T = 8 at delta = %.4f' % d2)
    print('(floating point: where the pair terms of prop:truncated reach 8, not a bound)')


def main():
    if sys.argv[1] == 'rays':
        rays()
        return
    M = int(sys.argv[1]); nstart = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(200):
        h1, h2 = 1 + 0.2 * rng.random(2); u = -0.9 + 1.8 * rng.random()
        worst = max(worst, abs(float(pair(h1, h2, u)) - pair_quad(h1, h2, u)))
    print('closed form of Pair against quadrature, 200 random points: largest difference %.1e' % worst)
    print('T at the root system (24 centres at distance 2): %.6f' % T(roots()))
    I, J = np.triu_indices(M, 1)

    def cons(x):
        Y = x.reshape(M, 4); n2 = np.sum(Y * Y, axis=1)
        D = Y[I] - Y[J]
        return np.concatenate([n2 - 4, 6 - 1e-6 - n2, np.sum(D * D, axis=1) - 4])

    def pen(x):
        Y = x.reshape(M, 4); n2 = np.sum(Y * Y, axis=1); D2 = np.sum((Y[I] - Y[J]) ** 2, axis=1)
        return (np.sum(np.maximum(0, 4.02 - n2) ** 2) + np.sum(np.maximum(0, n2 - 5.98) ** 2)
                + np.sum(np.maximum(0, 4.02 - D2) ** 2))

    def start(k):
        Y = []
        if k % 3 == 0 and M >= 24:
            Y = list(roots() * (1 + 0.03 * rng.random(24))[:, None] + 0.05 * rng.standard_normal((24, 4)))
        while len(Y) < M:
            v = rng.standard_normal(4); v /= np.linalg.norm(v)
            Y.append(v * (2 + (math.sqrt(6) - 2) * rng.random()))
        x = minimize(pen, np.array(Y[:M]).ravel(), method='L-BFGS-B', options={'maxiter': 5000}).x
        return x if np.min(cons(x)) > -1e-9 else None

    best = (np.inf, None); found = 0
    for k in range(nstart):
        x0 = start(k)
        if x0 is None:
            print('start %2d: no packing found' % k, flush=True)
            continue
        r = minimize(lambda x: T(x.reshape(M, 4)), x0, method='SLSQP', constraints=[{'type': 'ineq', 'fun': cons}],
                     options={'maxiter': 500, 'ftol': 1e-11})
        if np.min(cons(r.x)) <= -1e-6:
            print('start %2d: the local solver ended at an infeasible point (discarded)' % k, flush=True)
        else:
            found += 1
            v = T(r.x.reshape(M, 4))
            print('start %2d: T = %.6f' % (k, v), flush=True)
            if v < best[0]:
                best = (v, r.x.reshape(M, 4))
    if best[1] is None:
        print('M = %d: no feasible configuration found from %d starts' % (M, nstart))
        return
    d = np.sort(np.linalg.norm(best[1], axis=1))
    print('M = %d: least T over %d local minima (of %d starts) = %.6f; distances of that configuration:' % (M, found, nstart, best[0]))
    print('  ' + ' '.join('%.4f' % x for x in d))
    print('(floating point and local optimisation: evidence about where the truncated bound fails, not a bound)')


if __name__ == '__main__':
    main()
