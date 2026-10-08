#!/usr/bin/env python3
"""
facet_bounds_probe.py -- a numerical check of step (b) of the near-contact theorem
(thm:near-contact), facet by facet.  Floating point, exploration; the proof is the text,
and its constants are evaluated in exact arithmetic by explicit_eps0.py.

For random and adversarial perturbations of the root system (directions w_i = u_i +
eps_i on the sphere, support numbers 1 + delta_i/2) it computes, with exact polytope
routines in floating point (scipy):

  1. vol P(1) - 8 against the integral over t of the facet formula
         sum_i [(delta_i/2 + e_i^2/2) vol G_i(t) - <eps_i, m_i(t)>] / (1 - t e_i^2/2)^2 ,
     G_i(t) the facet projected along u_i, m_i its first moment (the two agree);
  2. for every facet and t in {1/4, 3/5, 1}, the ratios of 4/3 - vol G_i(t) and of
     |m_i(t)| to the bounds of step (b) (both must be at most 1).

Usage: python3 facet_bounds_probe.py [trials] [seed]
"""
import itertools
import sys

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

R4 = []
for a, b in itertools.combinations(range(4), 2):
    for sa in (1, -1):
        for sb in (1, -1):
            v = np.zeros(4); v[a] = sa; v[b] = sb; R4.append(v / np.sqrt(2))
U = np.array(R4)
G0 = U @ U.T
NB = [[j for j in range(24) if abs(G0[i, j] - 0.5) < 1e-9] for i in range(24)]
S3 = np.sqrt(3)


def vol_P(W, h):
    hi = HalfspaceIntersection(np.hstack([W, -h[:, None]]), np.zeros(4))
    return ConvexHull(hi.intersections).volume


def facet(i, t, eps, dlt):
    """vol G_i(t) and its first moment (as a vector of R^4)"""
    u = U[i]; Bp = np.linalg.svd(np.eye(4) - np.outer(u, u))[0][:, :3]
    e2 = eps[i] @ eps[i]; q = t / (1 - t * e2 / 2)
    Wt = U + t * eps; ht = 1 + t * dlt / 2
    rows = []
    for j in range(24):
        if j == i:
            continue
        w = Wt[j]
        A = (w - (w @ u) * u) - q * (u @ w) * (eps[i] - (eps[i] @ u) * u)
        B = ht[j] - (u @ w) * (1 + q * (dlt[i] / 2 + e2 / 2))
        rows.append(np.r_[Bp.T @ A, -B])
    pts = HalfspaceIntersection(np.array(rows), np.zeros(3)).intersections
    hull = ConvexHull(pts); c0 = pts.mean(0); vol = 0; mom = np.zeros(3)
    for s in hull.simplices:
        P = pts[s]; v = abs(np.linalg.det(P - c0)) / 6; vol += v; mom += v * (P.sum(0) + c0) / 4
    return vol, Bp @ mom


def config(rng, S, em, kind):
    dlt = rng.random(24) ** 3; dlt *= S / dlt.sum()
    eps = []
    v0 = rng.normal(size=4)
    for i in range(24):
        v = v0.copy() if kind == 1 else rng.normal(size=4)
        v -= (v @ U[i]) * U[i]; v *= (em if kind == 1 else em * rng.random()) / np.linalg.norm(v)
        w = U[i] + v; w /= np.linalg.norm(w); eps.append(w - U[i])
    if kind == 2:
        dlt = np.zeros(24); dlt[rng.choice(24, 3, replace=False)] = S / 3
    return np.array(eps), dlt


def bounds(i, t, eps, dlt):
    e = np.linalg.norm(eps, axis=1); emax, dmax = e.max(), dlt.max()
    r1 = np.sqrt(2) * (1 + dmax / 2) / (1 - np.sqrt(2) * emax); eta1 = dmax / 2 + r1 * emax
    R = 1 + 2 * (eta1 + dmax / 2 + r1 * emax)
    th = np.array([2 * (R * e[j] + e[j] ** 2 / 2 + dlt[j] / 2 + (0.5 + e[j]) * (dlt[i] / 2 + e[i] ** 2 / 2 + R * e[i])
                        / (1 - e[i] ** 2 / 2)) / (S3 * (1 - e[j])) for j in NB[i]])
    ain = 3 * S3 / 4 * (1 - (1 / S3 - th.max()) ** 2); lam = 1 + S3 * th.max(); aout = 3 * S3 / 4 * (lam ** 2 - 1 / 3)
    Dp = np.sqrt(5) * emax + emax ** 2 / 2 + dmax / 2 + emax * (dmax / 2 + emax ** 2 / 2 + emax) / (1 - emax ** 2 / 2)
    caps = 4 * (t * Dp) ** 3
    return t * th.sum() * ain + caps, t * th.sum() * (ain + R * aout) + caps


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 11)
    worst_formula, wv, wm, n = 0, 0, 0, 0
    for k in range(trials):
        S = 10 ** rng.uniform(-4, -2.6); em = 10 ** rng.uniform(-4, -2.2)
        eps, dlt = config(rng, S, em, k % 3)
        if k % 10 == 0:
            ts = np.linspace(0, 1, 17); vals = []
            for t in ts:
                tot = 0
                for i in range(24):
                    vg, m = facet(i, t, eps, dlt); e2 = eps[i] @ eps[i]
                    tot += ((dlt[i] / 2 + e2 / 2) * vg - eps[i] @ m) / (1 - t * e2 / 2) ** 2
                vals.append(tot)
            V = vol_P(U + eps, 1 + dlt / 2)
            worst_formula = max(worst_formula, abs(np.trapezoid(vals, ts) - (V - 8)) / (V - 8))
        for t in (0.25, 0.6, 1.0):
            for i in range(24):
                vg, m = facet(i, t, eps, dlt); bv, bm = bounds(i, t, eps, dlt)
                wv = max(wv, (4 / 3 - vg) / bv); wm = max(wm, np.linalg.norm(m) / bm); n += 1
        print('trial %3d  S %.2e  e %.2e  kind %d: worst ratios so far %.3f %.3f' % (k, S, em, k % 3, wv, wm), flush=True)
    print('1. the facet formula reproduces vol P(1) - 8 to relative %.1e (trapezoid rule, 17 nodes)' % worst_formula)
    print('2. over %d facets: max (4/3 - vol G)/bound = %.3f, max |m|/bound = %.3f (both at most 1)' % (n, wv, wm))


if __name__ == '__main__':
    main()
