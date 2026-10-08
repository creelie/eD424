#!/usr/bin/env python3
"""
link_census.py -- the links of 24-point configurations on S^3, groundwork for
steps 2 and 3 of the budgeted localisation (gap_closure/README.md).

The link of a direction w_i is the set of its near neighbours (inner product at
least 0.4 with w_i), projected to the 2-sphere of tangent directions at w_i.
At the root system every link is a cube: 8 neighbours at 60 degrees, pairwise
at 70.53 degrees in the link.  For each configuration the script reports, point
by point, the number of near neighbours, the root-sum-square distance of the
link from a cube (after the best rotation and matching, when there are 8), and
the volume of the point's Voronoi cell on S^3 (Monte Carlo), against
pi^2/12 = 0.82247 at the root system.

Configurations: the root system; ends of budget_far.py at T <= 8.00368; and
24-point codes pushed to the least largest inner product from random starts,
which mostly stop at the second code of prop:second-code (0.516978).
Floating point, exploration only.

    python3 link_census.py [near_ends] [codes] [samples]
"""
import itertools
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_far as bf  # noqa: E402

A = bf.A
n = 24
iu, ju = np.triu_indices(n, 1)
CUBE = np.array(list(itertools.product((-1, 1), repeat=3)), float) / math.sqrt(3)


def link(W, i, thr=0.4):
    u = W @ W[i]
    nb = [j for j in range(n) if j != i and u[j] >= thr]
    T_ = W[nb] - np.outer(u[nb], W[i])
    T_ /= np.linalg.norm(T_, axis=1)[:, None]
    # coordinates in the tangent space at W[i]
    Q, _ = np.linalg.qr(np.c_[W[i], np.eye(4)[:, :3]])
    return T_ @ Q[:, 1:4], u[nb]


def cube_dist(L):
    if len(L) != 8:
        return float('nan')
    best = 1e9
    rng = np.random.default_rng(0)
    for _ in range(20):
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        for _ in range(30):
            perm = linear_sum_assignment(-L @ (CUBE @ R.T).T)[1]
            U_, s, Vt = np.linalg.svd(L.T @ CUBE[perm])
            Rn = (U_ @ Vt)
            if np.allclose(Rn, R):
                break
            R = Rn
        d2 = np.sum((L - CUBE[perm] @ R.T) ** 2)
        best = min(best, d2)
    return math.sqrt(best)


def cells(W, N):
    rng = np.random.default_rng(3)
    vols = np.zeros(n)
    done = 0
    while done < N:
        m = min(2_000_000, N - done)
        g = rng.standard_normal((m, 4)); g /= np.linalg.norm(g, axis=1)[:, None]
        vols += np.bincount(np.argmax(g @ W.T, axis=1), minlength=n)
        done += m
    return 2 * math.pi ** 2 * vols / N


def code(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n * 4)

    def unit(v):
        P = v.reshape(n, 4)
        return P / np.linalg.norm(P, axis=1)[:, None]

    def smooth(v, beta):
        g = np.sum(unit(v)[iu] * unit(v)[ju], axis=1)
        return np.log(np.sum(np.exp(beta * (g - g.max())))) / beta + g.max()
    for beta in (50, 300, 2000, 10000):
        x = minimize(lambda v: smooth(v, beta), x, method='L-BFGS-B').x
    W = unit(x)
    return W, np.max(np.sum(W[iu] * W[ju], axis=1))


def report(name, W, N):
    v = cells(W, N)
    rows = []
    for i in range(n):
        L, u = link(W, i)
        rows.append((len(L), cube_dist(L), v[i]))
    ks = sorted(set(r[0] for r in rows))
    print('%s: largest inner product %.5f, distance to a root system %.4f' % (
        name, np.max(np.sum(W[iu] * W[ju], axis=1)), bf.true_dist(W)), flush=True)
    for k in ks:
        sel = [r for r in rows if r[0] == k]
        cd = [r[1] for r in sel]
        print('   %2d points with %d near neighbours: link-to-cube distance %s, cell volumes %.4f to %.4f' % (
            len(sel), k, 'n/a' if k != 8 else '%.3f to %.3f' % (min(cd), max(cd)),
            min(r[2] for r in sel), max(r[2] for r in sel)), flush=True)


def main():
    near_ends = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ncodes = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 8_000_000
    report('root system', A, N)
    for s, Y in enumerate(bf.ends(0.00368, near_ends, 11)):
        W = Y / np.linalg.norm(Y, axis=1)[:, None]
        report('budget end %d (T = %.5f)' % (s, bf.T(Y)), W, N)
    for s in range(ncodes):
        W, mx = code(100 + s)
        report('code %d' % s, W, N)


if __name__ == '__main__':
    main()
