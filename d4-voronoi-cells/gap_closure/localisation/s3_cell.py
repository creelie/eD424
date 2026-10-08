#!/usr/bin/env python3
"""
s3_cell.py -- is the Voronoi cell of a root direction the smallest cell on S^3?
For a code on S^3 with pairwise angles at least 60 degrees, the cell of a point
x is the set of points of S^3 closer to x than to every other point of the
code.  At the root system D4 every cell has volume 2 pi^2/24 = pi^2/12, and the
link of x (its 8 neighbours at 60 degrees, seen from x) is a cube.  This script
compares that cell with the cell bounded by 8 neighbours at 60 degrees whose
link is the square antiprism, the best 8-point code on S^2 (Tammes): further
points of a code can only make that cell smaller.  Monte Carlo on one shared
sample of S^3, floating point; it settles nothing, it shows that a per-cell
volume bound on S^3 cannot single out the root system.

    python3 s3_cell.py [samples]
"""
import math
import sys

import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(1)
# sample S^3 uniformly in the cap of angle <= 60 deg about e0 (Voronoi cell of e0 lies there when neighbours are at >= 60)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
g = rng.standard_normal((N, 4)); g /= np.linalg.norm(g, axis=1)[:, None]
CAPC = math.cos(math.radians(60))
pts = g[g[:, 0] >= CAPC]
frac = len(pts) / N
VOL = 2 * math.pi ** 2
def cellvol(Y):
    """volume of the Voronoi cell of e0 on S^3 among e0 and the points Y (rows)"""
    dots = pts @ np.asarray(Y).T
    inside = np.all(pts[:, :1] >= dots, axis=1)
    return VOL * inside.sum() / N
def tammes(n, starts=40):
    best = None
    for s in range(starts):
        x0 = rng.standard_normal((n, 3)).ravel()
        def f(v):
            P = v.reshape(n, 3); P = P / np.linalg.norm(P, axis=1)[:, None]
            G = P @ P.T; iu = np.triu_indices(n, 1)
            return np.max(G[iu])
        def smooth(v, beta=200):
            P = v.reshape(n, 3); P = P / np.linalg.norm(P, axis=1)[:, None]
            G = P @ P.T; iu = np.triu_indices(n, 1); g = G[iu]
            return np.log(np.sum(np.exp(beta * (g - g.max())))) / beta + g.max()
        r = minimize(smooth, x0, method='BFGS')
        for beta in (1000, 5000):
            r = minimize(lambda v: smooth(v, beta), r.x, method='BFGS')
        val = f(r.x)
        if best is None or val < best[0]:
            P = r.x.reshape(n, 3); best = (val, P / np.linalg.norm(P, axis=1)[:, None])
    return best
def lift(W, ang=60):
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    return np.c_[np.full(len(W), c), s * W]
# D4: roots normalised, e0 replaced by a root direction
roots = []
import itertools
for i, j in itertools.combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = np.zeros(4); v[i] = a; v[j] = b; roots.append(v / math.sqrt(2))
roots = np.array(roots)
x = roots[0]
# rotate so that x = e0
Q, _ = np.linalg.qr(np.c_[x, rng.standard_normal((4, 3))])
if Q[:, 0] @ x < 0: Q[:, 0] *= -1
Rm = Q.T
others = np.array([Rm @ r for r in roots[1:]])
print('pi^2/12 =', math.pi ** 2 / 12)
print('D4 cell (all 23 others):', cellvol(others))
near = others[others[:, 0] > 0.49]
print('D4 cell, only the 8 neighbours at 60 deg:', cellvol(near), len(near))
for n in (8, 9, 10):
    val, P = tammes(n)
    print('Tammes %d: max inner product %.5f (angle %.3f deg)' % (n, val, math.degrees(math.acos(val))))
    if val <= 1 / 3 + 1e-6:
        print('   cell with %d neighbours at 60 deg on a Tammes link: %.5f' % (n, cellvol(lift(P))))
