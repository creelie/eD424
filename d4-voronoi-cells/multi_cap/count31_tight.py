#!/usr/bin/env python3
"""
count31_tight.py -- where the certificate of thm:count31 is tight (rem:count31-tight).

Floating point, from radial_certificates/radial_31.json: the affine bound M m + t/2
for several counts M, the per-centre function of condition (c) against m on a grid
of distances, and the slack Pi - K of condition (a) on a grid of pairs of distances
and inner products, with its local minima for the end distances 2 and sqrt 6.
Nothing in a proof uses these values.  Log: runs/count31_tight.log.
"""
import json, math, os, sys
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truncated_search import pair, S
c = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radial_certificates', 'radial_31.json')))
D, r = c['D'], c['r']
c1, c2 = float(Fr(c['c1'])), float(Fr(c['c2']))
A = [np.array([[float(Fr(x)) for x in row] for row in Ak]) for Ak in c['A']]
z = np.array([float(Fr(x)) for x in c['z']]); m = float(Fr(c['m']))
A0inv_z = np.linalg.solve(A[0], z); t = z @ A0inv_z
print('m %.6f  t/2 %.6f  31m+t/2 %.6f  (level %.6f)' % (m, t / 2, 31 * m + t / 2, 9 * math.pi ** 2 / 8 - 8))
for M in (25, 28, 29, 30, 31, 32, 35, 40):
    print('   M=%d  M m + t/2 = %.5f' % (M, M * m + t / 2))
def pvec(d):
    x = (2 * np.asarray(d) - c1) / c2
    return np.polynomial.chebyshev.chebvander(x, r)            # rows T_0..T_r
def Uk(k, u):
    th = np.arccos(np.clip(u, -1, 1)); s = np.sin(th)
    return np.where(s < 1e-12, np.where(u > 0, k + 1.0, (-1.0) ** k * (k + 1)), np.sin((k + 1) * th) / np.maximum(s, 1e-300))
d = np.linspace(2, math.sqrt(6), 2001)
P = pvec(d)
f = S(d) + 0.5 * sum(np.einsum('na,ab,nb->n', P, A[k], P) for k in range(D + 1)) - P @ z
print('per-centre m - f(d): min %.2e at d=%.4f; below 1e-4 at d in %s' % ((m - f).min(), d[np.argmin(m - f)],
      np.round(d[(m - f) < 1e-4][[0, -1]], 4) if ((m - f) < 1e-4).any() else '-'))
for dd in (2.0, 2.05, 2.1, 2.2, 2.3, 2.4, math.sqrt(6)):
    i = np.argmin(abs(d - dd)); print('   d=%.4f  f=%.6f  m-f=%.2e  S=%.5f' % (d[i], f[i], m - f[i], S(d[i])))
# pair slack on a grid
def amax(a, b): return (a * a + b * b - 4) / (2 * a * b)
dg = np.linspace(2, math.sqrt(6), 41)
best = []
for i, a in enumerate(dg):
    for b in dg[i:]:
        u = np.linspace(-1, amax(a, b), 400)
        pa, pb = pvec(np.full_like(u, a)), pvec(np.full_like(u, b))
        K = sum(Uk(k, u) / (k + 1) * np.einsum('na,ab,nb->n', pa, A[k], pb) for k in range(D + 1))
        sl = pair(a / 2, b / 2, u) - K
        j = np.argmin(sl); best.append((sl[j], a, b, u[j]))
best.sort()
print('smallest pair slacks (slack, d, d\', u):')
for s_, a, b, u in best[:12]:
    print('   %.2e  %.4f %.4f  u=%.4f (amax %.4f)' % (s_, a, b, u, amax(a, b)))
print('--- slack along u for the three distance pairs (fine grid)')
R6 = math.sqrt(6)
for a, b in ((2, 2), (2, R6), (R6, R6), (2, 2.2), (2.2, 2.2)):
    u = np.linspace(-1, amax(a, b), 20001)
    pa, pb = pvec(np.full_like(u, a)), pvec(np.full_like(u, b))
    K = sum(Uk(k, u) / (k + 1) * np.einsum('na,ab,nb->n', pa, A[k], pb) for k in range(D + 1))
    sl = pair(a / 2, b / 2, u) - K
    loc = [i for i in range(1, len(u) - 1) if sl[i] <= sl[i - 1] and sl[i] <= sl[i + 1] and sl[i] < 2e-4]
    print('  (%.4f, %.4f): local minima of the slack %s; at u=amax %.2e; at u=-1 %.2e' % (a, b, [(round(float(u[i]), 4), '%.1e' % sl[i]) for i in loc], sl[-1], sl[0]))
