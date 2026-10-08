#!/usr/bin/env python3
"""
cardinality_tight.py -- where the three-point certificates of thm:kissing-stable and
thm:twenty-six are tight (rem:cardinality-tight).

Floating point, from cardinality_certificates/cert_d10_t*.npz: the value
B = 1 + f(1) + F(1,1,1), the largest values of P1(u) = f(u) + 3F(u,u,1) + 1 on [-1, t]
and of P2 = F on the admissible triples and the points where they are attained, the
values of P1 and P2 at the inner products of the root system, and how the room of the
final inequality is used.  Nothing in a proof uses these values.

Usage: python3 cardinality_tight.py cert.npz e1 e2 N        Logs: runs/cardinality_tight_*.log
"""
import itertools
import os
import sys
from fractions import Fraction as Fr

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certify_cardinality import build  # noqa: E402

path, e1, e2, N = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
Z = np.load(path)
d = len(Z['f'])
t = float(Z['t'])
f = [Fr(float(x)) for x in Z['f']]
F = [[[Fr(float(x)) for x in row] for row in Z['F%d' % k]] for k in range(d + 1)]
P1, P2, F111 = build(d, f, F)


def ev(P, u, v, w):
    out = np.zeros_like(np.asarray(u, float))
    for (a, b, c), co in P.items():
        out = out + float(co) * u ** a * v ** b * w ** c
    return out


f1 = float(ev(P1, np.array([1.0]), np.array([0.0]), np.array([0.0]))[0]) - 3 * float(F111) - 1   # P1(1) = f(1) + 3F(1,1,1) + 1
B = 1 + f1 + float(F111)
print('%s: degree %d, t = %.5f, B = 1 + f(1) + F(1,1,1) = %.6f (f(1) %.6f, F(1,1,1) %.6f)'
      % (os.path.basename(path), d, t, B, f1, float(F111)))
u = np.linspace(-1, t, 400001)
p1 = ev(P1, u, np.zeros_like(u), np.zeros_like(u))
loc = [i for i in range(1, len(u) - 1) if p1[i] >= p1[i - 1] and p1[i] >= p1[i + 1]]
loc = sorted(loc + [0, len(u) - 1], key=lambda i: -p1[i])
print('P1 = f + 3F(u,u,1) + 1 on [-1, t]: largest %.3e at u = %.5f; the local maxima above -1e-4:' % (p1.max(), u[np.argmax(p1)]))
for i in loc:
    if p1[i] > -1e-4:
        print('   u = %+.5f  P1 = %+.3e' % (u[i], p1[i]))

# P2 on the admissible domain
rng = np.random.default_rng(3)
pts = []
for dim in (4, 3, 2):
    X = rng.normal(size=(600000, 3, dim)); X /= np.linalg.norm(X, axis=2, keepdims=True)
    g = np.stack([np.einsum('ni,ni->n', X[:, 0], X[:, 1]), np.einsum('ni,ni->n', X[:, 0], X[:, 2]),
                  np.einsum('ni,ni->n', X[:, 1], X[:, 2])], 1)
    pts.append(g[(g <= t).all(1)])
ax = np.r_[-1.0, np.linspace(-1, t, 70), t]
G = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
G = G[1 + 2 * G[:, 0] * G[:, 1] * G[:, 2] - (G ** 2).sum(1) >= 0]
pts.append(G)
X = np.sort(np.concatenate(pts), axis=1)
p2 = ev(P2, X[:, 0], X[:, 1], X[:, 2])
from scipy.optimize import minimize  # noqa: E402
items = [(np.array(e), float(c)) for e, c in P2.items()]
E = np.array([e for e, _ in items]); Cf = np.array([c for _, c in items])


def fneg(x):
    return -float(np.sum(Cf * np.prod(np.asarray(x)[None, :] ** E, axis=1)))


det = {'type': 'ineq', 'fun': lambda x: 1 + 2 * x[0] * x[1] * x[2] - x[0] ** 2 - x[1] ** 2 - x[2] ** 2}
found = []
for i in np.argsort(p2)[-300:]:
    r = minimize(fneg, X[i], method='SLSQP', bounds=[(-1, t)] * 3, constraints=[det],
                 options={'maxiter': 300, 'ftol': 1e-15})
    x = np.sort(np.clip(r.x, -1, t))
    gd = 1 + 2 * x[0] * x[1] * x[2] - (x ** 2).sum()
    if gd < -1e-10:
        continue
    val = -fneg(x)
    if all(np.abs(x - y).sum() > 2e-3 for _, y in found):
        found.append((val, x))
found.sort(key=lambda z: -z[0])
print('P2 = F on the admissible triples: largest %.4e (samples %.4e); distinct local maxima (sorted u <= v <= w):'
      % (found[0][0], p2.max()))
for val, x in found[:8]:
    gd = 1 + 2 * x[0] * x[1] * x[2] - (x ** 2).sum()
    tag = [('t' if abs(y - t) < 1e-7 else ('-1' if abs(y + 1) < 1e-7 else '')) for y in x]
    print('   %.4e at (%.4f, %.4f, %.4f) %s, Gram determinant %.1e' % (val, *x, tag, gd))
print('at the inner products of the root system:')
for uu in (-1.0, -0.5, 0.0, 0.5):
    print('   P1(%+.1f) = %+.3e' % (uu, float(ev(P1, np.array([uu]), np.array([0.0]), np.array([0.0]))[0])))
R = []
for a in itertools.product([-1, 0, 1], repeat=4):
    if sum(x * x for x in a) == 2:
        R.append(np.array(a) / np.sqrt(2))
R = np.array(R)
Gm = R @ R.T
trip = set()
for i, j, k in itertools.combinations(range(24), 3):
    trip.add(tuple(sorted(np.round([Gm[i, j], Gm[i, k], Gm[j, k]], 6))))
for tr in sorted(trip):
    x = np.array(tr)
    print('   P2%s = %+.3e' % (tuple(float(y) for y in tr), -fneg(x)))
print('the final inequality (N-1)(1 - e1) - (N-1)(N-2) e2 > B - 1 at N = %d, with the proved e1 = %.0e, e2 = %.0e:' % (N, e1, e2))
print('   (N-1) = %d, (N-1) e1 = %.6f, (N-1)(N-2) e2 = %.6f, left side %.6f against B - 1 = %.6f'
      % (N - 1, (N - 1) * e1, (N - 1) * (N - 2) * e2, (N - 1) * (1 - e1) - (N - 1) * (N - 2) * e2, B - 1))
print('   with the floating-point maxima in place of e1, e2: left side %.6f'
      % ((N - 1) * (1 - max(p1.max(), 0)) - (N - 1) * (N - 2) * found[0][0]))
