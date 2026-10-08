#!/usr/bin/env python3
"""
certificate_tight.py -- where the certificate of thm:certificate (the pair inequality for
contact configurations of 23 directions) is tight.

Floating point, from continuation_out/certificate_d8.npz: the two parts of the bound
B = N (N f_0 - f(1)) / 2 - N F(1,1,1) / (6 (N - 2)), the least values of
Q = 1000 (omega(u) + omega(v) + omega(t)) - P(u, v, t) on the admissible domain and the
points where they are attained, the values of Q at the triples of the root system, and the
pair sum of omega over the root system with one root removed, against B.  Nothing in a
proof uses these values.

Usage: python3 certificate_tight.py        Log: runs/certificate_tight.log
"""
import itertools
import math
import os
import sys
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificate_check import N, SCALE, build_P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, 'continuation_out', 'certificate_d8.npz'))
d = len(Z['f']) - 1
f = [Fr(float(x)) for x in Z['f']]
f = [f[0]] + [max(x, Fr(0)) for x in f[1:]]
F = [[[Fr(float(x)) for x in row] for row in Z['F%d' % k]] for k in range(d + 1)]
P, F111 = build_P(d, f, F)
E = np.array(list(P.keys())); C = np.array([float(c) for c in P.values()])
TS = 1 / math.sqrt(2)


def omega(u):
    """The closed form of thm:certificate's proof; zero for u <= 1/3."""
    u = np.clip(np.asarray(u, float), -1 + 1e-12, 1)
    tau = np.sqrt((1 - u) / (1 + u))
    w = 4 * math.pi * (9 / 32 * np.arctan((TS - tau) / (1 + TS * tau)) - (4 * tau ** 3 - 24 * tau + 11 * math.sqrt(2)) / 96)
    return np.where(u > 1 / 3, w, 0.0)


def Pv(X):
    X = np.atleast_2d(X)
    return (C[None, :] * np.prod(X[:, None, :] ** E[None, :, :], axis=2)).sum(1)


def Q(X):
    X = np.atleast_2d(X)
    return SCALE * omega(X).sum(1) - Pv(X)


f1 = sum(f)
part1 = N * (N * f[0] - f1) / 2
part2 = -N * F111 / (6 * (N - 2))
B = part1 + part2
print('certificate_d8.npz: degree %d, N = %d; B / 1000 = %.9f' % (d, N, float(B) / SCALE))
print('   N (N f_0 - f(1)) / 2 / 1000 = %+.9f   (f_0 = %.6f, f(1) = %.6f)' % (float(part1) / SCALE, float(f[0]), float(f1)))
print('   -N F(1,1,1) / (6 (N-2)) / 1000 = %+.9f   (F(1,1,1) = %.6f)' % (float(part2) / SCALE, float(F111)))
print('   (f(1) - f_0) / 1000 = (f_1 + ... + f_8) / 1000 = %.2e: the two-point part is nearly constant' % (float(f1 - f[0]) / SCALE))

om_half = float(omega(0.5))
print('omega(1/2) = %.9f; omega vanishes for u <= 1/3' % om_half)
# the root system with one root removed: its pair sum of omega
R = [np.array(a) / math.sqrt(2) for a in itertools.product([-1, 0, 1], repeat=4) if sum(x * x for x in a) == 2]
R = np.array(R)
W = R[1:]
G = W @ W.T
iu = np.triu_indices(23, 1)
gu = G[iu]
n60 = int(np.sum(np.abs(gu - 0.5) < 1e-9))
print('the root system less one root: %d pairs at 60 degrees, pair sum of omega %.9f against B / 1000 = %.9f (8 - A_* = 0.092855570)'
      % (n60, float(omega(gu).sum()), float(B) / SCALE))
print('   one pair at 60 degrees is worth %.9f, so the pair sum exceeds 8 - A_* by %.1f such pairs' % (om_half, (float(omega(gu).sum()) - 0.0928555703) / om_half))

# Q on the admissible domain -1 <= u <= v <= t <= 1/2
rng = np.random.default_rng(5)
pts = []
for dim in (4, 3, 2):
    X = rng.normal(size=(400000, 3, dim)); X /= np.linalg.norm(X, axis=2, keepdims=True)
    g = np.stack([np.einsum('ni,ni->n', X[:, 0], X[:, 1]), np.einsum('ni,ni->n', X[:, 0], X[:, 2]),
                  np.einsum('ni,ni->n', X[:, 1], X[:, 2])], 1)
    pts.append(g[(g <= 0.5).all(1)])
ax = np.r_[-1.0, np.linspace(-1, 0.5, 61), 1 / 3, 0.5]
Gd = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
Gd = Gd[1 + 2 * Gd[:, 0] * Gd[:, 1] * Gd[:, 2] - (Gd ** 2).sum(1) >= -1e-12]
pts.append(Gd)
X = np.sort(np.concatenate(pts), axis=1)
q = np.concatenate([Q(X[i:i + 20000]) for i in range(0, len(X), 20000)])
det = {'type': 'ineq', 'fun': lambda x: 1 + 2 * x[0] * x[1] * x[2] - x[0] ** 2 - x[1] ** 2 - x[2] ** 2}
found = []
for i in np.argsort(q)[:300]:
    r = minimize(lambda x: float(Q(np.sort(np.clip(x, -1, 0.5)))[0]), X[i], method='SLSQP', bounds=[(-1, 0.5)] * 3,
                 constraints=[det], options={'maxiter': 300, 'ftol': 1e-15})
    x = np.sort(np.clip(r.x, -1, 0.5))
    if 1 + 2 * x[0] * x[1] * x[2] - (x ** 2).sum() < -1e-10:
        continue
    val = float(Q(x)[0])
    if all(np.abs(x - y).sum() > 2e-3 for _, y in found):
        found.append((val, x))
found.sort(key=lambda z: z[0])
print('Q = 1000 (omega(u) + omega(v) + omega(t)) - P on the admissible triples: least %.4e (samples %.4e); distinct local minima:'
      % (found[0][0], q.min()))
for val, x in found[:10]:
    gd = 1 + 2 * x[0] * x[1] * x[2] - (x ** 2).sum()
    print('   %.4e at (%+.4f, %+.4f, %+.4f), Gram determinant %.1e' % (val, *x, gd))
print('Q at the triples of the root system less one root (each kind once):')
trip = {}
for i, j, k in itertools.combinations(range(23), 3):
    key = tuple(sorted(np.round([G[i, j], G[i, k], G[j, k]], 6)))
    trip[key] = trip.get(key, 0) + 1
for key in sorted(trip):
    print('   (%+.1f, %+.1f, %+.1f): %4d triples, Q = %+.4e' % (*key, trip[key], float(Q(np.array(key))[0])))
