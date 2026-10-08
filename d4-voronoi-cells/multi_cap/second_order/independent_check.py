#!/usr/bin/env python3
"""
independent_check.py -- checks of the second-order statements of the paper that do not
use the volume code of this directory: every volume here is computed afresh, by
intersecting the halfspaces {x : <x, w_i> <= h_i} (scipy HalfspaceIntersection) and
taking the convex hull.  Floating point.

  1. The form H of rational_model.py is the Hessian of the volume at the 24-cell:
     second central differences of the exact volume along random directions xi, with
     w_i moved along geodesics with velocity tau_i and h_i = 1 + eta_i.
  2. The 120 rows of the cone are the first-order packing constraints: the derivative
     of |y_i - y_j|^2, y_i = 2(1 + eta_i) w_i, is 8 (B xi)_ij on the 96 tight pairs.
  3. The spectrum of H in orthonormal coordinates, and H on the push-outs = Adj - 4I.
  4. The one-centre formula vol = 25/3 - (1 - eta)^4/3 and the pure-push inequality
     vol >= 8 + (1/3) sum (1 - (1 - eta_i)^4) on random push patterns.

Usage: python3 independent_check.py     (from this directory)
"""
import itertools, sys, os
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rational_model as RM

H = RM.to_float(RM.build_H()); B = RM.to_float(RM.build_B())
U = np.array([[float(x) for x in a] for a in RM.ROOTS]) / np.sqrt(2)
BAS = [[np.array([float(x) for x in b]) for b in RM.BAS[i]] for i in range(24)]


def directions(xi):
    c = xi[:72].reshape(24, 3); W = np.zeros((24, 4))
    for i in range(24):
        tau = sum(c[i, k] * BAS[i][k] for k in range(3)) / np.sqrt(2)
        a = np.linalg.norm(tau)
        W[i] = U[i] if a == 0 else np.cos(a) * U[i] + np.sin(a) * tau / a
    return W


def vol(W, h):
    pts = HalfspaceIntersection(np.hstack([W, -h[:, None]]), np.zeros(4)).intersections
    return ConvexHull(pts).volume


rng = np.random.default_rng(11)
V0 = vol(U, np.ones(24))
print('1. volume of the 24-cell: %.12f' % V0)
worst = 0
for trial in range(8):
    xi = rng.standard_normal(96)
    if trial % 2:
        xi[72:] = np.abs(xi[72:])
    xi /= np.linalg.norm(xi); t = 1e-3
    vp = vol(directions(t * xi), 1 + t * xi[72:]); vm = vol(directions(-t * xi), 1 - t * xi[72:])
    worst = max(worst, abs((vp + vm - 2 * V0) / t**2 - xi @ H @ xi))
print('   second differences of the exact volume against xi^T H xi, 8 directions: largest difference %.1e' % worst)

worst = 0
for trial in range(5):
    xi = rng.standard_normal(96); t = 1e-6
    Yp = (2 + 2 * t * xi[72:])[:, None] * directions(t * xi)
    Ym = (2 - 2 * t * xi[72:])[:, None] * directions(-t * xi)
    for r, (i, j) in enumerate(RM.TIGHT):
        d = (np.sum((Yp[i] - Yp[j])**2) - np.sum((Ym[i] - Ym[j])**2)) / (2 * t)
        worst = max(worst, abs(d - 8 * (B[r] @ xi)))
print('2. derivative of |y_i - y_j|^2 against 8 (B xi)_ij on the 96 tight pairs: largest difference %.1e' % worst)

d = np.ones(96)
for i in range(24):
    d[3 * i + 1] = d[3 * i + 2] = np.sqrt(2)          # orthonormal coordinates of tau_i
ev = np.sort(np.linalg.eigvalsh(np.diag(d) @ H @ np.diag(d)))
vals, counts = np.unique(np.round(ev, 4), return_counts=True)
print('3. spectrum of H in orthonormal coordinates (value x multiplicity):')
print('   ' + ', '.join('%.4f x%d' % (v, c) for v, c in zip(vals, counts)))
Adj = np.array([[1.0 if (i, j) in RM.TIGHT or (j, i) in RM.TIGHT else 0 for j in range(24)] for i in range(24)])
print('   H on the push-outs is the matrix Adj - 4I:', np.allclose(H[72:, 72:], Adj - 4 * np.eye(24)))

worst = 0
for s in np.linspace(0, 1, 11):
    h = np.ones(24); h[5] += s
    worst = max(worst, abs(vol(U, h) - (25 / 3 - (1 - s)**4 / 3)))
print('4. one centre pushed out, 0 <= eta <= 1: largest |vol - 25/3 + (1 - eta)^4/3| = %.1e' % worst)
least = least2 = np.inf
for trial in range(400):
    k = rng.integers(1, 25)
    eta = np.zeros(24); idx = rng.choice(24, k, replace=False)
    eta[idx] = rng.random(k) ** rng.choice([1, 3, 6]) * rng.choice([0.05, 0.3, 1.0])
    v = vol(U, 1 + eta); S = 2 * eta.sum()
    least = min(least, v - 8 - np.sum(1 - (1 - eta)**4) / 3)
    least2 = min(least2, v - 8 - (2 / 3 * S - S**2 / 2))
print('   400 push patterns: least of vol - 8 - (1/3) sum(1 - (1 - eta_i)^4) = %.1e' % least)
print('                      least of vol - 8 - (2/3) S + S^2/2              = %.1e' % least2)
