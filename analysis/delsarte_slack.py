#!/usr/bin/env python3
"""
delsarte_slack.py -- the linear programming bound of Delsarte, Goethals and
Seidel for codes on S^3 with all inner products at most s.

If 25 <= |Y| and T(Y) <= 8, prop:C-radial of d4-voronoi-cells puts at least
17 centres within 2.05, 21 within 2.1, 22 within 2.15 and 23 within 2.2 (at
|Y| = 25).  Two centres within rho at distance at least 2 have directions
with inner product at most 1 - 2/rho^2: 0.5240 at 2.05, 0.5465 at 2.1,
0.5673 at 2.15, 0.5868 at 2.2.  A contradiction from functions of one inner
product alone would have to bound such codes by fewer than 25 points; this
script shows that the best such function, the Delsarte bound, allows 27.7 at
2.05 and 35.2 at 2.2.  Musin's improvement of the bound (Ann. of Math. 168
(2008), 1-32, doi:10.4007/annals.2008.168.1), which lets the function be
positive near -1 and counts the points there by hand, lowers it from 25.56 to
below 24.865 at s = 1/2; a gain of that size (0.7) does not bring the bound
below 25 at any of these radii.

The bound is min f(1)/f_0 over f = sum_k f_k U_k(t)/(k+1) with f_0 = 1,
f_k >= 0 and f(t) <= 0 on [-1, s], U_k the Chebyshev polynomials of the
second kind (the zonal functions of S^3), degree at most 30, constraints on
8000 points of [-1, s]: floating point, an approximation from below of the
programme over all t.
"""
import numpy as np
from scipy.optimize import linprog


def U(k, t):
    th = np.arccos(np.clip(t, -1, 1))
    s = np.sin(th)
    return np.where(np.abs(s) < 1e-12, (k + 1) * np.sign(np.cos(th)) ** k, np.sin((k + 1) * th) / np.where(np.abs(s) < 1e-12, 1, s))


def bound(s, d=30, m=8000):
    ts = np.linspace(-1, s, m)
    A = np.array([U(k, ts) / (k + 1) for k in range(1, d + 1)]).T
    r = linprog(np.ones(d), A_ub=A, b_ub=-np.ones(m), bounds=[(0, None)] * d, method='highs')
    return 1 + r.fun


print('  rho     s = 1 - 2/rho^2   Delsarte bound')
for rho in (2.0, 2.0161, 2.05, 2.1, 2.15, 2.2):
    s = 1 - 2 / rho ** 2
    print('  %-7s %.4f            %.4f' % (rho, s, bound(s)))
