#!/usr/bin/env python3
"""dual_where.py -- where the optimal fictitious configuration of the two-point
programme puts its pairs and its radii (the dual measure of radial_count_sdp.py).
Floating point, diagnosis only.   python3 dual_where.py M D r"""
import sys
import numpy as np
import cvxpy as cp
from musin_scan import samples, pair, pbasis, ubasis, cheb, S, R6, TARGET, kern

M, D, r = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
P, Q, W = samples(15, 36, -1.0)
pv = pair(P / 2, Q / 2, W)
A = [cp.Variable((r + 1, r + 1), symmetric=True) for _ in range(D + 1)]
z, t, m = cp.Variable(r + 1), cp.Variable(), cp.Variable()
dd = cheb(2, R6, 400); bd = pbasis(dd, r)
Kd = sum(cp.sum(cp.multiply(bd @ A[k], bd), axis=1) for k in range(D + 1))
Z = cp.bmat([[A[0], cp.reshape(z, (r + 1, 1), order='C')], [cp.reshape(z, (1, r + 1), order='C'), cp.reshape(t, (1, 1), order='C')]])
c_pair = kern(A, D, r, P, Q, W) <= pv
c_rad = S(dd) + Kd / 2 - bd @ z <= m
prob = cp.Problem(cp.Minimize(M * m + t / 2), [A[k] >> 0 for k in range(1, D + 1)] + [Z >> 0, c_pair, c_rad])
prob.solve(solver='CLARABEL')
print('M %d bound %.5f' % (M, prob.value))
lam = np.maximum(c_pair.dual_value, 0); mu = np.maximum(c_rad.dual_value, 0)
print('radial measure: total %.3f (should be M = %d)' % (mu.sum(), M))
for lo, hi in [(2, 2.0161), (2.0161, 2.05), (2.05, 2.1), (2.1, 2.2), (2.2, 2.3), (2.3, 2.4), (2.4, 2.45)]:
    s = mu[(dd >= lo) & (dd < hi)].sum()
    print('  radius in [%.4f, %.4f): %.3f centres' % (lo, hi, s))
print('pair measure (ordered pairs): total %.2f (M(M-1) = %d)' % (2 * lam.sum(), M * (M - 1)))
edges = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (W >= lo) & (W < hi)
    s = lam[sel].sum()
    if s > 1e-3:
        print('  u in [%5.2f, %5.2f): mass %.3f per centre, mean radii %.3f %.3f' % (lo, hi, 2 * s / M,
              (lam[sel] * P[sel]).sum() / s, (lam[sel] * Q[sel]).sum() / s))
