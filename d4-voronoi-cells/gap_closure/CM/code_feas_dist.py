#!/usr/bin/env python3
"""
code_feas_dist.py -- can centres at given distances from c be arranged as a packing set?
Centres y_i = d_i w_i and y_j = d_j w_j with |y_i - y_j| >= 2 need
<w_i, w_j> <= a(d_i, d_j) = (d_i^2 + d_j^2 - 4) / (2 d_i d_j), which increases with both
distances on [2, sqrt6]; so placing every centre at the largest distance its count allows
gives the loosest constraints, and a configuration that fails there fails at every
smaller distance.  Local search (L-BFGS on the squared violations) from random starts and
from rotated, perturbed roots of D_4 for the centres within 2.0161.  Floating point;
a failure to find a configuration proves nothing.

Usage: python3 code_feas_dist.py 'd1*n1,d2*n2,...' starts seed   (d may be 'sqrt6')
Log: c28_direction_search.log.
"""
import itertools, sys
import numpy as np
from scipy.optimize import minimize
spec, starts, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
R6 = 6 ** .5
ds = []
for part in spec.split(','):
    d, n = part.split('*')
    ds += [R6 if d == 'sqrt6' else float(d)] * int(n)
ds = np.array(ds); n = len(ds)
Tm = (ds[:, None] ** 2 + ds[None, :] ** 2 - 4) / (2 * ds[:, None] * ds[None, :])
iu = np.triu_indices(n, 1); tv = Tm[iu]
rng = np.random.default_rng(seed)
roots = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = np.zeros(4); v[i] = si; v[j] = sj; roots.append(v / 2 ** .5)
roots = np.array(roots)
def f(x):
    X = x.reshape(n, 4); nr = np.linalg.norm(X, axis=1); W = X / nr[:, None]; G = W @ W.T
    v = G[iu] - tv; vp = np.maximum(v, 0); val = (vp ** 2).sum()
    Gr = np.zeros((n, n)); Gr[iu] = 2 * vp; Gr = Gr + Gr.T; gW = Gr @ W
    gX = (gW - (gW * W).sum(1)[:, None] * W) / nr[:, None]
    return val, gX.ravel()
nclose = int((ds <= 2.0161 + 1e-12).sum())
best = None; nfeas = 0
for s in range(starts):
    if s % 2 == 0:
        x0 = rng.standard_normal(n * 4)
    else:
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        P = roots[rng.permutation(24)][:min(nclose, 24)] @ Q.T + 0.05 * rng.standard_normal((min(nclose, 24), 4))
        x0 = np.r_[P.ravel(), rng.standard_normal((n - len(P)) * 4)]
    r = minimize(f, x0, jac=True, method='L-BFGS-B', options={'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-16})
    if best is None or r.fun < best[0]:
        best = (r.fun, r.x)
    if r.fun < 1e-16:
        nfeas += 1
X = best[1].reshape(n, 4); W = X / np.linalg.norm(X, axis=1)[:, None]; G = W @ W.T
print('%s: feasible starts %d/%d, best max violation %.3e' % (spec, nfeas, starts, (G[iu] - tv).max()), flush=True)
