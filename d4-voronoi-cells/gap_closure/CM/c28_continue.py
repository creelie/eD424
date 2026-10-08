#!/usr/bin/env python3
"""
c28_continue.py m starts seed -- from codes with m centres within 2.0161 (under the radial
counts of 28, every centre at its bin maximum) found by local search, move one centre at
2.1 in to 2.0161 by continuation, re-minimising the squared violations at each step.
Reports the smallest largest violation reached with m + 1 close centres.  Floating point.
"""
import itertools, sys
import numpy as np
from scipy.optimize import minimize
m, starts, seed = map(int, sys.argv[1:4])
R6 = 6 ** .5
def dists(mm):
    return np.array([2.0161] * mm + [2.1] * (21 - mm) + [2.15, 2.2, 2.35] + [R6] * 4)
n = 28; iu = np.triu_indices(n, 1)
def tmat(ds):
    return ((ds[:, None] ** 2 + ds[None, :] ** 2 - 4) / (2 * ds[:, None] * ds[None, :]))[iu]
def make_f(tv):
    def f(x):
        X = x.reshape(n, 4); nr = np.linalg.norm(X, axis=1); W = X / nr[:, None]; G = W @ W.T
        v = G[iu] - tv; vp = np.maximum(v, 0); val = (vp ** 2).sum()
        Gr = np.zeros((n, n)); Gr[iu] = 2 * vp; Gr = Gr + Gr.T; gW = Gr @ W
        gX = (gW - (gW * W).sum(1)[:, None] * W) / nr[:, None]
        return val, gX.ravel()
    return f
def maxviol(x, tv):
    X = x.reshape(n, 4); W = X / np.linalg.norm(X, axis=1)[:, None]
    return ((W @ W.T)[iu] - tv).max()
opt = {'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-16}
rng = np.random.default_rng(seed)
d0 = dists(m); tv0 = tmat(d0); f0 = make_f(tv0)
found = []
for s in range(starts):
    r = minimize(f0, rng.standard_normal(n * 4), jac=True, method='L-BFGS-B', options=opt)
    if r.fun < 1e-16:
        found.append(r.x)
print('m = %d: %d codes found in %d starts' % (m, len(found), starts), flush=True)
best = (np.inf, None)
for k, x in enumerate(found):
    for j in range(m, 21):
        y = x.copy()
        for lam in np.linspace(0, 1, 26)[1:]:
            ds = d0.copy(); ds[j] = 2.1 - lam * (2.1 - 2.0161)
            tv = tmat(ds)
            y = minimize(make_f(tv), y, jac=True, method='L-BFGS-B', options=opt).x
        mv = maxviol(y, tv)
        if mv < best[0]:
            best = (mv, (k, j))
    print('   code %d: best so far %.3e' % (k, best[0]), flush=True)
print('m + 1 = %d by continuation: smallest largest violation %.3e' % (m + 1, best[0]), flush=True)
