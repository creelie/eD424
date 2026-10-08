#!/usr/bin/env python3
"""
c28_extend.py code.npy -- take a code of 28 directions (radial counts of 28, each centre at
its bin maximum) found by c28_maxclose.py, recover its close centres, and move the other
centres at 2.1 in to 2.0161 one at a time by continuation (40 steps each, every candidate
tried, the best kept), until no move stays feasible.  Floating point.
"""
import sys, itertools
import numpy as np
from scipy.optimize import minimize
R6 = 6 ** .5
n = 28; iu = np.triu_indices(n, 1)
base = np.r_[[2.1] * 21, [2.15, 2.2, 2.35], [R6] * 4]
def A(ds):
    return (ds[:, None] ** 2 + ds[None, :] ** 2 - 4) / (2 * ds[:, None] * ds[None, :])
def make_f(tv):
    def f(x):
        X = x.reshape(n, 4); nr = np.linalg.norm(X, axis=1); W = X / nr[:, None]; G = W @ W.T
        v = G[iu] - tv; vp = np.maximum(v, 0); val = (vp ** 2).sum()
        Gr = np.zeros((n, n)); Gr[iu] = 2 * vp; Gr = Gr + Gr.T; gW = Gr @ W
        gX = (gW - (gW * W).sum(1)[:, None] * W) / nr[:, None]
        return val, gX.ravel()
    return f
def maxviol(x, ds):
    X = x.reshape(n, 4); W = X / np.linalg.norm(X, axis=1)[:, None]
    return ((W @ W.T) - A(ds))[iu].max()
opt = {'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-16}
x = np.load(sys.argv[1])
X = x.reshape(n, 4); W = X / np.linalg.norm(X, axis=1)[:, None]; G = W @ W.T
cand = []
for j in range(21):
    d = base.copy(); d[j] = 2.0161
    if (G - A(d))[j][np.arange(n) != j].max() < 1e-8:
        cand.append(j)
close = None
for k in range(len(cand), 0, -1):
    for S in itertools.combinations(cand, k):
        d = base.copy(); d[list(S)] = 2.0161
        if (G - A(d))[iu].max() < 1e-8:
            close = list(S); break
    if close: break
ds = base.copy(); ds[close] = 2.0161
print('%s: %d close centres, largest violation %.3e' % (sys.argv[1], len(close), maxviol(x, ds)), flush=True)
y = x.copy()
while True:
    best = (np.inf, None, None)
    for j in [j for j in range(21) if j not in close]:
        z = y.copy()
        for lam in np.linspace(0, 1, 41)[1:]:
            dd = ds.copy(); dd[j] = 2.1 - lam * (2.1 - 2.0161)
            z = minimize(make_f(A(dd)[iu]), z, jac=True, method='L-BFGS-B', options=opt).x
        mv = maxviol(z, dd)
        if mv < best[0]:
            best = (mv, j, z)
    if best[0] > 1e-8:
        print('stops at %d close centres; best move to %d misses by %.3e' % (len(close), len(close) + 1, best[0]), flush=True)
        break
    close.append(best[1]); ds[best[1]] = 2.0161; y = best[2]
    print('%d close centres fit (largest violation %.3e)' % (len(close), best[0]), flush=True)
    np.save(sys.argv[1].replace('.npy', '_ext%d.npy' % len(close)), y)
