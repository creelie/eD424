#!/usr/bin/env python3
"""
c28_chain.py m starts seed -- from codes with m centres within 2.0161 (radial counts of 28,
every centre at its bin maximum) found by local search, move centres at 2.1 in to 2.0161
one at a time by continuation (25 steps each), trying every remaining centre at 2.1 and
keeping the move with the smallest largest violation; stop when no move stays feasible.
Floating point; feasibility means largest violation below 1e-8.
"""
import sys
import numpy as np
from scipy.optimize import minimize
m0, starts, seed = map(int, sys.argv[1:4])
R6 = 6 ** .5
n = 28; iu = np.triu_indices(n, 1)
base = np.array([2.1] * 21 + [2.15, 2.2, 2.35] + [R6] * 4)
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
ds0 = base.copy(); ds0[:m0] = 2.0161
f0 = make_f(tmat(ds0))
codes = []
for s in range(starts):
    r = minimize(f0, rng.standard_normal(n * 4), jac=True, method='L-BFGS-B', options=opt)
    if r.fun < 1e-16:
        codes.append(r.x)
print('m = %d: %d codes found in %d starts' % (m0, len(codes), starts), flush=True)
reach = []
for c, x in enumerate(codes):
    ds = ds0.copy(); y = x.copy(); close = list(range(m0))
    while True:
        best = (np.inf, None, None)
        for j in [j for j in range(21) if j not in close]:
            z = y.copy()
            for lam in np.linspace(0, 1, 26)[1:]:
                dd = ds.copy(); dd[j] = 2.1 - lam * (2.1 - 2.0161)
                z = minimize(make_f(tmat(dd)), z, jac=True, method='L-BFGS-B', options=opt).x
            mv = maxviol(z, tmat(dd))
            if mv < best[0]:
                best = (mv, j, z)
        if best[0] > 1e-8:
            print('   code %d: stops at %d close centres; best move to %d misses by %.3e' % (c, len(close), len(close) + 1, best[0]), flush=True)
            reach.append(len(close)); break
        close.append(best[1]); ds[best[1]] = 2.0161; y = best[2]
        print('   code %d: %d close centres fit (largest violation %.3e)' % (c, len(close), best[0]), flush=True)
        np.save('chain_m%d_code%d_%d.npy' % (m0, c, len(close)), y)
print('largest count reached: %s' % (max(reach) if reach else None), flush=True)
