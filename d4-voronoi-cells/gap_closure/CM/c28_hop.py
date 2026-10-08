#!/usr/bin/env python3
"""
c28_hop.py code.txt hops sigma seed -- basin hopping from a code of 28 directions with k close
centres (a witness file of c28_maxclose / c28_extend): perturb the directions by sigma, run the
close-count search of c28_maxclose.py from there with the current distances as the start, and
keep the code whenever the close count grows.  Floating point.
"""
import os, sys
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'c28_maxclose.py')).read().split('opt = {')[0]
path, hops, sigma, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
sys.argv = ['c28_maxclose.py', '1', '0', '28']
g = {}; exec(src, g)
make, viol_only, amat, n, nv, ups, lo, fixed = (g[k] for k in ('make', 'viol_only', 'amat', 'n', 'nv', 'ups', 'lo', 'fixed'))
iu = np.triu_indices(n, 1)
opt = {'maxiter': 20000, 'gtol': 1e-13, 'ftol': 1e-15}
bounds = [(None, None)] * (4 * n) + [(0, 1)] * nv
X0 = np.loadtxt(path); W = X0[:, :4]; d = X0[:, 4]
s = np.clip((ups - d[:nv]) / (ups - lo), 0, 1)
cur = (W.ravel(), s, int((s > 0.99).sum()))
rng = np.random.default_rng(seed)
print('start: %d close centres' % cur[2], flush=True)
for h in range(hops):
    z = np.r_[cur[0] + sigma * rng.standard_normal(4 * n), np.clip(cur[1] + 0.2 * rng.standard_normal(nv), 0, 1)]
    for mu in (1e2, 1e3, 1e4, 1e5):
        z = minimize(make(mu), z, jac=True, method='L-BFGS-B', bounds=bounds, options=opt).x
    sv = z[4 * n:]
    for k in range(int((sv > 0.7).sum()), cur[2], -1):
        order = np.argsort(-sv)
        ds = np.r_[np.where(np.isin(np.arange(nv), order[:k]), lo, ups), fixed]
        f, tv = viol_only(ds)
        r = minimize(f, z[:4 * n], jac=True, method='L-BFGS-B', options=opt)
        Xr = r.x.reshape(n, 4); Wr = Xr / np.linalg.norm(Xr, axis=1)[:, None]
        if ((Wr @ Wr.T)[iu] - tv).max() < 1e-8:
            cur = (Wr.ravel(), np.where(np.isin(np.arange(nv), order[:k]), 1.0, 0.0), k)
            np.save(os.path.join(HERE, 'hop_%d.npy' % k), Wr.ravel())
            print('hop %d: %d close centres fit' % (h, k), flush=True)
            break
    if (h + 1) % 100 == 0:
        print('after %d hops: %d close centres' % (h + 1, cur[2]), flush=True)
print('largest: %d' % cur[2], flush=True)
