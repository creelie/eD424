"""Floating point: minimise the largest violation of a typed code.
usage: code_feas_gen.py starts seed  r1:n1 r2:n2 ...   (radius:count; radius 'F' = sqrt6 shell)"""
import numpy as np, sys, itertools
from scipy.optimize import minimize
starts, seed = int(sys.argv[1]), int(sys.argv[2])
spec = [(float(x.split(':')[0]) if x.split(':')[0] != 'F' else 6 ** .5, int(x.split(':')[1])) for x in sys.argv[3:]]
rad = np.concatenate([[r] * n for r, n in spec]); n = len(rad)
def a(d, e): return (d*d + e*e - 4) / (2*d*e)
Tm = a(rad[:, None], rad[None, :]); iu = np.triu_indices(n, 1); tv = Tm[iu]
rng = np.random.default_rng(seed)
roots = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = np.zeros(4); v[i] = si; v[j] = sj; roots.append(v / 2 ** .5)
roots = np.array(roots)
nnear = int((rad < 2.4).sum())
def f(x, p=2):
    X = x.reshape(n, 4); nr = np.linalg.norm(X, axis=1); W = X / nr[:, None]; G = W @ W.T
    v = G[iu] - tv; vp = np.maximum(v, 0); val = (vp ** 2).sum()
    Gr = np.zeros((n, n)); Gr[iu] = 2 * vp; Gr = Gr + Gr.T; gW = Gr @ W
    gX = (gW - (gW * W).sum(1)[:, None] * W) / nr[:, None]
    return val, gX.ravel()
best = None; nfeas = 0
for s in range(starts):
    if s % 2 == 0:
        x0 = rng.standard_normal(n * 4)
    else:
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        k = min(nnear, 24)
        P = roots[rng.permutation(24)][:k] @ Q.T + 0.05 * rng.standard_normal((k, 4))
        x0 = np.r_[P.ravel(), rng.standard_normal((n - k) * 4)]
    r = minimize(f, x0, jac=True, method='L-BFGS-B', options={'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-16})
    if best is None or r.fun < best[0]:
        best = (r.fun, r.x)
    if r.fun < 1e-16: nfeas += 1
X = best[1].reshape(n, 4); W = X / np.linalg.norm(X, axis=1)[:, None]; G = W @ W.T
print(' '.join(sys.argv[3:]), ': feasible %d/%d, best max violation %.3e' % (nfeas, starts, (G[iu] - tv).max()), flush=True)
