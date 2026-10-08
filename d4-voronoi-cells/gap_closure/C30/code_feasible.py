"""Floating-point exploration: does a 30-point two/three-type spherical code exist
with 22 'A' points (pairwise <= tAA), 2 'B' points, 6 'F' points (far centres in holes)?
Minimise sum of squared violations from random starts (L-BFGS)."""
import numpy as np, sys
from scipy.optimize import minimize
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
def a(d, e): return (d*d + e*e - 4) / (2*d*e)
R6 = 6 ** .5
nA, nB, nF = 22, 2, 6
dA, dB, dF = 2.05, 2.25, 2.4   # A within 2.05, B within 2.25, F in [2.4, sqrt6)
# the largest admissible inner product for each type pair (a increases in each argument)
T = {('A','A'): a(dA,dA), ('A','B'): a(dA,dB), ('B','B'): a(dB,dB),
     ('A','F'): a(dA,R6), ('B','F'): a(dB,R6), ('F','F'): a(R6,R6)}
types = ['A']*nA + ['B']*nB + ['F']*nF
n = len(types)
Tm = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        key = tuple(sorted((types[i], types[j])))
        Tm[i, j] = T[key]
iu = np.triu_indices(n, 1)
tv = Tm[iu]
def f(x):
    X = x.reshape(n, 4)
    nr = np.linalg.norm(X, axis=1)
    W = X / nr[:, None]
    G = W @ W.T
    v = G[iu] - tv
    vp = np.maximum(v, 0)
    val = (vp ** 2).sum()
    # gradient
    Gr = np.zeros((n, n))
    Gr[iu] = 2 * vp
    Gr = Gr + Gr.T
    gW = Gr @ W
    gX = (gW - (gW * W).sum(1)[:, None] * W) / nr[:, None]
    return val, gX.ravel()
best = None
for s in range(int(sys.argv[2]) if len(sys.argv) > 2 else 200):
    x0 = rng.standard_normal(n * 4)
    r = minimize(f, x0, jac=True, method='L-BFGS-B', options={'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-16})
    if best is None or r.fun < best[0]:
        best = (r.fun, r.x)
    if r.fun < 1e-14:
        print('FEASIBLE found at start', s, r.fun); break
X = best[1].reshape(n, 4); W = X / np.linalg.norm(X, axis=1)[:, None]; G = W @ W.T
print('best sum sq violation %.3e, max violation %.3e' % (best[0], (G[iu] - tv).max()))
np.save('code_best_%s.npy' % (sys.argv[1] if len(sys.argv) > 1 else '0'), W)
print({k: round(v, 5) for k, v in T.items()})
