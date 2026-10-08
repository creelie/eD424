"""How many far points (holes) can a two-shell code hold?  Floating point exploration.
usage: code_feasible2.py nA nB nF dA dB starts seed"""
import numpy as np, sys
from scipy.optimize import minimize
nA, nB, nF = map(int, sys.argv[1:4]); dA, dB = map(float, sys.argv[4:6]); starts = int(sys.argv[6]); seed = int(sys.argv[7])
rng = np.random.default_rng(seed)
def a(d, e): return (d*d + e*e - 4) / (2*d*e)
R6 = 6 ** .5
T = {('A','A'): a(dA,dA), ('A','B'): a(dA,dB), ('B','B'): a(dB,dB),
     ('A','F'): a(dA,R6), ('B','F'): a(dB,R6), ('F','F'): a(R6,R6)}
types = ['A']*nA + ['B']*nB + ['F']*nF
n = len(types)
Tm = np.array([[T[tuple(sorted((ti, tj)))] for tj in types] for ti in types])
iu = np.triu_indices(n, 1); tv = Tm[iu]
roots = []
import itertools
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
best = None; nfeas = 0
for s in range(starts):
    if s % 2 == 0:
        x0 = rng.standard_normal(n * 4)
    else:  # near-root start: first nA+nB from the roots (shuffled, perturbed), far points random
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        P = roots[rng.permutation(24)][:nA + nB] @ Q.T + 0.05 * rng.standard_normal((nA + nB, 4))
        x0 = np.r_[P.ravel(), rng.standard_normal(nF * 4)]
    r = minimize(f, x0, jac=True, method='L-BFGS-B', options={'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-16})
    if best is None or r.fun < best[0]:
        best = (r.fun, r.x)
    if r.fun < 1e-16:
        nfeas += 1
X = best[1].reshape(n, 4); W = X / np.linalg.norm(X, axis=1)[:, None]; G = W @ W.T
print('nA=%d nB=%d nF=%d dA=%.4f dB=%.4f: feasible starts %d/%d, best max violation %.3e' % (nA, nB, nF, dA, dB, nfeas, starts, (G[iu] - tv).max()), flush=True)
np.save('cf2_%d_%d_%d_%s_%s.npy' % (nA, nB, nF, dA, dB), W)
