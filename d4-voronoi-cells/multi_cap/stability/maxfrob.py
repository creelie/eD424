# Largest Procrustes distance min_Q ||X - Q R||_F over 24-point codes X in S^3 with all inner
# products <= 1/2 + s (s = 0.008), labelled from the D4 roots R. Local maximisation (SLSQP) from
# random starts; the constraint R^T X symmetric keeps Q = I stationary, and the final distance
# is recomputed with a true Procrustes rotation.
import numpy as np, itertools, sys
from scipy.optimize import minimize
s = float(sys.argv[1]) if len(sys.argv) > 1 else 0.008
starts = int(sys.argv[2]) if len(sys.argv) > 2 else 20
R = []
for i, j in itertools.combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = np.zeros(4); v[i] = a; v[j] = b; R.append(v / np.sqrt(2))
R = np.array(R); k = 24; t = 0.5 + s
I, J = np.triu_indices(k, 1)
def procr(X):
    U, S_, Vt = np.linalg.svd(X.T @ R); Q = U @ Vt
    return np.linalg.norm(X - R @ Q.T), np.linalg.norm(X - R @ Q.T, axis=1)
def f(x):
    X = x.reshape(k, 4); D = X - R; return -np.sum(D * D), (-2 * D).ravel()
def cin(x): X = x.reshape(k, 4); return t - np.einsum('ij,ij->i', X[I], X[J])
def cin_j(x):
    X = x.reshape(k, 4); Jm = np.zeros((len(I), 4 * k))
    for r, (p, q) in enumerate(zip(I, J)): Jm[r, 4*p:4*p+4] = -X[q]; Jm[r, 4*q:4*q+4] = -X[p]
    return Jm
def ceq(x):
    X = x.reshape(k, 4); M = R.T @ X
    return np.concatenate([(X * X).sum(1) - 1, [M[a, b] - M[b, a] for a, b in itertools.combinations(range(4), 2)]])
def ceq_j(x):
    X = x.reshape(k, 4); Jm = np.zeros((k + 6, 4 * k))
    for p in range(k): Jm[p, 4*p:4*p+4] = 2 * X[p]
    for r, (a, b) in enumerate(itertools.combinations(range(4), 2)):
        for p in range(k): Jm[k + r, 4*p + b] += R[p, a]; Jm[k + r, 4*p + a] -= R[p, b]
    return Jm
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
best = 0
for st in range(starts):
    sc = rng.choice([0.02, 0.05, 0.1, 0.15])
    X0 = R + sc * rng.standard_normal(R.shape); X0 /= np.linalg.norm(X0, axis=1)[:, None]
    r = minimize(f, X0.ravel(), jac=True, method='SLSQP',
                 constraints=[{'type': 'ineq', 'fun': cin, 'jac': cin_j}, {'type': 'eq', 'fun': ceq, 'jac': ceq_j}],
                 options={'maxiter': 3000, 'ftol': 1e-14})
    X = r.x.reshape(k, 4); X /= np.linalg.norm(X, axis=1)[:, None]
    G = X @ X.T; mx = G[I, J].max()
    F, dev = procr(X)
    ok = mx <= t + 1e-7
    print('start %2d scale %.2f  %s  max ip %.6f  Procrustes F %.5f  max per-direction %.5f  sum dist(u,roots)^2 %.5f' % (
        st, sc, 'ok ' if ok else 'BAD', mx, F, dev.max(),
        np.sum(np.min(np.abs(G[I, J][:, None] - np.array([-1, -.5, 0, .5])[None]), axis=1) ** 2)), flush=True)
    if ok and F > best: best = F; np.save('maxfrob_best_s%g.npy' % s, X)
print('best F %.5f' % best)
