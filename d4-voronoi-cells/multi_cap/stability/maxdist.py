# D_k = max over k-point codes in S^3 with inner products <= 1/2 + s of sum_{pairs} dist(u_ij, {-1,-1/2,0,1/2})^2
# (lower bounds: every code found is a real code).  A level-2 certificate with bound B and pair penalty
# P2 >= c dist^2 gives c * D_k <= B - k for every k, so c <= min_k (B - k) / D_k.
import numpy as np, itertools, sys, json
from scipy.optimize import minimize
s = float(sys.argv[1]) if len(sys.argv) > 1 else 0.008
starts = int(sys.argv[2]) if len(sys.argv) > 2 else 12
t = 0.5 + s
def dist2(u): return np.where(u >= 0.5, 0.0, np.min((u[:, None] - np.array([-1, -.5, 0, .5]))**2, axis=1))  # [1/2, 1/2+s] is on-root
rng = np.random.default_rng(7)
R4 = []
for i_, j_ in itertools.combinations(range(4), 2):
    for a_ in (1, -1):
        for b_ in (1, -1):
            v_ = np.zeros(4); v_[i_] = a_; v_[j_] = b_; R4.append(v_ / np.sqrt(2))
R4 = np.array(R4)
res = {}
k0, k1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2, 24)
for k in range(k0, k1 + 1):
    P = np.array(list(itertools.combinations(range(k), 2))); I, J = P[:, 0], P[:, 1]
    def f(x):
        X = x.reshape(k, 4); u = np.einsum('ij,ij->i', X[I], X[J])
        val = -np.sum(np.sin(2*np.pi*u)**2); d = -4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*u)
        g = np.zeros_like(X); np.add.at(g, I, d[:, None]*X[J]); np.add.at(g, J, d[:, None]*X[I]); return val, g.ravel()
    def cin(x): X = x.reshape(k, 4); return t - np.einsum('ij,ij->i', X[I], X[J])
    def cin_j(x):
        X = x.reshape(k, 4); Jm = np.zeros((len(P), 4*k))
        for r, (a, b) in enumerate(P): Jm[r, 4*a:4*a+4] = -X[b]; Jm[r, 4*b:4*b+4] = -X[a]
        return Jm
    def ceq(x): X = x.reshape(k, 4); return (X*X).sum(1) - 1
    def ceq_j(x):
        X = x.reshape(k, 4); Jm = np.zeros((k, 4*k))
        for p in range(k): Jm[p, 4*p:4*p+4] = 2*X[p]
        return Jm
    best = 0.0
    for st in range(starts):
        if k >= 23:   # seed from D4 (minus 24-k roots) plus noise; random starts rarely reach feasibility here
            X0 = R4[rng.permutation(24)[:k]] + rng.choice([0.03, 0.08, 0.15, 0.25]) * rng.standard_normal((k, 4))
        else:
            X0 = rng.standard_normal((k, 4))
        X0 /= np.linalg.norm(X0, axis=1)[:, None]
        r = minimize(f, X0.ravel(), jac=True, method='SLSQP', constraints=[{'type': 'ineq', 'fun': cin, 'jac': cin_j},
                     {'type': 'eq', 'fun': ceq, 'jac': ceq_j}], options={'maxiter': 2000, 'ftol': 1e-12})
        X = r.x.reshape(k, 4); X /= np.linalg.norm(X, axis=1)[:, None]
        u = np.einsum('ij,ij->i', X[I], X[J])
        if u.max() <= t + 1e-9: best = max(best, dist2(u).sum())
    res[k] = best
    print('k=%2d  D_k >= %.5f' % (k, best), flush=True)
json.dump(res, open('maxdist_s%g_%d_%d.json' % (s, k0, k1), 'w'))
