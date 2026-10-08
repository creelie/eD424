# Best 23-point codes in S^3: minimise the largest inner product t (SLSQP on (X, t)) from random starts and from
# D4 minus one root. If t < 1/2 is reachable, 23-point codes exist that are not subsets of the root system,
# and a direction-only localisation of 23 points near D4 cannot hold at any slack.
import numpy as np, itertools, sys
from scipy.optimize import minimize
k = int(sys.argv[1]) if len(sys.argv) > 1 else 23
starts = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
R = []
for i, j in itertools.combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = np.zeros(4); v[i] = a; v[j] = b; R.append(v / np.sqrt(2))
R = np.array(R)
I, J = np.triu_indices(k, 1)
def obj(z): g = np.zeros_like(z); g[-1] = 1; return z[-1], g
def cin(z):
    X = z[:-1].reshape(k, 4); return z[-1] - np.einsum('ij,ij->i', X[I], X[J])
def cin_j(z):
    X = z[:-1].reshape(k, 4); Jm = np.zeros((len(I), 4 * k + 1)); Jm[:, -1] = 1
    for r, (p, q) in enumerate(zip(I, J)): Jm[r, 4*p:4*p+4] = -X[q]; Jm[r, 4*q:4*q+4] = -X[p]
    return Jm
def ceq(z): X = z[:-1].reshape(k, 4); return (X * X).sum(1) - 1
def ceq_j(z):
    X = z[:-1].reshape(k, 4); Jm = np.zeros((k, 4 * k + 1))
    for p in range(k): Jm[p, 4*p:4*p+4] = 2 * X[p]
    return Jm
def d4_distance(X):
    # smallest root-sum-square distance to k roots of a rotated root system: Procrustes with greedy/Hungarian labelling
    from scipy.optimize import linear_sum_assignment
    best = 9
    for trial in range(30):
        Q = np.linalg.qr(rng.standard_normal((4, 4)))[0] if trial else np.eye(4)
        for it in range(50):
            C = -(X @ (R @ Q.T).T); rows, cols = linear_sum_assignment(C)
            Y = R[cols]; U, S, Vt = np.linalg.svd(Y.T @ X); Qn = (U @ Vt).T
            if np.allclose(Qn, Q, atol=1e-12): break
            Q = Qn
        best = min(best, np.linalg.norm(X - Y @ Q.T))
    return best
res = []
for st in range(starts):
    if st % 4 == 0:
        drop = rng.choice(24, 24 - k, replace=False); X0 = np.delete(R, drop, 0) + 0.05 * rng.standard_normal((k, 4))
    else:
        X0 = rng.standard_normal((k, 4))
    X0 /= np.linalg.norm(X0, axis=1)[:, None]
    z0 = np.concatenate([X0.ravel(), [np.max((X0 @ X0.T)[I, J])]])
    r = minimize(obj, z0, jac=True, method='SLSQP', constraints=[{'type': 'ineq', 'fun': cin, 'jac': cin_j},
                 {'type': 'eq', 'fun': ceq, 'jac': ceq_j}], options={'maxiter': 2000, 'ftol': 1e-15})
    X = r.x[:-1].reshape(k, 4); X /= np.linalg.norm(X, axis=1)[:, None]
    t = np.max((X @ X.T)[I, J]); dd = d4_distance(X)
    res.append((t, dd, st)); print('start %2d (%s): max ip %.6f  angle %.3f deg  RSS distance to D4 subset %.4f' % (
        st, 'D4-' if st % 4 == 0 else 'rnd', t, np.degrees(np.arccos(t)), dd), flush=True)
    if st == 0 or t <= min(x[0] for x in res): np.save('code%d_best.npy' % k, X)
res.sort(); print('best max ip %.6f (angle %.3f deg), its D4 distance %.4f' % (res[0][0], np.degrees(np.arccos(res[0][0])), res[0][1]))
