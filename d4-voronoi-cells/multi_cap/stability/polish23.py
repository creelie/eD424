# Random starts for 23-point codes; keep those in the second family (max ip in (0.5005, 0.5095)) and polish each by
# repeated SLSQP restarts on (X, t). Prints the polished max ip, the number of active pairs and the Gram spectrum gap.
import numpy as np, itertools, sys
from scipy.optimize import minimize
k = 23; rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7); want = int(sys.argv[2]) if len(sys.argv) > 2 else 4
I, J = np.triu_indices(k, 1)
def obj(z): g = np.zeros_like(z); g[-1] = 1; return z[-1], g
def cin(z): X = z[:-1].reshape(k, 4); return z[-1] - np.einsum('ij,ij->i', X[I], X[J])
def cin_j(z):
    X = z[:-1].reshape(k, 4); Jm = np.zeros((len(I), 4 * k + 1)); Jm[:, -1] = 1
    Jm[np.arange(len(I))[:, None], 4 * I[:, None] + np.arange(4)] = -X[J]
    Jm[np.arange(len(I))[:, None], 4 * J[:, None] + np.arange(4)] = -X[I]
    return Jm
def ceq(z): X = z[:-1].reshape(k, 4); return (X * X).sum(1) - 1
def ceq_j(z):
    X = z[:-1].reshape(k, 4); Jm = np.zeros((k, 4 * k + 1))
    Jm[np.arange(k)[:, None], 4 * np.arange(k)[:, None] + np.arange(4)] = 2 * X
    return Jm
def solve(X0, it=3000):
    z0 = np.concatenate([X0.ravel(), [np.max((X0 @ X0.T)[I, J])]])
    r = minimize(obj, z0, jac=True, method='SLSQP', constraints=[{'type': 'ineq', 'fun': cin, 'jac': cin_j},
                 {'type': 'eq', 'fun': ceq, 'jac': ceq_j}], options={'maxiter': it, 'ftol': 1e-16})
    X = r.x[:-1].reshape(k, 4); X /= np.linalg.norm(X, axis=1)[:, None]; return X, np.max((X @ X.T)[I, J])
found = 0; tries = 0
while found < want and tries < 200:
    tries += 1
    X0 = rng.standard_normal((k, 4)); X0 /= np.linalg.norm(X0, axis=1)[:, None]
    X, t = solve(X0)
    if not (0.5005 < t < 0.5095): continue
    found += 1; hist = [t]
    for rep in range(12):
        Xn, tn = solve(X + 1e-7 * rng.standard_normal(X.shape))
        if tn < t: X, t = Xn, tn
        hist.append(t)
    u = (X @ X.T)[I, J]; act = np.sum(u > t - 1e-7)
    np.save('code23_fam2_%d.npy' % found, X)
    print('try %3d: polished max ip %.8f (slack %.6f), active pairs %d, history %s' % (
        tries, t, t - 0.5, act, ' '.join('%.7f' % h for h in hist[::3])), flush=True)
    # inner-product spectrum: which values occur
    vals = np.sort(u)[::-1]; print('   top inner products', np.round(vals[:12], 5), ' count >= 0.45:', np.sum(u > 0.45), flush=True)
