"""Multi-type Delsarte bound (floating point exploration).
Types s with counts n_s; ordered pairs of types (s,t) have inner products in [-1, T_st].
Find PSD F_k (ntypes x ntypes), k=1..D, and constants c_st with
   g_st(u) = sum_k F_k[s,t] G_k(u) <= c_st on [-1, T_st]
minimising  sum_s n_s g_ss(1) + sum_{s,t} N_st c_st,  N_st = n_s n_t - n_s delta_st,
normalised by sum_s n_s g_ss(1) = 1.  A value < 0 proves (after rigorous checking) that no code exists.
G_k = U_k/(k+1) Gegenbauer of S^3, G_k(1)=1."""
import numpy as np, cvxpy as cp, sys
def a(d, e): return (d*d + e*e - 4) / (2*d*e)
R6 = 6 ** .5
def Gk(u, D):
    u = np.asarray(u, float); out = [np.ones_like(u), 2*u]
    for j in range(2, D+1): out.append(2*u*out[-1] - out[-2])
    return np.stack([out[k]/(k+1) for k in range(D+1)], -1)
def run(types, D, ngrid=2000):
    names = list(types)
    n = np.array([types[s][0] for s in names], float)
    rad = {s: types[s][1] for s in names}
    m = len(names)
    F = [cp.Variable((m, m), symmetric=True) for _ in range(D+1)]
    c = cp.Variable((m, m), symmetric=True)
    cons = [F[k] >> 0 for k in range(1, D+1)]
    for i, s in enumerate(names):
        for j, t in enumerate(names):
            if j < i: continue
            T = min(a(rad[s][1], rad[t][1]), 1.0)
            if T < -1: continue
            u = np.linspace(-1, T, ngrid)
            G = Gk(u, D)
            g = sum(G[:, k] * F[k][i, j] for k in range(1, D+1))
            cons.append(g <= c[i, j])
    one = sum(n[i] * sum(F[k][i, i] for k in range(1, D+1)) for i in range(m))
    N = np.outer(n, n) - np.diag(n)
    obj = one + cp.sum(cp.multiply(N, c))
    cons.append(one == 1)
    pr = cp.Problem(cp.Minimize(obj), cons)
    pr.solve(solver='CLARABEL')
    return pr.status, pr.value
if __name__ == '__main__':
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    nF = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    # types: name -> (count, (dmin, dmax)) ; only dmax used for the inner-product bound
    types = {'A': (22, (2, 2.05)), 'B': (2, (2, 2.25)), 'F': (nF, (2.4, R6))}
    print(D, nF, run(types, D))
