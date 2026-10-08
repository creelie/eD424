# F* = sqrt(min_z sum_i (<z,r_i> - tau)_+^2) over unit z, r_i the 24 roots of D4 (unit).
# If a 24-point code X satisfies min_Q ||X - Q R||_F < F*, every direction z has max_i <z,x_i> > tau.
import numpy as np, itertools
from scipy.optimize import minimize
R = []
for i, j in itertools.combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = np.zeros(4); v[i] = a; v[j] = b; R.append(v / np.sqrt(2))
R = np.array(R)
rng = np.random.default_rng(1)
def cost(z, tau):
    z = z / np.linalg.norm(z); return np.sum(np.maximum(R @ z - tau, 0) ** 2)
for tau in (0.6141, 0.60, 0.62, 0.65):
    best = (9, None)
    for _ in range(4000):
        z = rng.standard_normal(4)
        r = minimize(cost, z, args=(tau,), method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 4000})
        if r.fun < best[0]: best = (r.fun, r.x / np.linalg.norm(r.x))
    z = best[1]; ips = np.sort(R @ z)[::-1]
    print('tau %.4f  F*^2 = %.6f  F* = %.5f  at z with top inner products %s' % (tau, best[0], np.sqrt(best[0]), np.round(ips[:8], 4)))
