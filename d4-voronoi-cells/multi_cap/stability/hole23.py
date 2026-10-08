# Two far centres beside 23 directions near the root system minus r0 (the 23-close case of (C)).
# Budget: if the 23 close directions are within RSS F of sigma(R minus r0), two further centres within sqrt6 need
# F^2 >= g(w1,w2) = sum_{i != 0} max_k (<w_k, r_i> - tau)_+^2 with <w1,w2> <= 2/3 (both at sqrt6, the easiest case).
# Floating point: min g over many starts (Nelder-Mead on a penalised objective).
import numpy as np, itertools
from scipy.optimize import minimize
R = []
for i, j in itertools.combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = np.zeros(4); v[i] = a; v[j] = b; R.append(v / np.sqrt(2))
R = np.array(R); r0 = R[0]; R23 = R[1:]
rho2 = 4 / 0.984; tau = (2 + rho2) / (2 * np.sqrt(6) * np.sqrt(rho2))
def g(z, pen=1e3):
    w1, w2 = z[:4] / np.linalg.norm(z[:4]), z[4:] / np.linalg.norm(z[4:])
    c = np.maximum(np.maximum(R23 @ w1 - tau, 0), np.maximum(R23 @ w2 - tau, 0))
    return np.sum(c ** 2) + pen * max(0, w1 @ w2 - 2 / 3) ** 2
rng = np.random.default_rng(3); best = (9, None)
for st in range(3000):
    if st % 3 == 0: z = np.concatenate([r0 + 0.5 * rng.standard_normal(4), r0 + 0.5 * rng.standard_normal(4)])
    else: z = rng.standard_normal(8)
    r = minimize(g, z, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 6000, 'maxfev': 12000})
    r = minimize(g, r.x, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-16, 'maxiter': 6000, 'maxfev': 12000})
    if r.fun < best[0]: best = (r.fun, r.x)
w1, w2 = best[1][:4] / np.linalg.norm(best[1][:4]), best[1][4:] / np.linalg.norm(best[1][4:])
print('tau %.6f; min g = %.6f -> F23* = %.5f' % (tau, best[0], np.sqrt(best[0])))
print('angles from r0: %.2f and %.2f deg, between them %.2f deg; <w1,w2> = %.5f' % (
    np.degrees(np.arccos(w1 @ r0)), np.degrees(np.arccos(w2 @ r0)), np.degrees(np.arccos(w1 @ w2)), w1 @ w2))
print('top ips w1:', np.round(np.sort(R23 @ w1)[::-1][:6], 4), ' w2:', np.round(np.sort(R23 @ w2)[::-1][:6], 4))
print('one far centre alone (24-close analogue, 6h^2) = %.6f' % (6 * (2 ** -0.5 - tau) ** 2))
