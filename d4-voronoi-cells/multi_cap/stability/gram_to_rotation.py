# Lemma P (Gram deviation to rotation distance). X: 24 x 4 with unit rows, R: normalised D4 roots (labelled),
# R^T R = 6 I. D^2 = ||XX^T - RR^T||_F^2 = 2 sum_{i<j} (u_ij - r_ij)^2, F^2 = min over orthogonal Q ||X - R Q||_F^2.
# Identity: D^2 = a^2 + 12 e, a = ||X^T X - 6I||_F, e = ||P_perp X||_F^2 (P = RR^T/6).
# Bound:    F^2 <= e + (a + e)^2 / (sqrt(6 - a - e) + sqrt6)^2   (when a + e < 6).
# Part 1: numerical check on random and extremal configurations. Part 2: rigorous threshold c* such that
# sum_{i<j} (u_ij - r_ij)^2 <= c* implies F^2 < 6 h^2 (Lemma H), so no further centre within sqrt6.
import numpy as np, itertools, sys
from fractions import Fraction as Fr
from math import isqrt
R = []
for i, j in itertools.combinations(range(4), 2):
    for a in (1, -1):
        for b in (1, -1):
            v = np.zeros(4); v[i] = a; v[j] = b; R.append(v / np.sqrt(2))
R = np.array(R); P = R @ R.T / 6
def quantities(X):
    G, G0 = X @ X.T, R @ R.T
    D2 = np.sum((G - G0) ** 2)
    a = np.linalg.norm(X.T @ X - 6 * np.eye(4)); e = np.sum(((np.eye(24) - P) @ X) ** 2)
    U, S, Vt = np.linalg.svd(R.T @ X); F2 = np.sum((X - R @ (U @ Vt)) ** 2)
    bnd = e + (a + e) ** 2 / (np.sqrt(max(6 - a - e, 0)) + np.sqrt(6)) ** 2 if a + e < 6 else np.inf
    return D2, a, e, F2, bnd
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
worst_id, worst_ratio = 0, 0
for trial in range(20000):
    sc = 10 ** rng.uniform(-3, 0)
    X = R + sc * rng.standard_normal(R.shape)
    if trial % 2: X = R @ np.linalg.qr(rng.standard_normal((4, 4)))[0] + sc * rng.standard_normal(R.shape)
    if trial % 3 == 0: X[:, :] += sc * np.outer(rng.standard_normal(24), rng.standard_normal(4)) / 3
    X /= np.linalg.norm(X, axis=1)[:, None]
    D2, a, e, F2, bnd = quantities(X)
    worst_id = max(worst_id, abs(D2 - a * a - 12 * e) / max(D2, 1e-30))
    if a + e < 6: worst_ratio = max(worst_ratio, F2 / bnd)
print('identity D^2 = a^2 + 12e: worst relative error %.1e' % worst_id)
print('bound: worst F^2 / bound over random trials = %.6f (must be <= 1)' % worst_ratio)
for f in ('maxfrob_best_s0.008.npy',):
    X = np.load(f); U, S, Vt = np.linalg.svd(X.T @ R); Q = U @ Vt
    # the file's labelling is the one from the roots it started at; F is rotation distance under that labelling
    D2, a, e, F2, bnd = quantities(X)
    print('%s: sum_{i<j}(u-r)^2 = %.5f  F = %.5f  bound on F = %.5f' % (f, D2 / 2, np.sqrt(F2), np.sqrt(bnd)))

# Part 2 (rigorous, rational arithmetic). phi(D) = max over e in [0, D^2/12], a = sqrt(D^2 - 12e) of the bound.
K = 10 ** 30
def sqrt_lo(q): n, d = q.numerator, q.denominator; return Fr(isqrt(n * d * K * K), d * K)
def sqrt_hi(q): return sqrt_lo(q) + Fr(1, q.denominator * K)
s0 = Fr(1, 125); rho2 = 4 / (1 - 2 * s0)
# lower bound for 6h^2 = 6 (1/sqrt2 - tau)^2, tau = (2 + rho^2)/(2 sqrt6 rho): use tau_hi, then h_lo
tau_hi = (2 + rho2) / (2 * sqrt_lo(Fr(6)) * sqrt_lo(rho2))
h_lo = 1 / sqrt_hi(Fr(2)) - tau_hi
target = 6 * h_lo * h_lo
def phi_upper(c, pieces=400):   # rigorous upper bound on max F^2 over sum (u-r)^2 <= c, i.e. D^2 <= 2c
    D2 = 2 * c; emax = D2 / 12; worst = Fr(0)
    for k in range(pieces):
        e0, e1 = emax * k / pieces, emax * (k + 1) / pieces
        a_hi = sqrt_hi(D2 - 12 * e0) if D2 - 12 * e0 > 0 else Fr(0)
        s = a_hi + e1; assert s < 6
        den = sqrt_lo(6 - s) + sqrt_lo(Fr(6))
        worst = max(worst, e1 + s * s / (den * den))
    return worst
lo, hi = Fr(0), Fr(1, 2)
for _ in range(30):
    mid = (lo + hi) / 2
    if phi_upper(mid, 60) < target: lo = mid
    else: hi = mid
c_star = Fr(int(lo * 10000), 10000)
val = phi_upper(c_star, 2000)
print('6h^2 >= %.8f; c* = %s = %.4f: rigorous max F^2 <= %.8f  -> %s' % (float(target), c_star, float(c_star), float(val), 'PASS' if val < target else 'FAIL'))
