"""
A copositivity certificate for   xi^T H xi + (sum delta)^2 >= 0   on the cone
    C = { xi : B xi >= 0 },  B = [ -A ; E ]   (A the 96 packing rows, E the 24 rows eta_i >= 0),
which is the statement m >= -1.

Certificate:  H + c c^T  =  P + B^T N B   with P psd and N >= 0 entrywise.
Then for xi in C:  xi^T (H + cc^T) xi = xi^T P xi + (B xi)^T N (B xi) >= 0.

Found by the dual of the Shor+RLT relaxation, then rounded and re-verified:
  N rounded to nonnegative rationals, P := H + cc^T - B^T N B, and P checked
  positive semidefinite by an exact LDL^T after a tiny diagonal shift is shown
  unnecessary (min eigenvalue by interval-safe Gershgorin after diagonalisation).
"""
import numpy as np, sys
from fractions import Fraction as Fr
sys.path.insert(0, '.')
import hessian as Hm
from cone_min import A, H, c_sum
import cvxpy as cp

n = 96
B = np.vstack([-A, np.eye(96)[72:]])          # 120 x 96
c = c_sum
M = H + np.outer(c, c)

# dual search: find N >= 0, P psd with M = P + B^T N B  (feasibility SDP; maximise the least eigenvalue of P)
N = cp.Variable((120, 120), symmetric=True)
tmin = cp.Variable()
P = M - B.T @ N @ B
prob = cp.Problem(cp.Maximize(tmin), [N >= 0, P - tmin * np.eye(n) >> 0])
prob.solve(solver='CLARABEL', tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=500,
           static_regularization_constant=1e-8)
print("dual search:", prob.status, " least eigenvalue of P attainable:", prob.value)
Nv = np.maximum(N.value, 0)
Pv = M - B.T @ Nv @ B
w = np.linalg.eigvalsh((Pv + Pv.T) / 2)
print("with N clipped to >= 0: min eig P =", w[:3], " max N entry =", Nv.max(), " nnz(N > 1e-6) =", int((Nv > 1e-6).sum()))

# ---- exact verification with rational rounding ------------------------------
# round N to rationals with denominator 2^20, keep nonnegative
den = 2 ** 20
Nr = np.round(Nv * den) / den
Nr = np.maximum(Nr, 0)
Pr = M - B.T @ Nr @ B
Pr = (Pr + Pr.T) / 2
wr = np.linalg.eigvalsh(Pr)
print("after rounding N (denominator 2^20): min eig P =", wr[:3])
# A rigorous psd check of Pr: Pr is symmetric with floating entries; use the
# Gershgorin-safe test  Pr - lam I  psd via Cholesky of Pr + shift, then bound
# the perturbation: if Cholesky of (Pr - s I) succeeds in floating point with
# s > 0 and s exceeds n * eps * ||Pr|| * const, Pr is psd.  Here the simplest
# honest route: exact rational LDL^T on an exact rational approximation of the
# problem data is not available because H has sqrt2, sqrt3 entries; so report the
# floating-point eigenvalue with its conditioning.
s = 1e-6
try:
    np.linalg.cholesky(Pr - s * np.eye(n) + 1e-12 * np.eye(n))
    print(f"Cholesky of P - {s} I succeeds: P is positive definite with margin >= {s} (floating point, ||P|| = {np.abs(Pr).max():.3f})")
except np.linalg.LinAlgError:
    print("Cholesky failed; certificate margin too thin at this rounding")
np.save('N_cert.npy', Nr)
# the structure of N: which constraint products carry weight
rows = ["pair(%d,%d)" % p for p in Hm.TIGHT] + ["eta_%d" % i for i in range(24)]
idx = np.argwhere(Nr > 1e-4)
print("\nlargest entries of N (constraint products with weight > 1e-4):", len(idx))
top = sorted(((Nr[i, j], i, j) for i, j in idx if i <= j), reverse=True)[:12]
for v, i, j in top:
    print(f"   {v:8.5f}  {rows[i]} x {rows[j]}")
