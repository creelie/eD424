"""
An exact certificate for  m = -1:   xi^T H xi + (sum delta)^2 >= 0  on the cone.

Steps
  1. the signed-permutation group W(B4) (order 384) acts on the roots, on the
     coordinates xi (orthogonally) and on the 120 constraints (by permutation);
     H, c and the cone are invariant.
  2. average the numerical dual N over the group -> an invariant N with few
     distinct values; snap them to rationals.
  3. rebuild everything in the scaled rational model (roots (+-1,+-1,0,0),
     vertices of the cell (+-2,0,0,0) and (+-1,+-1,+-1,+-1), rational bases of
     the tangent spaces) so that H, B, c are exact rationals, and verify
        P = H + c c^T - B^T N B   is positive semidefinite
     by an exact LDL^T over the rationals (pivots >= 0, a zero pivot only with a
     zero row, which is the tight direction).
"""
import itertools, sys
import numpy as np
from fractions import Fraction as Fr
sys.path.insert(0, '.')
import hessian as Hm
from cone_min import A, H, c_sum

U, BASIS, TIGHT = Hm.U, Hm.BASIS, Hm.TIGHT
n = 96
B = np.vstack([-A, np.eye(96)[72:]])
M = H + np.outer(c_sum, c_sum)
Nnum = np.load('N_cert.npy')

# ---- 1. the group -----------------------------------------------------------
def signed_perms():
    for perm in itertools.permutations(range(4)):
        for signs in itertools.product((1, -1), repeat=4):
            g = np.zeros((4, 4))
            for r in range(4):
                g[r, perm[r]] = signs[r]
            yield g

def root_index(v):
    d = np.linalg.norm(U - v, axis=1)
    k = int(np.argmin(d)); assert d[k] < 1e-9; return k

pair_index = {p: r for r, p in enumerate(TIGHT)}
Pis, Qs = [], []
for g in signed_perms():
    sigma = [root_index(g @ U[i]) for i in range(24)]
    Pi = np.zeros((96, 96))
    for i in range(24):
        j = sigma[i]
        Pi[3 * j:3 * j + 3, 3 * i:3 * i + 3] = BASIS[j].T @ g @ BASIS[i]
        Pi[72 + j, 72 + i] = 1.0
    Q = np.zeros((120, 120))
    for r, (i, j) in enumerate(TIGHT):
        a, b = sorted((sigma[i], sigma[j]))
        Q[pair_index[(a, b)], r] = 1.0
    for i in range(24):
        Q[96 + sigma[i], 96 + i] = 1.0
    Pis.append(Pi); Qs.append(Q)
print(f"group elements: {len(Pis)}")
# invariance checks (Pi maps xi -> Pi xi; H invariant means Pi^T H Pi = H; B Pi = Q B)
err_H = max(np.abs(Pi.T @ H @ Pi - H).max() for Pi in Pis[:40])
err_B = max(np.abs(B @ Pi - Q @ B).max() for Pi, Q in zip(Pis[:40], Qs[:40]))
print(f"invariance: max|Pi^T H Pi - H| = {err_H:.2e},  max|B Pi - Q B| = {err_B:.2e}")

# ---- 2. average N over the group -------------------------------------------
Navg = sum(Q.T @ Nnum @ Q for Q in Qs) / len(Qs)
Navg = (Navg + Navg.T) / 2
vals = np.unique(np.round(Navg, 6))
print(f"distinct values in the averaged N (rounded to 1e-6): {len(vals)}")
print("   ", vals[:20], "...")
P = M - B.T @ Navg @ B
print("min eig of P with averaged N:", np.linalg.eigvalsh((P + P.T) / 2)[:4])
np.save('N_avg.npy', Navg)
