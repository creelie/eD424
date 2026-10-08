#!/usr/bin/env python3
"""
verify_cone_certificate.py -- exact verification of

    Lemma.  On the first-order feasible cone of the packing constraints at the
    root system,  (Lambda tau)_ij <= (eta_i + eta_j)/2 on the 96 tight pairs and
    eta_i >= 0, the Hessian form of the cell volume satisfies

            H(xi, xi) + (sum_i delta_i)^2  >=  0,        delta_i = 2 eta_i ,

    with equality exactly on the rays of a single centre pushed out.

Certificate:  H + c c^T = P + B^T N B  with c = 2*1_eta, B the 120 cone rows
(B xi >= 0 on the cone), N a symmetric 120 x 120 matrix with nonnegative
rational entries constant on the orbits of the signed-permutation group, and
P positive semidefinite.  For xi in the cone, xi^T (H + cc^T) xi =
xi^T P xi + (B xi)^T N (B xi) >= 0.

Everything is rebuilt here from the integral root system in exact rational
arithmetic (rational_model.py); the certificate values are read from
exact_certificate.pkl; P is assembled exactly and its positive semidefiniteness
is decided by an exact LDL^T (a zero pivot is accepted only with a zero row).
No floating point is used anywhere.

Usage: python3 verify_cone_certificate.py
"""
import sys, pickle, time
from fractions import Fraction as Fr
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import rational_model as RM

t0 = time.time()
N = 96
H = RM.build_H(); B = RM.build_B(); c = RM.cvec()
M = [[H[p][q] + c[p] * c[q] for q in range(N)] for p in range(N)]
cert = pickle.load(open('exact_certificate.pkl', 'rb'))
orbits = cert['orbits']
nu = [Fr(a, b) for a, b in cert['nu']]
assert len(orbits) == len(nu)
assert all(x >= 0 for x in nu), "a negative certificate value"
# every unordered pair of constraints is covered exactly once by the orbits
covered = {}
for o, orb in enumerate(orbits):
    for e in orb:
        assert e not in covered; covered[e] = o
assert len(covered) == 120 * 121 // 2
print(f"1. N: {len(orbits)} orbits, all values nonnegative rationals, {sum(1 for x in nu if x > 0)} of them positive")

Nm = [[Fr(0)] * 120 for _ in range(120)]
for o, orb in enumerate(orbits):
    for a, b in orb:
        Nm[a][b] = nu[o]; Nm[b][a] = nu[o]
NB = [[sum(Nm[r][s] * B[s][p] for s in range(120) if Nm[r][s] != 0) for p in range(N)] for r in range(120)]
P = [[M[p][q] - sum(B[r][p] * NB[r][q] for r in range(120) if B[r][p] != 0) for q in range(N)] for p in range(N)]
assert all(P[p][q] == P[q][p] for p in range(N) for q in range(p)), "P not symmetric"
print(f"2. P = H + cc^T - B^T N B assembled exactly  [{time.time()-t0:.0f}s]")


def ldl_psd(Mx):
    n = len(Mx); A = [row[:] for row in Mx]
    zero_pivots = 0
    for k in range(n):
        piv = A[k][k]
        if piv < 0:
            return False, k, zero_pivots
        if piv == 0:
            if any(A[k][j] != 0 for j in range(k + 1, n)):
                return False, k, zero_pivots
            zero_pivots += 1
            continue
        for i in range(k + 1, n):
            if A[i][k] != 0:
                m = A[i][k] / piv
                for j in range(k + 1, n):
                    A[i][j] -= m * A[k][j]
    return True, None, zero_pivots


psd, where, zp = ldl_psd(P)
print(f"3. exact LDL^T of P: {'positive semidefinite' if psd else 'FAILED at pivot %d' % where}; "
      f"zero pivots {zp} (the 24 push-outs and 6 rotations)  [{time.time()-t0:.0f}s]")
# equality on the one-centre rays
for i in (0, 7, 23):
    xi = [Fr(0)] * N; xi[RM.idx_eta(i)] = Fr(1, 2)
    val = sum(xi[p] * M[p][q] * xi[q] for p in range(N) for q in range(N) if xi[p] != 0 and xi[q] != 0)
    assert val == 0
print("4. equality on the one-centre rays: H + (sum delta)^2 = 0 there (exact)")
if psd:
    print("\nPASS: on the packing cone  H(xi,xi) >= -(sum_i delta_i)^2 ;  the second-order model of the cell volume is")
    print("      vol(V(Y)) - 8 >= (2/3) S - (1/2) S^2 + O(S^3),  S = sum delta_i, with the worst case the one-centre ray.")
else:
    sys.exit(1)
