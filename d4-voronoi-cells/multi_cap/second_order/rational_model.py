"""
Exact rational model of the second-order problem at the root system.

Scaled coordinates: roots a_i in {(+-1,+-1,0,0) and permutations}, |a_i|^2 = 2,
u_i = a_i/sqrt2.  The cell Q = {<x,u_i> <= 1}; x' = sqrt2 x has integer vertices
(+-2,0,0,0)-type and (+-1,+-1,+-1,+-1).  Tangent vectors tau_i = t_i/sqrt2 with
t_i = sum_k c_{ik} b_{ik} in a rational basis b_{i0}, b_{i1}, b_{i2} of a_i^perp
(b_{i0} of norm^2 2, the others of norm^2 1, mutually orthogonal).  Then

    <x, tau_i>   = <x', t_i>/2                      (rational)
    |tau_i|^2    = (2 c_{i0}^2 + c_{i1}^2 + c_{i2}^2)/2
    (Lambda tau)_ij = (<a_i,t_j> + <a_j,t_i>)/2

and the Hessian form
    H(xi,xi) = (4/3) sum_i |tau_i|^2 + sum_{T} (1/3) sum_{midpoints m} [2 v_i v_j - (v_i^2+v_j^2)/2],
    v_i(m) = eta_i - <m', t_i>/2,   m' = (x'_a + x'_b)/2,
is a rational quadratic form in xi = (c, eta) in Q^96.
"""
import itertools
from fractions import Fraction as Fr
import numpy as np

# ---------------------------------------------------------------- data
ROOTS = []
for p, q in itertools.combinations(range(4), 2):
    for sp in (1, -1):
        for sq in (1, -1):
            v = [0, 0, 0, 0]; v[p], v[q] = sp, sq
            ROOTS.append(tuple(v))
ROOTS = [tuple(Fr(x) for x in r) for r in ROOTS]          # 24, |a|^2 = 2

VERTS = []
for k in range(4):
    for s in (1, -1):
        v = [0, 0, 0, 0]; v[k] = 2 * s; VERTS.append(tuple(Fr(x) for x in v))
for signs in itertools.product((1, -1), repeat=4):
    VERTS.append(tuple(Fr(x) for x in signs))               # 24 scaled vertices x'


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def basis(a):
    """rational orthogonal basis of a^perp: (s_p e_p - s_q e_q), e_r, e_s"""
    supp = [k for k in range(4) if a[k] != 0]
    p, q = supp
    out = [0, 0, 0, 0]
    b0 = [Fr(0)] * 4; b0[p] = a[p]; b0[q] = -a[q]
    others = [k for k in range(4) if k not in supp]
    b1 = [Fr(0)] * 4; b1[others[0]] = Fr(1)
    b2 = [Fr(0)] * 4; b2[others[1]] = Fr(1)
    for b in (b0, b1, b2):
        assert dot(b, a) == 0
    return [tuple(b0), tuple(b1), tuple(b2)]


BAS = [basis(a) for a in ROOTS]
NORM2 = [[dot(b, b) for b in B] for B in BAS]              # [2,1,1]

# tight pairs and their triangles (scaled: <x', a_i> = 2 on facet i)
TIGHT = [(i, j) for i in range(24) for j in range(i + 1, 24) if dot(ROOTS[i], ROOTS[j]) == 1]
assert len(TIGHT) == 96
ONF = [[k for k in range(24) if dot(VERTS[k], ROOTS[i]) == 2] for i in range(24)]
assert all(len(o) == 6 for o in ONF)
TRI = {}
for i, j in TIGHT:
    ks = [k for k in ONF[i] if k in ONF[j]]
    assert len(ks) == 3
    TRI[(i, j)] = ks

N = 96


def idx_c(i, k):
    return 3 * i + k


def idx_eta(i):
    return 72 + i


def row_v(i, mprime):
    """coefficient row (length 96) of v_i(m) = eta_i - <m', t_i>/2 with t_i = sum_k c_ik b_ik"""
    r = [Fr(0)] * N
    r[idx_eta(i)] = Fr(1)
    for k in range(3):
        r[idx_c(i, k)] = -dot(mprime, BAS[i][k]) / 2
    return r


def outer_add(Hm, r, s, coef):
    for p, rp in enumerate(r):
        if rp == 0:
            continue
        for q, sq in enumerate(s):
            if sq == 0:
                continue
            Hm[p][q] += coef * rp * sq


def build_H():
    """H as a symmetric 96x96 matrix of Fractions with xi^T H xi = H(xi,xi)"""
    Hm = [[Fr(0)] * N for _ in range(N)]
    # (4/3) |tau_i|^2 = (4/3) (2c0^2 + c1^2 + c2^2)/2 = (2/3)(2 c0^2 + c1^2 + c2^2)
    for i in range(24):
        for k in range(3):
            Hm[idx_c(i, k)][idx_c(i, k)] += Fr(2, 3) * NORM2[i][k]
    for (i, j), ks in TRI.items():
        P = [VERTS[k] for k in ks]
        mids = [tuple((P[a][d] + P[b][d]) / 2 for d in range(4)) for a, b in ((0, 1), (1, 2), (0, 2))]
        for m in mids:
            ri, rj = row_v(i, m), row_v(j, m)
            # (1/3) [2 v_i v_j - (v_i^2 + v_j^2)/2]  ->  symmetric matrix
            outer_add(Hm, ri, rj, Fr(1, 3))          # v_i v_j  (once each way = 2 v_i v_j total)
            outer_add(Hm, rj, ri, Fr(1, 3))
            outer_add(Hm, ri, ri, Fr(-1, 6))
            outer_add(Hm, rj, rj, Fr(-1, 6))
    return Hm


def build_B():
    """the 120 cone rows: 96 rows (Lambda tau)_ij - (eta_i+eta_j)/2 <= 0, then 24 rows -eta_i <= 0;
       returned as the matrix Bpos with Bpos xi >= 0 on the cone (i.e. negated first block)"""
    B = []
    for i, j in TIGHT:
        r = [Fr(0)] * N
        for k in range(3):
            r[idx_c(j, k)] += dot(ROOTS[i], BAS[j][k]) / 2     # <a_i, t_j>/2
            r[idx_c(i, k)] += dot(ROOTS[j], BAS[i][k]) / 2     # <a_j, t_i>/2
        r[idx_eta(i)] -= Fr(1, 2); r[idx_eta(j)] -= Fr(1, 2)
        B.append([-x for x in r])                             # -(...) >= 0
    for i in range(24):
        r = [Fr(0)] * N; r[idx_eta(i)] = Fr(1); B.append(r)
    return B


def cvec():
    c = [Fr(0)] * N
    for i in range(24):
        c[idx_eta(i)] = Fr(2)
    return c


def to_float(M):
    return np.array([[float(x) for x in row] for row in M])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    import hessian as Hf                                     # the floating model
    H = build_H(); B = build_B(); c = cvec()
    Hfl = to_float(H)
    print("rational H built; symmetric:", all(H[p][q] == H[q][p] for p in range(N) for q in range(p)))
    # compare with the floating model on random rational points
    rng = np.random.default_rng(5)
    Hfloat = np.load('H.npy')
    worst = 0
    for trial in range(5):
        c_ = rng.integers(-3, 4, size=(24, 3)); eta = rng.integers(0, 4, size=24)
        xi_r = np.r_[c_.reshape(-1), eta].astype(float)
        val_r = xi_r @ Hfl @ xi_r
        # physical tau_i = t_i / sqrt2, t_i = sum_k c_ik b_ik ; float coords = BASIS[i]^T tau_i
        xi_f = np.zeros(96)
        for i in range(24):
            t = sum(c_[i, k] * np.array([float(x) for x in BAS[i][k]]) for k in range(3))
            tau = t / np.sqrt(2)
            xi_f[3 * i:3 * i + 3] = Hf.BASIS[i].T @ tau
        xi_f[72:] = eta
        val_f = xi_f @ Hfloat @ xi_f
        worst = max(worst, abs(val_r - val_f))
        print(f"  trial {trial}: rational {val_r:.10f}   float {val_f:.10f}")
    print("max discrepancy:", worst)
    # one-centre ray
    xi = [Fr(0)] * N; xi[idx_eta(0)] = Fr(1, 2)
    v = sum(xi[p] * H[p][q] * xi[q] for p in range(N) for q in range(N))
    print("one-centre ray (delta = 1): H =", v)
