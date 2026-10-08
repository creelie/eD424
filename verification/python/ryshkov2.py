#!/usr/bin/env python3
"""
ryshkov2.py -- exact enumeration of the Ryshkov-like polyhedron R(4) of
two-periodic point sets in dimension four, and of the rank-four points on its
edges.

A two-periodic set Lambda u (Lambda + b) in R^4 is encoded by the symmetric
5 x 5 matrix J = [[Q, r], [r^T, s]], Q_ij = a_i.a_j, r_i = a_i.b, s = b.b, for a
basis a_1..a_4 of Lambda.  For k = (n, l) in M = Z^4 x {-1, 0, 1},
J[k] = |n_1 a_1 + ... + n_4 a_4 + l b|^2.  The polyhedron is

    R(4) = { J : J[k] >= 4 for all k in M, k != 0 },

and a point of R(4) is a packing of unit balls exactly when rank J = 4.  All
arithmetic here is exact (fractions.Fraction, and cddlib in GMP rationals for
the extreme rays); no floating-point number decides anything.
"""
import itertools
import json
import sys
from fractions import Fraction as F
from math import isqrt, floor, ceil, gcd

import cdd
import cdd.gmp

N = 5                     # size of J
D = 4                     # dimension
IDX = [(i, i) for i in range(N)] + [(i, j) for i in range(N) for j in range(i + 1, N)]


# ---------------------------------------------------------------- linear algebra
def mat(rows):
    return [[F(x) for x in row] for row in rows]


def qf(J, k):
    return sum(J[i][j] * k[i] * k[j] for i in range(len(k)) for j in range(len(k)))


def bil(J, a, b):
    return sum(J[i][j] * a[i] * b[j] for i in range(len(a)) for j in range(len(b)))


def det(A):
    A = [row[:] for row in A]
    n = len(A)
    d = F(1)
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None:
            return F(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            d = -d
        d *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            if f:
                for cc in range(c, n):
                    A[r][cc] -= f * A[c][cc]
    return d


def solve(A, b):
    n = len(A)
    M_ = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M_[r][c] != 0)
        M_[c], M_[p] = M_[p], M_[c]
        for r in range(n):
            if r != c and M_[r][c] != 0:
                f = M_[r][c] / M_[c][c]
                for cc in range(c, n + 1):
                    M_[r][cc] -= f * M_[c][cc]
    return [M_[i][n] / M_[i][i] for i in range(n)]


def ldl(Q):
    """Q[x] = sum_i d_i (x_i + sum_{j>i} m_ij x_j)^2, exact; None if Q is not positive definite."""
    n = len(Q)
    A = [row[:] for row in Q]
    d = [F(0)] * n
    m = [[F(0)] * n for _ in range(n)]
    for i in range(n):
        if A[i][i] <= 0:
            return None
        d[i] = A[i][i]
        for j in range(i + 1, n):
            m[i][j] = A[i][j] / A[i][i]
        for j in range(i + 1, n):
            for k in range(i + 1, n):
                A[j][k] -= A[i][j] * A[i][k] / A[i][i]
    return d, m


def sqrt_upper(r):
    """a rational number >= sqrt(r), for r >= 0, within 2^-30 of it."""
    K = 1 << 30
    return F(isqrt(floor(r * K * K)) + 1, K)


def inhom(Q, q, B):
    """all integer n with Q[n + q] <= B, exactly (Fincke-Pohst with rational bounds)."""
    if B < 0:
        return []
    dm = ldl(Q)
    assert dm is not None, 'Q not positive definite'
    d, m = dm
    n_ = len(Q)
    out = []
    x = [F(0)] * n_
    nn = [0] * n_

    def rec(i, rem):
        if i < 0:
            out.append(tuple(nn))
            return
        y = q[i] + sum(m[i][j] * x[j] for j in range(i + 1, n_))
        ub = sqrt_upper(rem / d[i])
        for v in range(ceil(-y - ub), floor(-y + ub) + 1):
            z = v + y
            val = d[i] * z * z
            if val <= rem:
                nn[i] = v
                x[i] = v + q[i]
                rec(i - 1, rem - val)
        nn[i] = 0
        x[i] = F(0)

    rec(n_ - 1, F(B))
    return out


def blocks(J):
    Q = [row[:D] for row in J[:D]]
    r = [J[i][D] for i in range(D)]
    s = J[D][D]
    return Q, r, s


def vectors_upto(J, B):
    """the vectors k = (n, l) of M, one of each pair +-k, with 0 < J[k] <= B; Q must be positive definite."""
    Q, r, s = blocks(J)
    out = []
    for n in inhom(Q, [F(0)] * D, B):
        if any(n):
            if next(v for v in n if v) > 0:
                out.append(n + (0,))
    q = solve(Q, r)               # J[n, 1] = Q[n + q] + s - Q[q]
    c = s - qf(Q, q)
    if B - c >= 0:
        for n in inhom(Q, q, B - c):
            out.append(n + (1,))
    return out


def minimum(J):
    """the minimum of J over M \\ {0} and the vectors attaining it (one of each +-pair)."""
    Q, r, s = blocks(J)
    B = min(min(Q[i][i] for i in range(D)), s)
    vs = vectors_upto(J, B)
    mval = min(qf(J, k) for k in vs)
    return mval, [k for k in vs if qf(J, k) == mval]


# ---------------------------------------------------------------- polyhedral part
def coords(k):
    """the linear functional X -> X[k] in the 15 coordinates of IDX."""
    return [F(k[i] * k[j]) * (1 if i == j else 2) for (i, j) in IDX]


def from_coords(x):
    X = [[F(0)] * N for _ in range(N)]
    for v, (i, j) in zip(x, IDX):
        X[i][j] = X[j][i] = F(v)
    return X


def rank_of(vecs):
    rows = [coords(k) for k in vecs]
    rows = [r[:] for r in rows]
    rank, col = 0, 0
    ncol = len(IDX)
    while rank < len(rows) and col < ncol:
        p = next((i for i in range(rank, len(rows)) if rows[i][col] != 0), None)
        if p is None:
            col += 1
            continue
        rows[rank], rows[p] = rows[p], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][col] != 0:
                f = rows[i][col] / rows[rank][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[rank])]
        rank += 1
        col += 1
    return rank


def extreme_rays(active):
    rows = [[0] + coords(k) for k in active]
    m = cdd.gmp.matrix_from_array(rows, rep_type=cdd.RepType.INEQUALITY)
    p = cdd.gmp.polyhedron_from_matrix(m)
    g = cdd.gmp.copy_generators(p)
    assert not g.lin_set, 'cone not pointed'
    rays = []
    for row in g.array:
        assert row[0] == 0
        x = [F(v) for v in row[1:]]
        den = 1
        for v in x:
            den = den * v.denominator // __import__('math').gcd(den, v.denominator)
        x = [v * den for v in x]
        g_ = 0
        for v in x:
            g_ = __import__('math').gcd(g_, int(v))
        rays.append(from_coords([v / g_ for v in x]))
    return rays


def add(J, X, t):
    return [[J[i][j] + t * X[i][j] for j in range(N)] for i in range(N)]


def is_pd(Q):
    return ldl(Q) is not None


def psd_split(A):
    """'pd', 'psd' (singular) or 'indef' for a rational symmetric matrix, exactly."""
    A = [row[:] for row in A]
    n = len(A)
    alive = list(range(n))
    while alive:
        p = next((i for i in alive if A[i][i] > 0), None)
        if p is None:
            if any(A[i][i] < 0 for i in alive):
                return 'indef'
            if any(A[i][j] != 0 for i in alive for j in alive):
                return 'indef'
            return 'psd'
        alive.remove(p)
        for i in alive:
            f = A[i][p] / A[p][p]
            for j in alive:
                A[i][j] -= f * A[p][j]
    return 'pd'


def column_hnf_unimodular(B):
    """unimodular V (integer, det +-1) with B V = [H | 0], H of full column rank."""
    n = len(B[0])
    B = [row[:] for row in B]
    V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def colop(i, j, a, b, c, d):          # (col_i, col_j) <- (a col_i + b col_j, c col_i + d col_j)
        for M_ in (B, V):
            for row in M_:
                x, y = row[i], row[j]
                row[i], row[j] = a * x + b * y, c * x + d * y

    piv = 0
    for r in range(len(B)):
        if piv == n:
            break
        for j in range(piv + 1, n):
            x, y = B[r][piv], B[r][j]
            if y == 0:
                continue
            # extended Euclid: g = u x + v y
            g, u, v = egcd(x, y)
            colop(piv, j, u, v, -y // g, x // g)
        if B[r][piv] != 0:
            piv += 1
    return V, piv


def egcd(a, b):
    if b == 0:
        return (a, 1, 0) if a >= 0 else (-a, -1, 0)
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y


def to_int_matrix(A):
    den = 1
    for row in A:
        for x in row:
            den = den * x.denominator // gcd(den, x.denominator)
    return [[int(x * den) for x in row] for row in A]


def transform(X, V):
    """T^T X T for T = diag(V, 1)."""
    T = [[F(V[i][j]) if i < D and j < D else F(1 if i == j else 0) for j in range(N)] for i in range(N)]
    return [[sum(T[a][i] * X[a][b] * T[b][j] for a in range(N) for b in range(N)) for j in range(N)] for i in range(N)], T


def violator_psd(X):
    """for X whose Q-block is positive semidefinite and singular: some k in M with X[k] < 0, or None if X >= 0 on M."""
    QX = [row[:D] for row in X[:D]]
    V, rk = column_hnf_unimodular(to_int_matrix(QX))
    Y, T = transform(X, V)
    assert all(Y[i][j] == 0 for i in range(D) for j in range(D) if i >= rk or j >= rk)
    r = [Y[i][D] for i in range(D)]
    sX = Y[D][D]

    def back(n):
        k = tuple(int(x) for x in matvec(T, list(n)))
        assert qf(X, k) < 0
        return k

    for j in range(rk, D):
        if r[j] != 0:
            c = -(abs(sX) // (2 * abs(r[j])) + 1) * (1 if r[j] > 0 else -1)
            n = [0] * N
            n[j] = c
            n[D] = 1
            return back(n)
    if rk == 0:
        return back([0] * D + [1]) if sX < 0 else None
    A = [row[:rk] for row in Y[:rk]]
    q = solve(A, r[:rk])
    c = sX - qf(A, q)
    if c >= 0:
        return None
    for n1 in inhom(A, q, -c):
        n = list(n1) + [0] * (D - rk) + [1]
        if qf(Y, n) < 0:
            return back(n)
    return None


def crossing(J, X, k):
    return (qf(J, k) - 4) / (-qf(X, k))


def descend(J, X, t):
    """given t > t* with Q + tQX positive definite, return t* exactly."""
    while True:
        mval, ks = minimum(add(J, X, t))
        if mval >= 4:
            assert mval == 4
            return t
        t = min(crossing(J, X, k) for k in ks)


def qblock(J, X, t):
    return [[J[i][j] + t * X[i][j] for j in range(D)] for i in range(D)]


def walk(J, X, log=None):
    """t* = max{t >= 0 : J + tX in R(4)}, exactly, or None if the ray stays in R(4).

    The ray stays in R(4) iff X[k] >= 0 for all k in M.  This needs the
    Q-block QX of X to be positive semidefinite; if QX is positive definite
    the minimum of X on M decides, and if QX is singular violator_psd does.
    If QX is indefinite the ray is bounded, and since every point of R(4)
    has a positive definite Q-block, t* lies below the first t at which
    Q + tQX stops being positive definite."""
    QX = [row[:D] for row in X[:D]]
    kind = psd_split(QX)
    if kind == 'pd':
        mval, ks = minimum(X)
        if mval >= 0:
            return None
        return descend(J, X, crossing(J, X, ks[0]))
    if kind == 'psd':
        k = violator_psd(X)
        if k is None:
            return None
        return descend(J, X, crossing(J, X, k))
    lo, hi = F(0), F(1)
    while psd_split(qblock(J, X, hi)) == 'pd':
        if minimum(add(J, X, hi))[0] < 4:
            return descend(J, X, hi)
        lo, hi = hi, 2 * hi
    while True:
        mid = (lo + hi) / 2
        if psd_split(qblock(J, X, mid)) == 'pd':
            if minimum(add(J, X, mid))[0] < 4:
                return descend(J, X, mid)
            lo = mid
        else:
            hi = mid


# ---------------------------------------------------------------- equivalence
def profile(J, act, k):
    """the multiset of squared inner products of k with the vectors of act, and the type of k."""
    return (k[D] == 0, tuple(sorted(bil(J, k, c) ** 2 for c in act)))


def invariants(J, act):
    Q, r, s = blocks(J)
    n0 = sum(1 for k in act if k[D] == 0)
    return (det(Q), det(J), len(act), n0, tuple(sorted(profile(J, act, k) for k in act)))


def gram_pair(J, a, b):
    return bil(J, a, b)


def find_iso(J1, act1, J2, act2):
    """T in Gamma with J2 = T^T J1 T (J2[k] = J1[Tk]), or None; act* are the perfect sets J*^{-1}(4)."""
    if invariants(J1, act1) != invariants(J2, act2):
        return None
    full1 = act1 + [tuple(-v for v in k) for k in act1]
    prof1 = {c: profile(J1, act1, c) for c in full1}
    # a basis of R^5 inside act2
    basis = []
    for k in act2:
        if rank5(basis + [k]) > len(basis):
            basis.append(k)
        if len(basis) == N:
            break
    G2 = [[bil(J2, a, b) for b in basis] for a in basis]
    pb = [profile(J2, act2, b) for b in basis]
    imgs = []

    def rec(i):
        if i == N:
            T = linmap(basis, imgs)
            if T is None:
                return None
            if not gamma_ok(T):
                return None
            # check J2 = T^T J1 T
            for a in range(N):
                for b in range(N):
                    ea = [1 if x == a else 0 for x in range(N)]
                    eb = [1 if x == b else 0 for x in range(N)]
                    if bil(J1, matvec(T, ea), matvec(T, eb)) != J2[a][b]:
                        return None
            return T
        for c in full1:
            if prof1[c] != pb[i]:
                continue
            if bil(J1, c, c) != G2[i][i]:
                continue
            if all(bil(J1, c, imgs[j]) == G2[i][j] for j in range(i)):
                imgs.append(c)
                T = rec(i + 1)
                if T is not None:
                    return T
                imgs.pop()
        return None

    return rec(0)


def all_isos(J1, act1, J2, act2):
    """all T in Gamma with J2 = T^T J1 T (for J1 = J2: the stabiliser of J1)."""
    if invariants(J1, act1) != invariants(J2, act2):
        return []
    full1 = act1 + [tuple(-v for v in k) for k in act1]
    prof1 = {c: profile(J1, act1, c) for c in full1}
    basis = []
    for k in act2:
        if rank5(basis + [k]) > len(basis):
            basis.append(k)
        if len(basis) == N:
            break
    G2 = [[bil(J2, a, b) for b in basis] for a in basis]
    pb = [profile(J2, act2, b) for b in basis]
    imgs, out = [], []

    def rec(i):
        if i == N:
            T = linmap(basis, imgs)
            if T is None or not gamma_ok(T):
                return
            for a in range(N):
                for b in range(N):
                    ea = [1 if x == a else 0 for x in range(N)]
                    eb = [1 if x == b else 0 for x in range(N)]
                    if bil(J1, matvec(T, ea), matvec(T, eb)) != J2[a][b]:
                        return
            out.append(T)
            return
        for c in full1:
            if prof1[c] != pb[i] or bil(J1, c, c) != G2[i][i]:
                continue
            if all(bil(J1, c, imgs[j]) == G2[i][j] for j in range(i)):
                imgs.append(c)
                rec(i + 1)
                imgs.pop()

    rec(0)
    return out


def rank5(vecs):
    A = [[F(x) for x in v] for v in vecs]
    rank = 0
    for col in range(N):
        p = next((i for i in range(rank, len(A)) if A[i][col] != 0), None)
        if p is None:
            continue
        A[rank], A[p] = A[p], A[rank]
        for i in range(len(A)):
            if i != rank and A[i][col] != 0:
                f = A[i][col] / A[rank][col]
                A[i] = [a - f * b for a, b in zip(A[i], A[rank])]
        rank += 1
    return rank


def matvec(T, v):
    return [sum(T[i][j] * v[j] for j in range(N)) for i in range(N)]


def linmap(src, dst):
    """T with T src_i = dst_i (src a basis), as a matrix; None if not integral."""
    S = [[F(src[j][i]) for j in range(N)] for i in range(N)]     # columns src_j
    Dm = [[F(dst[j][i]) for j in range(N)] for i in range(N)]
    # T = Dm S^{-1}
    Sinv = inverse(S)
    T = [[sum(Dm[i][k] * Sinv[k][j] for k in range(N)) for j in range(N)] for i in range(N)]
    if any(x.denominator != 1 for row in T for x in row):
        return None
    return [[int(x) for x in row] for row in T]


def inverse(A):
    n = len(A)
    M_ = [A[i][:] + [F(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M_[r][c] != 0)
        M_[c], M_[p] = M_[p], M_[c]
        piv = M_[c][c]
        M_[c] = [x / piv for x in M_[c]]
        for r in range(n):
            if r != c and M_[r][c] != 0:
                f = M_[r][c]
                M_[r] = [a - f * b for a, b in zip(M_[r], M_[c])]
    return [row[n:] for row in M_]


def gamma_ok(T):
    if any(T[D][j] != 0 for j in range(D)) or T[D][D] not in (1, -1):
        return False
    U = [row[:D] for row in T[:D]]
    return abs(det([[F(x) for x in row] for row in U])) == 1


# ---------------------------------------------------------------- main enumeration
def fmt(J):
    return [[str(x) for x in row] for row in J]


def main():
    J0 = mat([[4, 0, 0, 0, 2], [0, 4, 0, 0, -2], [0, 0, 4, 0, -2], [0, 0, 0, 4, -2], [2, -2, -2, -2, 4]])
    verts = []
    queue = []

    def register(J):
        mval, act = minimum(J)
        assert mval == 4
        assert rank_of(act) == 15, 'not a vertex'
        for i, (K, actK) in enumerate(verts):
            if find_iso(K, actK, J, act) is not None:
                return i, False
        verts.append((J, act))
        queue.append(len(verts) - 1)
        return len(verts) - 1, True

    register(J0)
    edges = []
    while queue:
        i = queue.pop(0)
        J, act = verts[i]
        rays = extreme_rays(act)
        print('vertex %d: %d active pairs, det Q = %s, det J = %s, %d extreme rays'
              % (i, len(act), det(blocks(J)[0]), det(J), len(rays)), flush=True)
        for X in rays:
            t = walk(J, X)
            if t is None:
                edges.append({'from': i, 'X': fmt(X), 't': None, 'to': None})
                continue
            J2 = add(J, X, t)
            j, new = register(J2)
            edges.append({'from': i, 'X': fmt(X), 't': str(t), 'to': j})
            if new:
                print('   new vertex %d along a ray (t = %s)' % (j, t), flush=True)
    out = {'vertices': [{'J': fmt(J), 'active': [list(k) for k in act]} for J, act in verts], 'edges': edges}
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'ryshkov2_d4.json', 'w'), indent=1)
    print('%d vertex classes, %d edges at the representatives' % (len(verts), len(edges)))


if __name__ == '__main__':
    main()
