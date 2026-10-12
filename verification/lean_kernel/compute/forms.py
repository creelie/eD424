"""Exact arithmetic for the polyhedron R in Sym_{d+1} (two lattice translates in R^d).

J is a (d+1)x(d+1) symmetric matrix of Fractions; the last row and column belong to b.
"""
from fractions import Fraction as F
import math, itertools

def frac_mat(rows):
    return [[F(x) for x in row] for row in rows]

def dim_of(J):
    return len(J) - 1

def qform(J, k):
    n = len(J)
    s = F(0)
    for i in range(n):
        ki = k[i]
        if ki == 0:
            continue
        row = J[i]
        t = row[i] * ki
        for j in range(i + 1, n):
            if k[j]:
                t += 2 * row[j] * k[j]
        s += t * ki
    return s

def sym_coords(k):
    """Linear form X -> X[k] in the coordinates X_ij (i <= j)."""
    n = len(k)
    out = []
    for i in range(n):
        for j in range(i, n):
            out.append(k[i] * k[j] * (1 if i == j else 2))
    return out

def sym_from_vec(v, n):
    X = [[F(0)] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i, n):
            X[i][j] = X[j][i] = F(v[idx])
            idx += 1
    return X

def vec_from_sym(X):
    n = len(X)
    return [X[i][j] for i in range(n) for j in range(i, n)]

def ldl(Q):
    """Q = L D L^T with unit lower L. Returns (L, D) or None if not positive definite."""
    n = len(Q)
    L = [[F(0)] * n for _ in range(n)]
    D = [F(0)] * n
    for j in range(n):
        s = Q[j][j] - sum(L[j][k] ** 2 * D[k] for k in range(j))
        if s <= 0:
            return None
        D[j] = s
        L[j][j] = F(1)
        for i in range(j + 1, n):
            L[i][j] = (Q[i][j] - sum(L[i][k] * L[j][k] * D[k] for k in range(j))) / s
    return L, D

def det(M):
    n = len(M)
    A = [row[:] for row in M]
    d = F(1)
    for c in range(n):
        p = None
        for r in range(c, n):
            if A[r][c] != 0:
                p = r
                break
        if p is None:
            return F(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            d = -d
        d *= A[c][c]
        inv = 1 / A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] * inv
            if f:
                for k in range(c, n):
                    A[r][k] -= f * A[c][k]
    return d

def solve(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        inv = 1 / M[c][c]
        M[c] = [x * inv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]

def blocks(J):
    d = dim_of(J)
    Q = [row[:d] for row in J[:d]]
    r = [J[i][d] for i in range(d)]
    s = J[d][d]
    return Q, r, s

def qvec_sigma(J):
    Q, r, s = blocks(J)
    q = solve(Q, r)
    sigma = s - sum(ri * qi for ri, qi in zip(r, q))
    return q, sigma

def _isqrt_floor_frac(x):
    """largest integer m with m*m <= x (x >= 0 Fraction)."""
    if x < 0:
        return -1
    m = math.isqrt(x.numerator // x.denominator)
    while (m + 1) * (m + 1) <= x:
        m += 1
    while m * m > x:
        m -= 1
    return m

def short_vectors(Q, c, B):
    """All x in Z^d with Q[x + c] <= B (Q positive definite, exact)."""
    d = len(Q)
    fac = ldl(Q)
    assert fac is not None, "Q not positive definite"
    L, D = fac
    # Q[y] = sum_j D[j] (y_j + sum_{i>j} L[i][j] y_i)^2
    out = []
    x = [0] * d

    def rec(j, rem):
        # coordinates j+1..d-1 fixed; y_i = x_i + c_i
        t = c[j] + sum(L[i][j] * (x[i] + c[i]) for i in range(j + 1, d))
        # need D[j] (x_j + t)^2 <= rem
        bound = rem / D[j]
        # x_j + t in [-sqrt(bound), sqrt(bound)]
        lo_f = -t - F(math.isqrt(int(bound) + 1) + 2)
        lo = math.floor(lo_f)
        hi = math.ceil(-t + math.isqrt(int(bound) + 1) + 2)
        for v in range(lo, hi + 1):
            w = (v + t) ** 2 * D[j]
            if w <= rem:
                x[j] = v
                if j == 0:
                    out.append(tuple(x))
                else:
                    rec(j - 1, rem - w)
        x[j] = 0

    rec(d - 1, F(B))
    return out

def values_upto(J, B):
    """All k in M* (one of each pair +-k) with J[k] <= B, as dict k -> value. Requires Q pos. def."""
    d = dim_of(J)
    Q, r, s = blocks(J)
    q, sigma = qvec_sigma(J)
    res = {}
    for x in short_vectors(Q, [F(0)] * d, B):
        if any(x):
            # one of +-x: first nonzero coordinate positive
            nz = next(v for v in x if v != 0)
            if nz > 0:
                k = tuple(x) + (0,)
                res[k] = qform(J, k)
    if B - sigma >= 0:
        for x in short_vectors(Q, q, B - sigma):
            k = tuple(x) + (1,)
            res[k] = qform(J, k)
    return res

def minimum(J):
    """min over M* of J[k], and the set of minimal vectors. Q must be pos. def."""
    B = F(4)
    while True:
        vals = values_upto(J, B)
        if vals:
            m = min(vals.values())
            return m, sorted(k for k, v in vals.items() if v == m)
        B *= 2

def in_R(J):
    """True if J[k] >= 4 for all k in M*."""
    Q, r, s = blocks(J)
    if ldl(Q) is None:
        return False
    vals = values_upto(J, F(4))
    return all(v >= 4 for v in vals.values())

def min_vectors(J):
    vals = values_upto(J, F(4))
    assert all(v >= 4 for v in vals.values())
    return sorted(k for k, v in vals.items() if v == 4)

def rank(rows):
    A = [[F(x) for x in r] for r in rows]
    if not A:
        return 0
    m = len(A[0])
    rk = 0
    for c in range(m):
        p = None
        for i in range(rk, len(A)):
            if A[i][c] != 0:
                p = i
                break
        if p is None:
            continue
        A[rk], A[p] = A[p], A[rk]
        inv = 1 / A[rk][c]
        for i in range(len(A)):
            if i != rk and A[i][c] != 0:
                f = A[i][c] * inv
                A[i] = [a - f * b for a, b in zip(A[i], A[rk])]
        rk += 1
    return rk
