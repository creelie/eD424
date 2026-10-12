"""Decide whether a ray J + tY (t >= 0) of a vertex cone stays in R, i.e. whether Y[k] >= 0 for all k in M*.

If not, return a witness k in M* with Y[k] < 0 (Andreanov-Kallus, Sec. 3.5, made exact)."""
from fractions import Fraction as F
import itertools
import numpy as np
from forms import qform, blocks, det, solve, short_vectors

def principal_minors_nonneg(Q):
    n = len(Q)
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            if det([[Q[i][j] for j in S] for i in S]) < 0:
                return False
    return True

def negative_direction(Q):
    """integer x with Q[x] < 0 (Q symmetric, not PSD)."""
    A = np.array([[float(x) for x in row] for row in Q])
    w, V = np.linalg.eigh(A)
    v = V[:, 0]
    m = 1.0
    while True:
        x = [int(round(m * c)) for c in v]
        if any(x) and qform(Q, x) < 0:
            return x
        m *= 2
        if m > 2 ** 40:
            raise RuntimeError("no negative integer direction found")

def column_echelon(A):
    """A: list of rows (integers), m x n. Returns (H, U) with A U = H, U unimodular n x n,
    H in column echelon form; the columns of U beyond rank(A) span ker(A) cap Z^n."""
    m, n = len(A), len(A[0])
    H = [row[:] for row in A]
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def colop(j, k, c):  # col_j -= c * col_k
        for row in H:
            row[j] -= c * row[k]
        for row in U:
            row[j] -= c * row[k]

    def swap(j, k):
        for row in H:
            row[j], row[k] = row[k], row[j]
        for row in U:
            row[j], row[k] = row[k], row[j]

    piv = 0
    for i in range(m):
        if piv == n:
            break
        while True:
            nz = [j for j in range(piv, n) if H[i][j] != 0]
            if not nz:
                break
            jmin = min(nz, key=lambda j: abs(H[i][j]))
            swap(piv, jmin)
            done = True
            for j in range(piv + 1, n):
                if H[i][j] != 0:
                    colop(j, piv, H[i][j] // H[i][piv])
                    if H[i][j] != 0:
                        done = False
            if done:
                break
        if any(H[i][j] != 0 for j in range(piv, n)):
            piv += 1
    return H, U, piv

def classify(Y):
    """('bounded', k) with Y[k] < 0, or ('unbounded', None)."""
    n = len(Y)
    d = n - 1
    QY, rY, sY = blocks(Y)
    if not principal_minors_nonneg(QY):
        x = negative_direction(QY)
        return 'bounded', tuple(x) + (0,)
    # QY is PSD; integer kernel basis via unimodular column reduction of the scaled matrix
    den = 1
    from math import lcm
    for row in QY:
        for x in row:
            den = lcm(den, x.denominator)
    A = [[int(x * den) for x in row] for row in QY]
    H, U, rk = column_echelon(A)
    Ucols = [[U[i][j] for i in range(d)] for j in range(d)]
    kernel = Ucols[rk:]
    for u in kernel:
        c = sum(rY[i] * u[i] for i in range(d))
        if c != 0:
            # Y[(m u, 1)] = sY + 2 m c, linear in m
            m = 1
            while sY + 2 * m * c >= 0 and sY - 2 * m * c >= 0:
                m *= 2
            mm = m if sY + 2 * m * c < 0 else -m
            k = tuple(mm * ui for ui in u) + (1,)
            assert qform(Y, k) < 0
            return 'bounded', k
    # rY orthogonal to the kernel: Y[(U x, 1)] = P[x1] + 2 r1.x1 + sY with P positive definite
    if rk == 0:
        return ('unbounded', None) if sY >= 0 else ('bounded', tuple([0] * d) + (1,))
    UT = [[F(U[i][j]) for i in range(d)] for j in range(d)]  # rows = columns of U
    Qp = [[sum(UT[a][i] * QY[i][j] * UT[b][j] for i in range(d) for j in range(d)) for b in range(rk)] for a in range(rk)]
    r1 = [sum(UT[a][i] * rY[i] for i in range(d)) for a in range(rk)]
    q1 = solve(Qp, r1)
    sig = sY - sum(a * b for a, b in zip(r1, q1))
    if sig >= 0:
        return 'unbounded', None
    # minimum of P[x1 + q1] over integer x1: search a growing radius
    B = F(1)
    while True:
        xs = short_vectors(Qp, q1, B)
        if xs:
            best = min(xs, key=lambda x: qform(Qp, [x[i] + q1[i] for i in range(rk)]))
            val = qform(Qp, [best[i] + q1[i] for i in range(rk)]) + sig
            if val < 0:
                x = [sum(U[i][a] * best[a] for a in range(rk)) for i in range(d)]
                k = tuple(x) + (1,)
                assert qform(Y, k) < 0
                return 'bounded', k
            return 'unbounded', None
        B *= 2
