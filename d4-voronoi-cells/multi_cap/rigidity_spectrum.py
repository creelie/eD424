"""
rigidity_spectrum.py -- the exact spectrum of the rigidity operator of the 24-point
kissing configuration, in integer arithmetic.

Write T for the tangent space of (S^3)^24 at the normalised D_4 roots and

    (Lambda tau)_{ij} = u_i . tau_j + u_j . tau_i ,   (i,j) a tight pair,

the derivative of the 96 tight inner products.  What the quantitative
uniqueness theorem needs is a lower bound on the smallest nonzero singular
value of Lambda restricted to T.

To keep everything integral, work with Lambda' = sqrt2 * Lambda, whose
entries are alpha_i . x with alpha the integer roots, and with the rational
orthogonal projector P onto T, whose blocks are I - alpha alpha^T / 2.  Then

    N = P Lambda'^T Lambda' P

is rational with 4N integral, its nonzero eigenvalues are the squares of the
nonzero singular values of Lambda' restricted to T, and

    sigma_min(Lambda|_T)^2 = lambda_min^+(N) / 2 .

The script proves, in integer arithmetic, that 4N annihilates the polynomial
x(x-8)(x-20)(x-24)(x-32) and has rank 66.  So the eigenvalues of N lie in
{0, 2, 5, 6, 8}, the kernel has dimension 30 (the 24 radial directions and
the 6 infinitesimal rotations), and the smallest nonzero eigenvalue is at
least 2.  Hence sigma_min(Lambda|_T) >= 1.

It then proves, again in integers, that the orthogonal projector onto the image
of Lambda has every diagonal entry equal to 11/16, the second constant of that
theorem (the lemma on the coordinates of the image).
"""
import itertools
import numpy as np
from fractions import Fraction

ROOTS = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = [0, 0, 0, 0]; v[i] = si; v[j] = sj
            ROOTS.append(tuple(v))
ROOTS = sorted(set(ROOTS))
A = np.array(ROOTS, dtype=np.int64)
assert A.shape == (24, 4)

TIGHT = [(i, j) for i in range(24) for j in range(i + 1, 24)
         if int(A[i] @ A[j]) == 1]
assert len(TIGHT) == 96

# Lambda' on the full space (R^4)^24, as a 96 x 96 integer matrix
Lam = np.zeros((96, 96), dtype=np.int64)
for r, (i, j) in enumerate(TIGHT):
    Lam[r, 4 * i:4 * i + 4] += A[j]
    Lam[r, 4 * j:4 * j + 4] += A[i]

# 2P, integral: blocks 2I - alpha alpha^T
P2 = np.zeros((96, 96), dtype=np.int64)
for i in range(24):
    P2[4 * i:4 * i + 4, 4 * i:4 * i + 4] = 2 * np.eye(4, dtype=np.int64) - np.outer(A[i], A[i])

N4 = P2 @ (Lam.T @ Lam) @ P2          # = 4 N, integral
assert np.array_equal(N4, N4.T)
print('max |4N| entry:', int(np.abs(N4).max()))

I96 = np.eye(96, dtype=np.int64)
prod = N4.copy()
for c in (8, 20, 24, 32):
    prod = prod @ (N4 - c * I96)
print('4N annihilates x(x-8)(x-20)(x-24)(x-32):', bool(np.all(prod == 0)))


def rank_mod(M, p):
    """rank over the field with p elements, p prime"""
    B = [[int(x) % p for x in row] for row in M]
    rows, cols, r = len(B), len(B[0]), 0
    for c in range(cols):
        piv = next((k for k in range(r, rows) if B[k][c]), None)
        if piv is None:
            continue
        B[r], B[piv] = B[piv], B[r]
        inv = pow(B[r][c], p - 2, p)
        B[r] = [(v * inv) % p for v in B[r]]
        for k in range(rows):
            if k != r and B[k][c]:
                f = B[k][c]
                B[k] = [(B[k][t] - f * B[r][t]) % p for t in range(cols)]
        r += 1
    return r


# rank over Q is at least the rank over F_p, and here the upper bound 66 is
# known from the 30-dimensional kernel exhibited above, so equality follows
MULT = {}
for c, lam in ((0, 0), (8, 2), (20, 5), (24, 6), (32, 8)):
    r = rank_mod(N4 - c * I96, 1000003)
    MULT[lam] = 96 - r
    print('eigenvalue %d of N: rank of 4N - %2dI is %2d, multiplicity %2d'
          % (lam, c, r, 96 - r))
assert sum(MULT.values()) == 96, MULT
assert MULT == {0: 30, 2: 29, 5: 8, 6: 21, 8: 8}, MULT
print()
print('The multiplicities sum to 96, so the ranks over a field of positive')
print('characteristic, which bound the ranks over Q from below, are met.')
print('The kernel of N is 30-dimensional: the 24 radial directions, on which')
print('P vanishes, and the 6 infinitesimal rotations.  Halving and taking')
print('square roots, the singular values of Lambda on the tangent space are')
print('0, 1, sqrt(5/2), sqrt3, 2 with multiplicities 6, 29, 8, 21, 8.')
print('The smallest nonzero singular value is exactly 1, the first of the')
print('two constants the radius of Theorem (a radius for the rigidity) is built from.')

# ---------------------------------------------------------------------------
# The image of Lambda: every one of the 96 coordinates carries the same weight.
#
# K = Lambda' P Lambda'^T is the Gram matrix of the image, with the nonzero
# eigenvalues of N; so 2K = Lambda' (2P) Lambda'^T is integral with eigenvalues in
# {0, 4, 10, 12, 16}.  If 2K annihilates x(x-4)(x-10)(x-12)(x-16), the projector
# onto ker K is E0 = (2K-4I)(2K-10I)(2K-12I)(2K-16I) / 7680 (Lagrange), and the
# projector onto im Lambda is I - E0.  Its diagonal is 11/16 at every pair exactly
# when every diagonal entry of 7680 E0 equals 7680 * 5/16 = 2400.  All in integers.
print()
Lo = Lam.astype(object)
K2 = Lo @ P2.astype(object) @ Lo.T
I96o = np.eye(96, dtype=np.int64).astype(object)
M = I96o.copy()
for c in (4, 10, 12, 16):
    M = M @ (K2 - c * I96o)
print('2K annihilates x(x-4)(x-10)(x-12)(x-16):', bool(np.all(K2 @ M == 0)))
diag = sorted(set(int(M[p, p]) for p in range(96)))
print('diagonal of 7680 E0 takes the values', diag, '(2400 means 11/16 on im Lambda)')
trace_im = sum(Fraction(7680 - int(M[p, p]), 7680) for p in range(96))
print('trace of the projector onto im Lambda:', trace_im, '(its rank, 72 - 6)')
assert bool(np.all(K2 @ M == 0)) and diag == [2400] and trace_im == 66
print('So |g_p| <= (sqrt11/4) |g|_2 for every g in im Lambda and every pair p, and')
print('|g|_1 >= (4/sqrt11) |g|_2: the constant behind the radius 2/sqrt(397).')

# ---------------------------------------------------------------------------
# How far the constant can be pushed: nu = min |Lambda tau|_1 / |tau|_2 over the
# tangent vectors orthogonal to the rotations.  The lemma gives nu >= 4/sqrt11; a
# displacement of one direction, with its rotational part removed, gives an upper
# bound, computed here exactly: ratio^2 = |Lambda' tau|_1^2 / (2 |tau|^2).
print()
ROT = []
for a, b in itertools.combinations(range(4), 2):
    Mab = np.zeros((4, 4), dtype=np.int64); Mab[a, b] = 1; Mab[b, a] = -1
    ROT.append([Fraction(int(x)) for x in (A @ Mab.T).reshape(-1)])
Gr = [[sum(x * y for x, y in zip(r, q)) for q in ROT] for r in ROT]
assert all(Gr[i][j] == (Gr[0][0] if i == j else 0) for i in range(6) for j in range(6))   # orthogonal, equal norms
x = [Fraction(0)] * 96
x[0:4] = [Fraction(int(v)) for v in (2 * np.eye(4, dtype=np.int64) - np.outer(A[0], A[0]))[:, 2]]   # 2P e_2 at root 0
for r in ROT:
    c = sum(a * b for a, b in zip(x, r)) / Gr[0][0]
    x = [a - c * b for a, b in zip(x, r)]
lx = [sum(Fraction(int(Lam[p, k])) * x[k] for k in range(96)) for p in range(96)]
ratio2 = sum(abs(v) for v in lx) ** 2 / (2 * sum(v * v for v in x))
print('one direction displaced, rotations removed: |Lambda tau|_1/|tau|_2 = sqrt(%s) = %.6f' % (ratio2, float(ratio2) ** 0.5))
print('so nu <= %.4f, and the radius (144/nu^2 + 1/4)^(-1/2) of this argument cannot pass %.4f'
      % (float(ratio2) ** 0.5, (144 / float(ratio2) + 0.25) ** -0.5))
