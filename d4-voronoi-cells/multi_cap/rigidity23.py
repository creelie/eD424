#!/usr/bin/env python3
"""
Infinitesimal rigidity of the deletion configuration, prop:deletion-rigid.

Write the D4 roots unnormalised, alpha = +-e_i +- e_j with |alpha|^2 = 2,
so that the contact condition <w_i,w_j> <= 1/2 reads <a_i,a_j> <= 1 and
every number below is an integer.  Delete one root and let T be the set
of the 88 pairs left at <a_i,a_j> = 1.

A first-order motion of the remaining 23 directions through contact
configurations is a d = (d_1,...,d_23) with

    <d_i, a_i> = 0                              for every i,
    <d_i, a_j> + <a_i, d_j> <= 0                for every (i,j) in T.

The infinitesimal rotations d_i = A a_i, A antisymmetric, satisfy all of
these with equality and span a 6-dimensional space.  Two exact linear
computations show there is nothing else:

  1  an integer stress: y_ij > 0 on T and mu_i with
     sum_j y_ij a_j + mu_i a_i = 0 for every i.  Pairing it with any
     feasible d gives sum_ij y_ij (<d_i,a_j> + <a_i,d_j>) = 0, a sum of
     nonpositive terms with positive weights, so every one of the 88
     inequalities is forced to hold with equality;

  2  the space of motions holding all of T at equality has dimension
     exactly 6, and the rotations already fill it.

So every first-order motion is a rotation.  The stress is given by the
inner product with the deleted root: y_ij depends only on the pair
(<a_i,a_0>, <a_j,a_0>) and takes the values printed below.

Run:  python3 rigidity23.py
Exits nonzero if any check fails.
"""

import sys
from fractions import Fraction

from sympy import Matrix, zeros

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("[%s] %s" % ("PASS" if ok else "FAIL", name))
    if detail:
        for line in detail.splitlines():
            print("       " + line)


def roots():
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i], v[j] = si, sj
                    out.append(tuple(v))
    return out


# stress weights, keyed by the sorted pair of inner products with the
# deleted root, and diagonal multipliers keyed by that inner product
YPAIR = {(-2, -1): 2, (-1, -1): 1, (-1, 0): 2,
         (-1, 1): 1, (0, 1): 2, (1, 1): 3}
YDIAG = {-2: -8, -1: -6, 0: -8, 1: -8}


def main():
    R = roots()
    a0 = R[0]
    A = [R[k] for k in range(1, 24)]
    n = len(A)
    Am = Matrix(A)

    def ip(u, v):
        return sum(u[k] * v[k] for k in range(4))

    cls = [ip(a, a0) for a in A]
    T = [(i, j) for i in range(n) for j in range(i + 1, n)
         if ip(A[i], A[j]) == 1]
    record("the deletion leaves 23 directions and 88 tight pairs",
           n == 23 and len(T) == 88,
           "directions %d, tight pairs %d\n"
           "inner products with the deleted root: %s"
           % (n, len(T),
              {v: cls.count(v) for v in sorted(set(cls))}))

    types = sorted({tuple(sorted((cls[i], cls[j]))) for (i, j) in T})
    record("only the six stated pair types occur in T",
           set(types) == set(YPAIR),
           "types present: %s" % (types,))

    # ---- 1. the integer stress -----------------------------------
    Y = zeros(n, n)
    for i in range(n):
        Y[i, i] = YDIAG[cls[i]]
    for (i, j) in T:
        w = YPAIR[tuple(sorted((cls[i], cls[j])))]
        Y[i, j] = w
        Y[j, i] = w
    positive = all(Y[i, j] > 0 for (i, j) in T)
    record("the stress is strictly positive on every tight pair",
           positive,
           "weights used: %s" % sorted(set(YPAIR.values())))
    record("the stress is in equilibrium: Y A = 0",
           (Y * Am).is_zero_matrix,
           "exact integer arithmetic, 23 x 4 residual matrix is zero")

    # ---- 2. the equality space -----------------------------------
    N = 4 * n
    B = zeros(n, N)
    for i in range(n):
        for k in range(4):
            B[i, 4 * i + k] = A[i][k]
    P = zeros(len(T), N)
    for p, (i, j) in enumerate(T):
        for k in range(4):
            P[p, 4 * i + k] = A[j][k]
            P[p, 4 * j + k] = A[i][k]
    L = (B.col_join(P)).nullspace()

    rot = []
    for a in range(4):
        for b in range(a + 1, 4):
            v = zeros(N, 1)
            for i in range(n):
                v[4 * i + a] = A[i][b]
                v[4 * i + b] = -A[i][a]
            rot.append(v)
    Rm = Matrix.hstack(*rot)
    both = Matrix.hstack(Rm, Matrix.hstack(*L)) if L else Rm
    record("motions holding every tight pair at equality are exactly the "
           "rotations",
           len(L) == 6 and Rm.rank() == 6 and both.rank() == 6,
           "dim of the equality space %d, dim of the rotation space %d, "
           "rank of the two together %d" % (len(L), Rm.rank(), both.rank()))

    ok = all(o for _, o in RESULTS)
    print("=" * 62)
    print("%d of %d checks passed" % (sum(o for _, o in RESULTS),
                                      len(RESULTS)))
    if ok:
        print("Every first-order motion of the deletion configuration "
              "through")
        print("contact configurations is an infinitesimal rotation.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
