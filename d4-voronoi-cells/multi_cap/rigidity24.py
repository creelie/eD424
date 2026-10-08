#!/usr/bin/env python3
"""
Infinitesimal rigidity of the root system itself as a contact
configuration of 24 directions, the corollary that follows
prop:deletion-rigid (the deletion case is rigidity23.py).

Write the D4 roots unnormalised, a = +-e_i +- e_j with |a|^2 = 2, so
that the contact condition <w_i,w_j> <= 1/2 reads <a_i,a_j> <= 1 and
every number below is an integer.  All 24 roots are kept, and T is the
set of the 96 pairs at <a_i,a_j> = 1, the edges of the 24-cell.

A first-order motion of the 24 directions through contact configurations
is a d = (d_1,...,d_24) with

    <d_i, a_i> = 0                              for every i,
    <d_i, a_j> + <a_i, d_j> <= 0                for every (i,j) in T.

The infinitesimal rotations d_i = A a_i, A antisymmetric, satisfy all of
these with equality and span a 6-dimensional space.  Two exact linear
computations show there is nothing else:

  1  the constant stress: weight 1 on every pair of T and -4 on the
     diagonal is in equilibrium, sum_{j ~ i} a_j - 4 a_i = 0 for every
     i, because the eight roots at sixty degrees from a root sum to four
     times that root.  Pairing it with a feasible d gives
     sum_T (<d_i,a_j> + <a_i,d_j>) = 0, a sum of nonpositive terms, so
     every one of the 96 inequalities holds with equality;

  2  the space of motions holding all of T at equality has dimension
     exactly 6, and the rotations already fill it.

So every first-order motion of the root system is a rotation, and by
the theorem of Roth and Whiteley the root system is rigid as a
tensegrity: no continuous deformation through contact configurations
moves it except by rotation.  That is the local half of the statement
of de Laat, Leijenhorst and de Muinck Keizer that every 24-point
contact configuration is a root system; the global half is what this
paper cites and does not prove.

Run:  python3 rigidity24.py
Exits nonzero if any check fails.
"""

import sys

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


def main():
    A = roots()
    n = len(A)
    Am = Matrix(A)

    def ip(u, v):
        return sum(u[k] * v[k] for k in range(4))

    T = [(i, j) for i in range(n) for j in range(i + 1, n)
         if ip(A[i], A[j]) == 1]
    degrees = [sum(1 for (i, j) in T if p in (i, j)) for p in range(n)]
    record("the root system has 24 directions and 96 tight pairs",
           n == 24 and len(T) == 96,
           "directions %d, tight pairs %d, every degree %s"
           % (n, len(T), sorted(set(degrees))))

    # ---- 1. the constant stress ----------------------------------
    Y = zeros(n, n)
    for i in range(n):
        Y[i, i] = -4
    for (i, j) in T:
        Y[i, j] = 1
        Y[j, i] = 1
    record("the stress is strictly positive on every tight pair",
           all(Y[i, j] > 0 for (i, j) in T),
           "weight 1 on each of the 96 pairs, -4 on the diagonal")
    record("the stress is in equilibrium: Y A = 0",
           (Y * Am).is_zero_matrix,
           "exact integer arithmetic, 24 x 4 residual matrix is zero; "
           "the eight neighbours of a root sum to four times the root")

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
        print("Every first-order motion of the root system through contact")
        print("configurations is an infinitesimal rotation.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
