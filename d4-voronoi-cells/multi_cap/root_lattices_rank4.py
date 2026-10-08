#!/usr/bin/env python3
"""
root_lattices_rank4.py -- the combinatorial half of the twenty-four-point
classification, checked exactly.

De Laat, Leijenhorst and de Muinck Keizer prove (their Lemma 5.1) that any
24 unit vectors in R^4 with pairwise inner products at most 1/2 have all
their pairwise inner products in {-1, -1/2, 0, 1/2}; that is the part of
their theorem that rests on the semidefinite computation.  From there the
classification is classical, and this script checks its finite content.

Scale the 24 vectors by sqrt 2.  They become vectors of norm 2 whose
pairwise inner products are integers, so they generate an integral
lattice L of rank at most 4, and they are among the norm-2 vectors of L.
The norm-2 vectors of an integral lattice form a simply laced root system
(inner products of norm-2 vectors are 0, +-1, +-2, and the reflection in a
norm-2 vector preserves the lattice), the lattice they generate is a root
lattice, an orthogonal sum of lattices of type A, D, E, and a root lattice
has a basis of roots (its simple roots).  So L has a basis of norm-2
vectors, whose Gram matrix G has 2 on the diagonal and entries -1, 0, 1
off it, and the number of norm-2 vectors of L is the number of integer
vectors x with x^T G x = 2.

The script enumerates every positive definite Gram matrix of that shape of
rank r = 1, 2, 3, 4, counts the norm-2 vectors of the lattice it spans,
and reports the largest count in each rank: 2, 6, 12, 24.  The count 24
occurs only at rank 4 and only for lattices of determinant 4 whose 24
roots each have 8 neighbours at inner product 1, 6 at 0, 8 at -1 and 1 at
-2, which is the D_4 root system; the next largest count at rank 4 is 20
(A_4, determinant 5).  Hence the 24 scaled vectors are all the roots of a
copy of D_4, and the original configuration is the D_4 root system up to
isometry.  Everything is integer arithmetic.  The search bound
|x_i| <= 15 is rigorous: for these matrices det G >= 1 and every
eigenvalue is at most 5 (Gershgorin), so the least eigenvalue is at
least det G / 5^3 >= 1/125 and x^T G x = 2 forces |x|^2 <= 250.

Usage: python3 root_lattices_rank4.py
"""
import itertools
import numpy as np

def det_exact(M):
    """Determinant of a small integer matrix by exact rational elimination."""
    from fractions import Fraction
    A = [[Fraction(int(x)) for x in row] for row in M]; n = len(A); d = Fraction(1)
    for k in range(n):
        p = next((i for i in range(k, n) if A[i][k] != 0), None)
        if p is None: return 0
        if p != k: A[k], A[p] = A[p], A[k]; d = -d
        d *= A[k][k]
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n): A[i][j] -= m * A[k][j]
    assert d.denominator == 1
    return int(d)

def pd_and_det(G):
    """Sylvester's criterion with exact integer determinants."""
    n = len(G)
    for k in range(1, n + 1):
        d = det_exact(G[:k, :k])
        if d <= 0: return False, 0
    return True, d

_GRID = {}
def count_roots(G, bound=15):
    r = len(G)
    if r not in _GRID:
        rng = np.arange(-bound, bound + 1)
        _GRID[r] = np.array(list(itertools.product(rng, repeat=r)), dtype=np.int64)
    X = _GRID[r]
    q = np.einsum('ni,ij,nj->n', X, G, X)
    R = X[q == 2]
    return R

def main():
    grand = {}
    for r in range(1, 5):
        pairs = [(i, j) for i in range(r) for j in range(i + 1, r)]
        best = 0; best_info = []
        n_pd = 0
        for off in itertools.product((-1, 0, 1), repeat=len(pairs)):
            G = 2 * np.eye(r, dtype=np.int64)
            for (i, j), v in zip(pairs, off):
                G[i, j] = G[j, i] = v
            ok, det = pd_and_det(G)
            if not ok: continue
            n_pd += 1
            R = count_roots(G)
            c = len(R)
            if c > best: best = c; best_info = []
            if c == best:
                # neighbour profile of the root system: inner products of one root with all others
                ip = R @ G @ R.T
                prof = tuple(sorted(np.bincount((ip[0] + 2).astype(int), minlength=5).tolist()))
                best_info.append((det, prof))
        grand[r] = (best, n_pd, sorted(set(best_info)))
        print(f"rank {r}: {n_pd} positive definite Gram matrices; largest number of norm-2 vectors {best}; "
              f"(determinant, neighbour profile) of the maximisers: {sorted(set(best_info))}")
    # the rank-4 census: every count that occurs, with determinants
    census = {}
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for off in itertools.product((-1, 0, 1), repeat=6):
        G = 2 * np.eye(4, dtype=np.int64)
        for (i, j), v in zip(pairs, off):
            G[i, j] = G[j, i] = v
        ok, det = pd_and_det(G)
        if not ok: continue
        c = len(count_roots(G))
        census.setdefault(c, set()).add(int(det))
    print("rank 4 census (number of norm-2 vectors: determinants that occur):",
          {k: sorted(v) for k, v in sorted(census.items())})
    assert grand[4][0] == 24 and grand[3][0] == 12 and grand[2][0] == 6 and grand[1][0] == 2
    assert census[24] == {4} and max(c for c in census if c < 24) == 20
    for det, prof in grand[4][2]:
        # profile counts of inner products -2, -1, 0, 1, 2 with the first root: 1, 8, 6, 8, 1
        assert det == 4 and prof == (1, 1, 6, 8, 8), (det, prof)
    print("PASS: a lattice of rank at most 4 generated by norm-2 vectors has at most 24 of them, "
          "with 24 only for the D_4 lattice; so 24 unit vectors with pairwise inner products in "
          "{-1, -1/2, 0, 1/2} form a copy of the D_4 root system.")

if __name__ == "__main__":
    main()
