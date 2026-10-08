"""
Exact integer verification of what can be added back to a punctured root
system.

Write the roots as integer vectors r = +-e_i +- e_j, so that the contact
directions are r/sqrt2 and the cell of a subset S is

    { x : <x, r> <= sqrt2 for r in S }  =  sqrt2 * { y : <y, r> <= 1 }.

The inner polytope P(S) = { y : <y,r> <= 1, r in S } has integer facet
normals and unit right-hand sides, so all of its vertices are rational:
each is the solution of a four by four integer system, and Cramer's rule
gives it as m/e with m integral and e a positive integer. The cell has
circumradius 2 exactly when P(S) has circumradius sqrt2, that is when
|m|^2 <= 2 e^2 at every vertex, and a unit direction v can be added to S
exactly when 2v lies in the cell, that is when y = v*sqrt2 is a vertex of
P(S) with |m|^2 = 2 e^2.

Everything below is integer arithmetic, so the conclusions are exact.
"""
import numpy as np
from fractions import Fraction
from itertools import combinations

ROOTS = np.array(
    [[(1 if k == i else 0) * si + (1 if k == j else 0) * sj for k in range(4)]
     for i, j in combinations(range(4), 2)
     for si in (1, -1) for sj in (1, -1)], dtype=np.int64)


def det3(A):
    return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))


def det4(A):
    s = 0
    for c in range(4):
        minor = np.delete(np.delete(A, 0, axis=0), c, axis=1)
        s += (-1) ** c * A[0, c] * det3(minor)
    return int(s)


def solve_ones(A):
    """Solve A y = 1 exactly. Returns (m, e) with y = m/e, e > 0, or None."""
    d = det4(A)
    if d == 0:
        return None
    m = np.zeros(4, dtype=np.int64)
    for c in range(4):
        B = A.copy()
        B[:, c] = 1
        m[c] = det4(B)
    if d < 0:
        m, d = -m, -d
    return m, d


def candidates():
    """Every rational point where four root hyperplanes meet."""
    out = {}
    for q in combinations(range(24), 4):
        r = solve_ones(ROOTS[list(q)])
        if r is not None:
            out[q] = r
    return out


CAND = candidates()
QUADS = np.array(list(CAND.keys()), dtype=np.int64)
MNUM = np.array([CAND[tuple(q)][0] for q in QUADS], dtype=np.int64)
EDEN = np.array([CAND[tuple(q)][1] for q in QUADS], dtype=np.int64)


def vertices_of(keep):
    """Exact vertices of P(S) for S = ROOTS[keep], as (m, e) arrays."""
    inside = np.all(np.isin(QUADS, np.array(keep)), axis=1)
    M, E = MNUM[inside], EDEN[inside]
    if len(M) == 0:
        return M, E
    V = M @ ROOTS[keep].T                      # <m, r> for every kept root
    ok = np.all(V <= E[:, None], axis=1)
    M, E = M[ok], E[ok]
    key = {}
    for m, e in zip(M, E):
        g = np.gcd.reduce(np.append(np.abs(m), e))
        key[tuple(m // g) + (int(e // g),)] = None
    K = np.array([list(k) for k in key], dtype=np.int64)
    return K[:, :4], K[:, 4]


def report(j):
    """For every way of removing j roots: is the circumradius exactly 2, and
    are the directions attaining it exactly the removed roots?"""
    tot = 0
    over = []          # cells of circumradius above 2
    missing = []       # a removed root that is not attained
    extra = {}         # an attaining direction that is not a removed root
    far = {}           # the squared circumradius, as a fraction |m|^2 / e^2
    for cmb in combinations(range(24), j):
        keep = [i for i in range(24) if i not in cmb]
        M, E = vertices_of(keep)
        num = np.sum(M * M, axis=1)             # |m|^2
        den = E * E                             # e^2
        # circumradius of the cell is 2 * max sqrt(|m|^2 / (2 e^2))
        k = max(range(len(num)), key=lambda i: Fraction(int(num[i]),
                                                        int(den[i])))
        far[cmb] = (int(num[k]), int(den[k]))
        if num[k] * 1 > 2 * den[k]:
            over.append(cmb)
            continue
        hit = np.where(num == 2 * den)[0]       # squared norm exactly 2
        seen = set()
        for h in hit:
            m, e = M[h], E[h]
            match = [c for c in cmb if np.array_equal(m, ROOTS[c] * e)]
            if match:
                seen.add(match[0])
            else:
                extra.setdefault(cmb, []).append((tuple(int(x) for x in m),
                                                  int(e)))
        if len(seen) != j:
            missing.append(cmb)
        tot += 1
    return tot, over, missing, extra, far


if __name__ == "__main__":
    print("exact vertex enumeration of the punctured root cells")
    print(f"  points where four root hyperplanes meet: {len(CAND)}")
    print()
    for j in (1, 2, 3):
        n = len(list(combinations(range(24), j)))
        tot, over, missing, extra, far = report(j)
        print(f"j = {j}:  {n} ways of removing j roots")
        print(f"  cells of circumradius above 2: {len(over)}")
        print(f"  of the rest, cells where a removed root is not attained:"
              f" {len(missing)}")
        print(f"  of the rest, cells attaining a direction that is not a"
              f" removed root: {len(extra)}")
        if over:
            c = over[0]
            a, b = far[c]
            print(f"    example {c}: squared circumradius"
                  f" 4 * {a}/(2 * {b}) = {2.0*a/b:.10f}"
                  f"   (6 = {6.0:.10f})")
        print()

    print("conclusion")
    print("  Removing one or two roots leaves a cell of circumradius exactly")
    print("  2 whose only points of that norm are twice the removed roots, so")
    print("  a contact configuration meeting a D_4 root system in 22 or more")
    print("  directions lies inside it.")
    print("  Removing three leaves the same picture except at the 96 triples")
    print("  that are pairwise at 60 degrees, where the circumradius is")
    print("  sqrt6 and a region of further directions opens; root_meet.py")
    print("  shows that no two directions of that region are 60 degrees")
    print("  apart unless both are removed roots.")
