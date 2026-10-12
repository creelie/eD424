"""Extreme rays of C_J = {X in Sym_n : X[k] >= 0 for k in Min(J)} with cddlib (GMP rationals)."""
from fractions import Fraction as F
import cdd.gmp as cg
from forms import sym_coords

def extreme_rays(minvecs):
    rows = [[0] + sym_coords(k) for k in minvecs]
    mat = cg.matrix_from_array([[F(x) for x in r] for r in rows], rep_type=cg.RepType.INEQUALITY)
    poly = cg.polyhedron_from_matrix(mat)
    gen = cg.copy_generators(poly)
    rays = []
    for row in gen.array:
        assert row[0] == 0, "cone has a vertex other than 0?"
        v = [F(x) for x in row[1:]]
        # normalise to primitive integer vector
        from math import gcd, lcm
        den = 1
        for x in v:
            den = lcm(den, x.denominator)
        iv = [int(x * den) for x in v]
        g = 0
        for x in iv:
            g = gcd(g, abs(x))
        rays.append(tuple(x // g for x in iv))
    assert not gen.lin_set, "cone has a lineality space"
    return rays
