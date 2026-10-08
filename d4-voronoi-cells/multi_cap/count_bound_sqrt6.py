#!/usr/bin/env python3
"""
count_bound_sqrt6.py -- at most 49 other centres lie within sqrt 6 of a centre
of a unit-ball packing of R^4 (exact arithmetic).

Two centres y, y' within sqrt 6 of c = 0, at distances d, d' in [2, sqrt 6)
and at least 2 apart, satisfy
    <y/d, y'/d'> <= (d^2 + d'^2 - 4) / (2 d d') < (6 + 6 - 4) / 12 = 2/3,
since the middle expression increases in d and d' on [2, sqrt 6] (its
derivative in d is (d^2 - d'^2 + 4) / (2 d^2 d') > 0).  Their directions
therefore form a spherical code on S^3 with inner products below 2/3, and
the linear programming bound of Delsarte, Goethals and Seidel applies: if
f = sum_k f_k G_k with f_0 > 0, f_k >= 0 for k >= 1 and f(t) <= 0 on
[-1, 2/3], where G_k(t) = U_k(t)/(k+1) are the Gegenbauer polynomials of
S^3, then every such code has at most f(1)/f_0 points.

The polynomial below, of degree 13, has f_0 = 1 and the rational
coefficients f_1, ..., f_16 listed (found by a linear programme and
rounded to nine decimals).  The script expands it exactly (sympy), checks
the signs of the coefficients, counts its real roots in [-1, 2/3] by
Sturm's theorem (none) and evaluates it at one point of the interval
(negative), so f < 0 on the whole interval, and prints f(1) = 49.577...
Hence at most 49 centres.

Exit status 0 when every check passes.
"""
import sys
from fractions import Fraction as Fr

import sympy as sp

F = [Fr(1)] + [Fr(c) for c in (
    '373905381/100000000', '3533131001/500000000', '987582551/100000000', '1029670669/100000000',
    '8741894339/1000000000', '5144212799/1000000000', '2250567441/1000000000', '0', '0', '0',
    '718880241/1000000000', '78673213/200000000', '350708567/1000000000', '0', '0', '0')]

x = sp.symbols('x')
U = [sp.Integer(1), 2 * x]
for k in range(1, len(F)):
    U.append(sp.expand(2 * x * U[-1] - U[-2]))
f = sp.expand(sum(sp.Rational(c.numerator, c.denominator) * U[k] / (k + 1) for k, c in enumerate(F)))

ok = True
signs = F[0] > 0 and all(c >= 0 for c in F[1:])
print('f_0 = 1 and f_k >= 0 for k >= 1: %s' % signs)
ok &= signs
roots = sp.Poly(f, x).count_roots(-1, sp.Rational(2, 3))
at0 = f.subs(x, 0)
neg = roots == 0 and at0 < 0
print('real roots of f in [-1, 2/3] (Sturm): %d; f(0) = %s < 0: so f < 0 on [-1, 2/3]: %s' % (roots, float(at0), neg))
ok &= neg
f1 = f.subs(x, 1)
print('degree %d; f(1) / f_0 = %s = %.9f' % (sp.degree(f, x), f1, float(f1)))
ok &= f1 < 50
if ok:
    print('PASS: at most %d centres lie within sqrt 6 of any centre' % int(f1))
    sys.exit(0)
print('FAILED')
sys.exit(1)
