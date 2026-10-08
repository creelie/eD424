"""
gegenbauer_check.py -- the check of zonal/README.md item 3, which the package
describes but does not contain.

For a signature lambda = (k, 0) and two single points x, y with <x, y> = u, the
zonal matrix Z_lambda({x}, {y}) (the entry k1 = k2 = 0, the only admissible one
for one-point sets) must be a multiple of the zonal spherical harmonic of
degree k on S^3, that is of the Chebyshev polynomial U_k(u), for k = 0..14.
This evaluates Z_lambda symbolically from psker's output through zonal.py,
exactly as verify45.py does, and reports the constant.

    python3 gegenbauer_check.py ../zonal ps.txt
"""
import os
import sys
from fractions import Fraction

import sympy as sp

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import zonal  # noqa: E402

ps = zonal.load_ps(sys.argv[2])
u = sp.Symbol('u')
one, zero = sp.Integer(1), sp.Integer(0)
coerce = lambda q: sp.Rational(q.numerator, q.denominator)
ok = True
for k in range(15):
    lam = (k, 0)
    if (lam, 0, 0) not in ps:
        print('k=%2d  missing from the psker output' % k)
        ok = False
        continue
    z = sp.expand(zonal.evaluate_zonal_matrix(ps, lam, 0, 0, [one, one, u, u, u, u],
                                              one, zero, coerce))
    U = sp.expand(sp.chebyshevu(k, u))
    ratio = sp.simplify(z / U)
    exact = ratio.is_constant()
    target = sp.Rational(8) ** k / (k + 1) ** 2
    ok &= bool(exact)
    print('k=%2d  Z_(k,0) = c * U_k(u) exactly: %-5s  c = %s   c / (8^k/(k+1)^2) = %s'
          % (k, exact, ratio, sp.nsimplify(ratio / target) if exact else '-'))
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
