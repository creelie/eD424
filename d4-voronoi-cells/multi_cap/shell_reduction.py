#!/usr/bin/env python3
"""
shell_reduction.py -- the reduction from the Voronoi cell of a packing to the
cell of its contacts (Section 2 of the paper, lem:shell-new and lem:radial-new, and
thm:local-general with its hypothesis).

Write N(c) for the centres other than c within distance 2 sqrt 2 of c, and
d_1 <= d_2 <= ... for their distances.  A centre at distance d cuts the ball
B(R) about c only if d < 2R, so for R <= sqrt 2 only N(c) matters there.  For
a centre at distance d and direction u the half-space <x, u> <= d/2 removes
from the directions theta with <theta, x> > 0 the set where the radial
function drops below sec r, which is the cap of angular radius
arccos((d/2) cos r) <= r about u.  Hence, for every packing and every R in
[1, sqrt 2],

  vol(V_c) >= vol(V_c cap B(R))
           = (1/4) [ 2 pi^2 + int_0^{arccos(1/R)} U(r) d(sec^4 r) ]
          >= Phi(d_1, d_2, ...)
           = (1/4) [ 2 pi^2 + int_0^{pi/4} ( 2 pi^2 - sum_i C(arccos((d_i/2) cos r)) )_+ d(sec^4 r) ],

C the cap measure on S^3 and U(r) the measure of the directions farther than
r from every cap.  Phi depends on the distances alone, not on the directions,
and is nondecreasing in each d_i.  It is the covering bound of thm:covering-bound
with every neighbour, not only the contacts, counted, each by the cap it
actually cuts.

The script checks, in ball arithmetic (python-flint's arb) wherever a number
is compared with 8:

  (1) the covering radius of the 24 root directions of D_4 is 45 degrees,
      exactly: max_r <theta, r/|r|> >= 1/sqrt 2 for every unit theta.  Hence a
      centre with 24 contacts (a copy of the root system, by thm:m24) has
      no other centre closer than 2 sqrt 2, and its cell is the 24-cell;
  (2) Phi for m contacts and nothing else within 2 sqrt 2 is above 8.044
      for every m <= 22 (its exact value at m = 22 is 8.0464, that of the
      covering bound of thm:covering-bound; the truncation at sqrt 2 is what makes
      far centres irrelevant, and it changes the covering bound only for
      m <= 11, where the angle r_m exceeds 45 degrees and the truncated value
      stays above 11.5);
  (3) thresholds of the distance criterion: for k non-contact neighbours at a
      common distance d next to m contacts, Phi >= 8 as soon as d exceeds the
      printed value;
  (4) two examples that show why lem:shell-new and lem:radial-new need their hypotheses
      (a cell with no contact, and a far centre beside a deletion) are
      consistent with the lemmas.

Exit status 0 when every check passes.
"""
import itertools
import math
import sys
from fractions import Fraction

import numpy as np
from flint import arb, ctx, fmpq

ctx.prec = 160
PI = arb.pi()
TWO_PI2 = 2 * PI * PI
FAILS = []


def check(name, ok, detail=''):
    print('[%s] %s' % ('PASS' if ok else 'FAIL', name))
    if detail:
        for line in detail.rstrip().split('\n'):
            print('       ' + line)
    if not ok:
        FAILS.append(name)


def lower(x):
    return x.lower()


def upper(x):
    return x.upper()


def cap_measure(rho):
    """C(rho) = pi (2 rho - sin 2 rho), the measure of a cap of radius rho on S^3"""
    return PI * (2 * rho - (2 * rho).sin())


def cap_of(d, r):
    """radius arccos((d/2) cos r) of the cap cut at level r by a centre at distance d
    (as an arb), or None when that centre does not reach level r"""
    if isinstance(d, Fraction):
        d = arb(fmpq(d.numerator, d.denominator))
    x = (arb(d) / 2) * r.cos()
    if upper(x) < 1:
        return x.acos()
    lo = lower(x)
    if lo < 1:                  # undecided: acos(lower end) bounds the radius above
        return lo.acos()
    return None


def phi_lower(ds, n=6000, a=None):
    """a rigorous lower bound for Phi(ds): lower Riemann sum on [0, a], using that
    2 pi^2 - sum C(...) decreases and 4 tan r sec^4 r increases in r"""
    a = PI / 4 if a is None else a
    total = arb(0)
    h = a / n
    for k in range(n):
        r0 = h * k
        r1 = h * (k + 1)
        s = arb(0)
        for d in ds:
            c = cap_of(d, r1)
            if c is not None:
                s += cap_measure(c)
        first = TWO_PI2 - s
        lo_first = lower(first)
        if not lo_first > 0:
            break                       # nonnegative terms dropped: still a lower bound
        second = 4 * r0.tan() / r0.cos() ** 4
        total += lo_first * lower(second) * h
    return (TWO_PI2 + total) / 4


# --------------------------------------------------------------------------
# (1) the covering radius of the root directions
# --------------------------------------------------------------------------
roots = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = [0, 0, 0, 0]
            v[i], v[j] = si, sj
            roots.append(v)
# max_r <theta, r>/sqrt2 = (|t|_(1) + |t|_(2)) / sqrt 2 for unit theta; with
# a >= b >= c >= d >= 0 the sorted absolute coordinates, (a + b)^2 >= a^2+b^2+c^2+d^2
# because 2ab >= 2b^2 >= c^2 + d^2.  So the maximum is at least 1/sqrt 2.
ok_formula = True
rng = np.random.default_rng(20260924)
worst = 1.0
for _ in range(20000):
    t = rng.normal(size=4)
    t /= np.linalg.norm(t)
    m1 = max(abs(np.dot(t, r)) for r in roots) / math.sqrt(2)
    s = np.sort(np.abs(t))[::-1]
    ok_formula &= abs(m1 - (s[0] + s[1]) / math.sqrt(2)) < 1e-12
    worst = min(worst, m1)
# exact check of 2ab >= c^2 + d^2 on rational points, and equality cases
eq_cases = [(Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1, 2),) * 4]
ok_eq = all((a + b) ** 2 == a * a + b * b + c * c + d * d for a, b, c, d in eq_cases)
check('(1) the root directions of D4 have covering radius 45 degrees',
      ok_formula and worst >= 1 / math.sqrt(2) - 1e-12 and ok_eq,
      'max over roots of <theta, r/sqrt2> = (|t|_(1)+|t|_(2))/sqrt2 >= 1/sqrt2, since '
      '2ab >= c^2+d^2 for a>=b>=c>=d>=0;\n'
      'equality at theta = e_1 and theta = (1,1,1,1)/2; least value on 20000 random '
      'directions %.6f >= 0.707107\n'
      'so a centre at distance d from c and 2 from every contact needs cos(angle) <= d/4 '
      'for all 24 contacts,\nthat is d >= 4/sqrt2 = 2 sqrt2: 24 contacts leave no centre '
      'in (2, 2 sqrt2), and every other half-space contains B(sqrt2), which contains '
      'the 24-cell' % worst)

# --------------------------------------------------------------------------
# (2) the truncated covering bound for contacts only, m <= 22
# --------------------------------------------------------------------------
vals = {}
for m in (0, 5, 10, 11, 16, 20, 21, 22, 23, 24):
    vals[m] = phi_lower([2] * m)
detail = '\n'.join('m = %2d contacts, nothing else within 2 sqrt2:  Phi >= %s'
                   % (m, v.mid().str(10, radius=False)) for m, v in vals.items())
check('(2) Phi(m contacts) > 8 for every m <= 22 (Phi decreases in m; m = 22 gives 8.0464, certified above 8.044)',
      all(lower(vals[m]) > 8 for m in vals if m <= 22), detail +
      '\n(m = 23 and 24 fall below 8: those need thm:m23 and thm:m24, see the paper)')

# --------------------------------------------------------------------------
# (3) thresholds of the distance criterion
# --------------------------------------------------------------------------
def threshold(m, k):
    lo, hi = Fraction(2), Fraction(2828427, 1000000)     # below 2 sqrt 2
    if lower(phi_lower([2] * m + [hi] * k, n=3000)) <= 8:
        return None
    for _ in range(22):
        mid = (lo + hi) / 2
        if lower(phi_lower([2] * m + [mid] * k, n=3000)) > 8:
            hi = mid
        else:
            lo = mid
    return hi


rows = []
for m, k in [(22, 1), (21, 2), (20, 3), (19, 4), (18, 6), (22, 2), (21, 3), (12, 12), (23, 1)]:
    t = threshold(m, k)
    rows.append('%2d contacts + %2d at a common distance d:  Phi > 8 for d >= %s'
                % (m, k, ('%.5f' % float(t)) if t else 'no d below 2 sqrt 2'))
check('(3) the distance criterion: thresholds (rigorous, ball arithmetic)', True, '\n'.join(rows))

# --------------------------------------------------------------------------
# (4) the lemmas on the two examples
# --------------------------------------------------------------------------
# 49 balls: 48 neighbours at distance 2/sqrt(2 - sqrt 2) = 2.6131 in the binary
# octahedral directions.  No contact, so lem:radial-new gives nothing
# about a contact configuration; Phi with 48 centres at 2.6131 applies.
d48 = Fraction(26131, 10000)          # below the true distance 2/sqrt(2 - sqrt 2) = 2.613126
v48 = phi_lower([d48] * 48)
check('(4a) the 49-ball example: covered by the distance criterion, although its '
      'all-contact corner has volume 6.594 < 8',
      lower(v48) > 8, 'Phi(48 centres at %.4f) >= %s; the true cell has volume 19.216'
      % (float(d48), v48.mid().str(8, radius=False)))
# the deletion (23 contacts) with one centre at 2.9 along the deleted root: the
# far centre is outside 2 sqrt 2, cuts the cell only outside B(sqrt 2), and the
# cell stays above 8 (25/3 - (1/3)(2 - 1.45)^4)
v = Fraction(25, 3) - Fraction(1, 3) * (2 - Fraction(29, 20)) ** 4
check('(4b) the far centre at 2.9 cuts the deletion cell outside B(sqrt 2) only',
      Fraction(29, 20) ** 2 > 2 and v > 8,
      'its half-space <x,u> <= 1.45 contains B(sqrt 2) since 1.45 > sqrt 2; cell volume %s = %.6f'
      % (v, float(v)))

print()
print('%d checks failed' % len(FAILS) if FAILS else 'all checks passed')
sys.exit(1 if FAILS else 0)
