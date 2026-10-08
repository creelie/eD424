#!/usr/bin/env python3
"""
closure_lemmas.py -- lem:holes, prop:inversion-hull, cor:root-subsets,
lem:no-triple and cor:count of the paper ((A) to (E) below), checked in
exact arithmetic (sympy, integers) and in ball arithmetic (python-flint's
arb).  The distance criterion printed under (E) and part (F) are further
checks that the paper does not use.

Throughout, c = 0 is a centre of a unit-ball packing of R^4, the other
centres are points z with |z| >= 2 and pairwise distances >= 2, and the
half-space of z is H_z = {x : <x, z> <= |z|^2 / 2}.  The contacts of c are
the centres at distance exactly 2, written 2w with w a unit vector.

  (A) the hole lemma: every centre z satisfies |z| >= 4 <z/|z|, w> for every
      contact direction w, because |z - 2w|^2 = |z|^2 - 4 <z, w> + 4 >= 4.  So
      every centre other than the contacts lies at distance at least
      4 g(W), g(W) = min over unit t of max over w of <t, w>, the cosine of the
      covering radius of the contact directions W on S^3;

  (B) the inversion hull: for any set Y of centres, every centre z outside Y
      satisfies <q, z> <= |z|^2 / 2 for q = 4y/|y|^2 (y in Y) and for q = 0,
      by the identity

        |z|^2/2 - <4y/|y|^2, z> = [(|z|^2-4)(|y|^2-4) + 4(|y-z|^2-4)] / (2|y|^2),

      so the convex hull K(Y) of 0 and the points 4y/|y|^2 lies in H_z, and
      V_c contains V(Y) cap K(Y), V(Y) the intersection of the H_y, y in Y;

  (C) root subsets: every vertex of the 24-cell Q = {x : <x, r> <= 1, r in R}
      (R the unit roots) is the sum r + r' of exactly three orthogonal pairs
      of roots.  If the contact directions contain, for every vertex, one of
      its three pairs, then Q lies in K(contacts) and in the half-space of
      every contact, so V_c contains Q and vol(V_c) >= 8.  The script finds
      every set D of at most four deleted roots for which some vertex loses
      all three of its pairs: none for |D| <= 2, exactly the 96 triples of
      pairwise 60-degree roots at a common vertex for |D| = 3;

  (D) no triple overlaps below sqrt(3/2): four points pairwise at least 2
      apart do not fit in a ball of radius less than sqrt(3/2), by the
      identity  sum_{i<j} |p_i - p_j|^2 = 4 sum_i |p_i - g|^2  (g the
      centroid of four points).  If |x| < sqrt(3/2) and x were closer to three
      centres than to 0, the four points 0, y_1, y_2, y_3 would lie in the ball
      of radius |x| about x.  So inside B(sqrt(3/2)) the cell is the ball minus
      the caps cut by single centres plus the overlaps of pairs, exactly;

  (E) the count criterion: with R = sqrt(3/2), vol(V_c) >= vol B(R) - sum over
      the centres of S(|z|), S(d) the volume of the cap of B(R) beyond the
      hyperplane at distance d/2, zero for d >= sqrt 6.  S decreases in d, so
      if at most 22 centres lie within sqrt 6, vol(V_c) >= 9 pi^2/8 - 22 S(2),
      which is certified above 8.046 here;  the same function
      Psi(d_1, d_2, ...) = 9 pi^2/8 - sum S(d_i) is a distance criterion,
      and its thresholds are printed;

  (F) 23 contacts: if their covering radius is at most arccos(sqrt 6 / 4),
      52.2388 degrees, every other centre is at distance >= sqrt 6 by (A), and
      the certificate of thm:certificate bounds vol(V_c cap B(sqrt(3/2))) above 8
      (the case quoted after thm:m23, which labelled_certificate_check.py
      proves for any 23 centres within sqrt 6, whatever their distances).

Exit status 0 when every check passes.
"""
import itertools
import math
import sys
from fractions import Fraction

import sympy as sp
from flint import arb, ctx

ctx.prec = 160
PI = arb.pi()
FAILS = []


def check(name, ok, detail=''):
    print('[%s] %s' % ('PASS' if ok else 'FAIL', name))
    for line in detail.rstrip().split('\n') if detail else []:
        print('       ' + line)
    if not ok:
        FAILS.append(name)


# --------------------------------------------------------------------------
# (A) and (B): the two identities, symbolically in R^4
# --------------------------------------------------------------------------
y = sp.Matrix(sp.symbols('y1:5', real=True))
z = sp.Matrix(sp.symbols('z1:5', real=True))
w = sp.Matrix(sp.symbols('w1:5', real=True))
Y2, Z2 = (y.T * y)[0], (z.T * z)[0]
lhs = Z2 / 2 - (4 * y.T * z)[0] / Y2
rhs = ((Z2 - 4) * (Y2 - 4) + 4 * (((y - z).T * (y - z))[0] - 4)) / (2 * Y2)
okB = sp.simplify(sp.together(lhs - rhs)) == 0
# (A): with |w| = 1 substituted, |z - 2w|^2 - 4 = |z|^2 - 4<z,w>
W2 = (w.T * w)[0]
okA = sp.expand(((z - 2 * w).T * (z - 2 * w))[0] - 4 - (Z2 - 4 * (z.T * w)[0] + 4 * W2 - 4)) == 0
check('(A) hole lemma: |z - 2w|^2 - 4 = |z|^2 - 4<z,w> for unit w (symbolic)', okA,
      'so <z/|z|, w> <= |z|/4 for every centre z and contact direction w;\n'
      'for the root system g = 1/sqrt 2 and 4g = 2 sqrt 2 (lem:holes)')
check('(B) inversion hull: |z|^2/2 - <4y/|y|^2, z> = [(|z|^2-4)(|y|^2-4) + 4(|y-z|^2-4)]/(2|y|^2)'
      ' (symbolic)', okB,
      'both brackets are >= 0 for centres y, z (|y|, |z| >= 2, |y - z| >= 2)')

# --------------------------------------------------------------------------
# (C) root subsets, in integer coordinates (roots scaled to norm^2 = 2)
# --------------------------------------------------------------------------
roots = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = [0, 0, 0, 0]
            v[i], v[j] = si, sj
            roots.append(tuple(v))
dot = lambda a, b: sum(p * q for p, q in zip(a, b))
# the 24-cell in these coordinates is {x : <x, r> <= 2}; its vertices
verts = [tuple(2 * s if k == i else 0 for k in range(4)) for i in range(4) for s in (1, -1)]
verts += list(itertools.product((1, -1), repeat=4))
ok_verts = all(max(dot(v, r) for r in roots) == 2 and sum(dot(v, r) == 2 for r in roots) == 6
               for v in verts) and len(verts) == 24
pairs = {v: [] for v in verts}
for a, b in itertools.combinations(range(24), 2):
    if dot(roots[a], roots[b]) == 0:
        s = tuple(p + q for p, q in zip(roots[a], roots[b]))
        assert s in pairs, s
        pairs[s].append((a, b))
ok_pairs = all(len(p) == 3 for p in pairs.values())
counts = {}
bad3 = []
for size in range(1, 5):
    bad = 0
    for D in itertools.combinations(range(24), size):
        Ds = set(D)
        if any(all(a in Ds or b in Ds for a, b in pairs[v]) for v in verts):
            bad += 1
            if size == 3:
                bad3.append(D)
    counts[size] = bad
ok_bad3 = (len(bad3) == 96 and all(
    all(dot(roots[a], roots[b]) == 1 for a, b in itertools.combinations(D, 2)) and
    any(all(dot(v, roots[a]) == 2 for a in D) for v in verts) for D in bad3))
from math import comb
check('(C) root subsets: every vertex of the 24-cell is r + r\' for exactly three orthogonal '
      'root pairs', ok_verts and ok_pairs and counts[1] == 0 and counts[2] == 0 and ok_bad3,
      '24 vertices, each on 6 facets, each the sum of 3 orthogonal pairs (72 pairs in all);\n'
      'deleted sets D for which some vertex loses all three pairs:\n'
      '  |D| = 1: %d of %d,  |D| = 2: %d of %d,  |D| = 3: %d of %d (the pairwise-60-degree\n'
      '  triples at a common vertex),  |D| = 4: %d of %d'
      % (counts[1], comb(24, 1), counts[2], comb(24, 2), counts[3], comb(24, 3),
         counts[4], comb(24, 4)))

# --------------------------------------------------------------------------
# (D) the four-point identity
# --------------------------------------------------------------------------
P = [sp.Matrix(sp.symbols('p%d_1:5' % k, real=True)) for k in range(4)]
g = (P[0] + P[1] + P[2] + P[3]) / 4
s1 = sum((((P[i] - P[j]).T * (P[i] - P[j]))[0] for i, j in itertools.combinations(range(4), 2)))
s2 = 4 * sum((((P[i] - g).T * (P[i] - g))[0] for i in range(4)))
okD = sp.expand(s1 - s2) == 0
check('(D) sum_{i<j} |p_i - p_j|^2 = 4 sum |p_i - g|^2 for four points (symbolic)', okD,
      'six distances >= 2 give 24 <= 4 sum |p_i - x|^2 <= 16 |x|^2, so |x| >= sqrt(3/2):\n'
      'no point of the open ball B(sqrt(3/2)) is cut off by three centres at once')

# --------------------------------------------------------------------------
# (E) the count criterion and the second distance criterion Psi
# --------------------------------------------------------------------------
R2 = arb(3) / 2
Rr = R2.sqrt()


def antider(t):
    """an antiderivative of (R^2 - t^2)^(3/2) on [0, R]"""
    s = (R2 - t * t)
    s = s.sqrt() if s > 0 else arb(0)
    return t / 8 * (5 * R2 - 2 * t * t) * s + 3 * R2 * R2 / 8 * (t / Rr).asin()


def S(d):
    """volume of {x in B(sqrt(3/2)) : <x, u> > d/2}, for 2 <= d <= sqrt 6"""
    h = arb(d) / 2
    if h >= Rr:
        return arb(0)
    return 4 * PI / 3 * (3 * R2 * R2 / 8 * PI / 2 - antider(h))


VB = 9 * PI * PI / 8
S2 = S(2)
bound22 = VB - 22 * S2
A_star = VB - 23 * S2
check('(E) count criterion: at most 22 centres within sqrt 6 give vol(V_c) > 8.046',
      bound22.lower() > arb('8.046'),
      'vol B(sqrt(3/2)) = 9 pi^2/8 = %s,  S(2) = %s\n'
      '9 pi^2/8 - 22 S(2) = %s\n'
      '9 pi^2/8 - 23 S(2) = %s  (= A_* of thm:certificate, which the pair terms lift above 8)'
      % (VB.str(12, radius=False), S2.str(12, radius=False), bound22.str(12, radius=False),
         A_star.str(12, radius=False)))


def psi_threshold(m, k):
    """least d (to 1e-6) with 9pi^2/8 - m S(2) - k S(d) > 8, or None"""
    need = VB - m * S2 - 8           # k S(d) must be below this
    if need.upper() <= 0:
        return None
    lo, hi = Fraction(2), Fraction(24495, 10000)
    for _ in range(40):
        mid = (lo + hi) / 2
        val = k * S(arb(mid.numerator) / mid.denominator)
        if (val.upper() < need.lower()):
            hi = mid
        else:
            lo = mid
    return hi


rows = []
for (m, k) in [(22, 1), (21, 2), (20, 3), (19, 4), (18, 6), (22, 2), (21, 3), (12, 12)]:
    t = psi_threshold(m, k)
    rows.append('%2d contacts + %2d at d:  Psi > 8 for d >= %s'
                % (m, k, ('%.5f' % float(t)) if t else 'none below sqrt 6'))
check('(E\') the distance criterion Psi (ball arithmetic)', True, '\n'.join(rows))

# --------------------------------------------------------------------------
# (F) 23 contacts: the covering-radius threshold
# --------------------------------------------------------------------------
c = arb(6).sqrt() / 4
ang = c.acos() * 180 / PI
check('(F) 23 contacts with covering radius <= arccos(sqrt6/4) leave nothing within sqrt 6',
      abs(float(ang) - 52.2388) < 1e-3 and abs(4 * c - arb(6).sqrt()).upper() < arb('1e-40'),
      'arccos(sqrt 6/4) = %s degrees; 4 cos of it = sqrt 6 = 2 sqrt(3/2), the diameter of\n'
      'the ball of thm:certificate, so V_c and V_c(W) agree inside it' % ang.str(8, radius=False))

print()
print('%d checks failed' % len(FAILS) if FAILS else 'all checks passed')
sys.exit(1 if FAILS else 0)
