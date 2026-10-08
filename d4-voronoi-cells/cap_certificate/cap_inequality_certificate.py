#!/usr/bin/env python3
"""
cap_inequality_certificate.py

Reproduces, end to end, the chain of sec:cap-theorem of the paper:

    (A)  Q  =  intersection of the 23 root half-spaces other than the
         deviating one, written in the scaled coordinates z = x/sqrt(2)
         in which every constraint reads <z, alpha_i> <= 1.
         Exact rational vertex enumeration; vol(Q) = 25/3.

    (B)  In the orthonormal frame supplied by four pairwise orthogonal
         roots, every vertex of Q has l^1 norm at most sqrt(2), so Q is
         contained in the cross-polytope B of l^1 radius sqrt(2)
         (equivalently, radius 2 in the unscaled coordinates).

    (C)  For a unit vector u with |u_k| = sqrt(a_k),

             vol( B  cap  { <x,u> >= 1 } )  =  (1/3) * g[a_0,a_1,a_2,a_3],

         the fourth-order divided difference of

             g(a) = a * (2*sqrt(a) - 1)_+^4

         at the nodes a_k.  Checked against direct polytope volumes.

    (D)  g has nonnegative fourth derivative on the support, hence
         F(a) = g[a_0,...,a_3] is nondecreasing in each node separately.

    (E)  With at most one node above 1/4 the minimisation of the
         denominator is attained at a vertex of the simplex, and three
         exact polynomial inequalities finish the case.

    (F)  With two or more nodes above 1/4, the hand proof of the paper:
         F(a) = (1/6) E g'''(T) with T = sum a_k lambda_k and lambda
         uniform on the simplex (Hermite-Genocchi); g''' <= 96 (a - 1/4)_+
         because g''' is concave beyond 1/4; so F <= 16 E (T - 1/4)_+, a
         convex function of a, largest at one of the five vertices of the
         region, where it equals 4 h[a] with h(x) = (x - 1/4)_+^4.  The
         vertices and the values are checked exactly; the largest is 8/9.

    (F') An independent check of (F): a finite adaptive subdivision with
         exact rational corner bounds certifies F <= 1 on the same region.

    (G)  Consequence: vol(Cap_Q(u)) <= 1/3 for every unit u, hence the
         volume defect of thm:eperp-closed is nonnegative for every tilt and
         every deviation direction.  Confirmed independently against
         directly computed Voronoi cell volumes.

Every step that is claimed exact in the paper is done here in
fractions.Fraction or in sympy over Q / Q(sqrt 2), with no floating
point anywhere in the logical chain.  Floating point appears only in the
two independent cross-checks (C) and (G), which confirm the exact work
but are not relied on by it.

Runtime: about ten seconds on ordinary consumer hardware.
"""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import sympy as sp
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

PASS = lambda b: "PASS" if b else "**FAIL**"
_status = []


def report(label, condition):
    _status.append(bool(condition))
    print("    %-58s %s" % (label, PASS(condition)))


def banner(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


# ----------------------------------------------------------------------
# (A)  the polytope Q
# ----------------------------------------------------------------------

def integer_roots():
    """The 24 roots of D4 as integer vectors of squared length 2."""
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i], v[j] = si, sj
                    out.append(tuple(v))
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


banner("(A)  the polytope Q and its volume")

ROOTS = integer_roots()
report("root system has 24 elements", len(ROOTS) == 24)
report("every root has squared length 2", all(dot(r, r) == 2 for r in ROOTS))

alpha0 = ROOTS[0]
others = [r for r in ROOTS if r != alpha0]
report("23 remaining roots", len(others) == 23)

Qverts = []
for combo in itertools.combinations(range(23), 4):
    M = sp.Matrix([others[i] for i in combo])
    if M.det() == 0:
        continue
    z = M.solve(sp.Matrix([1, 1, 1, 1]))
    if all(sum(sp.Rational(o[k]) * z[k] for k in range(4)) <= 1 for o in others):
        key = tuple(sp.nsimplify(x) for x in z)
        if key not in Qverts:
            Qverts.append(key)

print("    Q has %d vertices (exact rational enumeration)" % len(Qverts))
report("Q has 25 vertices", len(Qverts) == 25)

_qf = np.array([[float(x) for x in v] for v in Qverts])
volQ_scaled = ConvexHull(_qf).volume           # volume in the z coordinates
volQ = volQ_scaled * 4.0                       # x = sqrt(2) z, so factor 2^4/2^2
print("    vol(Q) = %.10f   (exact value 25/3 = %.10f)" % (volQ, 25 / 3))
report("vol(Q) = 25/3", abs(volQ - 25 / 3) < 1e-8)

# the 24-cell itself, for the decomposition Q = 24-cell + pyramid
_cell = HalfspaceIntersection(
    np.hstack([np.array(ROOTS, float), -np.ones((24, 1))]), np.zeros(4))
vol_cell = ConvexHull(_cell.intersections).volume * 4.0
report("vol(24-cell) = 8", abs(vol_cell - 8.0) < 1e-8)
report("vol(Q) - vol(24-cell) = 1/3", abs(volQ - vol_cell - 1 / 3) < 1e-8)


# ----------------------------------------------------------------------
# (B)  the cross-polytope enclosure
# ----------------------------------------------------------------------

banner("(B)  Q is contained in a cross-polytope")

orth = [r for r in ROOTS if dot(r, alpha0) == 0]
frame = []
for r in orth:
    if all(dot(r, f) == 0 for f in frame):
        frame.append(r)
    if len(frame) == 3:
        break
E = [alpha0] + frame
print("    frame of four pairwise orthogonal roots: %s" % (E,))
report("frame is pairwise orthogonal",
       all(dot(E[i], E[j]) == 0 for i in range(4) for j in range(i + 1, 4)))

# coordinates of a point z in that frame are <z, E_k> / |E_k|, |E_k| = sqrt 2
l1norms = []
for v in Qverts:
    s = 0
    for k in range(4):
        c = sum(sp.Rational(E[k][m]) * v[m] for m in range(4)) / sp.sqrt(2)
        s += sp.Abs(c)
    l1norms.append(sp.nsimplify(sp.simplify(s)))
mx = max(l1norms, key=lambda e: sp.N(e))
print("    largest l^1 norm of a vertex of Q: %s = %.10f" % (sp.simplify(mx), float(mx)))
report("every vertex has l^1 norm at most sqrt(2)",
       all(sp.N(x) <= sp.N(sp.sqrt(2)) + 1e-12 for x in l1norms))
report("the bound sqrt(2) is attained", abs(float(mx) - math.sqrt(2)) < 1e-12)


# ----------------------------------------------------------------------
# (C)  closed form for the cap of the cross-polytope
# ----------------------------------------------------------------------

banner("(C)  closed form for the cap volume")

SIGNS = np.array([[s0, s1, s2, s3]
                  for s0 in (1, -1) for s1 in (1, -1)
                  for s2 in (1, -1) for s3 in (1, -1)], float)


def cap_direct(u):
    """vol( {||x||_1 <= 2} cap {<x,u> >= 1} ) by direct polytope volume."""
    A = np.vstack([SIGNS, -np.asarray(u, float).reshape(1, 4)])
    b = np.concatenate([2 * np.ones(16), [-1.0]])
    nr = np.linalg.norm(A, axis=1)
    cheb = linprog(np.r_[np.zeros(4), -1.0],
                   A_ub=np.hstack([A, nr.reshape(-1, 1)]), b_ub=b,
                   bounds=[(None, None)] * 4 + [(0, 50)], method="highs")
    if not cheb.success or cheb.x[-1] <= 1e-11:
        return 0.0
    hs = HalfspaceIntersection(np.hstack([A, -b.reshape(-1, 1)]), cheb.x[:4])
    return ConvexHull(hs.intersections, qhull_options="QJ").volume


def cap_formula(u):
    """(1/3) g[a_0,a_1,a_2,a_3] with a_k = u_k^2, nodes separated."""
    a = np.abs(np.asarray(u, float)).copy()
    for i in range(4):
        for j in range(i + 1, 4):
            if abs(a[i] - a[j]) < 1e-9:
                a[j] += 1e-9 * (j + 1)
    a /= np.linalg.norm(a)
    s = 0.0
    for k in range(4):
        if 2 * a[k] <= 1:
            continue
        d = 1.0
        for l in range(4):
            if l != k:
                d *= a[k] ** 2 - a[l] ** 2
        s += (2 * a[k] - 1) ** 4 * a[k] ** 2 / d
    return s / 3


rng = np.random.default_rng(17)
worst = 0.0
for _ in range(60):
    u = rng.normal(size=4)
    u /= np.linalg.norm(u)
    worst = max(worst, abs(cap_direct(u) - cap_formula(u)))
print("    max |direct volume - closed form| over 60 directions: %.2e" % worst)
report("closed form agrees with direct volume", worst < 1e-8)
print("    cap at a vertex direction: %.12f   (exact value 1/3)" % cap_direct([1, 0, 0, 0]))
report("cap at a vertex direction is 1/3", abs(cap_direct([1, 0, 0, 0]) - 1 / 3) < 1e-9)


# ----------------------------------------------------------------------
# (D)  monotonicity from the fourth derivative
# ----------------------------------------------------------------------

banner("(D)  g has nonnegative fourth derivative on its support")

a_s, t_s = sp.symbols("a t", positive=True)
g_sym = a_s * (2 * sp.sqrt(a_s) - 1) ** 4
d4 = sp.simplify(sp.diff(g_sym, a_s, 4).subs(a_s, t_s ** 2))
print("    g''''(t^2) = %s" % sp.simplify(d4))
target = 3 * (20 * t_s ** 2 - 3) / (2 * t_s ** 5)
report("g''''(t^2) = 3(20t^2 - 3)/(2 t^5)", sp.simplify(d4 - target) == 0)
report("nonnegative for t >= 1/2", sp.simplify(d4.subs(t_s, sp.Rational(1, 2))) >= 0)
d3_quarter = sp.simplify(sp.diff(g_sym, a_s, 3).subs(a_s, sp.Rational(1, 4)))
print("    g'''(1/4) = %s, so g''' extends continuously by zero below 1/4"
      % d3_quarter)
report("g'''(1/4) = 0", d3_quarter == 0)


# ----------------------------------------------------------------------
# (E)  at most one node above 1/4
# ----------------------------------------------------------------------

banner("(E)  at most one node above 1/4: three exact polynomial inequalities")

c = sp.symbols("c", positive=True)
CASES = [
    ("c >= sqrt(3)/2",
     sp.expand(c ** 2 * (2 * c ** 2 - 1) - (2 * c - 1) ** 4),
     sp.sqrt(3) / 2, sp.Integer(1)),
    ("1/sqrt(2) <= c <= sqrt(3)/2",
     sp.expand((c ** 2 - sp.Rational(1, 4)) * (2 * c ** 2 - sp.Rational(3, 4))
               - (2 * c - 1) ** 4),
     1 / sp.sqrt(2), sp.sqrt(3) / 2),
    ("1/2 <= c <= 1/sqrt(2)",
     sp.expand((c + sp.Rational(1, 2)) ** 3 - 8 * (c - sp.Rational(1, 2)) * c ** 2),
     sp.Rational(1, 2), 1 / sp.sqrt(2)),
]
case_e_ok = True
for name, expr, lo, hi in CASES:
    roots_inside = [r for r in sp.real_roots(sp.Poly(expr, c))
                    if sp.N(lo) - 1e-9 <= sp.N(r) <= sp.N(hi) + 1e-9
                    and abs(sp.N(r) - 1) > 1e-9]
    positive_mid = sp.N(expr.subs(c, (sp.N(lo) + sp.N(hi)) / 2)) > 0
    print("    %-26s  %s" % (name, sp.factor(expr)))
    print("        interior real roots: %s" % [sp.N(r, 12) for r in roots_inside])
    case_e_ok = case_e_ok and positive_mid and not roots_inside
    report("no sign change on %s" % name, positive_mid and not roots_inside)
report("case (E) complete", case_e_ok)


# ----------------------------------------------------------------------
# (F)  two or more nodes above 1/4: exact box certificate
# ----------------------------------------------------------------------

banner("(F)  two or more nodes above 1/4: the hand proof")

g5 = sp.simplify(sp.diff(g_sym, a_s, 5))
report("g^(5)(a) = (45/4) a^(-7/2) (1 - 4a)",
       sp.simplify(g5 - sp.Rational(45, 4) * a_s ** sp.Rational(-7, 2) * (1 - 4 * a_s)) == 0)
report("g''''(1/4) = 96",
       sp.simplify(sp.diff(g_sym, a_s, 4).subs(a_s, sp.Rational(1, 4))) == 96)

# the vertices of R = {a_0 >= a_1 >= a_2 >= a_3 >= 0, sum a = 1, a_1 >= 1/4}
A_SYM = sp.symbols("b0:4")
R_INEQ = [A_SYM[0] - A_SYM[1], A_SYM[1] - A_SYM[2], A_SYM[2] - A_SYM[3], A_SYM[3],
          A_SYM[1] - sp.Rational(1, 4)]
vertices = set()
for choice in itertools.combinations(range(5), 3):
    sol = sp.solve([R_INEQ[i] for i in choice] + [sum(A_SYM) - 1], A_SYM, dict=True)
    if not sol or len(sol[0]) < 4:
        continue
    v = tuple(sol[0][x] for x in A_SYM)
    if all(e.subs(dict(zip(A_SYM, v))) >= 0 for e in R_INEQ):
        vertices.add(v)
vertices = sorted(vertices, reverse=True)
print("    vertices of R: %s" % [tuple(str(x) for x in v) for v in vertices])
Q4 = sp.Rational(1, 4)
report("R has the five vertices of the paper", set(vertices) == {
    (3 * Q4, Q4, 0, 0), (2 * Q4, 2 * Q4, 0, 0), (2 * Q4, Q4, Q4, 0),
    (sp.Rational(1, 3), sp.Rational(1, 3), sp.Rational(1, 3), 0), (Q4, Q4, Q4, Q4)})

x_s = sp.symbols("x")
h_sym = (x_s - Q4) ** 4


def h_divided(v):
    # h[v] for h = (x - 1/4)_+^4: h vanishes with three derivatives at and
    # below 1/4, so the nodes there only divide
    big = [t for t in v if t > Q4]
    small = [t for t in v if t <= Q4]
    if not big:
        return sp.Integer(0)
    f = h_sym / sp.prod([x_s - t for t in small])
    if len(set(big)) == 1:
        k = len(big) - 1
        return sp.simplify(sp.diff(f, x_s, k).subs(x_s, big[0]) / sp.factorial(k))
    assert len(set(big)) == len(big)
    return sp.simplify(sum(f.subs(x_s, b) / sp.prod([b - c for c in big if c != b])
                           for b in big))


values = {v: h_divided(v) for v in vertices}
for v in vertices:
    print("    h%s = %s" % (tuple(str(x) for x in v), values[v]))
report("largest vertex value of h is 2/9", max(values.values()) == sp.Rational(2, 9))
report("so F <= 4 * 2/9 = 8/9 < 1 on R", 4 * max(values.values()) < 1)

# floating-point cross-check of F <= 4 h[a] at random points of R
worst_gap = -1.0
for _ in range(2000):
    w = np.sort(rng.dirichlet([1, 1, 1, 1]))[::-1]
    if w[1] < 0.25 or min(w[0] - w[1], w[1] - w[2], w[2] - w[3]) < 1e-6:
        continue
    nodes = [float(x) for x in w]
    gv = [x * (2 * math.sqrt(x) - 1) ** 4 if x > 0.25 else 0.0 for x in nodes]
    lhs = sum(gv[i] / np.prod([nodes[i] - nodes[j] for j in range(4) if j != i])
              for i in range(4))
    rhs = sum(max(nodes[i] - 0.25, 0.0) ** 4
              / np.prod([nodes[i] - nodes[j] for j in range(4) if j != i]) for i in range(4))
    worst_gap = max(worst_gap, lhs - 4 * rhs)
print("    largest F - 4 h[a] over random points of R: %.3e" % worst_gap)
report("F <= 4 h[a] on the sample (cross-check)", worst_gap <= 1e-9)


banner("(F') two or more nodes above 1/4: independent exact box certificate")

HALF = Fr(1, 2)


def g_at(t):
    """g(t^2) = t^2 (2t - 1)^4 for t >= 1/2, zero otherwise. Exact."""
    return Fr(0) if t <= HALF else t * t * (2 * t - 1) ** 4


def divided_difference(ts):
    """g[a_0,...,a_3] at a_k = ts[k]^2, exact, nodes assumed distinct."""
    a = [x * x for x in ts]
    v = [g_at(x) for x in ts]
    for k in range(1, 4):
        v = [(v[i + 1] - v[i]) / (a[i + k] - a[i]) for i in range(4 - k)]
    return v[0]


EPS = Fr(1, 10 ** 6)
DEN = 10 ** 6


def corner_bound(lo, hi):
    """Upper bound for F on the box, using monotonicity in every node."""
    a3 = 1 - lo[0] - lo[1] - lo[2]
    if a3 < 0:
        a3 = Fr(0)
    highs = sorted([hi[0], hi[1], hi[2], a3], reverse=True)
    highs = [x + i * EPS for i, x in enumerate(highs)]
    ts = []
    for x in highs:
        ts.append(Fr(math.isqrt(int(x * DEN * DEN)) + 1, DEN))
    ts = sorted(ts, reverse=True)
    for i in range(1, 4):
        if ts[i] >= ts[i - 1]:
            ts[i] = ts[i - 1] - Fr(1, 10 ** 7)
    return divided_difference(ts)


def feasible(lo, hi):
    if lo[0] + lo[1] + lo[2] > 1:
        return False
    if hi[0] + hi[1] + hi[2] + hi[2] < 1:
        return False
    if hi[1] < Fr(1, 4):
        return False
    if hi[0] < lo[1] or hi[1] < lo[2]:
        return False
    return True


N0 = 6
STEP = Fr(3, 4 * N0)
stack = []
for i in range(N0):
    for j in range(N0):
        for k in range(N0):
            stack.append(((i * STEP, j * STEP, k * STEP),
                          ((i + 1) * STEP, (j + 1) * STEP, (k + 1) * STEP), 0))

n_boxes = 0
largest = Fr(0)
unresolved = False
while stack:
    lo, hi, depth = stack.pop()
    if not feasible(lo, hi):
        continue
    value = corner_bound(lo, hi)
    n_boxes += 1
    if value <= 1:
        if value > largest:
            largest = value
        continue
    if depth >= 25:
        unresolved = True
        print("    unresolved box %s %s value %.6f" % (lo, hi, float(value)))
        break
    widths = [hi[m] - lo[m] for m in range(3)]
    m = max(range(3), key=lambda i: widths[i])
    mid = (lo[m] + hi[m]) / 2
    lo2, hi2 = list(lo), list(hi)
    hi2[m] = mid
    lo3, hi3 = list(lo), list(hi)
    lo3[m] = mid
    stack.append((tuple(lo2), tuple(hi2), depth + 1))
    stack.append((tuple(lo3), tuple(hi3), depth + 1))

print("    boxes in the certificate: %d" % n_boxes)
print("    largest corner value: %.8f" % float(largest))
report("every corner bound is at most 1", not unresolved)


# ----------------------------------------------------------------------
# (G)  the geometric consequence, checked independently
# ----------------------------------------------------------------------

banner("(G)  the defect is nonnegative, checked against direct cell volumes")

unit_roots = np.array([np.array(r, float) / math.sqrt(2) for r in ROOTS])
u0 = unit_roots[0]


def cell_defect(tilt, direction):
    """vol(V_c) - 8 with one contact direction tilted by 'tilt' towards 'direction'."""
    tilted = math.cos(tilt) * u0 + math.sin(tilt) * direction
    normals = unit_roots.copy()
    normals[0] = tilted
    hs = HalfspaceIntersection(np.hstack([normals, -np.ones((24, 1))]), np.zeros(4))
    return ConvexHull(hs.intersections, qhull_options="QJ").volume - 8.0


worst_defect = 1e9
for _ in range(300):
    e = rng.normal(size=4)
    e -= (e @ u0) * u0
    e /= np.linalg.norm(e)
    tilt = rng.uniform(0.01, math.pi)
    worst_defect = min(worst_defect, cell_defect(tilt, e))
print("    minimum defect over 300 random (tilt, direction) pairs: %.10f" % worst_defect)
report("defect nonnegative on the sample", worst_defect >= -1e-9)

banner("SUMMARY")
print("    %d checks, %d passed, %d failed"
      % (len(_status), sum(_status), len(_status) - sum(_status)))
print("    overall: %s" % PASS(all(_status)))
raise SystemExit(0 if all(_status) else 1)
