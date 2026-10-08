#!/usr/bin/env python3
"""
hexagon_loop.py -- a closed curve of direction sets through the root system,
not rotations of it, along which the contact cell {x : <x, w_i> <= 1} has
volume exactly 8 (prop:hexagon-loop).  Exact (sympy) with numerical
cross-checks (qhull, and 50-digit volumes).

Coordinates: R^4 = P + P', P the plane of an A2 hexagon of unit roots, e(a) =
(cos a, sin a).  The 24 unit roots of D_4 are, up to rotation,

    (e(60m), 0),  m = 0..5,
    (s e(a), c e(b)),  (a, b) in {30,150,270} x {0,120,240}
                                or {90,210,330} x {60,180,300}    (degrees),

with s = sin(psi) = 1/sqrt3, c = cos(psi).  The family w(theta, psi) turns the
hexagon to (e(theta + 60m), 0) and changes psi.  For p in P, the six
hexagon constraints say p lies in the hexagon Hex_theta of inradius 1, and
the slice of the cell over p is the hexagon in P' with normals e(60k) and
support numbers h_+ = (1 - s m_+(p))/c (k even), h_- = (1 - s m_-(p))/c
(k odd), m_+ and m_- the support functions of the triangles {e(30), e(150),
e(270)} and {e(90), e(210), e(330)}.  Its area is
sqrt3 (4 h_+ h_- - h_+^2 - h_-^2) when h_+ <= 2 h_- and h_- <= 2 h_+, and exceeds
that by sqrt3 (h_+ - 2 h_-)^2 otherwise.  Integrating over Hex_theta,

    vol >= (12 - 2 sqrt3 s J1 + sqrt3 s^2 J2) / (1 - s^2),
    J1 = int (m_+ + m_-),   J2 = int (4 m_+ m_- - m_+^2 - m_-^2),

with equality when no slice degenerates.  The script computes J1 and J2
exactly (six congruent sectors, two triangles each), finds, with C =
cos(pi/6 - theta),

    J1 = sqrt3 (4C^2 + 3) / (3C),   J2 = sqrt3 (16C^4 - 8C^2 + 9) / (12 C^2),

so that vol - 8 >= (2 - s (4C^2 + 3)/(2C))^2 / (1 - s^2), and checks that at
s = 4C/(4C^2 + 3) no slice degenerates, so vol = 8 there.  At theta = 0 and
pi/3 the family is the root system; for 0 < theta < pi/3 its largest inner
product is 4C^2/(4C^2 + 3) > 1/2.
Usage: python3 hexagon_loop.py
"""
import itertools
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.spatial import HalfspaceIntersection, ConvexHull

S3 = sp.sqrt(3)
A_PLUS, A_MINUS = (30, 150, 270), (90, 210, 330)
B_PLUS, B_MINUS = (0, 120, 240), (60, 180, 300)


def config(theta, psi):
    """The 24 directions w(theta, psi), floating point, coordinates (P, P')."""
    w = [[np.cos(theta + m * np.pi / 3), np.sin(theta + m * np.pi / 3), 0, 0] for m in range(6)]
    for A, B in ((A_PLUS, B_PLUS), (A_MINUS, B_MINUS)):
        for a, b in itertools.product(np.radians(A), np.radians(B)):
            w.append([np.sin(psi) * np.cos(a), np.sin(psi) * np.sin(a), np.cos(psi) * np.cos(b), np.cos(psi) * np.sin(b)])
    return np.array(w)


def sigma_star(theta):
    C = np.cos(np.pi / 6 - theta)
    return 4 * C / (4 * C * C + 3)


def vol_qhull(w):
    P = HalfspaceIntersection(np.hstack([w, -np.ones((len(w), 1))]), np.zeros(4)).intersections
    for opt in ('Qt', 'Qt Q12', 'QJ'):
        try:
            return ConvexHull(P, qhull_options=opt).volume
        except Exception:
            pass
    raise RuntimeError('qhull')


def exact_part():
    print('1. Exact part (sympy)')
    # (a) at theta = 0, psi = asin(1/sqrt3) the family is the root system: equal Gram matrices
    e = lambda deg: sp.Matrix([sp.cos(sp.rad(deg)), sp.sin(sp.rad(deg))])
    s2 = sp.Rational(1, 3)
    pts = [('h', 60 * m, None) for m in range(6)]
    pts += [('o', a, b) for A, B in ((A_PLUS, B_PLUS), (A_MINUS, B_MINUS)) for a, b in itertools.product(A, B)]
    def ip(x, y):
        if x[0] == 'h' and y[0] == 'h':
            return (e(x[1]).T * e(y[1]))[0]
        if x[0] == 'h' or y[0] == 'h':
            h, o = (x, y) if x[0] == 'h' else (y, x)
            return sp.sqrt(s2) * (e(h[1]).T * e(o[1]))[0]
        return s2 * (e(x[1]).T * e(y[1]))[0] + (1 - s2) * (e(x[2]).T * e(y[2]))[0]
    G = sp.Matrix(24, 24, lambda i, j: sp.nsimplify(sp.simplify(ip(pts[i], pts[j]))))
    roots = [np.eye(4)[i] * a + np.eye(4)[j] * b for i in range(4) for j in range(i + 1, 4) for a in (1, -1) for b in (1, -1)]
    G0 = sp.Matrix(24, 24, lambda i, j: sp.Rational(int(roots[i] @ roots[j]), 2))
    Gf = np.array(G, dtype=float); G0f = np.array(G0, dtype=float)
    # labelling: match rows greedily by backtracking on Gram entries
    perm = [None] * 24
    def extend(k):
        if k == 24:
            return True
        for j in range(24):
            if j in perm[:k]:
                continue
            if all(abs(Gf[k, i] - G0f[j, perm[i]]) < 1e-12 for i in range(k)):
                perm[k] = j
                if extend(k + 1):
                    return True
        perm[k] = None
        return False
    assert extend(0)
    same = all(G[i, j] == G0[perm[i], perm[j]] for i in range(24) for j in range(24))
    print('   theta = 0: Gram matrix equals that of the unit D4 roots under a labelling, exactly:', same)
    assert same

    # (b) J1, J2 over one sector (angles 30..90 deg), two triangles, times 6
    c, s = sp.symbols('c s', real=True)          # cos theta, sin theta
    h = sp.Rational(1, 2)
    a30, a90 = sp.Matrix([S3 / 2, h]), sp.Matrix([0, 1])
    u = S3 / 2 * c + h * s                        # cos(pi/6 - theta) = C
    B1, B2 = a30 / u, a90 / u                     # where the rays at 30 and 90 deg leave Hex_theta
    V = sp.Matrix([c - s / S3, s + c / S3])       # the vertex (2/sqrt3) e(theta + 30)
    O = sp.zeros(2, 1)
    sarea = lambda P, Q, R: h * ((Q - P)[0] * (R - P)[1] - (Q - P)[1] * (R - P)[0])
    lin = lambda L, T: sarea(*T) / 3 * sum((L.T * v)[0] for v in T)
    def prod(L, M, T):
        Lv = [(L.T * v)[0] for v in T]; Mv = [(M.T * v)[0] for v in T]
        return sarea(*T) / 12 * (sum(x * y for x, y in zip(Lv, Mv)) + sum(Lv) * sum(Mv))
    tris = [(O, B1, V), (O, V, B2)]               # on the sector m_+ = <p, e(30)>, m_- = <p, e(90)>
    J1 = 6 * sum(lin(a30 + a90, T) for T in tris)
    J2 = 6 * sum(4 * prod(a30, a90, T) - prod(a30, a30, T) - prod(a90, a90, T) for T in tris)
    circle = [c ** 2 + s ** 2 - 1]
    def zero_mod_circle(expr):
        n, _ = sp.fraction(sp.together(sp.expand(expr)))
        return sp.reduced(sp.expand(n), circle, c, s)[1] == 0
    C = u
    ok1 = zero_mod_circle(J1 - S3 * (4 * C ** 2 + 3) / (3 * C))
    ok2 = zero_mod_circle(J2 - S3 * (16 * C ** 4 - 8 * C ** 2 + 9) / (12 * C ** 2))
    print('   J1 = sqrt3 (4C^2+3)/(3C):', ok1, '   J2 = sqrt3 (16C^4-8C^2+9)/(12C^2):', ok2)
    assert ok1 and ok2
    # (c) the perfect square
    sg = sp.symbols('sigma')
    J1c = S3 * (4 * C ** 2 + 3) / (3 * C); J2c = S3 * (16 * C ** 4 - 8 * C ** 2 + 9) / (12 * C ** 2)
    sq = sp.simplify(12 - 2 * S3 * sg * J1c + S3 * sg ** 2 * J2c - 8 * (1 - sg ** 2) - (2 - sg * (4 * C ** 2 + 3) / (2 * C)) ** 2)
    print('   12 - 2 sqrt3 s J1 + sqrt3 s^2 J2 - 8 (1 - s^2) = (2 - s (4C^2+3)/(2C))^2:', sq == 0)
    assert sq == 0
    # (d) no slice degenerates at s = 4C/(4C^2+3): the conditions are linear on each triangle
    sig = 4 * C / (4 * C ** 2 + 3)
    mp_ = lambda p: (a30.T * p)[0]; mm = lambda p: (a90.T * p)[0]
    Cs, Ss = sp.symbols('C S', real=True)         # cos phi, sin phi, phi = pi/6 - theta in [-pi/6, pi/6]
    to_phi = {c: S3 / 2 * Cs + Ss / 2, s: Cs / 2 - S3 / 2 * Ss}
    def in_phi(expr):
        n, d = sp.fraction(sp.together(sp.expand(expr.subs(to_phi, simultaneous=True))))
        n = sp.reduced(sp.expand(n), [Cs ** 2 + Ss ** 2 - 1], Ss, Cs)[1]
        d = sp.reduced(sp.expand(d), [Cs ** 2 + Ss ** 2 - 1], Ss, Cs)[1]
        return sp.factor(sp.simplify(n / d), extension=S3)
    conds = {}
    for name, p in (('B1', B1), ('V', V), ('B2', B2)):
        conds[name + ', 1 - s (2 m_+ - m_-)'] = in_phi(1 - sig * (2 * mp_(p) - mm(p)))
        conds[name + ', 1 - s (2 m_- - m_+)'] = in_phi(1 - sig * (2 * mm(p) - mp_(p)))
    for k, v in conds.items():
        print('   %-24s = %s' % (k, v))
    # each is 1, (C^2 - 3/4)/(C^2 + 3/4) or sqrt3 (sqrt3/4 -+ C S)/(C^2 + 3/4); C^2 >= 3/4 and |2 C S| = |sin 2 phi| <= sqrt3/2
    allowed = {sp.Integer(1), sp.factor((Cs ** 2 - sp.Rational(3, 4)) / (Cs ** 2 + sp.Rational(3, 4)), extension=S3),
               sp.factor(S3 * (S3 / 4 - Cs * Ss) / (Cs ** 2 + sp.Rational(3, 4)), extension=S3),
               sp.factor(S3 * (S3 / 4 + Cs * Ss) / (Cs ** 2 + sp.Rational(3, 4)), extension=S3)}
    good = all(any(sp.simplify(v - a) == 0 for a in allowed) for v in conds.values())
    print('   each condition is 1, (C^2-3/4)/(C^2+3/4) or sqrt3 (sqrt3/4 -+ CS)/(C^2+3/4), all >= 0 on the loop:', good)
    assert good
    # (e) largest inner product on the loop: hexagon root e(theta) against (s e(30), c e(0))
    top = sp.simplify(sig * C - 4 * C ** 2 / (4 * C ** 2 + 3))
    print('   s C = 4C^2/(4C^2+3) (> 1/2 iff C^2 > 3/4, i.e. 0 < theta < pi/3):', top == 0)
    assert top == 0


def numerical_part():
    print('2. Numerical cross-checks')
    rng = np.random.default_rng(7)
    worst = 0.0
    for theta in np.linspace(0, np.pi / 3, 13):
        worst = max(worst, abs(vol_qhull(config(theta, np.arcsin(sigma_star(theta)))) - 8))
    print('   qhull along the loop, 13 values of theta: max |vol - 8| = %.1e' % worst)
    # the lower bound on the two-parameter family, against qhull
    def bound(theta, psi):
        C = np.cos(np.pi / 6 - theta); s = np.sin(psi)
        J1 = np.sqrt(3) * (4 * C * C + 3) / (3 * C); J2 = np.sqrt(3) * (16 * C ** 4 - 8 * C * C + 9) / (12 * C * C)
        return (12 - 2 * np.sqrt(3) * s * J1 + np.sqrt(3) * s * s * J2) / (1 - s * s)
    gaps = []
    for _ in range(200):
        theta = rng.uniform(0, np.pi / 3); psi = np.arcsin(sigma_star(theta)) + rng.uniform(-0.15, 0.15)
        gaps.append(vol_qhull(config(theta, psi)) - bound(theta, psi))
    gaps = np.array(gaps)
    print('   200 random (theta, psi) near the loop: qhull volume minus the bound in [%.1e, %.1e]' % (gaps.min(), gaps.max()))
    print('   (zero where no slice degenerates, positive otherwise)')
    # 50-digit volumes at three points of the loop
    mp.mp.dps = 50
    for theta in (mp.mpf('0.1'), mp.pi / 6, mp.mpf('0.9')):
        print('   theta = %s: vol - 8 = %s at 50 digits' % (mp.nstr(theta, 6), mp.nstr(vol_mp(theta), 3)))


def vol_mp(theta):
    """Volume of the contact cell at a point of the loop, in mpmath: vertices from the
    tight facets found by qhull, then a cone decomposition over the facets."""
    C = mp.cos(mp.pi / 6 - theta); sg = 4 * C / (4 * C * C + 3); cg = mp.sqrt(1 - sg * sg)
    w = [mp.matrix([mp.cos(theta + m * mp.pi / 3), mp.sin(theta + m * mp.pi / 3), 0, 0]) for m in range(6)]
    for A, B in ((A_PLUS, B_PLUS), (A_MINUS, B_MINUS)):
        for a, b in itertools.product(A, B):
            a, b = mp.radians(a), mp.radians(b)
            w.append(mp.matrix([sg * mp.cos(a), sg * mp.sin(a), cg * mp.cos(b), cg * mp.sin(b)]))
    wf = np.array([[float(x) for x in v] for v in w])
    Vf = np.unique(np.round(HalfspaceIntersection(np.hstack([wf, -np.ones((24, 1))]), np.zeros(4)).intersections, 9), axis=0)
    Vm, tight = [], []
    for v in Vf:
        T = [i for i in range(24) if abs(wf[i] @ v - 1) < 1e-7]
        for S in itertools.combinations(T, 4):
            M = mp.matrix([[w[i][k] for k in range(4)] for i in S])
            if abs(mp.det(M)) > mp.mpf(10) ** -20:
                x = mp.lu_solve(M, mp.matrix([1, 1, 1, 1]))
                break
        assert max(abs((w[i].T * x)[0] - 1) for i in T) < mp.mpf(10) ** -40
        Vm.append(x); tight.append(T)
    total = mp.mpf(0)
    for i in range(24):
        idx = [n for n in range(len(Vm)) if i in tight[n]]
        B = np.linalg.svd(np.eye(4) - np.outer(wf[i], wf[i]))[0][:, :3]
        hull = ConvexHull((Vf[idx] - wf[i]) @ B, qhull_options='Qt')
        for tri in hull.simplices:
            if 0 in tri:
                continue
            try:
                total += abs(mp.det(mp.matrix([[Vm[idx[q]][k] for k in range(4)] for q in (0, *tri)])))
            except (TypeError, ZeroDivisionError):
                pass              # an exactly flat tetrahedron (coplanar vertices at theta = pi/6)
    return total / 24 - 8


if __name__ == '__main__':
    exact_part()
    numerical_part()
    print('PASS')
