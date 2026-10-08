#!/usr/bin/env python3
"""
edges.py -- the candidates of Proposition 5.4 and the value of det Q at each, in
exact arithmetic.

Reads the output of ryshkov2.py.  The candidates are

  (a) the vertices J of R(4) with det J <= 0, and
  (b) the points J + tX with det(J + tX) = 0 on the edges
      {J + tX : 0 <= t <= t*} (or t >= 0 for an unbounded edge) at the vertex
      representatives.

On an edge, p(t) = det(J + tX) is a polynomial of degree at most 5 with
rational coefficients; its real roots in the range of the edge are the points
of type (b).  At each of them g(t) = det Q(J + tX) is compared with 256
exactly: through a common factor of p and g - 256 when they share the root,
and otherwise by isolating the root of p in a rational interval on which
g - 256 has no root (Sturm), and reading the sign at an end point.  If p
vanishes identically on an edge, the whole edge consists of packings and
g - 256 is shown to be positive on it.

A candidate with g = 256 is checked to be a representation of D4: J has rank
four, 2q is integral for q = Q^{-1} r, and then Lambda + Z b is a lattice of
determinant 256 / 4 = 64 and minimum 4, which by Korkine and Zolotarev is
sqrt(2) D4.
"""
import json
import sys

import sympy as sp

t = sp.symbols('t')


def symmat(rows):
    return sp.Matrix([[sp.Rational(x) for x in row] for row in rows])


def sign(x):
    return int(bool(x > 0)) - int(bool(x < 0))


def sign_at_root(fp, gp, lo, hi):
    """sign of g at the unique root of the square-free f in [lo, hi]."""
    h = sp.gcd(fp, gp)
    if h.degree() > 0 and h.count_roots(lo, hi) > 0:
        return 0
    while gp.count_roots(lo, hi) > 0:
        mid = (lo + hi) / 2
        if fp.eval(mid) == 0:
            lo = hi = mid
            break
        if fp.count_roots(lo, mid) > 0:
            hi = mid
        else:
            lo = mid
    return sign(gp.eval(lo))


def is_d4(J):
    """J has rank four, det Q = 256, and 2 Q^{-1} r is integral."""
    Q, r = J[:4, :4], J[:4, 4]
    if J.det() != 0 or Q.det() != 256:
        return False
    q = Q.inv() * r
    return all((2 * x).is_integer for x in q)


def positive_on(gp, lo, hi):
    """g > 0 on [lo, hi] (hi = None: on [lo, oo))."""
    if hi is None:
        if gp.count_roots(lo, None) > 0:
            return False
        lead = gp.LC() if gp.degree() > 0 else gp.eval(0)
        return gp.eval(lo) > 0 and lead > 0
    return gp.count_roots(lo, hi) == 0 and gp.eval(lo) > 0


def main():
    data = json.load(open(sys.argv[1]))
    V = data['vertices']
    cands = []
    for i, v in enumerate(V):
        J = symmat(v['J'])
        if J.det() <= 0:
            g = J[:4, :4].det()
            cands.append({'where': 'vertex %d' % i, 'gram': g, 'cmp256': sign(g - 256),
                          'd4': bool(g == 256 and is_d4(J)), 'detJ': J.det()})
    whole = 0
    for e in data['edges']:
        J = symmat(V[e['from']]['J'])
        X = symmat(e['X'])
        Jt = J + t * X
        p = sp.Poly(sp.expand(Jt.det()), t)
        gp = sp.Poly(sp.expand(Jt[:4, :4].det()), t)
        g256 = gp - 256
        hi = sp.Rational(e['t']) if e['t'] is not None else None
        if p.is_zero:
            whole += 1
            assert positive_on(g256, sp.Integer(0), hi), 'an edge of packings reaches 256'
            continue
        for f, _ in p.factor_list()[1]:
            if f.degree() == 0:
                continue
            for (a, b), _ in f.intervals():
                a, b = sp.Rational(a), sp.Rational(b)
                if b < 0 or (hi is not None and a > hi):
                    continue
                # refine until the interval decides whether the root lies in (0, hi)
                while (a < 0 < b and f.eval(0) != 0) or (hi is not None and a < hi < b and f.eval(hi) != 0):
                    r = f.refine_root(a, b, eps=(b - a) / 4)
                    a, b = sp.Rational(r[0]), sp.Rational(r[1])
                if b < 0 or (hi is not None and a > hi):
                    continue
                if f.eval(0) == 0 and a <= 0 <= b:
                    continue             # the vertex itself, type (a)
                if hi is not None and f.eval(hi) == 0 and a <= hi <= b:
                    continue             # the other end point, a vertex, type (a)
                s = sign_at_root(f, g256, a, b)
                a, b = (sp.Rational(x) for x in f.refine_root(a, b, eps=sp.Rational(1, 10**12)))
                c = {'where': 'edge %d from vertex %d' % (data['edges'].index(e), e['from']),
                     'cmp256': s, 'factor': str(f.as_expr()),
                     'gram_approx': float(gp.eval((a + b) / 2)), 't_approx': float((a + b) / 2)}
                if f.degree() == 1:
                    root = sp.solve(f.as_expr(), t)[0]
                    c['t'] = root
                    c['gram'] = gp.eval(root)
                    c['d4'] = bool(is_d4(J + root * X))
                else:
                    c['d4'] = False
                cands.append(c)
    print('%d candidates; %d edges consist of packings' % (len(cands), whole))
    for c in sorted(cands, key=lambda c: float(c.get('gram', c.get('gram_approx', 0)))):
        val = c['gram'] if 'gram' in c else '%.6f' % c['gram_approx']
        print('  %-26s Gram det %-12s %s%s' % (c['where'], val,
              {1: '> 256', 0: '= 256', -1: '< 256'}[c['cmp256']], '  (D4)' if c['d4'] else ''))
    assert all(c['cmp256'] >= 0 for c in cands), 'a candidate below 256'
    assert all(c['d4'] for c in cands if c['cmp256'] == 0), 'equality at a candidate that is not D4'
    print('every candidate has Gram determinant >= 256, with equality only at representations of D4')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
