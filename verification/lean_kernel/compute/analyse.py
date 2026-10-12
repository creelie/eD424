"""Exact check of det Q >= 256 on the set C (vertices with det J <= 0, and points with det J = 0 on edges).

Reads the JSON written by enumerate.py.  For each edge J + tY, 0 <= t <= t* (or t >= 0 when unbounded),
det J(t) (degree <= 5) and det Q(t) (degree <= 4) are found by exact interpolation; the real roots of det J
in (0, t*) are isolated by sympy with rational intervals, and the sign of det Q - 256 at each root is decided
exactly (gcd for common roots, Sturm counts to separate the others)."""
import sys, json
from fractions import Fraction as F
import sympy as sp
from forms import det, sym_from_vec, blocks

N = 5
t = sp.Symbol('t')

def interp(vals):
    """coefficients (low to high) of the polynomial through (i, vals[i]), i = 0..len-1."""
    pts = [sp.Rational(i) for i in range(len(vals))]
    return sp.Poly(sp.interpolate(list(zip(pts, [sp.Rational(v.numerator, v.denominator) for v in vals])), t), t, domain='QQ')

def edge_polys(J, Y):
    def at(s):
        return [[J[a][b] + s * Y[a][b] for b in range(N)] for a in range(N)]
    dj = [det(at(F(i))) for i in range(N + 1)]
    dq = []
    for i in range(N):
        Q, r, s_ = blocks(at(F(i)))
        dq.append(det(Q))
    return interp(dj), interp(dq)

def roots_info(pJ, pQ, T):
    """list of (status, interval) for the roots of det J in the open edge (0, T); T None means infinity.
    status: 'gt' (det Q > 256 there), 'eq' (= 256), 'lt' (< 256)."""
    if pJ.is_zero:
        # the whole edge lies in {det J = 0}: det Q - 256 must be >= 0 on [0, T]
        g = pQ - 256
        hi = None if T is None else sp.Rational(T.numerator, T.denominator)
        ivs = g.intervals(inf=0, sup=hi) if hi is not None else g.intervals(inf=0)
        pts = [sp.Rational(0)] + ([hi] if hi is not None else [])
        for (a, b), m in ivs:
            pts += [sp.Rational(a), sp.Rational(b)]
        pts = sorted(set(pts))
        mids = [(u + v) / 2 for u, v in zip(pts, pts[1:])]
        if hi is None:
            mids.append(pts[-1] + 1)
        neg = any(g.eval(x) < 0 for x in pts + mids)
        return [('zero-edge-lt' if neg else 'zero-edge-ok', None)]
    p = pJ.sqf_part()
    g = pQ - 256
    hi = None if T is None else sp.Rational(T.numerator, T.denominator)
    out = []
    h = sp.gcd(p, g)
    parts = []
    if h.degree() > 0:
        parts.append(('eq', h))
        p = sp.quo(p, h)
    if p.degree() > 0:
        parts.append(('sep', p))
    for kind, q in parts:
        ivs = q.intervals(inf=0, sup=hi) if hi is not None else q.intervals(inf=0)
        for (a, b), m in ivs:
            a, b = sp.Rational(a), sp.Rational(b)
            # discard the endpoints t = 0 and t = T (vertices, treated separately)
            if a == b and (a == 0 or (hi is not None and a == hi)):
                continue
            if a == 0 and q.eval(0) == 0:
                while a <= 0 < b and q.eval(0) == 0 and a != b:
                    a, b = q.refine_root(a, b, eps=(b - a) / 4)
                    a, b = sp.Rational(a), sp.Rational(b)
                if a <= 0:
                    continue
            if hi is not None and b == hi and q.eval(hi) == 0:
                while a < hi <= b and a != b:
                    a, b = q.refine_root(a, b, eps=(b - a) / 4)
                    a, b = sp.Rational(a), sp.Rational(b)
                if b >= hi:
                    continue
            if kind == 'eq':
                out.append(('eq', (a, b)))
                continue
            while a != b and g.count_roots(a, b) > 0:
                a, b = q.refine_root(a, b, eps=(b - a) / 4)
                a, b = sp.Rational(a), sp.Rational(b)
            v = g.eval(a)
            assert v != 0
            out.append(('gt' if v > 0 else 'lt', (a, b)))
    return out

def main(path):
    data = json.load(open(path))
    seen = {}
    worst = None
    bad = []
    eqs = []
    for V in data:
        J = [[F(x) for x in row] for row in V['J']]
        if F(V['detJ']) <= 0 and F(V['detQ']) < 256:
            bad.append(('vertex', V['index']))
        for e in V['edges']:
            Y = sym_from_vec([F(x) for x in e['ray']], N)
            T = None if e['tstar'] is None else F(e['tstar'])
            pJ, pQ = edge_polys(J, Y)
            key = (tuple(pJ.all_coeffs()), tuple(pQ.all_coeffs()), e['tstar'])
            if key in seen:
                continue
            info = roots_info(pJ, pQ, T)
            seen[key] = info
            for st, iv in info:
                if st in ('lt', 'zero-edge-lt'):
                    bad.append((V['index'], e['ray'], iv))
                if st == 'eq':
                    eqs.append((V['index'], e['ray'], iv))
                if st in ('gt', 'eq', 'lt') and iv is not None:
                    a, b = iv
                    r = (a + b) / 2
                    val = float(pQ.eval(r))
                    if worst is None or val < worst[0]:
                        worst = (val, V['index'], e['tstar'])
        print(f"vertex {V['index']}: {len(V['edges'])} edges, {len(seen)} distinct edge types so far", flush=True)
    print("vertices with det J <= 0:", [(V['index'], V['detQ'], V['detJ']) for V in data if F(V['detJ']) <= 0])
    print("smallest det Q at an interior root of det J:", worst)
    print("equality roots:", len(eqs))
    for x in eqs[:20]:
        print("  ", x)
    print("violations:", len(bad))
    for x in bad[:20]:
        print("  ", x)

if __name__ == '__main__':
    main(sys.argv[1])
