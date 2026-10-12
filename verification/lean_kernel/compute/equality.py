"""The equality points of C: vertices with det J = 0 and det Q = 256, and points with det J = 0 = det Q - 256
on edges.  For each: the exact point, its Q, the squared covering radius of the lattice Q and its deep holes."""
import sys, json
from fractions import Fraction as F
import sympy as sp
from forms import sym_from_vec, blocks, det, min_vectors
from analyse import edge_polys, t
from deepholes import deep_holes

N = 5

def equality_points(data):
    pts = []
    for V in data:
        J = [[F(x) for x in row] for row in V['J']]
        if F(V['detJ']) == 0 and F(V['detQ']) == 256:
            pts.append({'from': V['index'], 'ray': None, 't': '0', 'J': J})
        for e in V['edges']:
            Y = sym_from_vec([F(x) for x in e['ray']], N)
            T = None if e['tstar'] is None else F(e['tstar'])
            pJ, pQ = edge_polys(J, Y)
            if pJ.is_zero:
                continue
            h = sp.gcd(pJ.sqf_part(), pQ - 256)
            if h.degree() <= 0:
                continue
            for r in sp.real_roots(h):
                if not r.is_rational:
                    raise RuntimeError("irrational equality root")
                tr = F(int(r.p), int(r.q))
                if tr <= 0 or (T is not None and tr >= T):
                    continue
                M = [[J[a][b] + tr * Y[a][b] for b in range(N)] for a in range(N)]
                pts.append({'from': V['index'], 'ray': e['ray'], 'tstar': e['tstar'], 't': str(tr), 'J': M})
    return pts

def main(path):
    data = json.load(open(path))
    pts = equality_points(data)
    zero_edges = 0
    for V in data:
        J = [[F(x) for x in row] for row in V['J']]
        for e in V['edges']:
            pJ, pQ = edge_polys(J, sym_from_vec([F(x) for x in e['ray']], N))
            if pJ.is_zero:
                zero_edges += 1
    print("edges on which det J vanishes identically:", zero_edges)
    for P in pts:
        M = P['J']
        Q, r, s = blocks(M)
        mu2, holes, nv = deep_holes(Q)
        q = None
        from forms import solve
        q = solve(Q, r)
        print(f"from vertex {P['from']} t={P['t']} tstar={P.get('tstar')}: detQ={det(Q)} detJ={det(M)} "
              f"|Min J|={len(min_vectors(M))} mu^2={mu2} deep holes={len(holes)} Voronoi vertices={nv} "
              f"q={[str(x) for x in q]}")
    json.dump([{'from': P['from'], 'ray': P['ray'], 't': P['t'], 'J': [[str(x) for x in row] for row in P['J']]}
               for P in pts], open(sys.argv[2], 'w'))

if __name__ == '__main__':
    main(sys.argv[1])
