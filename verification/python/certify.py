#!/usr/bin/env python3
"""
certify.py -- turn the output of ryshkov2.py into plain integer certificates
that the Julia, C and Lean checks read.

For every vertex representative: J (integer 5 x 5) and its minimal vectors.
For every edge at a representative, with direction X (integer 5 x 5):
  * bounded edges (t* in {1, 2}): the end point J + t* X and a matrix T in
    Gamma with J + t* X = T^T J_j T, where J_j is the representative of the
    end point's class;
  * unbounded edges: an integer vector u and an integer a with
    X = p q^T + q p^T, p = (u, -a), q = (u, -(a + 1)) (Definition 5.2),
    so that X[(n, l)] = 2 (u.n - a l)(u.n - (a + 1) l) >= 0 for l in
    {-1, 0, 1}.

It also writes, for every unbounded edge, an element T of the stabiliser of
J_i in Gamma that maps the representative ray of its orbit (Table 5.3 of the
paper) to it: T^T J_i T = J_i and T^T X_rep T = X.

Output: ../data/vertices.txt, ../data/edges.txt, ../data/unbounded_orbits.txt
(whitespace separated integers, one record per line, see ../data/FORMAT.md).
"""
import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(__file__))
from ryshkov2 import mat, minimum, add, find_iso, all_isos, D, N  # noqa: E402


def ints(A):
    out = []
    for row in A:
        for x in row:
            x = F(x)
            assert x.denominator == 1
            out.append(int(x))
    return out


def unbounded_form(X):
    """(u, a) with X[(n, l)] = 2 (u.n - a l)(u.n - (a + 1) l), or None.

    Equivalently Q_X = 2 u u^T, r_X = -(2a + 1) u and s_X = 2a(a + 1).  For an
    integer vector k = (n, l) the two factors are integers differing by l, so
    for l in {-1, 0, 1} their product is >= 0 and X >= 0 on M."""
    Q = [[X[i][j] for j in range(D)] for i in range(D)]
    r = [X[i][D] for i in range(D)]
    i0 = next(i for i in range(D) if Q[i][i] != 0)
    a2 = Q[i0][i0] / 2
    s = int(round(float(a2) ** 0.5))
    if F(s * s) != a2:
        return None
    u = [Q[i0][j] / (2 * s) for j in range(D)]
    if any(x.denominator != 1 for x in u):
        return None
    u = [int(x) for x in u]
    if any(Q[i][j] != 2 * u[i] * u[j] for i in range(D) for j in range(D)):
        return None
    c = -r[i0] / u[i0]                       # = 2a + 1
    if c.denominator != 1 or int(c) % 2 == 0:
        return None
    a = (int(c) - 1) // 2
    if any(r[i] != -(2 * a + 1) * u[i] for i in range(D)) or X[D][D] != 2 * a * (a + 1):
        return None
    return u, a


def main():
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'data')
    data = json.load(open(src))
    V = [(mat([[F(x) for x in row] for row in v['J']]), [tuple(k) for k in v['active']]) for v in data['vertices']]
    with open(os.path.join(outdir, 'vertices.txt'), 'w') as fh:
        for i, (J, act) in enumerate(V):
            fh.write(' '.join(map(str, [i] + ints(J) + [len(act)] + [x for k in act for x in k])) + '\n')
    with open(os.path.join(outdir, 'edges.txt'), 'w') as fh:
        for ei, e in enumerate(data['edges']):
            i = e['from']
            J, act = V[i]
            X = mat([[F(x) for x in row] for row in e['X']])
            if e['t'] is None:
                ue = unbounded_form(X)
                assert ue is not None, 'edge %d: unbounded edge of unexpected form' % ei
                u, a = ue
                fh.write(' '.join(map(str, [ei, i] + ints(X) + [0, -1] + u + [a])) + '\n')
                continue
            t = F(e['t'])
            assert t.denominator == 1
            J2 = add(J, X, t)
            m2, act2 = minimum(J2)
            assert m2 == 4
            j = e['to']
            Jj, actj = V[j]
            T = find_iso(Jj, actj, J2, act2)
            assert T is not None, 'edge %d: no isometry found' % ei
            fh.write(' '.join(map(str, [ei, i] + ints(X) + [int(t), j] + [x for row in T for x in row])) + '\n')
    # orbits of the unbounded rays under the stabilisers
    reps = {}
    with open(os.path.join(outdir, 'unbounded_orbits.txt'), 'w') as fh:
        stab = {}
        for ei, e in enumerate(data['edges']):
            if e['t'] is not None:
                continue
            i = e['from']
            J, act = V[i]
            if i not in stab:
                stab[i] = all_isos(J, act, J, act)
            X = mat([[F(x) for x in row] for row in e['X']])
            found = None
            for r in reps.get(i, []):
                Xr = mat([[F(x) for x in row] for row in data['edges'][r]['X']])
                for T in stab[i]:
                    Y = [[sum(T[a][p] * Xr[a][b] * T[b][q] for a in range(N) for b in range(N))
                          for q in range(N)] for p in range(N)]
                    if Y == X:
                        found = (r, T)
                        break
                if found:
                    break
            if found is None:
                reps.setdefault(i, []).append(ei)
                found = (ei, [[1 if a == b else 0 for b in range(N)] for a in range(N)])
            r, T = found
            fh.write(' '.join(map(str, [ei, i, r] + [x for row in T for x in row])) + '\n')
    print('unbounded orbit representatives:', reps)
    print('wrote %d vertices and %d edges to %s' % (len(V), len(data['edges']), outdir))


if __name__ == '__main__':
    main()
