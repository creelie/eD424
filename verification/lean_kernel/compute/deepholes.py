"""Voronoi cell and deep holes of a lattice with Gram matrix Q (exact).

The Voronoi-relevant vectors lie among the shortest vectors of the nonzero classes of Z^d / 2Z^d
(Voronoi); the cell is cut out by 2 y^T Q x <= Q[x] for those x, its vertices come from cddlib, and the
squared covering radius is the largest value of Q[y] at a vertex.  Since Q[y] is strictly convex, the
points y of the cell with Q[y] >= mu^2 for mu the covering radius are exactly the vertices attaining it."""
import itertools
from fractions import Fraction as F
import cdd.gmp as cg
from forms import qform, short_vectors

def class_minima(Q):
    d = len(Q)
    out = []
    for c in itertools.product((0, 1), repeat=d):
        if not any(c):
            continue
        # shortest vectors of c + 2Z^d: x = c + 2z, Q[x] = 4 Q[z + c/2]
        half = [F(ci, 2) for ci in c]
        B = F(1)
        while True:
            zs = short_vectors(Q, half, B)
            if zs:
                vals = [qform(Q, [z[i] + half[i] for i in range(d)]) for z in zs]
                m = min(vals)
                out.extend(tuple(2 * z[i] + c[i] for i in range(d)) for z, v in zip(zs, vals) if v == m)
                break
            B *= 2
    return out

def voronoi_vertices(Q):
    d = len(Q)
    xs = class_minima(Q)
    rows = []
    for x in xs:
        Qx = [sum(Q[i][j] * x[j] for j in range(d)) for i in range(d)]
        # Q[x] - 2 (Qx) . y >= 0
        rows.append([qform(Q, list(x))] + [-2 * v for v in Qx])
    mat = cg.matrix_from_array(rows, rep_type=cg.RepType.INEQUALITY)
    poly = cg.polyhedron_from_matrix(mat)
    gen = cg.copy_generators(poly)
    verts = []
    for row in gen.array:
        assert row[0] == 1, "Voronoi cell must be bounded"
        verts.append([F(v) for v in row[1:]])
    return verts, xs

def deep_holes(Q):
    verts, xs = voronoi_vertices(Q)
    vals = [qform(Q, v) for v in verts]
    mu2 = max(vals)
    return mu2, [v for v, w in zip(verts, vals) if w == mu2], len(verts)
