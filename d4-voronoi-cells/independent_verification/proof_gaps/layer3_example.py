"""
A local unit-ball packing configuration whose active neighbours (distance in
[2, 2*sqrt2)) do NOT form a contact configuration at the all-contact corner.

Centre at 0; 48 neighbours at distance d in the 48 directions of the binary
octahedral group (the vertices of two 24-cells in dual position), pairwise at
least 45 degrees apart.  d is the least distance keeping all 49 balls disjoint.
"""
import itertools, math
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

dirs = set()
for i, j in itertools.combinations(range(4), 2):          # 24 roots / sqrt2
    for si in (1, -1):
        for sj in (1, -1):
            v = [0.0] * 4; v[i] = si / math.sqrt(2); v[j] = sj / math.sqrt(2)
            dirs.add(tuple(v))
for i in range(4):                                          # 8 axis vectors
    for s in (1, -1):
        v = [0.0] * 4; v[i] = float(s); dirs.add(tuple(v))
for signs in itertools.product((0.5, -0.5), repeat=4):      # 16 half-vectors
    dirs.add(signs)
U = np.array(sorted(dirs))
assert len(U) == 48 and np.allclose((U * U).sum(1), 1)

G = U @ U.T
off = G[~np.eye(48, dtype=bool)]
cmax = off.max()
d = 2 / math.sqrt(2 - 2 * cmax)             # least radius with |y_i - y_j| >= 2
Y = d * U
D = np.linalg.norm(Y[:, None] - Y[None], axis=2)[~np.eye(48, dtype=bool)]
print("neighbours: %d, all at distance d = %.6f (2*sqrt2 = %.6f)" % (len(U), d, 2 * math.sqrt(2)))
print("largest inner product between neighbour directions: %.6f  (> 1/2: %s)" % (cmax, cmax > 0.5))
print("least distance between neighbour centres: %.9f  (>= 2: %s)" % (D.min(), D.min() >= 2 - 1e-12))
print("so this is a valid packing of 49 unit balls, every neighbour in the shell [2, 2 sqrt2)")


def cell_volume(U, h):
    # {x : <x,u> <= h} over all u, as a polytope volume
    hs = np.hstack([U, -h * np.ones((len(U), 1))])
    P = HalfspaceIntersection(hs, np.zeros(4)).intersections
    return ConvexHull(P).volume


v_corner = cell_volume(U, 1.0)                # every active radius pulled to 2
v_true = cell_volume(U, d / 2)                # the actual Voronoi cell
print()
print("cell at the all-contact corner (all 48 radii set to 2): volume %.6f  (< 8: %s)" % (v_corner, v_corner < 8))
print("  its direction set has %d > 24 elements and inner products up to %.4f > 1/2," % (len(U), cmax))
print("  so it is not a contact configuration and no result of the paper applies to it")
print("actual Voronoi cell of the centre in this packing:        volume %.6f" % v_true)
