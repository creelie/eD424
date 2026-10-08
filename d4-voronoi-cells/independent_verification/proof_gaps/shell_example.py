"""
Lemma "Shell localisation" says a neighbour at distance >= 2 sqrt2 cannot
affect vol(V_c).  Test: the 23-root deletion configuration (contacts at
distance 2 in 23 root directions) plus one neighbour at distance 2.9 > 2 sqrt2
in the direction of the deleted root.
"""
import itertools, math
import numpy as np
from fractions import Fraction
from scipy.spatial import ConvexHull, HalfspaceIntersection

R = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = [0.0] * 4; v[i] = si; v[j] = sj; R.append(np.array(v) / math.sqrt(2))
R = np.array(R)
r0 = R[0]
W = R[1:]                                   # the deletion: 23 contacts
dfar = 2.9
Y = np.vstack([2 * W, dfar * r0])          # centres of the 24 neighbours
allc = np.vstack([np.zeros(4), Y])
Dm = np.linalg.norm(allc[:, None] - allc[None], axis=2)[~np.eye(25, dtype=bool)]
print("least distance between any two of the 25 centres: %.6f (>= 2: %s)" % (Dm.min(), Dm.min() >= 2 - 1e-12))
print("far neighbour at distance %.3f >= 2 sqrt2 = %.6f" % (dfar, 2 * math.sqrt(2)))


def vol(U, h):
    hs = np.hstack([U, -np.asarray(h)[:, None]])
    return ConvexHull(HalfspaceIntersection(hs, np.zeros(4)).intersections).volume


v23 = vol(W, np.ones(23))
v24 = vol(np.vstack([W, r0]), np.r_[np.ones(23), dfar / 2])
h = Fraction(29, 20)
exact = Fraction(25, 3) - Fraction(1, 3) * (2 - h) ** 4
print("cell of the 23 contacts alone:        %.6f  (exact 25/3 = %.6f)" % (v23, 25 / 3))
print("cell with the far neighbour included: %.6f  (exact 25/3 - (1/3)(2 - 1.45)^4 = %s = %.6f)"
      % (v24, exact, float(exact)))
print("the far neighbour changes the volume: %s" % (abs(v23 - v24) > 1e-6))
