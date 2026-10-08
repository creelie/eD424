"""
hull_along_rays.py -- the part of V(Y) outside the inversion hull K(Y),
measured along the two rays of the root system that the paper uses:

  one centre pushed out:  y_1 = (2 + d) u_1,  y_i = 2 u_i (i > 1),   S = d;
  all pushed out evenly:  y_i = (2 + S/24) u_i,                        S = sum of the push-outs.

Here V(Y) = {x : <x, y> <= |y|^2 / 2 for all y in Y} and K(Y) is the convex
hull of 0 and the inverted centres 4 y / |y|^2 (prop:inversion-hull of the paper).
The exact volumes are computed by intersecting half-spaces (scipy), and
vol(V \ K) = vol(V) - vol(V cap K) is compared with the estimate
13958 Theta^4 of part (c) of the proof of thm:near-contact, with Theta = S on
these rays (the tilts vanish).

Usage:  python3 hull_along_rays.py
"""
import numpy as np
from itertools import combinations, product
from scipy.spatial import ConvexHull, HalfspaceIntersection

# the 24 normalised roots of D4
U = []
for i, j in combinations(range(4), 2):
    for si, sj in product((1, -1), repeat=2):
        v = np.zeros(4); v[i] = si; v[j] = sj
        U.append(v / np.sqrt(2))
U = np.array(U)


def cell_halfspaces(Y):
    # <x, y> <= |y|^2/2  written as  a.x + b <= 0
    n2 = np.sum(Y * Y, axis=1)
    return np.hstack([Y, -(n2 / 2)[:, None]])


def volume_of(halfspaces):
    hs = HalfspaceIntersection(halfspaces, np.zeros(4))
    return ConvexHull(hs.intersections).volume


def hull_halfspaces(Y):
    P = np.vstack([np.zeros(4), 4 * Y / np.sum(Y * Y, axis=1)[:, None]])
    return ConvexHull(P).equations           # rows a.x + b <= 0 inside


def outside_hull(Y):
    hv = cell_halfspaces(Y)
    vol_V = volume_of(hv)
    vol_VK = volume_of(np.vstack([hv, hull_halfspaces(Y)]))
    return vol_V, vol_V - vol_VK


print("S        one centre pushed out: vol(V)     vol(V \\ K)   | even push: vol(V)     vol(V \\ K)   | 13958 S^4")
rows = []
for S in [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.155, 0.197, 0.25, 0.3, 0.4]:
    Y1 = 2 * U.copy(); Y1[0] *= (2 + S) / 2
    Ye = (2 + S / 24) * U
    v1, o1 = outside_hull(Y1)
    ve, oe = outside_hull(Ye)
    rows.append((S, o1, oe))
    print(f"{S:<8} {v1:.9f}  {o1:.3e}   | {ve:.9f}  {oe:.3e}   | {13958 * S**4:.3e}")

# growth exponents from the last few points of each ray
r = np.array(rows)
for k, name in ((1, "one centre"), (2, "even push")):
    m = r[:, k] > 1e-13
    if m.sum() >= 3:
        x, y = np.log(r[m, 0][-4:]), np.log(r[m, k][-4:])
        print(f"growth exponent of vol(V \\ K) on the {name} ray, from the last four points with a measurable value: "
              f"{np.polyfit(x, y, 1)[0]:.2f}")
print(f"largest value of vol(V \\ K) for S <= 0.197: one centre {max(o for S,o,_ in rows if S<=0.197):.3e}, "
      f"even push {max(o for S,_,o in rows if S<=0.197):.3e}")
