#!/usr/bin/env python3
"""
Numerical active-facet-set transition scan for the w1-v2 arc, reduced to
t in [0,pi/4] by Lemma s4symmetry (S=diag(1,1,1,-1) swaps w1<->v2, fixes
u0 and v1, giving the arc its own midpoint symmetry about t=pi/4 exactly
as verified: S*w1==v2, S*v2==w1, S*v1==v1, S*u0==u0, checked to
float64 precision in w1v2_explore.py).

Goal: trace active-facet-set transitions across a fine theta-grid at many
t values, and see how many distinct breakpoint curves appear, comparing
against the two candidates found by direct sign-vertex inner products
(universal theta=pi/3 via Z*=2u0, and theta=2*arctan(cos t) via
W=(1,1,1,1)/sqrt2) in w1v2_explore.py. Also checked there: neither
Y=(1,-1,1,1)/sqrt2 nor the Hadamard image A=(sqrt2,0,0,0) give an
interior touching condition on this arc (their inner products with
u1(theta,t) never reach 1 except at the domain corner) -- so unlike the
v1-w1 arc, those two do not contribute curves here.
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

roots = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si
                v[j] = sj
                roots.append(v / np.sqrt(2))
roots = np.array(roots)


def find(vec):
    v = np.array(vec, dtype=float)
    v /= np.linalg.norm(v)
    d = roots @ v
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx


idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
w1 = roots[find([0, 0, 1, 1])]
v2 = roots[find([0, 0, 1, -1])]


def e_perp_arc(t):
    return np.cos(t) * w1 + np.sin(t) * v2


def active_set(theta, t):
    e_perp = e_perp_arc(t)
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    Am = dirs
    bm = -np.ones(len(dirs))
    hs = np.hstack([Am, bm.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
    cap_verts_idx = np.where(on_cap)[0]
    neighbor = set()
    for vi in cap_verts_idx:
        vv = verts[vi]
        for j in range(len(dirs)):
            if j == idx0:
                continue
            if abs(np.dot(vv, dirs[j]) - 1.0) < 1e-9:
                neighbor.add(j)
    return frozenset(neighbor)


def find_breaks(t, n_scan=400, th_lo=0.01, th_hi=np.pi / 2 - 0.005):
    thetas = np.linspace(th_lo, th_hi, n_scan)
    prev = active_set(thetas[0], t)
    breaks = []
    for th in thetas[1:]:
        cur = active_set(th, t)
        if cur != prev:
            breaks.append(th)
            prev = cur
    return breaks


# 1) how many distinct combinatorial types across a grid?
t_samples = np.linspace(0.02, np.pi / 4 - 0.02, 25)
all_types_at_reps = set()
break_counts = []
for ti in t_samples:
    breaks = find_breaks(ti)
    break_counts.append(len(breaks))

print("t_samples:", np.round(t_samples, 3))
print("num breakpoints found per t:", break_counts)

# 2) compare traced breakpoints against the two known candidate curves
W_curve = lambda tt: 2 * np.arctan(np.cos(tt))
U_curve = lambda tt: np.pi / 3

for ti in t_samples[::5]:
    breaks = find_breaks(ti)
    print(f"\nt={ti:.4f}: breaks at theta={[round(b,4) for b in breaks]}")
    print(f"   candidates: universal={U_curve(ti):.4f}, W-curve={W_curve(ti):.4f}")

# 3) grid of combinatorial types (theta,t) -> facet-set, count distinct types
type_grid = {}
ts2 = np.linspace(0.02, np.pi / 4 - 0.02, 12)
th2 = np.linspace(0.02, np.pi / 2 - 0.02, 16)
types_seen = set()
for ti in ts2:
    for thi in th2:
        s = active_set(thi, ti)
        types_seen.add(s)
print(f"\ndistinct combinatorial types over {len(ts2)}x{len(th2)} grid: {len(types_seen)}")
