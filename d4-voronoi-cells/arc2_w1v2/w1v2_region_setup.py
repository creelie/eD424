#!/usr/bin/env python3
"""Determine the true combinatorial region count for the w1-v2 arc,
using the 4 exact curves (universal, W, B, C) from
eperp_w1v2_breakpoint_classification.py, analogous to how the v1-w1
arc's region count was corrected from 9 to 8 (Round 13)."""
import numpy as np
from scipy.spatial import HalfspaceIntersection

roots = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si; v[j] = sj
                roots.append(v / np.sqrt(2))
roots = np.array(roots)

def find(vec):
    v = np.array(vec, dtype=float); v /= np.linalg.norm(v)
    d = roots @ v; idx = np.argmax(d); assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1,1,0,0]); u0 = roots[idx0]
w1 = roots[find([0,0,1,1])]; v2 = roots[find([0,0,1,-1])]

def active_set(theta,t):
    e = np.cos(t)*w1 + np.sin(t)*v2
    u1 = np.cos(theta)*u0 + np.sin(theta)*e
    dirs = roots.copy(); dirs[idx0]=u1
    hs = np.hstack([dirs, -np.ones((len(dirs),1))])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    on_cap = np.abs(verts@dirs[idx0]-1.0)<1e-9
    neigh=set()
    for vi in np.where(on_cap)[0]:
        vv=verts[vi]
        for j in range(len(dirs)):
            if j==idx0: continue
            if abs(np.dot(vv,dirs[j])-1.0)<1e-9: neigh.add(j)
    return frozenset(neigh)

U_curve = lambda tt: np.pi/3
W_curve = lambda tt: 2*np.arctan(np.cos(tt))
B_curve = lambda tt: 2*np.arctan(np.sin(tt))
def C_curve(tt):
    s = np.cos(tt)+np.sin(tt)
    if s>1: return np.arcsin(1.0/s)
    return None

# exact crossing points (all derived/verified earlier)
t_cross = 0.1699184547270611   # C meets universal
t_star  = 0.4636476090003471   # B meets C  (=arctan(1/2))
t_3     = 0.6154797086703875   # B meets universal (=arcsin(1/sqrt3))

# Sample representative points strictly inside each candidate band,
# away from all curves by a safety margin, and identify the exact
# active_set (as a canonical signature) for each band.
bands_t = [
    ("t in (0, t_cross)", (0.0+0.02, t_cross-0.01)),
    ("t in (t_cross, t*)", (t_cross+0.01, t_star-0.01)),
    ("t in (t*, t_3)", (t_star+0.01, t_3-0.01)),
    ("t in (t_3, pi/4)", (t_3+0.01, np.pi/4-0.02)),
]

print("Ordering of the 4 curves' theta-values changes at t_cross, t*, t_3.")
print("Within each t-band the curve ORDER (bottom to top) is fixed, so the")
print("vertical strips between consecutive curves + the two edges (0,pi/2)")
print("are the candidate combinatorial regions. Checking each stack:\n")

region_id = 0
all_types = {}
for label, (tlo, thi) in bands_t:
    tt = 0.5*(tlo+thi)
    vals = {
        'universal': U_curve(tt),
        'W': W_curve(tt),
        'B': B_curve(tt),
        'C': C_curve(tt),
    }
    # sort curves present at this t by theta value
    present = [(name, val) for name, val in vals.items() if val is not None]
    present.sort(key=lambda x: x[1])
    order = [name for name, _ in present]
    print(f"{label}: curve order (low to high theta) = {order}, "
          f"values = {[round(v,4) for _,v in present]}")
    # sample a representative theta strictly between consecutive curves
    edges = [0.0] + [v for _, v in present] + [np.pi/2]
    edge_names = ['0'] + order + ['pi/2']
    band_types = []
    for k in range(len(edges)-1):
        lo, hi = edges[k], edges[k+1]
        if hi - lo < 0.01:
            band_types.append(None)
            continue
        th = 0.5*(lo+hi)
        typ = active_set(th, tt)
        band_types.append(typ)
        print(f"    band theta in ({edge_names[k]},{edge_names[k+1]}): "
              f"{len(typ)} active facets")
    for typ in band_types:
        if typ is not None:
            all_types[typ] = all_types.get(typ, 0)

print(f"\ntotal distinct combinatorial types found across the 4 t-bands' "
      f"representative points: {len(all_types)}")
