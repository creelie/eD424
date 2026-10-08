#!/usr/bin/env python3
"""Rigorous region-count determination for the w1-v2 arc: for many t
samples within each of the 4 open sub-intervals bounded by
t_cross, t*, t_3, compute the exact theta-values of all 4 curves
(universal, W, B, C) AT THAT t, sort them, and sample the midpoint of
every resulting band (including thin ones near a curve or the domain
edge) to get its combinatorial type. Collect all distinct types."""
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
    return np.arcsin(1.0/s) if s > 1 else None

t_cross = 0.1699184547270611
t_star  = 0.4636476090003471
t_3     = 0.6154797086703875
t_max   = np.pi/4

subints = [(1e-4, t_cross - 1e-4), (t_cross + 1e-4, t_star - 1e-4),
           (t_star + 1e-4, t_3 - 1e-4), (t_3 + 1e-4, t_max - 1e-4)]

all_types = {}
type_id = {}
region_pattern_by_band = {}  # (subint_idx, band_idx_by_curve_order) -> set of type-ids seen

for si_idx, (tlo, thi) in enumerate(subints):
    ts = np.linspace(tlo, thi, 9)
    band_orders_seen = set()
    for tt in ts:
        vals = {'universal': U_curve(tt), 'W': W_curve(tt), 'B': B_curve(tt), 'C': C_curve(tt)}
        present = sorted([(k, v) for k, v in vals.items() if v is not None], key=lambda x: x[1])
        order = tuple(k for k, _ in present)
        band_orders_seen.add(order)
        edges = [0.0] + [v for _, v in present] + [np.pi/2]
        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k+1]
            if hi - lo < 1e-4:
                continue
            th = 0.5 * (lo + hi)
            typ = active_set(th, tt)
            if typ not in type_id:
                type_id[typ] = len(type_id)
            key = (si_idx, k, order)
            region_pattern_by_band.setdefault(key, set()).add(type_id[typ])
    print(f"subinterval {si_idx} (t in [{tlo:.4f},{thi:.4f}]): curve orders seen = {band_orders_seen}")

print(f"\ntotal distinct combinatorial types (Bernstein/certificate-relevant) "
      f"found: {len(type_id)}")

# check: does every (subinterval, band-index) pair map to a SINGLE type
# across all its t-samples (i.e. no internal splitting within a band)?
inconsistent = {k: v for k, v in region_pattern_by_band.items() if len(v) > 1}
print(f"bands with >1 type across their t-samples (would indicate a missed "
      f"internal crossing): {len(inconsistent)}")
for k, v in inconsistent.items():
    print(f"  {k}: types {v}")

# Now map which (subinterval, band) keys share the SAME type id, to see how
# the true regions extend across subintervals (mirroring how v1-w1 arc's
# region 4 extended across its own two subintervals).
type_to_bands = {}
for key, typeset in region_pattern_by_band.items():
    for tid in typeset:
        type_to_bands.setdefault(tid, []).append(key)

print(f"\ntotal distinct type-ids mapped to bands: {len(type_to_bands)}")
for tid, bands in sorted(type_to_bands.items()):
    print(f"  type {tid}: appears in bands {bands}")
