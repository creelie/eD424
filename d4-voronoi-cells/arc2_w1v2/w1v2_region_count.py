#!/usr/bin/env python3
"""Quick region-count scan for the w1-v2 arc now that its breakpoint
classification is complete (4 curves: universal, W, B, C). Purely a
scoping check for how large the region-by-region program would be --
NOT an attempt at volume formulas or positivity certificates."""
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

ts = np.linspace(0.01, np.pi/4-0.01, 40)
ths = np.linspace(0.01, np.pi/2-0.01, 60)
types = set()
type_counts_per_t = []
for ti in ts:
    row_types = set()
    for thi in ths:
        # skip too close to the known curves to avoid degenerate double-active grid noise
        row_types.add(active_set(thi, ti))
    types |= row_types
    type_counts_per_t.append(len(row_types))

print("distinct combinatorial region types over 40x60 grid:", len(types))
print("min/max distinct types per t-slice:", min(type_counts_per_t), max(type_counts_per_t))
