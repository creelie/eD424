#!/usr/bin/env python3
"""
region_top_interior_stability.py
==================================
Task: check whether the "region top" combinatorial type (bounded above by
theta=pi/2, below by curve W' on both boundary arcs; 32 vertices, 8 moving
under the deviating cap, 24 fixed) persists at GENERIC INTERIOR points of
the fundamental triangle, not just on its three edges.

If it does persist across the whole interior (for theta in some open band
below pi/2), that is the necessary (not sufficient) precondition for
attempting a genuine 2-parameter closed-form derivation the way region top
was already handled in 1 parameter on each named arc.
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
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]
v2 = roots[find([0, 0, 1, -1])]

def n_vertices(theta, e_perp):
    e_perp = e_perp / np.linalg.norm(e_perp)
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    Am = dirs
    bm = -np.ones(len(dirs))
    hs = np.hstack([Am, bm.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    hull = ConvexHull(hi.intersections, qhull_options='QJ')
    return len(hull.vertices), hi

def active_facet_set(hi, tol=1e-7):
    # facets whose equation is tight at at least one vertex: use the offsets
    # returned by qhull's dual to identify which of the ORIGINAL halfspaces
    # (rows of hi.halfspaces) are represented among the intersection's
    # generators. We instead just track n_vertices as the primary invariant
    # (as done in earlier breakpoint scans in this project), since exact
    # facet-set tracking requires vertex-to-facet incidence which scipy
    # does not expose directly for HalfspaceIntersection duals here.
    pass

print("=== Region-top interior stability scan ===")
print("theta near pi/2 (region top band); scanning (a,b,c) over the full")
print("closed triangle including deep interior points, checking n_vertices")
print("stays at 32 (8 moving + 24 fixed) as on the boundary arcs.")
print()

rng = np.random.default_rng(7)

# reference: boundary point value (arc1 midpoint) for comparison
def dev_abc(a, b, c):
    v = a * v1 + b * w1 + c * v2
    return v / np.linalg.norm(v)

thetas = [1.3, 1.4, 1.45, 1.5, 1.55]
mismatches = []
tested = 0
for theta in thetas:
    for _ in range(15):
        a, b, c = rng.uniform(0.02, 1, 3)
        nrm = np.sqrt(a*a+b*b+c*c)
        a, b, c = a/nrm, b/nrm, c/nrm
        nv, hi = n_vertices(theta, dev_abc(a, b, c))
        tested += 1
        if nv != 32:
            mismatches.append((theta, a, b, c, nv))

print(f"Tested {tested} random interior points across 5 theta values.")
print(f"Mismatches (n_vertices != 32): {len(mismatches)}")
for m in mismatches[:20]:
    print("  ", m)

# Also scan a dense GRID at one fixed theta to see where (if anywhere)
# the combinatorial type changes within the triangle's interior.
print()
print("=== Dense grid scan at theta=1.45 ===")
theta = 1.45
grid_mismatches = []
grid_total = 0
for ia in np.linspace(0.02, 1, 12):
    for ib in np.linspace(0.02, 1, 12):
        ic2 = 1 - ia**2 - ib**2
        if ic2 <= 0:
            continue
        ic = np.sqrt(ic2)
        nv, hi = n_vertices(theta, dev_abc(ia, ib, ic))
        grid_total += 1
        if nv != 32:
            grid_mismatches.append((ia, ib, ic, nv))
print(f"Grid points tested: {grid_total}, mismatches: {len(grid_mismatches)}")
for m in grid_mismatches[:30]:
    print("  ", m)
