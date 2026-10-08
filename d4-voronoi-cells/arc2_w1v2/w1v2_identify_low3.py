#!/usr/bin/env python3
"""
At the UNEXPLAINED breakpoint where facet v2=(0,0,1,-1)/sqrt2 leaves the
neighbor set of u1's cap, find the actual vertex (in R^4) where u1's cap
and v2's facet meet just before the transition, and identify which OTHER
fixed facets it lies on -- this pins down the exact touching condition
(an inner-product-equals-1 equation in theta,t against a genuine, fully
identified polytope vertex), analogous to Z*/W/Y/A on the other arc.
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection
import sympy as sp

roots = []
labels = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si
                v[j] = sj
                roots.append(v / np.sqrt(2))
                labels.append((i, si, j, sj))
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
idx_v2 = find([0, 0, 1, -1])


def e_perp_arc(t):
    return np.cos(t) * w1 + np.sin(t) * v2


def dirs_at(theta, t):
    e_perp = e_perp_arc(t)
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    return dirs


def cap_vertices_full(theta, t):
    dirs = dirs_at(theta, t)
    hs = np.hstack([dirs, -np.ones((len(dirs), 1))])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    return hi.intersections, dirs


def active_set(theta, t):
    verts, dirs = cap_vertices_full(theta, t)
    on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
    neighbor = set()
    for vi in np.where(on_cap)[0]:
        vv = verts[vi]
        for j in range(len(dirs)):
            if j == idx0:
                continue
            if abs(np.dot(vv, dirs[j]) - 1.0) < 1e-9:
                neighbor.add(j)
    return frozenset(neighbor)


def bisect(t, lo, hi, tol=1e-13):
    s_lo = active_set(lo, t)
    for _ in range(75):
        mid = 0.5 * (lo + hi)
        if active_set(mid, t) == s_lo:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo, hi


t_val = 0.3
lo, hi = 0.5, 0.6  # bracket around the unexplained break at t=0.3 (0.57468)
lo, hi = bisect(t_val, 0.55, 0.60)
print("bisected bracket:", lo, hi, "theta_b ~", 0.5 * (lo + hi))

# at theta = lo (just before break), v2 is still active: find the vertex(es)
# on the u1-cap that also touch v2
verts, dirs = cap_vertices_full(lo, t_val)
on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
for vi in np.where(on_cap)[0]:
    vv = verts[vi]
    if abs(np.dot(vv, v2) - 1.0) < 1e-7:
        # which other facets does this vertex lie on?
        touch = []
        for j in range(len(dirs)):
            if abs(np.dot(vv, dirs[j]) - 1.0) < 1e-7:
                touch.append(j)
        touch_lbl = [("u1" if j == idx0 else labels[j]) for j in touch]
        print("vertex:", np.round(vv, 6), "touches facets:", touch_lbl)

# Now do the same scan across several t values to see if the same OTHER
# facets consistently accompany v2, giving us the vertex's defining set.
print("\n--- scan across t ---")
for t_val in [0.1, 0.2, 0.3, 0.4636]:
    lo, hi = bisect(t_val, 0.02, np.pi/2 - 0.001) if False else (None, None)

def find_unexplained_break(t_val):
    thetas = np.linspace(0.001, 1.04, 400)
    prev = active_set(thetas[0], t_val)
    for i in range(1, len(thetas)):
        cur = active_set(thetas[i], t_val)
        if cur != prev and (prev - cur) == {idx_v2}:
            lo2, hi2 = bisect(t_val, thetas[i-1], thetas[i])
            return lo2, hi2
        prev = cur
    return None

for t_val in [0.1, 0.2, 0.3, 0.4636]:
    res = find_unexplained_break(t_val)
    if res is None:
        print(f"t={t_val}: no v2-loss break found in scan range")
        continue
    lo, hi = res
    verts, dirs = cap_vertices_full(lo, t_val)
    on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
    for vi in np.where(on_cap)[0]:
        vv = verts[vi]
        if abs(np.dot(vv, v2) - 1.0) < 1e-6:
            touch = []
            for j in range(len(dirs)):
                if abs(np.dot(vv, dirs[j]) - 1.0) < 1e-6:
                    touch.append(j)
            touch_lbl = [("u1" if j == idx0 else labels[j]) for j in touch]
            print(f"t={t_val}: theta_b~{0.5*(lo+hi):.6f}  vertex={np.round(vv,5)}  touches={touch_lbl}")
