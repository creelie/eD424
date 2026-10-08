#!/usr/bin/env python3
"""Identify the vertex responsible for the second unexplained transition
(the one where facets v2,(0,-1,2,1),(1,-1,2,1) or a subset are GAINED)."""
import numpy as np
from scipy.spatial import HalfspaceIntersection

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
w1 = roots[find([0, 0, 1, 1])]
v2 = roots[find([0, 0, 1, -1])]


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


def find_breaks(t, n_scan=1500, th_lo=0.0005, th_hi=np.pi/2-0.0002):
    thetas = np.linspace(th_lo, th_hi, n_scan)
    prev = active_set(thetas[0], t)
    out = []
    for i in range(1, len(thetas)):
        cur = active_set(thetas[i], t)
        if cur != prev:
            lo, hi = bisect(t, thetas[i-1], thetas[i])
            out.append((lo, hi, prev, cur))
            prev = cur
    return out

W_curve = lambda tt: 2*np.arctan(np.cos(tt))
U_curve = lambda tt: np.pi/3
B_curve = lambda tt: 2*np.arctan(np.sin(tt))

for t_val in [0.1, 0.2, 0.3, 0.4636, 0.6, 0.75]:
    print(f"\n=== t={t_val} ===")
    for (lo, hi, s_lo, s_hi) in find_breaks(t_val):
        theta_b = 0.5*(lo+hi)
        tag = []
        if abs(theta_b - U_curve(t_val)) < 1e-4: tag.append("universal")
        if abs(theta_b - W_curve(t_val)) < 1e-4: tag.append("W")
        if abs(theta_b - B_curve(t_val)) < 1e-4: tag.append("B(new)")
        gained = s_hi - s_lo
        lost = s_lo - s_hi
        if not tag:
            # identify vertex: look at side where the changed facets ARE present
            side_theta = hi if gained else lo
            verts, dirs = cap_vertices_full(side_theta, t_val)
            on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
            changed_idx = gained if gained else lost
            for vi in np.where(on_cap)[0]:
                vv = verts[vi]
                if all(abs(np.dot(vv, dirs[j]) - 1.0) < 1e-6 for j in changed_idx):
                    touch = [ ("u1" if j==idx0 else labels[j]) for j in range(len(dirs)) if abs(np.dot(vv,dirs[j])-1.0)<1e-6]
                    print(f"  theta={theta_b:.6f} [UNEXPLAINED] vertex={np.round(vv,5)} touches={touch}")
        else:
            print(f"  theta={theta_b:.6f} [{','.join(tag)}]")
