#!/usr/bin/env python3
"""
Identify the "low" breakpoint curve on the w1-v2 arc that
eperp_w1v2_first_curves.py found present but unexplained.

Strategy: at a given t, bisect precisely onto the low breakpoint theta_b
(the smallest breakpoint, below the W-curve). Then look at the actual
polytope vertex (in R^4) that is on the moving cap at theta_b+eps but not
at theta_b-eps (or vice versa) -- this is the vertex responsible for the
transition. Print its exact-looking coordinates and try to match against
candidate exact vectors (roots, sign vertices) to guess its identity.
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection
import itertools

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


def e_perp_arc(t):
    return np.cos(t) * w1 + np.sin(t) * v2


def dirs_at(theta, t):
    e_perp = e_perp_arc(t)
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    return dirs


def cap_vertices(theta, t):
    dirs = dirs_at(theta, t)
    hs = np.hstack([dirs, -np.ones((len(dirs), 1))])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
    return verts[on_cap], dirs


def active_set(theta, t):
    verts, dirs = cap_vertices(theta, t)
    neighbor = set()
    for vv in verts:
        for j in range(len(dirs)):
            if j == idx0:
                continue
            if abs(np.dot(vv, dirs[j]) - 1.0) < 1e-9:
                neighbor.add(j)
    return frozenset(neighbor)


def bisect(t, lo, hi, tol=1e-12):
    s_lo = active_set(lo, t)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if active_set(mid, t) == s_lo:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def find_breaks(t, n_scan=1200, th_lo=0.0005, th_hi=np.pi / 2 - 0.0002):
    thetas = np.linspace(th_lo, th_hi, n_scan)
    prev = active_set(thetas[0], t)
    breaks = []
    for i in range(1, len(thetas)):
        cur = active_set(thetas[i], t)
        if cur != prev:
            breaks.append((bisect(t, thetas[i - 1], thetas[i]), thetas[i - 1], thetas[i]))
            prev = cur
    return breaks


W_curve = lambda tt: 2 * np.arctan(np.cos(tt))
U_curve = lambda tt: np.pi / 3

samples = [0.1, 0.2, 0.3, 0.4636, 0.6]

# candidate identification vectors: all 24 roots, plus the four
# "sign vertex" style points already known (norm sqrt2 vectors with all
# four entries +-1/sqrt2 *2 = +-1 form) -- build ALL 16 sign patterns of
# (+-1,+-1,+-1,+-1)/sqrt2, plus all 8 "two-nonzero-entry, scaled by sqrt2"
# points like Z*, plus the 8 single-axis points (+-sqrt2,0,0,0)-type
# (A is one of these).
sign4 = []
for signs in itertools.product([1, -1], repeat=4):
    sign4.append(np.array(signs) / np.sqrt(2))
sign4 = np.array(sign4)  # 16 points, norm sqrt2

twoaxis = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si * np.sqrt(2)
                v[j] = sj * np.sqrt(2)
                twoaxis.append(v)
twoaxis = np.array(twoaxis)  # 24 points, norm 2 (like Z*)

oneaxis = []
for i in range(4):
    for si in (1, -1):
        v = np.zeros(4)
        v[i] = si * np.sqrt(2)
        oneaxis.append(v)
oneaxis = np.array(oneaxis)  # 8 points, norm sqrt2 (like A)

candidates = np.vstack([sign4, twoaxis, oneaxis])
cand_names = (["sign4:" + str(tuple(np.sign(s).astype(int))) for s in sign4]
              + ["twoaxis:" + str(tuple(np.round(v, 3))) for v in twoaxis]
              + ["oneaxis:" + str(tuple(np.round(v, 3))) for v in oneaxis])

print("Candidate vertex bank size:", len(candidates))

for ti in samples:
    breaks = find_breaks(ti)
    print(f"\n=== t={ti:.4f} ===  all breaks: {[round(b[0],6) for b in breaks]}")
    for (theta_b, lo, hi) in breaks:
        u_hit = abs(theta_b - U_curve(ti)) < 1e-4
        w_hit = abs(theta_b - W_curve(ti)) < 1e-4
        if u_hit or w_hit:
            continue
        # unexplained breakpoint -- identify the vertex responsible
        print(f"  UNEXPLAINED break at theta={theta_b:.8f}")
        verts_before, dirs = cap_vertices(lo, ti)
        verts_after, _ = cap_vertices(hi, ti)

        def key(v):
            return tuple(np.round(v, 6))
        before_keys = set(key(v) for v in verts_before)
        after_keys = set(key(v) for v in verts_after)
        changed = (before_keys ^ after_keys)
        print(f"    vertices present on one side only ({len(changed)}):")
        for ck in changed:
            v = np.array(ck)
            # match against candidate bank
            dists = np.linalg.norm(candidates - v, axis=1)
            jbest = np.argmin(dists)
            print(f"      v={v}  |v|={np.linalg.norm(v):.6f}  "
                  f"closest candidate: {cand_names[jbest]} (dist={dists[jbest]:.2e})")
