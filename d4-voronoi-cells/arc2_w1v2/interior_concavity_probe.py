#!/usr/bin/env python3
"""
concavity_probe.py
===================
Testing a candidate analytical route to close the fundamental triangle's
INTERIOR gap in the original (non-restated) Direction-of-Deviation
Positivity conjecture.

IDEA (not yet proved -- this script is a probe, not a certificate):
If, for fixed theta, defect(theta, e_perp) is CONCAVE as a function of
arclength along EVERY great-circle geodesic segment whose two endpoints
lie in the closed fundamental triangle, then for any interior point Q,
picking a geodesic through Q with both endpoints on the triangle's
boundary (e.g. from vertex v1 through Q to the point where that geodesic
exits through the opposite edge w1-v2) gives, by concavity:

    defect(theta, Q) >= min(defect(theta, endpoint1), defect(theta, endpoint2))

Since the ENTIRE boundary is now certified positive (Sec 7.6-7.7,
S3-symmetry result), both endpoints have defect >= 0, hence defect(Q) >= 0.
This would close the interior gap WITHOUT a two-parameter closed-form
derivation over the whole interior -- reducing the interior case to a
single new lemma (the concavity-along-every-geodesic property).

This script does NOT prove that lemma. It tests it numerically as a
falsifiable hypothesis: if we find even one geodesic segment along which
defect is not concave, the whole approach is dead and we should not
pursue it further. If it survives many probes, it becomes a genuine
candidate for a real analytical proof attempt (not a proof itself).
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

def defect(theta, e_perp):
    e_perp = e_perp / np.linalg.norm(e_perp)
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    Am = dirs
    bm = -np.ones(len(dirs))
    hs = np.hstack([Am, bm.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    hull = ConvexHull(hi.intersections, qhull_options='QJ')
    return hull.volume - 8.0

def geodesic(A, B, n=41):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    ang = np.arccos(np.clip(A @ B, -1, 1))
    ss = np.linspace(0, ang, n)
    pts = []
    for s in ss:
        if ang < 1e-9:
            pts.append(A)
        else:
            pt = (np.sin(ang - s) * A + np.sin(s) * B) / np.sin(ang)
            pts.append(pt)
    return ss, pts

def concavity_report(name, A, B, thetas, n=41):
    ss, pts = geodesic(A, B, n=n)
    worst_violation = 0.0
    worst_theta = None
    worst_s = None
    for theta in thetas:
        vals = np.array([defect(theta, p) for p in pts])
        # discrete second difference (proportional to -concavity defect)
        d2 = vals[:-2] + vals[2:] - 2 * vals[1:-1]
        # concave means d2 <= 0 everywhere; report the max positive violation
        idx = np.argmax(d2)
        if d2[idx] > worst_violation:
            worst_violation = d2[idx]
            worst_theta = theta
            worst_s = ss[1:-1][idx]
    print(f"  [{name}] worst second-difference (should be <=0 if concave): "
          f"{worst_violation:.3e}  (theta={worst_theta}, s~{worst_s})")
    return worst_violation

print("=== Concavity-along-geodesics probe ===")
print("(negative/~0 => consistent with concavity; positive => VIOLATION, kills the idea)")
print()

thetas_scan = np.linspace(0.05, 1.55, 25)

rng = np.random.default_rng(12345)

print("-- Geodesics from vertex v1 to random points on the far edge (w1,v2) --")
max_viol = 0.0
for k in range(8):
    c = rng.uniform(0, 1)
    P = np.cos(c * np.pi/4) * w1 + np.sin(c * np.pi/4) * v2  # point on far edge (using arc param)
    # careful: w1,v2 are orthogonal so this is a valid unit vector on that edge's great circle
    v = concavity_report(f"v1 -> edge(w1,v2) pt{k}", v1, P, thetas_scan)
    max_viol = max(max_viol, v)

print()
print("-- Geodesics from vertex w1 to random points on the far edge (v1,v2) --")
for k in range(8):
    c = rng.uniform(0, 1)
    P = np.cos(c * np.pi/4) * v1 + np.sin(c * np.pi/4) * v2
    v = concavity_report(f"w1 -> edge(v1,v2) pt{k}", w1, P, thetas_scan)
    max_viol = max(max_viol, v)

print()
print("-- Geodesics from vertex v2 to random points on the far edge (v1,w1) --")
for k in range(8):
    c = rng.uniform(0, 1)
    P = np.cos(c * np.pi/4) * v1 + np.sin(c * np.pi/4) * w1
    v = concavity_report(f"v2 -> edge(v1,w1) pt{k}", v2, P, thetas_scan)
    max_viol = max(max_viol, v)

print()
print("-- Fully generic chords: random interior point to random interior point --")
def rand_interior_point(rng):
    a, b, c = rng.uniform(0.05, 1, 3)
    n = np.sqrt(a*a+b*b+c*c)
    a, b, c = a/n, b/n, c/n
    return a*v1 + b*w1 + c*v2

for k in range(10):
    A = rand_interior_point(rng)
    B = rand_interior_point(rng)
    v = concavity_report(f"generic chord {k}", A, B, thetas_scan)
    max_viol = max(max_viol, v)

print()
print(f"OVERALL max violation found across all probes: {max_viol:.3e}")
print("(if this is comfortably negative/zero across the board, concavity survives")
print(" this probe battery; a single clearly-positive value falsifies it)")
