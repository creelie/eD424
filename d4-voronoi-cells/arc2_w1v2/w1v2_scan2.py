import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

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
    d = roots @ v; idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1,1,0,0]); u0 = roots[idx0]
w1 = roots[find([0,0,1,1])]; v2 = roots[find([0,0,1,-1])]

def e_perp_arc(t): return np.cos(t)*w1 + np.sin(t)*v2

def active_set(theta, t):
    e_perp = e_perp_arc(t)
    u1 = np.cos(theta)*u0 + np.sin(theta)*e_perp
    dirs = roots.copy(); dirs[idx0] = u1
    hs = np.hstack([dirs, -np.ones((len(dirs),1))])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
    neighbor = set()
    for vi in np.where(on_cap)[0]:
        vv = verts[vi]
        for j in range(len(dirs)):
            if j == idx0: continue
            if abs(np.dot(vv, dirs[j]) - 1.0) < 1e-9:
                neighbor.add(j)
    return frozenset(neighbor)

def bisect(t, lo, hi, tol=1e-10):
    s_lo = active_set(lo, t)
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if active_set(mid, t) == s_lo: lo = mid
        else: hi = mid
        if hi-lo < tol: break
    return 0.5*(lo+hi)

def find_breaks_precise(t, n_scan=800, th_lo=0.002, th_hi=np.pi/2-0.0005):
    thetas = np.linspace(th_lo, th_hi, n_scan)
    prev = active_set(thetas[0], t)
    breaks = []
    for i in range(1, len(thetas)):
        cur = active_set(thetas[i], t)
        if cur != prev:
            b = bisect(t, thetas[i-1], thetas[i])
            breaks.append(b)
            prev = cur
    return breaks

W_curve = lambda tt: 2*np.arctan(np.cos(tt))
U_curve = lambda tt: np.pi/3

for ti in [0.02, 0.1, 0.2, 0.3, 0.4636, 0.6, 0.75]:
    breaks = find_breaks_precise(ti)
    print(f"t={ti:.4f}: breaks={[round(b,5) for b in breaks]}  | U={U_curve(ti):.5f} W={W_curve(ti):.5f}")
