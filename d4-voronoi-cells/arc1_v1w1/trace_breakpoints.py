import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
for _p in (_PKG_ROOT, _os.path.join(_PKG_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
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

def active_set(theta, t):
    e_perp = np.cos(t) * v1 + np.sin(t) * w1
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    Am = dirs
    bm = -np.ones(len(dirs))
    hs = np.hstack([Am, bm.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-6
    cap_verts_idx = np.where(on_cap)[0]
    neighbor = set()
    for vi in cap_verts_idx:
        v = verts[vi]
        for j in range(len(dirs)):
            if j == idx0:
                continue
            if abs(np.dot(v, dirs[j]) - 1.0) < 1e-6:
                neighbor.add(j)
    return frozenset(neighbor)

def bisect_transition(t, th_lo, th_hi, tol=1e-7):
    s_lo = active_set(th_lo, t)
    s_hi = active_set(th_hi, t)
    if s_lo == s_hi:
        return None
    while th_hi - th_lo > tol:
        mid = 0.5*(th_lo+th_hi)
        if active_set(mid, t) == s_lo:
            th_lo = mid
        else:
            th_hi = mid
    return 0.5*(th_lo+th_hi)

# For each t, scan theta finely and bisect every transition found.
ts = np.linspace(0.02, np.pi/2-0.02, 40)
n_scan = 200
theta_scan = np.linspace(0.02, np.pi/2-0.005, n_scan)

all_breaks = []
for t in ts:
    prev_set = active_set(theta_scan[0], t)
    breaks_here = []
    for i in range(1, n_scan):
        cur_set = active_set(theta_scan[i], t)
        if cur_set != prev_set:
            br = bisect_transition(t, theta_scan[i-1], theta_scan[i])
            if br is not None:
                breaks_here.append(br)
        prev_set = cur_set
    all_breaks.append((t, breaks_here))

print("Number of breakpoints found per t (should be constant if curves are")
print("continuous and don't appear/disappear):")
counts = [len(b) for _,b in all_breaks]
print(f"  counts: {counts}")
print(f"  min={min(counts)} max={max(counts)}")
print()
print("Sample rows (t, breakpoint thetas):")
for t, b in all_breaks[::5]:
    print(f"  t={t:.4f}: {[round(x,4) for x in b]}")

np.save(_os.path.join(_DATA_DIR, 'all_breaks.npy'), np.array(all_breaks, dtype=object), allow_pickle=True)
