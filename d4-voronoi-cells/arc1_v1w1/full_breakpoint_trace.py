import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

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
                sgn = lambda s: '+' if s == 1 else '-'
                labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
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
    on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
    cap_verts_idx = np.where(on_cap)[0]
    neighbor = set()
    for vi in cap_verts_idx:
        vv = verts[vi]
        for j in range(len(dirs)):
            if j == idx0:
                continue
            if abs(np.dot(vv, dirs[j]) - 1.0) < 1e-9:
                neighbor.add(j)
    return frozenset(neighbor)

def bisect(t, lo, hi, tol=1e-9):
    s_lo = active_set(lo, t)
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if active_set(mid, t) == s_lo:
            lo = mid
        else:
            hi = mid
        if hi-lo < tol:
            break
    return 0.5*(lo+hi)

def find_breaks(t, n_scan=400, th_lo=0.02, th_hi=np.pi/2-0.005):
    thetas = np.linspace(th_lo, th_hi, n_scan)
    prev = active_set(thetas[0], t)
    breaks = []
    for i in range(1, n_scan):
        cur = active_set(thetas[i], t)
        if cur != prev:
            br = bisect(t, thetas[i-1], thetas[i])
            breaks.append(br)
        prev = cur
    return breaks

print("Breakpoints at t=0 (pure root-aligned v1 case):")
b0 = find_breaks(0.0)
print(f"  {[round(x,6) for x in b0]}")
print(f"  compare: pi/4={np.pi/4:.6f}, pi/3={np.pi/3:.6f}, arccos(1/3)={np.arccos(1/3):.6f}")
print()
print("Breakpoints at t=pi/2 (pure root-aligned w1 case):")
b1 = find_breaks(np.pi/2)
print(f"  {[round(x,6) for x in b1]}")
print()

# Trace all breakpoint curves across many t values
ts = np.linspace(0.0, np.pi/2, 30)
print("Full breakpoint sets across t:")
for t in ts:
    b = find_breaks(t)
    print(f"  t={t:.4f}: {[round(x,4) for x in b]}")
