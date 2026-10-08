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

def bisect(t, lo, hi, tol=1e-11):
    s_lo = active_set(lo, t)
    for _ in range(70):
        mid = 0.5*(lo+hi)
        if active_set(mid, t) == s_lo:
            lo = mid
        else:
            hi = mid
        if hi-lo < tol:
            break
    return 0.5*(lo+hi)

t_mid = np.pi/4
arccos13 = np.arccos(1/3)
print(f"pi/4={t_mid:.10f}, arccos(1/3)={arccos13:.10f}")
br = bisect(t_mid, 1.15, 1.30)
print(f"Breakpoint near arccos(1/3) at t=pi/4 exactly: theta={br:.10f}")
print(f"Diff from arccos(1/3): {br-arccos13:.2e}")

# Also check the "low branch" value at t=pi/4:
br2 = bisect(t_mid, 0.70, 0.85)
print(f"Breakpoint (low branch) at t=pi/4: theta={br2:.10f}, diff from pi/4: {br2-np.pi/4:.2e}")
