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
        v = verts[vi]
        for j in range(len(dirs)):
            if j == idx0:
                continue
            if abs(np.dot(v, dirs[j]) - 1.0) < 1e-9:
                neighbor.add(j)
    return frozenset(neighbor)

# bisect to find the EXACT (high-precision) transition theta near pi/3, for
# several t, to see how close to pi/3 = 1.0471975511965976 it really sits.
def bisect_near(t, lo, hi, tol=1e-11):
    s_lo = active_set(lo, t)
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if active_set(mid, t) == s_lo:
            lo = mid
        else:
            hi = mid
        if hi-lo < tol:
            break
    return 0.5*(lo+hi)

pi3 = np.pi/3
print(f"pi/3 = {pi3:.12f}")
for t in [0.0, 0.3, 0.6614, 0.9, 0.9921, 1.2, np.pi/2]:
    br = bisect_near(t, pi3-0.01, pi3+0.01)
    print(f"  t={t:.4f}: transition at theta={br:.12f}  (diff from pi/3: {br-pi3:.2e})")
