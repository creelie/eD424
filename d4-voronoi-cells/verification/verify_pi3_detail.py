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

theta_c = np.pi/3
eps = 1e-4
ts = np.linspace(0.0, np.pi/2, 20)
for t in ts:
    s_before = active_set(theta_c - eps, t)
    s_after = active_set(theta_c + eps, t)
    entered = s_after - s_before
    left = s_before - s_after
    ent_labs = sorted(labels[i] for i in entered)
    left_labs = sorted(labels[i] for i in left)
    print(f"t={t:.4f}  entered={ent_labs}  left={left_labs}")
