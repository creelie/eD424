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

def polytope_vertices(theta, t):
    e_perp = np.cos(t) * v1 + np.sin(t) * w1
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    Am = dirs
    bm = -np.ones(len(dirs))
    hs = np.hstack([Am, bm.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    return hi.intersections, dirs

# At t=0 (root-aligned, e_perp=v1), find vertex present just below pi/3,
# absent just above (or vice versa), on the +e2+e3 / +e2+e4 / +e2-e3 / +e2-e4 facets.
theta_lo, theta_hi = np.pi/3 - 1e-4, np.pi/3 + 1e-4
verts_lo, dirs = polytope_vertices(theta_lo, 0.0)
verts_hi, _ = polytope_vertices(theta_hi, 0.0)

target_labels = ['+e2+e3','+e2+e4','+e2-e3','+e2-e4']
target_idx = [labels.index(l) for l in target_labels]

def on_facets(v, dirs, tol=1e-6):
    return [j for j in range(24) if abs(np.dot(v,dirs[j])-1.0)<tol]

print("Vertices at theta=pi/3-eps lying on any of +e2+-e3, +e2+-e4:")
for v in verts_lo:
    f = on_facets(v, dirs)
    flabs = [labels[j] for j in f]
    if any(l in target_labels for l in flabs):
        print(f"  v={np.round(v,6)}  facets={flabs}")

print()
print("Vertices at theta=pi/3+eps lying on any of +e2+-e3, +e2+-e4:")
for v in verts_hi:
    f = on_facets(v, dirs)
    flabs = [labels[j] for j in f]
    if any(l in target_labels for l in flabs):
        print(f"  v={np.round(v,6)}  facets={flabs}")
