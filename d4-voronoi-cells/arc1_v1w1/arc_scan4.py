import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import minimize

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
    vec = np.array(vec, dtype=float)
    vec /= np.linalg.norm(vec)
    d = roots @ vec
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]
v2 = roots[find([0, 0, 1, -1])]

def F_direct(theta, e_perp):
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

def arc_point(t, A, B):
    a, b = np.cos(t), np.sin(t)
    return a * A + b * B

# BOUNDED refinement (L-BFGS-B) within the flagged transition window,
# strictly away from all already-resolved boundaries (theta->0, theta->pi/2,
# t=0, t=pi/2 or pi/4).
def obj_factory(A, B):
    def obj(x):
        theta, t = x
        e = arc_point(t, A, B)
        return F_direct(theta, e)
    return obj

print("v1-w1 arc, bounded search theta in [0.8,1.5], t in [0.1,pi/2-0.1]:")
best_vw = (1e9, None)
for theta0 in np.linspace(0.85, 1.45, 7):
    for t0 in np.linspace(0.2, np.pi/2-0.2, 7):
        res = minimize(obj_factory(v1, w1), x0=[theta0, t0], method='L-BFGS-B',
                        bounds=[(0.8, 1.5), (0.1, np.pi/2-0.1)],
                        options={'maxiter':200})
        if res.fun < best_vw[0]:
            best_vw = (res.fun, tuple(res.x))
print(f"  best: F={best_vw[0]:.8f} at (theta,t)={best_vw[1]}")

print("w1-v2 arc, bounded search theta in [0.8,1.5], t in [0.1,pi/4-0.05]:")
best_wv = (1e9, None)
for theta0 in np.linspace(0.85, 1.45, 7):
    for t0 in np.linspace(0.15, np.pi/4-0.05, 6):
        res = minimize(obj_factory(w1, v2), x0=[theta0, t0], method='L-BFGS-B',
                        bounds=[(0.8, 1.5), (0.05, np.pi/4-0.02)],
                        options={'maxiter':200})
        if res.fun < best_wv[0]:
            best_wv = (res.fun, tuple(res.x))
print(f"  best: F={best_wv[0]:.8f} at (theta,t)={best_wv[1]}")
