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

# Focus away from theta->0 (trivial small-angle regime, already known F~theta^2/3
# for ANY e_perp). Scan theta in [0.3, pi/2), fine grid, both reduced arcs.
thetas = np.linspace(0.3, np.pi/2 - 0.01, 80)

def scan_arc(A, B, tmax, name):
    ts = np.linspace(0.0, tmax, 80)
    worst = (1e9, None, None)
    for theta in thetas:
        for t in ts:
            e = arc_point(t, A, B)
            F = F_direct(theta, e)
            if F < worst[0]:
                worst = (F, theta, t)
    print(f"{name}: min F={worst[0]:.6f} at theta={worst[1]:.4f}, t={worst[2]:.4f}")
    return worst

w1_ = scan_arc(v1, w1, np.pi/2, "v1-w1 arc")
w2_ = scan_arc(w1, v2, np.pi/4, "w1-v2 arc")

# Local refine around the overall worst point with Nelder-Mead
best = w1_ if w1_[0] < w2_[0] else w2_
A, B = (v1, w1) if w1_[0] < w2_[0] else (w1, v2)
name = "v1-w1" if w1_[0] < w2_[0] else "w1-v2"
print(f"\nRefining near global worst ({name} arc) with local optimisation...")

def neg_F(x):
    theta, t = x
    if not (0.05 < theta < np.pi/2 - 0.001):
        return 1e3
    e = arc_point(t, A, B)
    return F_direct(theta, e)

res = minimize(neg_F, x0=[best[1], best[2]], method='Nelder-Mead',
                options={'xatol':1e-7, 'fatol':1e-10, 'maxiter':500})
print(f"Refined minimum: F={res.fun:.8f} at theta={res.x[0]:.6f}, t={res.x[1]:.6f}")
print(f"(theta in radians; for reference pi/3={np.pi/3:.4f}, pi/2={np.pi/2:.4f})")
