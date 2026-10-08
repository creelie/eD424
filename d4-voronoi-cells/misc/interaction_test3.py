import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

def build_roots():
    roots = []
    for i in range(4):
        for j in range(i+1,4):
            for si in (1,-1):
                for sj in (1,-1):
                    v=[0,0,0,0]; v[i]=si; v[j]=sj
                    roots.append(np.array(v,dtype=float)/np.sqrt(2))
    return np.array(roots)

roots = build_roots()

def random_perp(root_idx, rng):
    r = roots[root_idx]
    v = rng.normal(size=4)
    v = v - np.dot(v,r)*r
    return v/np.linalg.norm(v)

def hull_F(dirs):
    A=dirs; b=-np.ones(len(dirs))
    hs=np.hstack([A,b.reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    hull=ConvexHull(hi.intersections, qhull_options='QJ')
    return hull.volume - 8.0

def F_config(active_idx, perps, thetas):
    dirs = roots.copy()
    for idx, perp, th in zip(active_idx, perps, thetas):
        dirs[idx] = np.cos(th)*roots[idx] + np.sin(th)*perp
    return hull_F(dirs)

rng = np.random.default_rng(777)
n_trials = 2000
min_f = 1e9
min_info = None
n_neg = 0
for trial in range(n_trials):
    m = rng.integers(2,7)
    idx = rng.choice(24, size=m, replace=False)
    perps = [random_perp(i, rng) for i in idx]
    thetas = rng.uniform(0.02, 1.0, size=m)
    f = F_config(idx, perps, thetas)
    if f < 0:
        n_neg += 1
    if f < min_f:
        min_f = f
        min_info = (m, idx.copy(), thetas.copy())

print(f"trials={n_trials}  negative F_joint count={n_neg}")
print(f"min F_joint found: {min_f:.8f}")
m, idx, thetas = min_info
print(f"  at m={m} idx={idx} thetas={np.round(thetas,4)}")
