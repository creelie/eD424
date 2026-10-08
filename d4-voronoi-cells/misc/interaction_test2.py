"""
Follow-up, wider test of the superadditivity idea: is F_joint(active set)
>= sum of single-direction F's ALWAYS true, across many random e_perp
choices (not just root-aligned ones), many theta, and m=2,3,4 simultaneous
directions? If this ever fails (goes negative net of the sum), the whole
idea is dead. Bounded run: a few hundred random trials.
"""
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

rng = np.random.default_rng(12345)
n_trials = 300
worst_rel = 1e9
worst_info = None
n_negative_interaction = 0
n_total = 0

for trial in range(n_trials):
    m = rng.integers(2,5)  # 2,3, or 4 simultaneous directions
    idx = rng.choice(24, size=m, replace=False)
    perps = [random_perp(i, rng) for i in idx]
    thetas = rng.uniform(0.05, 0.9, size=m)

    f_joint = F_config(idx, perps, thetas)
    f_singles = []
    for i, p, t in zip(idx, perps, thetas):
        f_singles.append(F_config([i],[p],[t]))
    f_sum = sum(f_singles)
    interaction = f_joint - f_sum
    denom = sum(abs(x) for x in f_singles) + 1e-12
    rel = interaction/denom
    n_total += 1
    if interaction < -1e-6:
        n_negative_interaction += 1
    if rel < worst_rel:
        worst_rel = rel
        worst_info = (m, idx.copy(), thetas.copy(), f_singles, f_joint, interaction)

print(f"trials={n_total}  negative-interaction count={n_negative_interaction}")
print(f"worst (most negative) relative interaction found: {worst_rel:+.4%}")
m, idx, thetas, f_singles, f_joint, interaction = worst_info
print(f"  m={m} idx={idx} thetas={np.round(thetas,3)}")
print(f"  f_singles={np.round(f_singles,6)}  sum={sum(f_singles):.6f}  f_joint={f_joint:.6f}  interaction={interaction:+.6f}")
