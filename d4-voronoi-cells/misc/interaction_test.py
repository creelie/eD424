"""
One bounded, honest test of a genuinely different strategic idea before
giving up on full closure: does the free-volume "interaction" between two
simultaneously-deviating active directions vanish (or become provably small)
when their base roots are far apart (Gram = 0, -1/2, -1), so that only the
Gram = 1/2 (adjacent / clique) case actually needs the hard case-by-case
treatment already attempted? If true in general this would shrink the
Multi-Direction conjecture to a finite list of clique configurations.
This is NOT a proof attempt -- it's a 10-minute numerical scan to see if the
idea is even plausible before investing more real effort in it.
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
import itertools

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

def find(vec):
    v=np.array(vec,dtype=float); v/=np.linalg.norm(v)
    d=roots@v
    k=int(np.argmax(d))
    assert d[k]>1-1e-9
    return k

def perp_dir(root_idx, other_root_vec):
    # a direction within the plane of root and some other reference vector,
    # orthogonal to root, used to deviate root smoothly
    r = roots[root_idx]
    raw = other_root_vec - np.dot(other_root_vec, r)*r
    n = np.linalg.norm(raw)
    if n < 1e-9:
        # pick any orthogonal direction
        for k in range(4):
            e = np.zeros(4); e[k]=1.0
            raw = e - np.dot(e,r)*r
            if np.linalg.norm(raw) > 1e-6:
                break
        n = np.linalg.norm(raw)
    return raw/n

def F_single(root_idx, perp, theta, active_override=None):
    """Volume of V0 with ONLY root_idx replaced by a deviated direction."""
    dirs = roots.copy()
    u = np.cos(theta)*roots[root_idx] + np.sin(theta)*perp
    dirs[root_idx] = u
    return hull_F(dirs)

def hull_F(dirs):
    A=dirs; b=-np.ones(len(dirs))
    hs=np.hstack([A,b.reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    hull=ConvexHull(hi.intersections, qhull_options='QJ')
    return hull.volume - 8.0

def F_joint(i, perp_i, theta_i, j, perp_j, theta_j):
    dirs = roots.copy()
    dirs[i] = np.cos(theta_i)*roots[i] + np.sin(theta_i)*perp_i
    dirs[j] = np.cos(theta_j)*roots[j] + np.sin(theta_j)*perp_j
    return hull_F(dirs)

# pick a base root, and partners at each of the 4 possible Gram values
i0 = find([1,1,0,0])
base = roots[i0]
gram_targets = {}
for k in range(24):
    if k == i0: continue
    g = round(float(np.dot(base, roots[k])*2)/1)/1  # <alpha_i,alpha_j> in {2,1,0,-1,-2}; unit gram = /2
    g_unit = round(float(np.dot(base, roots[k])), 4)
    gram_targets.setdefault(g_unit, []).append(k)

print("Available unit-root Gram values from i0 and example partner indices:")
for g, ks in sorted(gram_targets.items()):
    print(f"  gram={g:+.3f}  count={len(ks)}  example partner={ks[0]}")

perp_i0 = perp_dir(i0, roots[find([1,0,1,0])])

print()
print("Testing additivity F_joint(theta,theta) vs F_single(i0,theta)+F_single(j,theta) ")
print("for one representative partner at each Gram value, theta=0.3 and theta=0.5:")
print()
for g, ks in sorted(gram_targets.items()):
    j = ks[0]
    perp_j = perp_dir(j, roots[i0])
    for theta in [0.3, 0.5]:
        f1 = F_single(i0, perp_i0, theta)
        f2 = F_single(j, perp_j, theta)
        fj = F_joint(i0, perp_i0, theta, j, perp_j, theta)
        interaction = fj - (f1+f2)
        rel = interaction/(abs(f1)+abs(f2)+1e-12)
        print(f"gram={g:+.3f} theta={theta:.2f}: F1={f1:.6f} F2={f2:.6f} sum={f1+f2:.6f} "
              f"F_joint={fj:.6f}  interaction={interaction:+.6f}  rel={rel:+.4%}")
    print()
