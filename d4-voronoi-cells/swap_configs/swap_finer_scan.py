import numpy as np
from scipy.spatial import HalfspaceIntersection

roots = []
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v = np.zeros(4); v[i]=si; v[j]=sj
                roots.append(v/np.sqrt(2))
roots = np.array(roots)

def find(vec):
    v = np.array(vec,dtype=float)/np.linalg.norm(vec)
    d = roots @ v
    k = np.argmax(d)
    assert d[k] > 1-1e-9
    return k

ia = find([1,1,0,0]); ib = find([1,0,1,0])
alpha, beta = roots[ia], roots[ib]
fixed_idx = [k for k in range(24) if k not in (ia,ib)]
fixed = roots[fixed_idx]
s = alpha@beta
beta_perp = beta - s*alpha; beta_perp/=np.linalg.norm(beta_perp)
alpha_perp = alpha - s*beta; alpha_perp/=np.linalg.norm(alpha_perp)
thetastar = np.pi/3

def dirs_at(theta):
    u1 = np.cos(theta)*alpha + np.sin(theta)*beta_perp
    u2 = np.cos(theta)*beta + np.sin(theta)*alpha_perp
    return np.vstack([fixed, u1.reshape(1,-1), u2.reshape(1,-1)])

def vertex_supports(theta, tol=1e-6):
    dirs = dirs_at(theta)
    A = dirs; b=-np.ones(len(dirs))
    hs = np.hstack([A,b.reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    vals = dirs @ verts.T
    tight = vals > 1-tol
    supports = [frozenset(np.where(tight[:,k])[0].tolist()) for k in range(verts.shape[0])]
    return supports

ts = np.linspace(0.02, thetastar-0.02, 80)
prev = None
transitions = []
for t in ts:
    supp = set(vertex_supports(t))
    if prev is not None and supp != prev:
        transitions.append((t, prev-supp, supp-prev))
    prev = supp

print(f"{len(transitions)} transitions found over 80 samples in (0.02, {thetastar-0.02:.4f})")
for t, removed, added in transitions:
    print(f"  t={t:.4f}  removed={[sorted(x) for x in removed]}  added={[sorted(x) for x in added]}")
