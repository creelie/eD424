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

i0 = find([1,1,0,0])
u0 = roots[i0]
v1 = np.array([1,-1,0,0])/np.sqrt(2)
w1 = np.array([0,0,1,1])/np.sqrt(2)
fixed_idx = [k for k in range(24) if k != i0]
fixed = roots[fixed_idx]

def dirs_at(theta,t):
    eperp = np.cos(t)*v1 + np.sin(t)*w1
    u1 = np.cos(theta)*u0 + np.sin(theta)*eperp
    return np.vstack([fixed, u1.reshape(1,-1)])

def tight_set(theta,t,tol=1e-7):
    dirs = dirs_at(theta,t)
    A = dirs; b = -np.ones(len(dirs))
    hs = np.hstack([A,b.reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    vals = dirs @ verts.T
    present = np.where((vals > 1-tol).any(axis=1))[0]
    return set(present.tolist()), len(verts)

theta = 0.8
for t in [0.40, 0.45, 0.457, 0.46, 0.50]:
    s, nv = tight_set(theta, t)
    print(f"t={t:.4f}  n_active_facets={len(s)}  n_verts={nv}  facets={sorted(s)}")
