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
    return np.vstack([fixed, u1.reshape(1,-1)])  # index 23 = u1

def vertices_with_support(theta,t,tol=1e-6):
    dirs = dirs_at(theta,t)
    A = dirs; b = -np.ones(len(dirs))
    hs = np.hstack([A,b.reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    vals = dirs @ verts.T
    tight = vals > 1-tol
    supports = [frozenset(np.where(tight[:,k])[0].tolist()) for k in range(verts.shape[0])]
    return verts, supports

theta = 0.8
v40, s40 = vertices_with_support(theta, 0.40)
v50, s50 = vertices_with_support(theta, 0.50)
print("t=0.40 supports (unique quadruples/higher):")
for s in sorted(set(s40), key=lambda x: sorted(x)):
    print(" ", sorted(s))
print()
print("t=0.50 supports:")
for s in sorted(set(s50), key=lambda x: sorted(x)):
    print(" ", sorted(s))

new_supports = set(s50) - set(s40)
print()
print("NEW supports appearing between t=0.40 and t=0.50:", [sorted(x) for x in new_supports])
