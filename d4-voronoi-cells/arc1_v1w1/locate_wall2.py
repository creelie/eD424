import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

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
fixed = roots[[k for k in range(24) if k != i0]]

def dirs_at(theta,t):
    eperp = np.cos(t)*v1 + np.sin(t)*w1
    u1 = np.cos(theta)*u0 + np.sin(theta)*eperp
    return np.vstack([fixed, u1.reshape(1,-1)])

def vertex_count_and_degrees(theta,t,tol=1e-7):
    dirs = dirs_at(theta,t)
    A = dirs; b = -np.ones(len(dirs))
    hs = np.hstack([A,b.reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    vals = dirs @ verts.T
    tight = vals > 1-tol
    degrees = tight.sum(axis=0)
    return len(verts), tuple(sorted(degrees.tolist()))

for theta in [0.3,0.5,0.8,1.0,1.2,1.4]:
    ts = np.linspace(0.01, np.pi/2-0.01, 60)
    prev = None
    transitions = []
    for t in ts:
        sig = vertex_count_and_degrees(theta, t)
        if prev is not None and sig != prev:
            transitions.append(t)
        prev = sig
    print(f"theta={theta}: {len(transitions)} transitions at t~{[f'{x:.3f}' for x in transitions]}")
