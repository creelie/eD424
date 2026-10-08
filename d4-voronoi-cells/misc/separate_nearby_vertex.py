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
    v = np.array(vec, dtype=float); v/=np.linalg.norm(v)
    d = roots@v; idx=np.argmax(d); assert d[idx]>1-1e-9
    return idx
idx0 = find([1,1,0,0])
u0 = roots[idx0]
v1 = roots[find([1,-1,0,0])]
w1 = roots[find([0,0,1,1])]

def polytope_vertices(theta,t):
    e_perp=np.cos(t)*v1+np.sin(t)*w1
    u1=np.cos(theta)*u0+np.sin(theta)*e_perp
    dirs=roots.copy(); dirs[idx0]=u1
    Am=dirs; bm=-np.ones(len(dirs))
    hs=np.hstack([Am,bm.reshape(-1,1)])
    hi=HalfspaceIntersection(hs,np.zeros(4))
    return hi.intersections, dirs

def full_facet_list(v,dirs,tol=1e-5):
    return [j for j in range(24) if abs(np.dot(v,dirs[j])-1.0)<tol]

t0 = 0.1
theta_c = 0.1990074415
# look further away to separate the two vertices
for theta in [theta_c-0.05, theta_c-0.02, theta_c+0.02, theta_c+0.05]:
    verts,dirs = polytope_vertices(theta,t0)
    print(f"theta={theta:.4f}:")
    for v in verts:
        if abs(v[1]-np.sqrt(2))<0.05 and abs(v[2])<0.05 and abs(v[3])<0.05:
            f = full_facet_list(v,dirs)
            flabs = [labels[j] if j!=idx0 else 'u1' for j in f]
            print(f"    v={np.round(v,5)}  facets={flabs}")
    print()
