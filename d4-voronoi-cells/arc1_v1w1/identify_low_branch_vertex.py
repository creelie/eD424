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
    v = np.array(vec, dtype=float)
    v /= np.linalg.norm(v)
    d = roots @ v
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]

def active_set(theta, t):
    e_perp = np.cos(t) * v1 + np.sin(t) * w1
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy(); dirs[idx0]=u1
    Am=dirs; bm=-np.ones(len(dirs))
    hs=np.hstack([Am,bm.reshape(-1,1)])
    hi=HalfspaceIntersection(hs,np.zeros(4))
    verts=hi.intersections
    on_cap=np.abs(verts@dirs[idx0]-1.0)<1e-9
    cap_verts_idx=np.where(on_cap)[0]
    neighbor=set()
    for vi in cap_verts_idx:
        vv=verts[vi]
        for j in range(len(dirs)):
            if j==idx0: continue
            if abs(np.dot(vv,dirs[j])-1.0)<1e-9:
                neighbor.add(j)
    return frozenset(neighbor)

def bisect(t,lo,hi,tol=1e-10):
    s_lo=active_set(lo,t)
    for _ in range(70):
        mid=0.5*(lo+hi)
        if active_set(mid,t)==s_lo: lo=mid
        else: hi=mid
        if hi-lo<tol: break
    return 0.5*(lo+hi)

t0 = 0.3222
br = bisect(t0, 0.55, 0.68)
print(f"Low-branch breakpoint at t={t0}: theta={br:.8f}")

def polytope_vertices(theta,t):
    e_perp=np.cos(t)*v1+np.sin(t)*w1
    u1=np.cos(theta)*u0+np.sin(theta)*e_perp
    dirs=roots.copy(); dirs[idx0]=u1
    Am=dirs; bm=-np.ones(len(dirs))
    hs=np.hstack([Am,bm.reshape(-1,1)])
    hi=HalfspaceIntersection(hs,np.zeros(4))
    return hi.intersections, dirs

verts_lo, dirs = polytope_vertices(br-1e-6, t0)
verts_hi, _ = polytope_vertices(br+1e-6, t0)

def near(v,w,tol=1e-4):
    return np.max(np.abs(v-w))<tol

# find vertices that appear/disappear
set_lo = [tuple(np.round(v,5)) for v in verts_lo]
set_hi = [tuple(np.round(v,5)) for v in verts_hi]
only_lo = set(set_lo)-set(set_hi)
only_hi = set(set_hi)-set(set_lo)
print("Vertices only below breakpoint:")
for v in only_lo: print(f"  {v}")
print("Vertices only above breakpoint:")
for v in only_hi: print(f"  {v}")
