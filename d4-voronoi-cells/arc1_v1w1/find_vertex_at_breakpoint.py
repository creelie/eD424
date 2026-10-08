import numpy as np
from scipy.spatial import HalfspaceIntersection

roots = []
labels = []
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v = np.zeros(4)
                v[i]=si; v[j]=sj
                roots.append(v/np.sqrt(2))
                sgn=lambda s: '+' if s==1 else '-'
                labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
roots=np.array(roots)

def find(vec):
    v=np.array(vec,dtype=float); v/=np.linalg.norm(v)
    d=roots@v
    idx=np.argmax(d)
    assert d[idx]>1-1e-9
    return idx

idx0=find([1,1,0,0])
u0=roots[idx0]
v1=roots[find([1,-1,0,0])]
w1=roots[find([0,0,1,1])]

def u1_vec(theta,t):
    eperp = np.cos(t)*v1+np.sin(t)*w1
    return np.cos(theta)*u0+np.sin(theta)*eperp

t=0.1
theta_c=0.199008393643  # from bisection

for theta in [theta_c-1e-6, theta_c+1e-6]:
    u1=u1_vec(theta,t)
    dirs=roots.copy()
    dirs[idx0]=u1
    Am=dirs
    bm=-np.ones(len(dirs))
    hs=np.hstack([Am,bm.reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    verts=hi.intersections
    on_cap=[v for v in verts if abs(np.dot(v,u1)-1)<1e-8]
    print(f"theta={theta:.9f}: {len(on_cap)} vertices on moving cap")
    for v in on_cap:
        vals = dirs@v
        touching = [labels[k] for k,val in enumerate(vals) if k!=idx0 and abs(val-1)<1e-6]
        if '+e3+e4' in touching:
            print(f"  vertex {np.round(v,6)}  touching: {touching}")
    print()

# Now precisely at theta_c, find vertex touching u1, +e3+e4 and others via near-exact bisection
theta=theta_c
u1=u1_vec(theta,t)
dirs=roots.copy()
dirs[idx0]=u1
Am=dirs
bm=-np.ones(len(dirs))
hs=np.hstack([Am,bm.reshape(-1,1)])
hi=HalfspaceIntersection(hs, np.zeros(4))
verts=hi.intersections
on_cap=[v for v in verts if abs(np.dot(v,u1)-1)<1e-7]
print(f"At theta_c={theta_c}: {len(on_cap)} on-cap vertices")
for v in on_cap:
    vals=dirs@v
    touching=[labels[k] for k,val in enumerate(vals) if k!=idx0 and abs(val-1)<1e-5]
    print(f"  v={np.round(v,6)}  touching={touching}")
