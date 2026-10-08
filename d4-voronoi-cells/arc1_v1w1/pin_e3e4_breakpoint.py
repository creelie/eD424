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

def active_set(theta,t):
    u1=u1_vec(theta,t)
    dirs=roots.copy()
    dirs[idx0]=u1
    Am=dirs
    bm=-np.ones(len(dirs))
    hs=np.hstack([Am,bm.reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    verts=hi.intersections
    on_cap=[v for v in verts if abs(np.dot(v,u1)-1)<1e-8]
    other_facets=set()
    for v in on_cap:
        vals = dirs@v
        for k,val in enumerate(vals):
            if k==idx0: continue
            if abs(val-1)<1e-7:
                other_facets.add(labels[k])
    return other_facets, on_cap

def has_e3e4(theta,t):
    s,_=active_set(theta,t)
    return '+e3+e4' in s

# bisect at t=0.1
t=0.1
lo,hi=0.15,0.25
assert has_e3e4(lo,t) and not has_e3e4(hi,t)
for _ in range(60):
    mid=(lo+hi)/2
    if has_e3e4(mid,t):
        lo=mid
    else:
        hi=mid
print(f"t={t}: breakpoint theta_c = {(lo+hi)/2:.12f}")
theta_c = (lo+hi)/2

# Also bisect for several other t values
print()
print("Scanning theta_c(t) across several t values:")
results={}
for t in [0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.35,np.pi/4-1e-6]:
    lo,hi=0.001,np.pi/2-0.01
    # find bracket where has_e3e4 transitions from True to False
    # first check has_e3e4(lo) True
    if not has_e3e4(lo,t):
        print(f"t={t}: e3e4 not active even at theta={lo}, skipping")
        continue
    # find hi where false
    hi_candidates=np.linspace(0.05,np.pi/2-0.01,60)
    hi_found=None
    for hc in hi_candidates:
        if not has_e3e4(hc,t):
            hi_found=hc
            break
    if hi_found is None:
        print(f"t={t}: never becomes inactive in range, skipping")
        continue
    lo2=0.001
    hi2=hi_found
    for _ in range(50):
        mid=(lo2+hi2)/2
        if has_e3e4(mid,t):
            lo2=mid
        else:
            hi2=mid
    tc=(lo2+hi2)/2
    results[t]=tc
    print(f"t={t:.6f}: theta_c={tc:.10f}   theta_c/t={tc/t if t>0 else float('nan'):.6f}")
