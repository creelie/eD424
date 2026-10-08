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

def has_e3e4(theta,t):
    u1=u1_vec(theta,t)
    dirs=roots.copy()
    dirs[idx0]=u1
    Am=dirs
    bm=-np.ones(len(dirs))
    hs=np.hstack([Am,bm.reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    verts=hi.intersections
    on_cap=[v for v in verts if abs(np.dot(v,u1)-1)<1e-8]
    for v in on_cap:
        vals = dirs@v
        for k,val in enumerate(vals):
            if k==idx0: continue
            if abs(val-1)<1e-7 and labels[k]=='+e3+e4':
                return True
    return False

def formula_theta(t):
    return 2*np.arctan(np.sin(t))

def bisect_near(t, target, halfwidth, n=40):
    ts = np.linspace(target-halfwidth, target+halfwidth, n)
    vals = [has_e3e4(th,t) for th in ts]
    # find sign change closest to target
    best=None
    for i in range(len(ts)-1):
        if vals[i] and not vals[i+1]:
            mid_est = (ts[i]+ts[i+1])/2
            d = abs(mid_est-target)
            if best is None or d<best[0]:
                best=(d, ts[i], ts[i+1])
    if best is None:
        return None
    lo,hi=best[1],best[2]
    for _ in range(50):
        mid=(lo+hi)/2
        if has_e3e4(mid,t):
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2

print(f"{'t':>8} {'theta_c(bisect)':>18} {'theta_c(formula)':>18} {'abs diff':>12}")
maxdiff=0
ts = list(np.linspace(0.02, np.pi/4-0.001, 20))
for t in ts:
    tf = formula_theta(t)
    tc = bisect_near(t, tf, 0.03)
    if tc is None:
        print(f"{t:8.5f}  no crossing found near formula value {tf:.6f}")
        continue
    d=abs(tc-tf)
    maxdiff=max(maxdiff,d)
    print(f"{t:8.5f} {tc:18.12f} {tf:18.12f} {d:12.2e}")
print(f"\nmax abs diff over all sampled t: {maxdiff:.3e}")
print("At t=pi/4:", formula_theta(np.pi/4), " vs arccos(1/3)=", np.arccos(1/3), " diff=", abs(formula_theta(np.pi/4)-np.arccos(1/3)))
