import numpy as np
from scipy.spatial import HalfspaceIntersection

roots, labels = [], []
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
    idx=np.argmax(roots@v)
    assert (roots@v)[idx]>1-1e-9
    return idx

idx0=find([1,1,0,0]); u0=roots[idx0]
v1=roots[find([1,-1,0,0])]; w1=roots[find([0,0,1,1])]

def u1_vec(theta,t):
    return np.cos(theta)*u0+np.sin(theta)*(np.cos(t)*v1+np.sin(t)*w1)

def n_oncap_vertices(theta,t):
    u1=u1_vec(theta,t)
    dirs=roots.copy(); dirs[idx0]=u1
    hs=np.hstack([dirs,(-np.ones(len(dirs))).reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    return len([v for v in hi.intersections if abs(np.dot(v,u1)-1)<1e-7])

def formula_theta_track0(t):
    val = 1.0/(np.sin(t)+np.cos(t))
    return np.arcsin(np.clip(val,-1,1))

def bisect_within_window(t, tf, halfwidth=0.01, n=50):
    ts = np.linspace(tf-halfwidth, tf+halfwidth, n)
    ns = [n_oncap_vertices(th,t) for th in ts]
    changes = [i for i in range(len(ts)-1) if ns[i]!=ns[i+1]]
    if not changes:
        return None
    best = min(changes, key=lambda i: abs((ts[i]+ts[i+1])/2 - tf))
    a,b = ts[best], ts[best+1]
    na = n_oncap_vertices(a,t)
    for _ in range(50):
        mid=(a+b)/2
        if n_oncap_vertices(mid,t)==na:
            a=mid
        else:
            b=mid
    return (a+b)/2

print(f"{'t':>10} {'theta_c(bisect)':>18} {'theta_c(formula)':>18} {'abs diff':>12}")
maxdiff=0
n_matched=0
for t in np.linspace(0.005, np.pi/4-0.0003, 40):
    tf = formula_theta_track0(t)
    tc = bisect_within_window(t, tf)
    if tc is None:
        print(f"{t:10.6f}  NO transition found near formula theta={tf:.6f}")
        continue
    d = abs(tc-tf)
    maxdiff = max(maxdiff,d)
    n_matched += 1
    print(f"{t:10.6f} {tc:18.12f} {tf:18.12f} {d:12.2e}")

print(f"\nMatched {n_matched}/40. max abs diff: {maxdiff:.3e}")
