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

def all_transitions(t, n_theta=400):
    thetas = np.linspace(0.01, np.pi/2-0.01, n_theta)
    ns = [n_oncap_vertices(th,t) for th in thetas]
    trans = []
    for i in range(len(thetas)-1):
        if ns[i]!=ns[i+1]:
            # bisect
            a,b = thetas[i], thetas[i+1]
            na = ns[i]
            for _ in range(40):
                mid=(a+b)/2
                if n_oncap_vertices(mid,t)==na:
                    a=mid
                else:
                    b=mid
            trans.append(((a+b)/2, ns[i], ns[i+1]))
    return trans

for t in [0.0, 0.05, 0.15, 0.3, 0.45, 0.6, 0.623, 0.65, 0.7, np.pi/4]:
    trans = all_transitions(max(t,1e-4))
    print(f"t={t:.4f}: transitions:")
    for theta, na, nb in trans:
        print(f"    theta={theta:.6f}  {na} -> {nb}")
