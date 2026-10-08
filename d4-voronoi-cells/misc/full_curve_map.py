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

def active_facets(theta,t):
    u1=u1_vec(theta,t)
    dirs=roots.copy(); dirs[idx0]=u1
    hs=np.hstack([dirs,(-np.ones(len(dirs))).reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    verts=hi.intersections
    on_cap=[v for v in verts if abs(np.dot(v,u1)-1)<1e-7]
    fset = set()
    for v in on_cap:
        vals=dirs@v
        for k,val in enumerate(vals):
            if k!=idx0 and abs(val-1)<1e-6:
                fset.add(labels[k])
    return frozenset(fset), len(on_cap)

def formula_theta_signvertex(t):
    return 2*np.arctan(np.sin(t))

# For each t, find ALL theta-transitions (fine grid + bisection),
# then classify each as "pi/3 line", "sign-vertex curve", or "OTHER" (new).
PI3 = np.pi/3

def all_transitions(t, n_theta=600):
    thetas = np.linspace(0.005, np.pi/2-0.005, n_theta)
    prev_set, _ = active_facets(thetas[0], t)
    trans = []
    for i in range(1,len(thetas)):
        cur_set,_ = active_facets(thetas[i], t)
        if cur_set != prev_set:
            a,b = thetas[i-1], thetas[i]
            fa = prev_set
            for _ in range(45):
                mid=(a+b)/2
                fm,_ = active_facets(mid,t)
                if fm==fa:
                    a=mid
                else:
                    b=mid
            trans.append((a+b)/2)
        prev_set = cur_set
    return trans

print(f"{'t':>8}  transitions (theta values)")
for t in np.linspace(0.02, np.pi/4-0.001, 12):
    trans = all_transitions(t)
    tf_sv = formula_theta_signvertex(t)
    tags = []
    for th in trans:
        if abs(th-PI3)<2e-3:
            tags.append(f"{th:.5f}(pi/3)")
        elif abs(th-tf_sv)<2e-3:
            tags.append(f"{th:.5f}(signvertex)")
        else:
            tags.append(f"{th:.5f}(OTHER)")
    print(f"{t:8.4f}  {tags}")
