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
    fset=set()
    for v in on_cap:
        vals=dirs@v
        for k,val in enumerate(vals):
            if k!=idx0 and abs(val-1)<1e-6:
                fset.add(labels[k])
    return frozenset(fset)

def all_transitions(t, n_theta=1000):
    thetas = np.linspace(0.003, np.pi/2-0.003, n_theta)
    prev = active_facets(thetas[0], t)
    trans = []
    for i in range(1,len(thetas)):
        cur = active_facets(thetas[i], t)
        if cur != prev:
            a,b = thetas[i-1], thetas[i]
            fa = prev
            for _ in range(45):
                mid=(a+b)/2
                fm = active_facets(mid,t)
                if fm==fa: a=mid
                else: b=mid
            trans.append((a+b)/2)
        prev = cur
    return trans

PI3 = np.pi/3
def formula_W(t): return 2*np.arctan(np.sin(t))
def formula_A(t): return 2*np.arctan(np.cos(t))
def formula_Y(t):
    val = 1.0/(np.sin(t)+np.cos(t))
    return np.arcsin(np.clip(val,-1,1))

TOL = 3e-3
def classify(t, trans):
    unexplained = []
    for th in trans:
        if abs(th-PI3)<TOL: continue
        if abs(th-formula_W(t))<TOL: continue
        if abs(th-formula_A(t))<TOL: continue
        if abs(th-formula_Y(t))<TOL: continue
        unexplained.append(th)
    return unexplained

print("Checking for ANY unexplained transitions across full t in (0,pi/4):")
max_unexplained = 0
for t in np.linspace(0.01, np.pi/4-0.0005, 40):
    trans = all_transitions(t)
    unexp = classify(t, trans)
    n_expected = 4  # pi/3, W, A, Y all expected present (generically)
    if unexp:
        print(f"  t={t:.5f}: {len(trans)} total transitions, "
              f"UNEXPLAINED: {[f'{x:.6f}' for x in unexp]}")
        max_unexplained = max(max_unexplained, len(unexp))
    else:
        pass  # silent if fully explained
print(f"\nMax unexplained transitions found at any sampled t: {max_unexplained}")
print("(0 means: pi/3 + W-curve + A-curve + Y-curve fully account for the")
print(" combinatorial breakpoint structure at every sampled point)")
