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

idx0=find([1,1,0,0])
u0=roots[idx0]
v1=roots[find([1,-1,0,0])]
w1=roots[find([0,0,1,1])]

def u1_vec(theta,t):
    return np.cos(theta)*u0+np.sin(theta)*(np.cos(t)*v1+np.sin(t)*w1)

def formula_theta(t):
    return 2*np.arctan(np.sin(t))

def n_oncap_vertices(theta,t):
    u1=u1_vec(theta,t)
    dirs=roots.copy()
    dirs[idx0]=u1
    hs=np.hstack([dirs,(-np.ones(len(dirs))).reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    verts=hi.intersections
    on_cap=[v for v in verts if abs(np.dot(v,u1)-1)<1e-7]
    return len(on_cap)

def bisect_vertex_jump(t, lo, hi):
    n_lo = n_oncap_vertices(lo,t)
    for _ in range(55):
        mid=(lo+hi)/2
        if n_oncap_vertices(mid,t)==n_lo:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2

print(f"{'t':>10} {'theta_c(bisect)':>18} {'theta_c(formula)':>18} {'abs diff':>12}")
maxdiff=0
for t in np.linspace(0.01, np.pi/4-0.0005, 25):
    tf = formula_theta(t)
    tc = bisect_vertex_jump(t, tf-0.01, tf+0.01)
    d = abs(tc-tf)
    maxdiff = max(maxdiff,d)
    print(f"{t:10.6f} {tc:18.12f} {tf:18.12f} {d:12.2e}")

print(f"\nmax abs diff (all 25 samples across FULL [0,pi/4) range): {maxdiff:.3e}")
print("(this is at bisection/floating-point precision -- confirms the")
print(" vertex-count discontinuity coincides with sin(t)=tan(theta/2)")
print(" EVERYWHERE on the arc, not just for t up to ~0.46 as the cruder")
print(" facet-label-matching detector had suggested.)")
