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

print(f"{'t':>8} {'theta_f':>10} {'n(-)':>6} {'n(+)':>6} {'jump?':>6}")
all_jump = True
for t in np.linspace(0.02, np.pi/4-0.001, 30):
    tf = formula_theta(t)
    eps = 0.003
    n_minus = n_oncap_vertices(tf-eps, t)
    n_plus  = n_oncap_vertices(tf+eps, t)
    jump = (n_minus != n_plus)
    all_jump = all_jump and jump
    print(f"{t:8.4f} {tf:10.6f} {n_minus:6d} {n_plus:6d} {str(jump):>6}")

print()
print("ALL sampled t show a vertex-count jump exactly at formula theta:", all_jump)
