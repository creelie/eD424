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
W = np.array([1,1,1,1])/np.sqrt(2)

def u1_vec(theta,t):
    return np.cos(theta)*u0+np.sin(theta)*(np.cos(t)*v1+np.sin(t)*w1)

def formula_theta(t):
    return 2*np.arctan(np.sin(t))

def W_status(theta,t):
    u1=u1_vec(theta,t)
    val = np.dot(W,u1)
    return val

def vertices_and_facets(theta,t):
    u1=u1_vec(theta,t)
    dirs=roots.copy()
    dirs[idx0]=u1
    hs=np.hstack([dirs,(-np.ones(len(dirs))).reshape(-1,1)])
    hi=HalfspaceIntersection(hs, np.zeros(4))
    verts=hi.intersections
    on_cap=[v for v in verts if abs(np.dot(v,u1)-1)<1e-7]
    result=[]
    for v in on_cap:
        vals=dirs@v
        touching=[labels[k] for k,val in enumerate(vals) if k!=idx0 and abs(val-1)<1e-6]
        result.append((v,touching))
    return result

for t in [0.3, 0.46, 0.5, 0.6, 0.7, np.pi/4-1e-4]:
    tf = formula_theta(t)
    print(f"\n=== t={t:.4f}, formula theta_c={tf:.6f} ===")
    print(f"<W,u1> at theta=tf: {W_status(tf,t):.10f}  (should be 1)")
    for theta in [tf-0.005, tf+0.005]:
        verts = vertices_and_facets(theta,t)
        wverts = [ (v,touch) for v,touch in verts if np.max(np.abs(v-W))<1e-3]
        print(f"  theta={theta:.6f}: total on-cap vertices={len(verts)}, near-W vertices={len(wverts)}")
        for v,touch in wverts:
            print(f"      W-vertex touching: {touch}")
