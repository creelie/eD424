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
    return frozenset(fset)

def formula_theta_signvertex(t):
    return 2*np.arctan(np.sin(t))

PI3 = np.pi/3

def all_transitions(t, n_theta=800):
    thetas = np.linspace(0.005, np.pi/2-0.005, n_theta)
    prev_set = active_facets(thetas[0], t)
    trans = []
    for i in range(1,len(thetas)):
        cur_set = active_facets(thetas[i], t)
        if cur_set != prev_set:
            a,b = thetas[i-1], thetas[i]
            fa = prev_set
            for _ in range(45):
                mid=(a+b)/2
                fm = active_facets(mid,t)
                if fm==fa:
                    a=mid
                else:
                    b=mid
            trans.append((a+b)/2)
        prev_set = cur_set
    return trans

def classify_and_strip(t, trans):
    """Remove transitions matching pi/3 or the sign-vertex formula (tight
    tolerance), return the leftover 'other' transitions."""
    tf_sv = formula_theta_signvertex(t)
    leftover = []
    for th in trans:
        if abs(th-PI3) < 1e-3:
            continue
        if abs(th-tf_sv) < 1e-3:
            continue
        leftover.append(th)
    return leftover

# Trace the leftover curves across a fine t grid using nearest-neighbor
# continuity matching.
ts = np.linspace(0.01, np.pi/4-0.0005, 60)
tracks = []  # list of dict: {'t':[], 'theta':[]}
prev_leftover = None
for t in ts:
    trans = all_transitions(t)
    leftover = classify_and_strip(t, trans)
    leftover.sort()
    if prev_leftover is None:
        for th in leftover:
            tracks.append({'t':[t], 'theta':[th]})
    else:
        # match each new leftover to nearest existing track's last theta
        used = set()
        for th in leftover:
            best_i, best_d = None, None
            for i,tr in enumerate(tracks):
                if tr['t'][-1] < t - 0.05:  # track died out
                    continue
                if i in used: continue
                d = abs(tr['theta'][-1]-th)
                if best_d is None or d<best_d:
                    best_d, best_i = d, i
            if best_i is not None and best_d < 0.15:
                tracks[best_i]['t'].append(t)
                tracks[best_i]['theta'].append(th)
                used.add(best_i)
            else:
                tracks.append({'t':[t],'theta':[th]})
    prev_leftover = leftover

print(f"Found {len(tracks)} leftover track(s) (excluding pi/3 and sign-vertex curves):")
for i,tr in enumerate(tracks):
    if len(tr['t']) < 3:
        continue
    print(f"\nTrack {i}: {len(tr['t'])} points, t range [{tr['t'][0]:.4f},{tr['t'][-1]:.4f}], "
          f"theta range [{min(tr['theta']):.4f},{max(tr['theta']):.4f}]")
    # print first and last few points
    for k in list(range(min(3,len(tr['t'])))) :
        print(f"    t={tr['t'][k]:.4f} theta={tr['theta'][k]:.6f}")
    print("    ...")
    for k in range(max(0,len(tr['t'])-3), len(tr['t'])):
        print(f"    t={tr['t'][k]:.4f} theta={tr['theta'][k]:.6f}")
