import numpy as np
from scipy.spatial import HalfspaceIntersection

def _build_root_system():
    roots, labels = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i], v[j] = si, sj
                    roots.append(v / np.sqrt(2))
                    sgn = lambda s: '+' if s == 1 else '-'
                    labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
    roots = np.array(roots)
    def find(vec):
        v = np.array(vec, dtype=float); v /= np.linalg.norm(v)
        idx = np.argmax(roots @ v)
        assert (roots @ v)[idx] > 1 - 1e-9
        return idx
    idx0 = find([1, 1, 0, 0])
    return roots, labels, idx0, roots[idx0], roots[find([1, -1, 0, 0])], roots[find([0, 0, 1, 1])]

roots, labels, idx0, u0, v1, w1 = _build_root_system()

def u1_vec(theta, t):
    return np.cos(theta)*u0 + np.sin(theta)*(np.cos(t)*v1 + np.sin(t)*w1)

def combinatorial_type(theta, t):
    u1 = u1_vec(theta, t)
    dirs = roots.copy(); dirs[idx0] = u1
    hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    sig = set()
    for v in hi.intersections:
        vals = dirs @ v
        touching = frozenset(labels[k] if k!=idx0 else 'u1' for k,val in enumerate(vals) if abs(val-1)<1e-6)
        sig.add(touching)
    return frozenset(sig), len(hi.intersections)

def theta_W(t): return 2*np.arctan(np.sin(t))
def theta_A(t): return 2*np.arctan(np.cos(t))
def theta_Y(t):
    val = 1.0/(np.sin(t)+np.cos(t)); return np.arcsin(min(val,1.0))
theta_U = np.pi/3

t1, t2, t3 = 0.169918, 0.463648, 0.615480

# representative points for each named band (well inside, away from boundaries)
bands = {
 '4a (0,W) t<t2':        (0.3, 0.5*theta_W(0.3)),
 '4b (0,Y) t>t2':        (0.6, 0.5*theta_Y(0.6)),
 'R1a (W,U) t<t1':       (0.1, 0.5*(theta_W(0.1)+theta_U)),
 'region2 (U,Y) t<t1':   (0.1, 0.5*(theta_U+theta_Y(0.1))),
 'R1c (Y,A) t<t1':       (0.1, 0.5*(theta_Y(0.1)+theta_A(0.1))),
 'P (W,Y) t1<t<t2':      (0.3, 0.5*(theta_W(0.3)+theta_Y(0.3))),
 'R2b (Y,U) t1<t<t2':    (0.3, 0.5*(theta_Y(0.3)+theta_U)),
 'M (U,A) t1<t<t3':      (0.3, 0.5*(theta_U+theta_A(0.3))),
 'M (U,A) t2<t<t3 chk':  (0.55, 0.5*(theta_U+theta_A(0.55))),
 'Q (Y,W) t2<t<t3':      (0.55, 0.5*(theta_Y(0.55)+theta_W(0.55))),
 'R (W,U) t2<t<t3':      (0.55, 0.5*(theta_W(0.55)+theta_U)),
 'S (Y,U) t>t3':         (0.7, 0.5*(theta_Y(0.7)+theta_U)),
 'T (U,W) t>t3':         (0.7, 0.5*(theta_U+theta_W(0.7))),
 'N (W,A) t>t3':         (0.7, 0.5*(theta_W(0.7)+theta_A(0.7))),
 'top (A,pi/2) t small': (0.1, 0.5*(theta_A(0.1)+np.pi/2)),
 'top (A,pi/2) t large': (0.7, 0.5*(theta_A(0.7)+np.pi/2)),
}

seen = {}
for name, (t,theta) in bands.items():
    sig, n = combinatorial_type(theta, t)
    key = None
    for k,v in seen.items():
        if v[0] == sig:
            key = k; break
    if key is None:
        key = f"TYPE{len(seen)}"
        seen[key] = (sig, n)
    print(f"{name:28s} theta={theta:.4f} t={t:.4f}  nverts={n:3d}  -> {key}")

print()
print(f"Total distinct types found among these samples: {len(seen)}")
for k,(sig,n) in seen.items():
    print(f"  {k}: nverts={n}")
