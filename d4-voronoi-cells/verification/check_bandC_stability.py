import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

roots, labels = [], []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                vv = np.zeros(4)
                vv[i], vv[j] = si, sj
                roots.append(vv / np.sqrt(2))
                sgn = lambda s: '+' if s == 1 else '-'
                labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
roots = np.array(roots)

def find(vec):
    v_ = np.array(vec, dtype=float)
    v_ /= np.linalg.norm(v_)
    idx = np.argmax(roots @ v_)
    assert (roots @ v_)[idx] > 1 - 1e-9
    return idx

idx0 = find([1, 1, 0, 0])
U0 = roots[idx0]
W1 = roots[find([0, 0, 1, 1])]
V2 = roots[find([0, 0, 1, -1])]

def u1_vec(th, tt):
    return np.cos(th) * U0 + np.sin(th) * (np.cos(tt) * W1 + np.sin(tt) * V2)

def polytope_full(th, tt):
    u1 = u1_vec(th, tt)
    dirs = roots.copy()
    dirs[idx0] = u1
    hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    out = []
    for vpt in hi.intersections:
        vals = dirs @ vpt
        touching = frozenset(
            labels[k] if k != idx0 else 'u1'
            for k, val in enumerate(vals) if abs(val - 1) < 1e-6)
        out.append((vpt, touching))
    return out

def theta_W(tt): return 2 * np.arctan(np.cos(tt))
def theta_C(tt): return np.arcsin(1.0/(np.cos(tt)+np.sin(tt)))
def theta_B(tt): return 2*np.arctan(np.sin(tt))
theta_U = np.pi/3

t_cross, t3 = 0.169918, 0.615480

print("=== Sub-part C<->W, t in (0, t_cross) ===")
ts = np.linspace(0.005, t_cross-0.005, 8)
pats=[]
for tt in ts:
    thC = theta_C(tt); thW = theta_W(tt)
    th = 0.5*(thC+thW)
    info = polytope_full(th, tt)
    pat = frozenset(p for _,p in info)
    pats.append((len(info), pat))
    print(f"  t={tt:.4f} thetaC={thC:.4f} thetaW={thW:.4f} theta={th:.4f}: {len(info)} vertices")
same = all(p==pats[0][1] for _,p in pats)
print(f"pattern identical across samples: {same}")
print()

print("=== Sub-part B<->W, t in (t3, pi/4) ===")
ts2 = np.linspace(t3+0.005, np.pi/4-0.005, 8)
pats2=[]
for tt in ts2:
    thB = theta_B(tt); thW = theta_W(tt)
    th = 0.5*(thB+thW)
    info = polytope_full(th, tt)
    pat = frozenset(p for _,p in info)
    pats2.append((len(info), pat))
    print(f"  t={tt:.4f} thetaB={thB:.4f} thetaW={thW:.4f} theta={th:.4f}: {len(info)} vertices")
same2 = all(p==pats2[0][1] for _,p in pats2)
print(f"pattern identical across samples: {same2}")
