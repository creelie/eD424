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

curve_fns = [theta_W, theta_A, theta_Y, lambda t: theta_U]

types_seen = {}
type_reps = {}
n_t, n_theta = 150, 200
skipped = 0
for t in np.linspace(0.005, np.pi/4-0.002, n_t):
    curve_vals = [f(t) for f in curve_fns]
    for theta in np.linspace(0.01, np.pi/2-0.005, n_theta):
        # skip points too close to any boundary curve (avoid grid artifacts)
        if any(abs(theta-cv) < 0.01 for cv in curve_vals):
            skipped += 1
            continue
        sig, n = combinatorial_type(theta, t)
        if sig not in types_seen:
            types_seen[sig] = len(types_seen)
            type_reps[sig] = (theta, t, n)

print(f"Distinct combinatorial types (away from boundary curves by >=0.01 rad): {len(types_seen)}")
print(f"(skipped {skipped} near-boundary grid points)")
for sig, idx in sorted(types_seen.items(), key=lambda kv: kv[1]):
    theta,t,n = type_reps[sig]
    print(f"  type {idx}: nverts={n}  rep(theta={theta:.4f}, t={t:.4f})")
