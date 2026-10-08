import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

roots = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si
                v[j] = sj
                roots.append(v / np.sqrt(2))
roots = np.array(roots)

def find(vec):
    v = np.array(vec, dtype=float)
    v /= np.linalg.norm(v)
    d = roots @ v
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]

def active_set(theta, t):
    e_perp = np.cos(t) * v1 + np.sin(t) * w1
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    Am = dirs
    bm = -np.ones(len(dirs))
    hs = np.hstack([Am, bm.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    on_cap = np.abs(verts @ dirs[idx0] - 1.0) < 1e-9
    cap_verts_idx = np.where(on_cap)[0]
    neighbor = set()
    for vi in cap_verts_idx:
        vv = verts[vi]
        for j in range(len(dirs)):
            if j == idx0:
                continue
            if abs(np.dot(vv, dirs[j]) - 1.0) < 1e-9:
                neighbor.add(j)
    return frozenset(neighbor)

def bisect(t, lo, hi, tol=1e-9):
    s_lo = active_set(lo, t)
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if active_set(mid, t) == s_lo:
            lo = mid
        else:
            hi = mid
        if hi-lo < tol:
            break
    return 0.5*(lo+hi)

def find_breaks(t, n_scan=500, th_lo=0.005, th_hi=np.pi/2-0.002):
    thetas = np.linspace(th_lo, th_hi, n_scan)
    prev = active_set(thetas[0], t)
    breaks = []
    for i in range(1, n_scan):
        cur = active_set(thetas[i], t)
        if cur != prev:
            br = bisect(t, thetas[i-1], thetas[i])
            breaks.append(br)
        prev = cur
    return breaks

# Reduced domain: t in [0, pi/4] only (by the H-midpoint-symmetry of the
# v1-w1 arc itself, Lemma lem:hadamardsymmetry -- F(theta,e_perp(t)) =
# F(theta,e_perp(pi/2-t)) exactly, so the breakpoint structure on
# [pi/4,pi/2] is the mirror image of [0,pi/4] and needs no separate scan).
ts = np.linspace(0.0, np.pi/4, 40)
all_rows = []
for t in ts:
    b = find_breaks(t)
    all_rows.append((t, b))

# Continuity-based curve tracking: greedily match breakpoints across
# consecutive t by nearest value.
curves = []  # each: list of (t, theta)
active_curves = []  # list of (last_theta, curve_index)
for t, b in all_rows:
    b_sorted = sorted(b)
    used = [False]*len(active_curves)
    new_active = []
    assigned = [False]*len(b_sorted)
    # try to match each active curve to nearest unassigned breakpoint
    for ci, (last_theta, idx) in enumerate(active_curves):
        best_j, best_d = None, 1e9
        for j, th in enumerate(b_sorted):
            if assigned[j]:
                continue
            d = abs(th - last_theta)
            if d < best_d:
                best_d = d; best_j = j
        if best_j is not None and best_d < 0.15:
            assigned[best_j] = True
            curves[idx].append((t, b_sorted[best_j]))
            new_active.append((b_sorted[best_j], idx))
    # any unassigned breakpoints start new curves
    for j, th in enumerate(b_sorted):
        if not assigned[j]:
            curves.append([(t, th)])
            new_active.append((th, len(curves)-1))
    active_curves = new_active

print(f"Total curve segments tracked: {len(curves)}")
for i, c in enumerate(curves):
    if len(c) < 3:
        continue
    t_start, th_start = c[0]
    t_end, th_end = c[-1]
    print(f"  Curve {i}: {len(c)} pts, t in [{t_start:.3f},{t_end:.3f}], "
          f"theta from {th_start:.4f} to {th_end:.4f}")

print()
print("Testing hypothesis: Curve 1 (the 'low' branch) is exactly theta = t:")
for t, th in curves[1][::5]:
    print(f"  t={t:.4f}  theta={th:.4f}  diff from t: {th-t:.2e}")
