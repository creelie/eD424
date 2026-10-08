import sys
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

def build_roots():
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i] = si; v[j] = sj
                    roots.append(v / np.sqrt(2))
    return np.array(roots)

def tangent_basis(root):
    B = []
    for e in np.eye(4):
        v = e - (e @ root) * root
        for b in B:
            v = v - (v @ b) * b
        n = np.linalg.norm(v)
        if n > 1e-8:
            B.append(v / n)
        if len(B) == 3:
            break
    return np.array(B)

def u_of_v(root, v):
    t = np.linalg.norm(v)
    if t < 1e-14:
        return root.copy()
    ep = v / t
    return np.cos(t) * root + np.sin(t) * ep

def poly_volume(dirs):
    A = dirs
    b = -np.ones(len(dirs))
    hs = np.hstack([A, b.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    hull = ConvexHull(hi.intersections, qhull_options='QJ')
    return hull.volume

def make_F(roots, active_idx):
    fixed_idx = [k for k in range(24) if k not in active_idx]
    fixed = roots[fixed_idx]
    active_roots = [roots[k] for k in active_idx]
    bases = [tangent_basis(r) for r in active_roots]
    m = len(active_idx)

    def F(coeffs):
        dirs_active = []
        for i in range(m):
            v = coeffs[3 * i:3 * i + 3] @ bases[i]
            dirs_active.append(u_of_v(active_roots[i], v))
        dirs = np.vstack([fixed] + [d.reshape(1, -1) for d in dirs_active])
        return poly_volume(dirs) - 8.0

    return F, 3 * m

def numeric_hessian(F, dim, h):
    H = np.zeros((dim, dim))
    F0 = F(np.zeros(dim))
    for i in range(dim):
        ei = np.zeros(dim)
        ei[i] = h
        H[i, i] = (F(ei) - 2 * F0 + F(-ei)) / h ** 2
    for i in range(dim):
        for j in range(i + 1, dim):
            eij = np.zeros(dim); eij[i] = h; eij[j] = h
            eimj = np.zeros(dim); eimj[i] = h; eimj[j] = -h
            mij = np.zeros(dim); mij[i] = -h; mij[j] = h
            mimj = np.zeros(dim); mimj[i] = -h; mimj[j] = -h
            H[i, j] = H[j, i] = (F(eij) - F(eimj) - F(mij) + F(mimj)) / (4 * h ** 2)
    return H, F0

roots = build_roots()
full_chain = [0, 4, 1, 15, 22, 10, 2, 11]   # length-8 induced path found exactly above

for length in range(3, 9):
    sub = full_chain[:length]
    Fp, dp = make_F(roots, sub)
    Hp, F0 = numeric_hessian(Fp, dp, 0.02)
    ev = np.linalg.eigvalsh(Hp)
    print(f"  length {length}: min eigenvalue = {ev.min():.6f}  (F0={F0:.2e})")

print()
print("cross-check length 8 at two more step sizes h (finite-difference robustness):")
Fp, dp = make_F(roots, full_chain[:8])
for h in (0.01, 0.03, 0.05):
    Hp, F0 = numeric_hessian(Fp, dp, h)
    ev = np.linalg.eigvalsh(Hp)
    print(f"  h={h}: min eigenvalue = {ev.min():.6f}")
