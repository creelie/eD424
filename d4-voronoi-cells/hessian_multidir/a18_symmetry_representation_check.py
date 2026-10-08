#!/usr/bin/env python3
"""
Follow-up: (1) compute the commutant dimension of the 48-element group's
action on the 4D near-null eigenspace at A_18 (tells us how it decomposes
into irreducibles); (2) directly verify, as a strong consistency check,
that the quartic coefficient a4(v) = (F(sigma v)+F(-sigma v))/(2 sigma^4)
is invariant under the group's action on directions v in the near-null
subspace -- i.e. a4(rho(g) v) == a4(v) for random v and several g in the
stabilizer, which must hold exactly if F itself is truly group-invariant
(it should be, since every stabilizer element is an exact isometry of the
whole configuration).
"""
import numpy as np
import itertools
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

roots = build_roots()
roots_int = np.round(roots*np.sqrt(2)).astype(int)
root_to_idx = {tuple(r): i for i, r in enumerate(roots_int)}
A_18 = [0,2,4,5,6,7,8,9,10,11,12,13,16,17,20,21,22,23]
fixed_idx = [k for k in range(24) if k not in A_18]
fixed = roots[fixed_idx]
active_roots = [roots[k] for k in A_18]
bases = [tangent_basis(r) for r in active_roots]
m = len(A_18)
dim = 3*m
A18_pos = {r: i for i, r in enumerate(A_18)}

def F(coeffs):
    dirs_active = []
    for i in range(m):
        v = coeffs[3*i:3*i+3] @ bases[i]
        dirs_active.append(u_of_v(active_roots[i], v))
    dirs = np.vstack([fixed] + [d.reshape(1,-1) for d in dirs_active])
    return poly_volume(dirs) - 8.0

def numeric_hessian(F, dim, h):
    H = np.zeros((dim, dim))
    F0 = F(np.zeros(dim))
    for i in range(dim):
        ei = np.zeros(dim); ei[i] = h
        H[i,i] = (F(ei) - 2*F0 + F(-ei)) / h**2
    for i in range(dim):
        for j in range(i+1, dim):
            eij = np.zeros(dim); eij[i]=h; eij[j]=h
            eimj = np.zeros(dim); eimj[i]=h; eimj[j]=-h
            mij = np.zeros(dim); mij[i]=-h; mij[j]=h
            mimj = np.zeros(dim); mimj[i]=-h; mimj[j]=-h
            H[i,j]=H[j,i]=(F(eij)-F(eimj)-F(mij)+F(mimj))/(4*h**2)
    return H

print("Recomputing 54x54 Hessian and near-null eigenspace (h=0.02)...")
H = numeric_hessian(F, dim, 0.02)
eigvals, eigvecs = np.linalg.eigh(H)
nullspace = eigvecs[:, :4]

group_mats = []
for perm in itertools.permutations(range(4)):
    for signs in itertools.product([1,-1], repeat=4):
        M = np.zeros((4,4), dtype=int)
        for out_idx in range(4):
            M[out_idx, perm[out_idx]] = signs[out_idx]
        group_mats.append(M)

stabilizer = []
for M in group_mats:
    image_idxs = {}
    ok = True
    for i, r in enumerate(roots_int):
        rimg = tuple(M @ r)
        if rimg not in root_to_idx:
            ok = False; break
        image_idxs[i] = root_to_idx[rimg]
    if not ok:
        continue
    if set(image_idxs[i] for i in A_18) == set(A_18):
        stabilizer.append((M.astype(float), image_idxs))

def induced_rep(M, image_idxs):
    Rep = np.zeros((dim, dim))
    for i, root_idx in enumerate(A_18):
        sigma_root_idx = image_idxs[root_idx]
        j = A18_pos[sigma_root_idx]
        R = bases[j] @ M @ bases[i].T
        Rep[3*j:3*j+3, 3*i:3*i+3] = R
    return Rep

# restrict each group element to the 4D near-null eigenspace
rho = []
for M, image_idxs in stabilizer:
    Rep = induced_rep(M, image_idxs)
    r4 = nullspace.T @ Rep @ nullspace   # should be ~orthogonal 4x4
    rho.append(r4)

traces = [np.trace(r) for r in rho]
print("\n=== Step 1: character of the 4D near-null representation ===")
from collections import Counter
tr_rounded = [round(t,3) for t in traces]
print("Trace value counts:", Counter([round(t) for t in traces]))
print("Sample raw traces:", sorted(set(tr_rounded))[:15])

# commutant dimension: solve for 4x4 X with rho(g) X = X rho(g) for all g
print("\n=== Step 2: commutant dimension (Schur decomposition check) ===")
constraints = []
for r in rho:
    # (I kron r - r.T kron I) vec(X) = 0  <=>  r@X - X@r = 0
    K = np.kron(np.eye(4), r) - np.kron(r.T, np.eye(4))
    constraints.append(K)
Big = np.vstack(constraints)
u, s, vt = np.linalg.svd(Big)
tol = 1e-6
rank = np.sum(s > tol)
commutant_dim = 16 - rank
print(f"Commutant dimension: {commutant_dim}  (1=>irreducible 4D rep or 4 inequiv factors summing weirdly; "
      f"if reducible into pieces X of dims d_i with mult m_i, commutant=sum(m_i^2))")

print("\n=== Step 3: direct a4-invariance spot check ===")
rng = np.random.default_rng(3)
raw_v = rng.normal(size=4)
raw_v /= np.linalg.norm(raw_v)
v54 = nullspace @ raw_v  # embed into 54-dim space

def a4_estimate(direction54, sigma=0.03):
    Fp = F(sigma*direction54)
    Fm = F(-sigma*direction54)
    return (Fp+Fm)/(2*sigma**4)

base_a4 = a4_estimate(v54)
print(f"a4(v) = {base_a4:.6f}")
for k in range(6):
    M, image_idxs = stabilizer[3+7*k]  # scatter through the list, skip identity
    Rep = induced_rep(M, image_idxs)
    gv54 = Rep @ v54
    a4_g = a4_estimate(gv54)
    print(f"  g#{3+7*k}: a4(g.v) = {a4_g:.6f}   diff = {abs(a4_g-base_a4):.2e}")
