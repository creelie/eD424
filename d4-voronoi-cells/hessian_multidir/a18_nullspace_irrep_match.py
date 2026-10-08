#!/usr/bin/env python3
"""
Identify exactly which representation of A_18's order-48 stabiliser
group the confirmed 4-dimensional near-null subspace of the joint
Hessian realises.

a18_symmetry_representation_check.py (elsewhere in this directory)
already showed the near-null subspace's character has aggregate trace
distribution {4: 2 elements, 1: 16, 0: 18, -2: 12} and a commutant
dimension of 2 -- i.e. exactly two irreducible pieces, each with
multiplicity 1. But TWO of the ten irreducible pairs from
a18_exact_character_table.py match that same AGGREGATE distribution:
  eps_perm + std3*eps_signs  ->  values [4,0,0,4,0,-2,0,-2,1,1]  (some order)
  eps_perm + std3*det        ->  values [4,0,0,4,-2,0,-2,0,1,1]  (some order)
These two only differ by swapping which of two equal-size class pairs
gets which value -- indistinguishable in aggregate, so this script
recomputes the near-null character PER CONJUGACY CLASS (tagging all 48
stabiliser elements individually, not a sample) and compares directly.

Run: python a18_nullspace_irrep_match.py
(takes under a minute; only double precision is needed here since we are
comparing finite-difference Hessian eigenvector traces against each
other, not extracting a small quartic signal)
"""
import numpy as np
import itertools
from collections import defaultdict
from scipy.spatial import HalfspaceIntersection, ConvexHull

# ---------------------------------------------------------------------
# Root system / joint Hessian machinery (shared with the rest of this
# directory's scripts).
# ---------------------------------------------------------------------
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

print("Recomputing 54x54 joint Hessian at A_18 (h=0.02)...")
H = numeric_hessian(F, dim, 0.02)
eigvals, eigvecs = np.linalg.eigh(H)
print("Smallest 6 eigenvalues:", eigvals[:6])
nullspace = eigvecs[:, :4]

# ---------------------------------------------------------------------
# The 48-element stabiliser, embedded in 4x4, tagged by (perm3, signs3),
# and its induced action on the 54-dim tangent space.
# ---------------------------------------------------------------------
FREE_COORDS = [0, 2, 3]; FIXED_COORD = 1
group_mats_tagged = []
for perm3 in itertools.permutations(range(3)):
    for signs3 in itertools.product([1,-1], repeat=3):
        M = np.zeros((4,4))
        M[FIXED_COORD, FIXED_COORD] = 1
        for a in range(3):
            for b in range(3):
                if perm3[a] == b:
                    M[FREE_COORDS[a], FREE_COORDS[b]] = signs3[a]
        group_mats_tagged.append((M, perm3, signs3))

stabilizer = []
for M, perm3, signs3 in group_mats_tagged:
    image_idxs = {}
    ok = True
    for i, r in enumerate(roots_int):
        rimg = tuple(np.round(M @ r).astype(int))
        if rimg not in root_to_idx:
            ok = False; break
        image_idxs[i] = root_to_idx[rimg]
    if not ok:
        continue
    if set(image_idxs[i] for i in A_18) == set(A_18):
        stabilizer.append((M, image_idxs, perm3, signs3))
print(f"Stabiliser size: {len(stabilizer)} (must be 48)")
assert len(stabilizer) == 48

def induced_rep(M, image_idxs):
    Rep = np.zeros((dim, dim))
    for i, root_idx in enumerate(A_18):
        sigma_root_idx = image_idxs[root_idx]
        j = A18_pos[sigma_root_idx]
        R = bases[j] @ M @ bases[i].T
        Rep[3*j:3*j+3, 3*i:3*i+3] = R
    return Rep

traces_by_elem = []
for M, image_idxs, perm3, signs3 in stabilizer:
    Rep = induced_rep(M, image_idxs)
    r4 = nullspace.T @ Rep @ nullspace
    traces_by_elem.append((perm3, signs3, np.trace(r4)))

# ---------------------------------------------------------------------
# Exact conjugacy classes of the abstract 3x3 group, with THE SAME
# ordering convention as a18_exact_character_table.py (identity class
# first, then sorted by (size, representative)) so class indices here
# line up directly with that script's printed character table.
# ---------------------------------------------------------------------
def mat_mult3(A,B):
    return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)) for i in range(3))
def mat_transpose3(A):
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))

group3 = []
elem_to_ps = {}
for perm in itertools.permutations(range(3)):
    for signs in itertools.product([1,-1], repeat=3):
        Mrow = [[0]*3 for _ in range(3)]
        for out_idx in range(3):
            Mrow[out_idx][perm[out_idx]] = signs[out_idx]
        Mt = tuple(tuple(row) for row in Mrow)
        group3.append(Mt)
        elem_to_ps[Mt] = (perm, signs)

visited = set(); classes = []
for g in group3:
    if g in visited:
        continue
    cls = set()
    for k in group3:
        kt = mat_transpose3(k)
        cls.add(mat_mult3(mat_mult3(k, g), kt))
    classes.append(cls); visited |= cls

identity3 = tuple(tuple(1 if i==j else 0 for j in range(3)) for i in range(3))
classes.sort(key=lambda c: (0 if identity3 in c else 1, len(c), sorted(c)[0]))
assert identity3 in classes[0]

ps_to_class = {}
for ci, cls in enumerate(classes):
    for elem in cls:
        ps_to_class[elem_to_ps[elem]] = ci

class_traces = defaultdict(list)
for perm3, signs3, tr in traces_by_elem:
    ci = ps_to_class[(perm3, signs3)]
    class_traces[ci].append(tr)

print("\nNumeric near-null character, grouped by exact conjugacy class")
print("(perfect per-class constancy -- min==max in every class -- is the")
print(" representation-theoretic prediction; any spread here would signal a bug):")
numeric_by_class = []
for ci in sorted(class_traces):
    vals = class_traces[ci]
    lo, hi, mean = min(vals), max(vals), np.mean(vals)
    numeric_by_class.append(mean)
    print(f"  class {ci} (size {len(vals)}): mean={mean:+.6f}  spread={hi-lo:.2e}")

# ---------------------------------------------------------------------
# Compare against the two candidate irreducible pairs (character values
# copied from a18_exact_character_table.py's printed table, same class
# ordering).
# ---------------------------------------------------------------------
eps_perm_vals       = [1, 1, 1, 1, -1, -1, -1, -1, 1, 1]
std3_eps_signs_vals = [3, 3, -1, -1, 1, -1, 1, -1, 0, 0]
std3_det_vals       = [3, 3, -1, -1, -1, 1, -1, 1, 0, 0]

cand_A = [a+b for a, b in zip(eps_perm_vals, std3_eps_signs_vals)]  # eps_perm + std3*eps_signs
cand_B = [a+b for a, b in zip(eps_perm_vals, std3_det_vals)]        # eps_perm + std3*det

print("\nCandidate eps_perm + std3*eps_signs:", cand_A)
print("Candidate eps_perm + std3*det:      ", cand_B)
print("Numeric (rounded):                  ", [round(v) for v in numeric_by_class])

def max_err(cand):
    return max(abs(a-b) for a, b in zip(cand, numeric_by_class))

errA, errB = max_err(cand_A), max_err(cand_B)
print(f"\nmax|numeric - candidate A| = {errA:.2e}")
print(f"max|numeric - candidate B| = {errB:.2e}")
assert errB < 1e-3 and errA > 0.5, "expected an exact, unambiguous match to candidate B only"
print("\n-> exact match (to finite-difference precision) is UNIQUELY candidate B:")
print("   the near-null subspace realises eps_perm (+) (std3 (x) det).")
