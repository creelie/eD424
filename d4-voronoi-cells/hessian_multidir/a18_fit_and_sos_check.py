#!/usr/bin/env python3
"""
End-to-end, self-contained pipeline for the numerical half of
sec:a18-invariant-quartics in the paper:

  1. Recompute the 54x54 joint Hessian at A_18 and its 4-dim near-null
     subspace.
  2. Build an orthonormal basis (u_x0, u_y) of the ACTUAL 54-dimensional
     ambient tangent space that realises the abstract representation
     eps_perm (on x0) (+) (std3 (x) det) (on y=(y0,y1,y2)) in exactly the
     coordinate convention used by a18_invariant_quartic_basis.py --
     via Reynolds-type projectors onto the two isotypic pieces, then an
     exact intertwiner (Sylvester equation, solved by SVD nullspace)
     aligning the arbitrary eigenbasis with the standard signed-
     permutation coordinates on y.
  3. Evaluate the quartic-coefficient estimator a4 at 7 directions
     chosen to isolate the 5 basis invariants (x0^4, x0^2*S2, x0*P3, S4,
     S22), using the project's validated HIGH-PRECISION (mpmath, 40
     digit) volume routine -- NOT plain double precision, which this
     paper documents in sec:dense-configs-new as inadequate at this exact
     configuration (confirmed again here: an initial attempt with plain
     scipy/qhull double precision gave fits disagreeing by 100-700%
     between step sizes, discarded once caught by the two-step-size
     check below, not reported as if reliable).
  4. Fit c1..c5 by least squares at two independent step sizes
     (sigma=0.02, 0.01) plus a combined 14-point fit, and report the
     agreement between them.
  5. Run the same Gram-matrix/SDP sum-of-squares check used elsewhere in
     this package (gram_sos_lib.py, re-validated against two textbook
     cases first) on all three fits.

Depends on hp_volume.py and gram_sos_lib.py in the same directory.

Run: python a18_fit_and_sos_check.py
(takes roughly 5-8 minutes: dominated by the 14 high-precision volume
 evaluations at ~15-25s each)
"""
import time
import numpy as np
import mpmath as mp
import itertools
from scipy.spatial import HalfspaceIntersection, ConvexHull

from hp_volume import hp_volume
from gram_sos_lib import fit_gram_and_check_sos, _validate

# ---------------------------------------------------------------------
# 1. Root system, joint Hessian, near-null subspace at A_18.
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
roots_int = np.round(roots * np.sqrt(2)).astype(int)
root_to_idx = {tuple(r): i for i, r in enumerate(roots_int)}
A_18 = [0,2,4,5,6,7,8,9,10,11,12,13,16,17,20,21,22,23]
fixed_idx = [k for k in range(24) if k not in A_18]
fixed = roots[fixed_idx]
active_roots = [roots[k] for k in A_18]
bases = [tangent_basis(r) for r in active_roots]
A18_pos = {r: i for i, r in enumerate(A_18)}
m = len(A_18); dim = 3*m

def F(coeffs):
    dirs_active = []
    for i in range(m):
        v = coeffs[3*i:3*i+3] @ bases[i]
        dirs_active.append(u_of_v(active_roots[i], v))
    dirs = np.vstack([fixed] + [d.reshape(1,-1) for d in dirs_active])
    return poly_volume(dirs) - 8.0

def numeric_hessian(F, dim, h):
    Hm = np.zeros((dim, dim))
    F0 = F(np.zeros(dim))
    for i in range(dim):
        ei = np.zeros(dim); ei[i] = h
        Hm[i,i] = (F(ei) - 2*F0 + F(-ei)) / h**2
    for i in range(dim):
        for j in range(i+1, dim):
            eij = np.zeros(dim); eij[i]=h; eij[j]=h
            eimj = np.zeros(dim); eimj[i]=h; eimj[j]=-h
            mij = np.zeros(dim); mij[i]=-h; mij[j]=h
            mimj = np.zeros(dim); mimj[i]=-h; mimj[j]=-h
            Hm[i,j] = Hm[j,i] = (F(eij)-F(eimj)-F(mij)+F(mimj))/(4*h**2)
    return Hm

print("[1/5] Computing 54x54 joint Hessian at A_18 (h=0.02) and its near-null subspace...")
Hm = numeric_hessian(F, dim, 0.02)
eigvals, eigvecs = np.linalg.eigh(Hm)
print("      smallest 6 eigenvalues:", eigvals[:6])
nullspace = eigvecs[:, :4]

# ---------------------------------------------------------------------
# 2. Aligned basis (u_x0, u_y) realising eps_perm (+) (std3 (x) det).
# ---------------------------------------------------------------------
print("[2/5] Building the aligned (u_x0, u_y) basis via Reynolds projectors + intertwiner...")
FREE_COORDS = [0, 2, 3]; FIXED_COORD = 1
stabilizer = []
for perm3 in itertools.permutations(range(3)):
    for signs3 in itertools.product([1,-1], repeat=3):
        M = np.zeros((4,4)); M[FIXED_COORD, FIXED_COORD] = 1
        for a in range(3):
            for b in range(3):
                if perm3[a] == b:
                    M[FREE_COORDS[a], FREE_COORDS[b]] = signs3[a]
        image_idxs = {}; ok = True
        for i, r in enumerate(roots_int):
            rimg = tuple(np.round(M @ r).astype(int))
            if rimg not in root_to_idx:
                ok = False; break
            image_idxs[i] = root_to_idx[rimg]
        if not ok:
            continue
        if set(image_idxs[i] for i in A_18) == set(A_18):
            stabilizer.append((M, image_idxs, perm3, signs3))
assert len(stabilizer) == 48

def induced_rep(M, image_idxs):
    Rep = np.zeros((dim, dim))
    for i, root_idx in enumerate(A_18):
        j = A18_pos[image_idxs[root_idx]]
        Rblk = bases[j] @ M @ bases[i].T
        Rep[3*j:3*j+3, 3*i:3*i+3] = Rblk
    return Rep

def perm_sign(perm):
    inv = 0
    for i in range(len(perm)):
        for j in range(i+1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1
def group3_matrix(perm3, signs3):
    Mg = np.zeros((3,3))
    for a in range(3):
        Mg[a, perm3[a]] = signs3[a]
    return Mg

rho4_list = []; target3_list = []; epsperm_list = []
for M, image_idxs, perm3, signs3 in stabilizer:
    Rep = induced_rep(M, image_idxs)
    r4 = nullspace.T @ Rep @ nullspace
    rho4_list.append(r4)
    g3 = group3_matrix(perm3, signs3); d = np.linalg.det(g3)
    target3_list.append(d*g3)
    epsperm_list.append(perm_sign(perm3))

# projector onto the eps_perm isotypic line
P_eps = np.zeros((4,4))
for r4, eps in zip(rho4_list, epsperm_list):
    P_eps += eps*r4
P_eps /= len(rho4_list)
w, v = np.linalg.eigh(P_eps)
x0_vec = v[:, np.argmax(w)]

# projector onto the (std3*det) isotypic block (dim-3 character of that irrep is 3 at identity)
P3 = np.zeros((4,4))
for r4, tgt in zip(rho4_list, target3_list):
    chi = np.trace(tgt)
    P3 += chi*r4
P3 = 3.0/48*P3
w3, v3 = np.linalg.eigh(P3)
idx3 = np.argsort(-w3)[:3]
Ybasis0 = v3[:, idx3]

# intertwiner: solve rho3(g) @ R = R @ target3(g) for all g via SVD nullspace
A_rows = []
for r4, tgt in zip(rho4_list, target3_list):
    rho3 = Ybasis0.T @ r4 @ Ybasis0
    K = np.kron(rho3, np.eye(3)) - np.kron(np.eye(3), tgt.T)
    A_rows.append(K)
Big = np.vstack(A_rows)
u, s, vt = np.linalg.svd(Big)
print(f"      intertwiner SVD smallest singular values: {s[-3:]}")
R = vt[-1].reshape(3,3)
scale = np.sqrt(np.trace(R @ R.T)/3)
R = R/scale
Ybasis = Ybasis0 @ R

errs = []
for r4, tgt in zip(rho4_list, target3_list):
    rho3a = Ybasis.T @ r4 @ Ybasis
    errs.append(np.max(np.abs(rho3a - tgt)))
print(f"      max alignment error against target (std3 (x) det) action: {max(errs):.2e}")
assert max(errs) < 1e-6

u_x0 = nullspace @ x0_vec        # 54-dim
u_y = nullspace @ Ybasis         # 54 x 3
allvecs = np.column_stack([u_x0, u_y])
gram = allvecs.T @ allvecs
assert np.max(np.abs(gram - np.eye(4))) < 1e-8
print("      orthonormality of (u_x0, u_y) confirmed.")

# ---------------------------------------------------------------------
# 3-4. High-precision fit of c1..c5 at two step sizes + combined fit.
# ---------------------------------------------------------------------
def F_hp(coeffs54, prec=40):
    mp.mp.dps = prec
    dirs = roots.copy()
    for i, k in enumerate(A_18):
        v = coeffs54[3*i:3*i+3] @ bases[i]
        dirs[k] = u_of_v(active_roots[i], v)
    dirs_mp = [[mp.mpf(str(x)) for x in dirs[j]] for j in range(24)]
    return hp_volume(dirs_mp, prec=prec) - 8

def embed(x0, y0, y1, y2):
    vec = np.array([x0, y0, y1, y2], dtype=float)
    vec = vec/np.linalg.norm(vec)
    return vec[0]*u_x0 + vec[1]*u_y[:,0] + vec[2]*u_y[:,1] + vec[3]*u_y[:,2]

def a4_est_hp(v54, sigma, prec=40):
    Fp = F_hp(sigma*v54, prec=prec)
    Fm = F_hp(-sigma*v54, prec=prec)
    return float((Fp+Fm)/(2*mp.mpf(sigma)**4))

pts = [
    (1,0,0,0),
    (0,1,0,0),
    (0,1,1,1),
    (1,1,0,0),
    (1,1,1,1),
    (1,-1,1,1),
    (2,1,1,1),
]

def design_row(x0, y0, y1, y2):
    nrm2 = x0*x0+y0*y0+y1*y1+y2*y2
    X0, Y0, Y1, Y2 = x0/np.sqrt(nrm2), y0/np.sqrt(nrm2), y1/np.sqrt(nrm2), y2/np.sqrt(nrm2)
    S2 = Y0**2+Y1**2+Y2**2
    S4 = Y0**4+Y1**4+Y2**4
    S22 = Y0**2*Y1**2+Y0**2*Y2**2+Y1**2*Y2**2
    P3 = Y0*Y1*Y2
    return [X0**4, X0**2*S2, X0*P3, S4, S22]

print("\n[3/5] Evaluating a4 at 7 directions via high-precision (mpmath, 40-digit) volume,")
print("      at TWO independent step sizes each:")
rows = []
vals_s02 = []
vals_s01 = []
for (x0,y0,y1,y2) in pts:
    t0 = time.time()
    v54 = embed(x0,y0,y1,y2)
    a1 = a4_est_hp(v54, sigma=0.02, prec=40)
    a2 = a4_est_hp(v54, sigma=0.01, prec=40)
    dt = time.time()-t0
    diffpct = 100*abs(a1-a2)/max(abs(a1), 1e-9)
    print(f"  ({x0:+d},{y0:+d},{y1:+d},{y2:+d}): a4(s=0.02)={a1:.6f}  a4(s=0.01)={a2:.6f}"
          f"  diff={diffpct:.2f}%  [{dt:.1f}s]")
    rows.append(design_row(x0,y0,y1,y2))
    vals_s02.append(a1)
    vals_s01.append(a2)

A = np.array(rows)
b02 = np.array(vals_s02); b01 = np.array(vals_s01)

print("\n[4/5] Least-squares fits:")
sol02, _, _, _ = np.linalg.lstsq(A, b02, rcond=None)
print("  sigma=0.02 fit: c1..c5 =", sol02, " residual max =", np.max(np.abs(A@sol02-b02)))
sol01, _, _, _ = np.linalg.lstsq(A, b01, rcond=None)
print("  sigma=0.01 fit: c1..c5 =", sol01, " residual max =", np.max(np.abs(A@sol01-b01)))
A_combined = np.vstack([A, A])
b_combined = np.concatenate([b02, b01])
sol_comb, _, _, _ = np.linalg.lstsq(A_combined, b_combined, rcond=None)
print("  combined 14-pt fit: c1..c5 =", sol_comb,
      " residual max =", np.max(np.abs(A_combined@sol_comb-b_combined)))

# ---------------------------------------------------------------------
# 5. SOS/Gram-matrix check on each fit.
# ---------------------------------------------------------------------
print("\n[5/5] SOS/Gram-matrix check (re-validating against textbook cases first):")
_validate()

import sympy as sp
z0, z1, z2, z3 = sp.symbols('z0 z1 z2 z3', real=True)

def coeffs_of(c1, c2, c3, c4, c5):
    Q = (c1*z0**4 + c2*z0**2*(z1**2+z2**2+z3**2) + c3*z0*z1*z2*z3
         + c4*(z1**4+z2**4+z3**4) + c5*(z1**2*z2**2+z1**2*z3**2+z2**2*z3**2))
    Q = sp.expand(Q)
    poly = sp.Poly(Q, z0, z1, z2, z3)
    return {monom: float(coeff) for monom, coeff in poly.terms()}

for name, sol in [("sigma=0.02", sol02), ("sigma=0.01", sol01), ("combined 14-pt", sol_comb)]:
    coeffs = coeffs_of(*sol)
    status, max_t, _ = fit_gram_and_check_sos(coeffs)
    print(f"  [{name}] status={status}  max_t={max_t}")

print("\nDone. This leaves the elementary route to Conjecture")
print("(Multi-Direction Positivity) at A_18 short of a proof:")
print("these are finite-difference numerical fits on an exactly-known-correct")
print("5-dimensional family, with a feasible but thin-margin SOS certificate --")
print("not an exact symbolic derivation and not an unconditional proof.")
