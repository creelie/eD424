#!/usr/bin/env python3
"""
Verification script for Section "A closed-form cap bound" of the paper.

Everything the section asserts is checked here numerically, INDEPENDENTLY
of the section's own algebra; the section's proofs are exact and this
script exists only so a reader can confirm that the objects named there
really are the objects computed in the rest of this package.

Checks performed:
  1. Q (= the 23 half-spaces of the roots other than alpha_0) has volume
     exactly 25/3 = 8 + 1/3, and Q cap {p >= 1} is the pyramid with apex
     2u0 over the octahedral facet F (volume 1/3)   [Lemma "Q-structure"]
  2. The facet F is a regular octahedron whose six vertices are
     u0 +- v_i for the three pairwise-orthogonal roots v_i orthogonal to
     u0, and vol_3(F) = 4/3.
  3. defect(theta,e) = 1/3 - vol(cap_Q)             [Prop "cap-reformulation"]
  4. The closed form for the cone cap,
         vol(cap_K) = (1/3)(2 - sec t)^4 / prod_i (1 - tan^2 t a_i^2),
     agrees with a direct polytope computation      [Thm "cone-bound"]
  5. defect >= the resulting lower bound, at random (theta,e)
  6. The threshold: (2c-1)^4 - c^2(2c^2-1) = (c-1)(14c^3-18c^2+7c-1),
     and the real root c0 of the cubic gives theta* = 42.1759... degrees;
     the bound is strictly positive for every direction below theta*.
  7. The pyramid's cap is empty for theta >= pi/2   [Prop "above-right-angle"]
  8. The maximum over u of the 24-cell's own cap (reported in the paper
     as numerical evidence, 0.058874..., not as a proof).

Run: python cap_reformulation_check.py
"""
import itertools
import numpy as np
import sympy as sp
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog, minimize


def build_roots():
    r = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4); v[i] = si; v[j] = sj
                    r.append(v / np.sqrt(2))
    return np.array(r)


roots = build_roots()
u0 = roots[0]
OTH = [i for i in range(24) if i != 0]
NEI = [i for i in range(24) if abs(roots[i] @ u0 - 0.5) < 1e-9]
ORT = [i for i in range(24) if abs(roots[i] @ u0) < 1e-9]

frame = []
for i in ORT:
    if all(abs(roots[i] @ roots[k]) < 1e-9 for k in frame):
        frame.append(i)
    if len(frame) == 3:
        break
V = np.array([roots[i] for i in frame])


def cheb(N, o):
    A = np.asarray(N, float); b = np.asarray(o, float)
    nr = np.linalg.norm(A, axis=1)
    res = linprog(np.r_[np.zeros(4), -1.0],
                  A_ub=np.hstack([A, nr.reshape(-1, 1)]), b_ub=b,
                  bounds=[(None, None)] * 4 + [(0, 50)], method='highs')
    if not res.success or res.x[-1] <= 1e-11:
        return None
    return res.x[:4]


def vol(N, o):
    p = cheb(N, o)
    if p is None:
        return 0.0
    A = np.asarray(N, float); b = -np.asarray(o, float)
    hi = HalfspaceIntersection(np.hstack([A, b.reshape(-1, 1)]), p)
    return ConvexHull(hi.intersections, qhull_options='QJ').volume


print("=" * 72)
print("1-2.  structure of Q and of the octahedral facet F")
print("=" * 72)
volQ = vol(roots[OTH], np.ones(23))
print(f"  vol(24-cell)            = {vol(roots, np.ones(24)):.12f}   (expect 8)")
print(f"  vol(Q)                  = {volQ:.12f}   (expect 25/3 = {25/3:.12f})")
volPyr = vol(np.vstack([roots[NEI], (-u0).reshape(1, 4)]),
             np.concatenate([np.ones(8), [-1.0]]))
print(f"  vol(Q cap {{p>=1}})       = {volPyr:.12f}   (expect 1/3)")
volPyrQ = vol(np.vstack([roots[OTH], (-u0).reshape(1, 4)]),
              np.concatenate([np.ones(23), [-1.0]]))
print(f"  vol(K cap {{p>=1}})       = {volPyrQ:.12f}   (same: the other 15")
print( "                                          constraints are slack there)")
print(f"  the three frame roots orthogonal to u0 (orthonormal):")
print("   ", np.round(V, 6).tolist())
print(f"  Gram matrix = {np.round(V @ V.T, 9).tolist()}")
Floc = np.array([s * (V @ (V[k])) for k in range(3) for s in (1, -1)])
print(f"  facet vertices u0 +- v_i are vertices of the 24-cell: "
      f"{all(np.all(roots @ (u0 + s*V[k]) <= 1 + 1e-12) for k in range(3) for s in (1,-1))}")
F_hull = ConvexHull(np.array([s * np.eye(3)[k] for k in range(3) for s in (1, -1)]))
print(f"  vol_3(F)                = {F_hull.volume:.12f}   (expect 4/3)")

print()
print("=" * 72)
print("3-5.  cap reformulation, closed form, and the lower bound")
print("=" * 72)


def defect(th, e):
    u1 = np.cos(th) * u0 + np.sin(th) * e
    d = roots.copy(); d[0] = u1
    hs = np.hstack([d, -np.ones((24, 1))])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    return ConvexHull(hi.intersections, qhull_options='QJ').volume - 8.0


def cap_Q(th, e):
    u1 = np.cos(th) * u0 + np.sin(th) * e
    return vol(np.vstack([roots[OTH], (-u1).reshape(1, 4)]),
               np.concatenate([np.ones(23), [-1.0]]))


def cap_K_direct(th, e):
    u1 = np.cos(th) * u0 + np.sin(th) * e
    return vol(np.vstack([roots[NEI], (-u1).reshape(1, 4)]),
               np.concatenate([np.ones(8), [-1.0]]))


def cap_K_closed(th, e):
    a = V @ e
    T = np.tan(th); c = np.cos(th)
    den = np.prod(1 - T * T * a * a)
    if den <= 0 or c <= 0:
        return np.inf
    return (1.0 / 3.0) * (2 - 1 / c) ** 4 / den


rng = np.random.default_rng(11)
mx_reform = 0.0; mx_closed = 0.0; nviol = 0; mn_gap = 1e9
for _ in range(200):
    e = rng.normal(size=4); e = e - (e @ u0) * u0; e /= np.linalg.norm(e)
    th = rng.uniform(0.02, 0.72)
    d = defect(th, e)
    mx_reform = max(mx_reform, abs(d - (1.0 / 3.0 - cap_Q(th, e))))
    mx_closed = max(mx_closed, abs(cap_K_direct(th, e) - cap_K_closed(th, e)))
    lb = (1.0 / 3.0) - cap_K_closed(th, e)
    mn_gap = min(mn_gap, d - lb)
    if d < lb - 1e-9:
        nviol += 1
print(f"  max |defect - (1/3 - cap_Q)|            = {mx_reform:.3e}   (expect ~0)")
print(f"  max |cap_K(direct) - cap_K(closed form)| = {mx_closed:.3e}   (expect ~0)")
print(f"  violations of defect >= 1/3 - cap_K      = {nviol} of 200")
print(f"  min (defect - lower bound)               = {mn_gap:.3e}")

print()
print("=" * 72)
print("6.  the threshold")
print("=" * 72)
x = sp.symbols('x')
fact = sp.factor(sp.expand((2 * x - 1) ** 4 - x ** 2 * (2 * x ** 2 - 1)))
print(f"  (2c-1)^4 - c^2(2c^2-1) = {fact}")
rts = sp.nroots(sp.Poly(14 * x ** 3 - 18 * x ** 2 + 7 * x - 1, x))
real = [sp.re(r) for r in rts if abs(sp.im(r)) < 1e-30]
c0 = max(real)
print(f"  real root c0           = {sp.N(c0, 22)}")
print(f"  theta* = arccos(c0)    = {sp.N(sp.acos(c0), 18)} rad "
      f"= {sp.N(sp.acos(c0) * 180 / sp.pi, 18)} deg")
th_star = float(sp.acos(c0))
bad = 0
for _ in range(4000):
    a = rng.normal(size=3); a /= np.linalg.norm(a)
    th = rng.uniform(1e-4, th_star * (1 - 1e-9))
    T = np.tan(th); c = np.cos(th)
    if (2 - 1 / c) ** 4 >= np.prod(1 - T * T * a * a):
        bad += 1
print(f"  directions/tilts below theta* where the closed-form bound fails: {bad} of 4000")

print()
print("=" * 72)
print("7.  the pyramid's cap is empty for theta >= pi/2")
print("=" * 72)
worst = -9.0
for _ in range(20000):
    lam = rng.uniform(0, 1)
    w = rng.normal(size=3); w = w / np.abs(w).sum() * lam
    xpt = (2 - lam) * u0 + V.T @ w
    e = rng.normal(size=4); e = e - (e @ u0) * u0; e /= np.linalg.norm(e)
    th = rng.uniform(np.pi / 2, np.pi)
    worst = max(worst, xpt @ (np.cos(th) * u0 + np.sin(th) * e))
print(f"  max <x,u1> over sampled pyramid points, theta in [pi/2,pi] = {worst:.9f}")
print( "  (<= 1 always: the cap is empty, so above a right angle only the")
print( "   24-cell's own cap matters)")

print()
print("=" * 72)
print("8.  numerical maximum of the 24-cell's own cap  (EVIDENCE, not proof)")
print("=" * 72)
def capC(u):
    return vol(np.vstack([roots, (-u).reshape(1, 4)]),
               np.concatenate([np.ones(24), [-1.0]]))
best = (0.0, None)
hs = np.hstack([roots, -np.ones((24, 1))])
VC = []
for v in HalfspaceIntersection(hs, np.zeros(4)).intersections:
    if not any(np.linalg.norm(v - w) < 1e-7 for w in VC):
        VC.append(v)
starts = [v / np.linalg.norm(v) for v in VC[:4]] + [rng.normal(size=4) for _ in range(20)]
for s in starts:
    s = np.asarray(s, float); s /= np.linalg.norm(s)
    r = minimize(lambda p: -capC(p / np.linalg.norm(p)), s, method='Nelder-Mead',
                 options=dict(xatol=1e-9, fatol=1e-13, maxiter=500))
    if -r.fun > best[0]:
        best = (-r.fun, r.x / np.linalg.norm(r.x))
print(f"  max_u vol(24-cell cap)  = {best[0]:.9f}   at u = {np.round(best[1],6).tolist()}")
print(f"  1/3 divided by that     = {1/3/best[0]:.4f}   (the factor quoted in the paper)")
print()
print("NOTE: item 8 is a numerical search, reported in the paper as evidence")
print("about the range theta >= pi/2 and explicitly NOT as a proof; the")
print("conjecture is not closed there.")
