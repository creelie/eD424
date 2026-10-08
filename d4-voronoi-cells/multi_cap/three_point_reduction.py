#!/usr/bin/env python3
"""
three_point_reduction.py -- the cell volume inside a ball, as a function of
the pair and triple angles, and what the pair angles alone deliver.

Setting.  W is a contact configuration of m = 23 directions with cell
V = {x : <x, w_i> <= 1}, delta(theta) the angular distance from theta in
S^3 to the nearest w_i, and U(r) the measure of {delta > r}.  For any
R >= 1 the volume of V inside the ball of radius R is
    vol(V cap B(R)) = (1/4) [ 2 pi^2 + int_0^{arccos(1/R)} U(r) d(sec^4 r) ],
a lower bound for vol(V).  By the cap lemma (k contact directions in a cap
of angular radius r force sin^2 r >= (k-1)/(2k)) no point of S^3 lies
within r of three contact directions when r < r_* = arcsin(1/sqrt3) =
35.2644 degrees, nor within r of four when r < r_4 = arcsin sqrt(3/8) =
37.7612 degrees.  Inclusion-exclusion is therefore exact with two terms
below r_* and with three terms below r_4:

  (P)  vol(V) >= vol(V cap B(sqrt(3/2)))
              =  A_*  + sum_{i<j} omega_*(gamma_ij),

  (T)  vol(V) >= vol(V cap B(sqrt(8/5)))
              =  A_4  + sum_{i<j} omega_4(gamma_ij) - sum_{i<j<k} omega_3(gamma_ij, gamma_ik, gamma_jk),

with A_R = (1/4)[2 pi^2 + int_0^R (2 pi^2 - 23 C(r)) d(sec^4 r)],
omega_R(g) = (1/4) int_0^R Lambda(r, g) d(sec^4 r) for the lens measure
Lambda of two caps, and omega_3 the same integral of the triple measure.
(P) is prop:truncated of the paper for twenty-three contacts; (T), in the
ball of radius sqrt(8/5), is not used there.

Part 1 prints the constants of (P): A_* = 7.907144, omega_*(60) =
0.00144541, the value 8.034340 at a deletion of a root, and the number of
pairs at 60 degrees, 64.24, that carry (P) past 8; and the same for the
integration stopped at r_23 = 34.6106 degrees (7.997885 at the deletion, 91
pairs needed).

Part 2 prints the constants of (T): A_4 = 7.647558, omega_4(60) =
0.00565947, omega_3(60,60,60) = 5.65e-5 by Monte Carlo, the value
8.140848 at a deletion (against an independent quadrature, 8.14068,
quoted from an earlier version of this package) and 7.968684 at the root
system itself (24 contacts).

Part 3 is the pair-angle relaxation of (P): minimise sum omega_*(gamma)
over all distributions of 253 pair angles in [60, 180] degrees subject to
the constraints every configuration satisfies (total mass, the
second-order Bonferroni inequality at every radius, positive definiteness
of the Gegenbauer polynomials up to degree 24), and compare the minimum
with the target 8 - A_* = 0.092856.  It reaches 0.073797, four fifths of
the target; the r_23 version reaches one half.  The minimiser is
printed.  Nothing in Part 3 is a proof of anything beyond the value of
the relaxation on the grid.

Usage: python3 three_point_reduction.py
"""
import numpy as np
from itertools import combinations
from scipy.integrate import quad
from scipy.optimize import brentq, linprog

PI = np.pi
T3 = 2 * PI ** 2

def C(r): return PI * (2 * r - np.sin(2 * r))

def lens(r, g):
    if r <= g / 2: return 0.0
    f = lambda t: np.sin(t) ** 2 * (1 - np.tan(g / 2) / np.tan(t))
    v, _ = quad(f, g / 2, r, limit=300)
    return 4 * PI * v

def dsec4(r): return 4 * np.tan(r) / np.cos(r) ** 4

def gegenbauer_S3(k, x):
    x = np.clip(x, -1.0, 1.0); t = np.arccos(x); s = np.sin(t)
    return np.where(np.abs(s) < 1e-12, np.sign(x) ** k, np.sin((k + 1) * t) / ((k + 1) * np.maximum(s, 1e-300)))

def roots():
    V = []
    for i, j in combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4); v[i], v[j] = si, sj; V.append(v / np.sqrt(2))
    return np.array(V)

def Aconst(m, R): return 0.25 * (T3 + quad(lambda r: (T3 - m * C(r)) * dsec4(r), 0, R, limit=300)[0])
def omega(g, R): return 0.25 * quad(lambda r: lens(r, g) * dsec4(r), g / 2, R, limit=300)[0] if g < 2 * R else 0.0

def main():
    r_star = np.arcsin(np.sqrt(1 / 3)); r4 = np.arcsin(np.sqrt(3 / 8))
    r23 = brentq(lambda r: 2 * r - np.sin(2 * r) - 2 * PI / 23, 1e-9, PI)
    R = roots(); Wdel = R[1:]
    def counts(W):
        G = W @ W.T; n = len(W)
        tight = sum(1 for i in range(n) for j in range(i + 1, n) if abs(G[i, j] - .5) < 1e-9)
        tri = sum(1 for i, j, k in combinations(range(n), 3)
                  if abs(G[i, j] - .5) < 1e-9 and abs(G[i, k] - .5) < 1e-9 and abs(G[j, k] - .5) < 1e-9)
        return tight, tri
    t23, tr23 = counts(Wdel); t24, tr24 = counts(R)

    print("=" * 70); print("Part 1: the pair-only bound (P), integration to r_*"); print("=" * 70)
    print(f"  r_23 = {np.degrees(r23):.4f} deg, r_* = {np.degrees(r_star):.4f} deg, r_4 = {np.degrees(r4):.4f} deg")
    for name, Rr in (("r_23", r23), ("r_*", r_star)):
        A = Aconst(23, Rr); w = omega(PI / 3, Rr); need = 8 - A
        print(f"  [{name}] A = {A:.6f}, omega(60) = {w:.8f}, deletion ({t23} tight pairs) = {A + t23 * w:.6f}, "
              f"pairs at 60 needed for 8: {need / w:.2f}, sec(R) = {1/np.cos(Rr):.6f}")
    print("  omega_*(gamma) at 60, 62, 64, 66, 68, 70 degrees: " +
          ", ".join(f"{omega(np.radians(d), r_star):.3e}" for d in (60, 62, 64, 66, 68, 70)) +
          f"; zero from {np.degrees(2*r_star):.3f} degrees")
    print()

    print("=" * 70); print("Part 2: the three-point bound (T), integration to r_4"); print("=" * 70)
    A4 = Aconst(23, r4); A4_24 = Aconst(24, r4); w4 = omega(PI / 3, r4)
    rng = np.random.default_rng(1)
    w = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]]) / np.sqrt(2)
    N = 8_000_000
    X = rng.normal(size=(N, 4)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    mn = (X @ w.T).min(axis=1)
    rs = np.linspace(r_star, r4, 41)
    L3 = np.array([T3 * np.mean(mn >= np.cos(r)) for r in rs])
    om3 = 0.25 * np.trapezoid(L3 * dsec4(rs), rs)
    print(f"  A_4 = {A4:.6f}, omega_4(60) = {w4:.8f}, omega_3(60,60,60) = {om3:.3e} (Monte Carlo, {N} points)")
    print(f"  deletion: {t23} tight pairs, {tr23} tight triangles: {A4 + t23*w4 - tr23*om3:.6f}  (quadrature 8.14068)")
    print(f"  root system, 24 contacts: {t24} tight pairs, {tr24} tight triangles: {A4_24 + t24*w4 - tr24*om3:.6f} (quadrature 7.96841; the truncation drops the deep holes)")
    print(f"  target for (T): sum omega_4 - sum omega_3 >= 8 - A_4 = {8 - A4:.6f}, that is {(8-A4)/w4:.2f} pairs at 60 degrees before the triple term")
    print()

    print("=" * 70); print("Part 3: the pair-angle relaxation of (P)"); print("=" * 70)
    grid = np.radians(np.linspace(60.0, 180.0, 1201))
    tgrid = np.linspace(r23, np.radians(90.0), 60)
    Arows = [[-lens(t, g) for g in grid] for t in tgrid]; blb = [-(23 * C(t) - T3) for t in tgrid]
    for k in range(1, 25):
        Arows.append([-gegenbauer_S3(k, np.cos(g)) for g in grid]); blb.append(23 / 2.0)
    Arows = np.array(Arows); blb = np.array(blb)
    for name, Rr in (("r_23", r23), ("r_*", r_star)):
        c = np.array([omega(g, Rr) for g in grid]); need = 8 - Aconst(23, Rr)
        res = linprog(c, A_ub=Arows, b_ub=blb, A_eq=np.ones((1, len(grid))), b_eq=[253],
                      bounds=[(0, None)] * len(grid), method='highs')
        sup = [(round(float(np.degrees(g)), 1), round(float(x), 1)) for g, x in zip(grid, res.x) if x > 1e-3]
        print(f"  [{name}] minimum of the relaxation {res.fun:.6f} against the target {need:.6f}: "
              f"{100*res.fun/need:.1f} per cent; support (degrees, mass): {sup}")
    print()
    print("(P) and (T) are exact identities for the truncated volume and valid lower bounds for vol(V).")
    print("The relaxation of Part 3 is a lower bound for the pair sum over a grid of angles; it is not a proof.")

if __name__ == "__main__":
    main()
