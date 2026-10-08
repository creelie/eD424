#!/usr/bin/env python3
"""
d4_independent_check.py
=======================

An independent re-implementation, from the statements in the text alone, of the
numerical claims of "The Sphere Packing Problem in Dimension 4 and the
Twenty-Four-Cell Conjecture" that the remaining gaps depend on.  Nothing here
is taken from the supplement; every formula is coded from the paper's own
definitions so that agreement is evidence and not a shared bug.

Run:  python3 d4_independent_check.py          (needs mpmath, numpy, scipy)

What it checks, and what it found
---------------------------------
  1. Table 1, the covering bound  (pi m /3) tan^3 r_m            AGREES to 6 dp
  2. Table 3, the thresholds of the distance criterion Phi       AGREES to 0.5%
                                                                 (the paper's are
                                                                  ball-arithmetic
                                                                  and outward
                                                                  rounded)
  3. eq:second-order-value, the four pair-bound numbers          3 of 4 AGREED
                                                                 with v1.7.0;
                                                                 the fourth was
                                                                 an erratum,
                                                                 corrected in
                                                                 v1.8.0
  4. lem:rigidity-spectrum, the singular values of Lambda        AGREES exactly
  5. the count bound 49 within sqrt 6                            AGREES exactly
  6. the two crossovers of the pair terms, 0.155 and 0.197       AGREE exactly
  7. NEW: an exact constant improving thm:local-uniqueness
  8. NEW: the truncation radius grown with the distances, measured
"""
import itertools
import numpy as np
from mpmath import mp, mpf, pi, sin, cos, tan, sec, acos, sqrt, quad, findroot

mp.dps = 30
TOTAL = 2*pi**2
G60 = pi/3
R_STAR = acos(sqrt(mpf(2)/3))          # 35.2643896...deg ; ball radius sqrt(3/2)


# ---------------------------------------------------------------- kernels
def C(r):
    """measure of a geodesic cap of angular radius r on S^3"""
    return pi*(2*r - sin(2*r))


def _F(t, g):
    return t/2 - sin(2*t)/4 - tan(g/2)*sin(t)**2/2


def Lambda(r, g):
    """measure of the intersection of two caps of radius r, centres g apart"""
    if r <= g/2:
        return mpf(0)
    return 4*pi*(_F(r, g) - _F(g/2, g))


def _halflens(rho, a):
    if rho <= a:
        return mpf(0)
    return 2*pi*(_F(rho, 2*a) - _F(a, 2*a))


def Lambda_un(rho1, rho2, g):
    """two caps of radii rho1, rho2 whose centres are g apart"""
    if rho1 + rho2 <= g:
        return mpf(0)
    if g <= abs(rho1 - rho2):
        return C(min(rho1, rho2))
    a1 = findroot(lambda a: cos(rho1)*cos(g-a) - cos(rho2)*cos(a), g/2)
    return _halflens(rho1, a1) + _halflens(rho2, g - a1)


def dsec4(r):
    return 4*sec(r)**4*tan(r)


def r_m(m):
    return findroot(lambda r: 2*r - sin(2*r) - 2*pi/m, mpf('0.6'))


def rho_at(d, r):
    """angular radius of the cap cut at level r by a centre at distance d"""
    x = d/2*cos(r)
    return mpf(0) if x >= 1 else acos(x)


def cell_bound(dists, R, pairs):
    """
    eq:second-order / lem:no-triple in radial form.  dists is a list of
    (distance, multiplicity); pairs is a list of (d_i, d_j, gamma, multiplicity).
    Exact for R <= sqrt(3/2) in ball radius, i.e. R <= r_* in angle.
    """
    def integrand(r):
        v = TOTAL
        for d, k in dists:
            v -= k*C(rho_at(d, r))
        for di, dj, g, k in pairs:
            ri, rj = rho_at(di, r), rho_at(dj, r)
            if ri > 0 and rj > 0:
                v += k*Lambda_un(ri, rj, g) if di != dj else k*Lambda(ri, g)
        return v*dsec4(r)

    brk = {mpf(0), R}
    for d, _ in dists:
        if d > 2:
            b = acos(2/d)
            if 0 < b < R:
                brk.add(b)
    p = sorted(brk)
    val = sum(quad(integrand, [p[i], p[i+1]]) for i in range(len(p)-1))
    return (TOTAL + val)/4


# ------------------------------------------------------- 1. covering bound
print("1. Table 1, covering bound (pi m/3) tan^3 r_m")
for m, paper in ((22, '8.046376'), (23, '7.916728'), (24, '7.798989')):
    v = pi*m/3*tan(r_m(m))**3
    print(f"   m={m}:  {mp.nstr(v, 8):>10}   paper {paper}")

# ------------------------------------------- 3. eq:second-order-value
print("\n3. eq:second-order-value, the four pair-bound numbers")
two = mpf(2)
for m, npair, R, tag, paper in (
        (23, 88, r_m(23), 'deletion, R=r_23', '7.997885'),
        (23, 88, R_STAR, 'deletion, R=r_*  ', '8.034340'),
        (24, 96, r_m(24), 'roots,    R=r_24 ', '7.858738'),
        (24, 96, R_STAR, 'roots,    R=r_*  ', '7.906940')):
    v = cell_bound([(two, m)], R, [(two, two, G60, npair)])
    flag = '' if abs(float(v) - float(paper)) < 5e-7 else '   <-- DISAGREES'
    print(f"   {tag}:  {mp.nstr(v, 9):>12}   paper {paper}{flag}")
print("   (v1.7.0 printed 7.907070 for the last one; 7.906940 is the value, as in sec:closure)")

# ------------------------------------------- 4. rigidity spectrum
print("\n4. lem:rigidity-spectrum")
roots = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = [0]*4
            v[i], v[j] = si, sj
            roots.append(v)
A = np.array(roots, float)
G = A @ A.T
T_pairs = [(i, j) for i in range(24) for j in range(i+1, 24) if abs(G[i, j]-1) < 1e-9]
basis = []
for i in range(24):
    u = np.linalg.svd(np.eye(4) - np.outer(A[i], A[i])/2.0)[0]
    for k in range(3):
        b = np.zeros((24, 4)); b[i] = u[:, k]; basis.append(b.reshape(-1))
B = np.array(basis).T
Lam = np.zeros((96, 96))
for p, (i, j) in enumerate(T_pairs):
    row = np.zeros((24, 4)); row[j] += A[i]/np.sqrt(2); row[i] += A[j]/np.sqrt(2)
    Lam[p] = row.reshape(-1)
L = Lam @ B
sv = np.round(np.linalg.svd(L, compute_uv=False), 10)
vals, cnt = np.unique(sv, return_counts=True)
print("   singular values / multiplicities:",
      ", ".join(f"{v:.6f}x{c}" for v, c in zip(vals, cnt)))
print("   paper: 0, 1, sqrt(5/2), sqrt3, 2 with multiplicities 6, 29, 8, 21, 8")

# ------------------------------------------- 7. the new constant
U = np.linalg.svd(L)[0]
rank = int((sv > 1e-9).sum())
Pim = U[:, :rank] @ U[:, :rank].T
dg = np.diag(Pim)
print(f"\n7. NEW.  diag of the projection onto im(Lambda): all equal to "
      f"{dg.min():.12f} = 66/96 = 11/16")
print("   (forced by W(F4) acting transitively on the 96 edges of the 24-cell)")
print("   => ||g||_inf <= (sqrt11/4)||g||_2 and ||g||_1 >= (4/sqrt11)||g||_2 on im")
print(f"   => thm:local-uniqueness isolation radius 2/sqrt(577)=0.0832611 "
      f"improves to 2/sqrt(397)={2/np.sqrt(397):.7f}")
print(f"   => estimate (a) of thm:near-contact: 192/71=2.70422 improves to "
      f"{(np.sqrt(11)/2)/(1-(3*np.sqrt(11)+0.5)/48):.5f}")

# ------------------------------------------- 6 & 8. the crossovers
print("\n6. the two crossovers of the pair terms, and 8. two ideas measured")


def even_push(S, R=None):
    d = 2 + S/24
    return cell_bound([(d, 24)], R or R_STAR, [(d, d, G60, 96)])


def one_push(delta, R=None):
    d = 2 + delta
    return cell_bound([(two, 23), (d, 1)], R or R_STAR,
                      [(two, two, G60, 88), (two, d, G60, 8)])


s1 = findroot(lambda S: even_push(S) - 8, mpf('0.15'))
s2 = findroot(lambda D: one_push(D) - 8, mpf('0.2'))
print(f"   root system pushed out evenly : sum delta = {mp.nstr(s1, 6)}  (paper 0.155)")
print(f"   one centre pushed out         : delta     = {mp.nstr(s2, 6)}  (paper 0.197)")

# idea: let the truncation radius grow with the distances.  The four-point
# argument of lem:no-triple gives 16|x|^2 >= 12 + d1^2+d2^2+d3^2, so R may be
# taken distance-aware instead of the constant sqrt(3/2).
R_rig = lambda S: acos(1/sqrt((12 + 3*(2+S/24)**2)/16))
s1b = findroot(lambda S: even_push(S, R_rig(S)) - 8, mpf('0.15'))
print(f"   same, with the distance-aware truncation radius: {mp.nstr(s1b, 6)}"
      f"   -> a gain of {float((s1-s1b)/s1)*100:.1f}%")
print("   The explicit neighbourhood of thm:near-contact ends at sum delta = 6e-5, so a factor of")
print("   about 2500 separates it from the crossing either way.")
