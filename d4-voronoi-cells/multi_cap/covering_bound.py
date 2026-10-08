#!/usr/bin/env python3
"""
The covering bound of thm:covering-bound and cor:m22.

For a contact configuration of m unit vectors in R^4 with pairwise inner
products at most 1/2 and bounded cell,

    vol(V_c) >= (pi m / 3) tan^3(r_m),     2 r_m - sin 2 r_m = 2 pi / m.

Checks, in order:

  1  the cap-area formula C(r) = pi (2r - sin 2r) integrates to the full
     measure 2 pi^2 of S^3, and agrees with direct Monte Carlo;
  2  the radial identity vol(V_c) = (1/4) * integral over S^3 of
     sec^4(delta), delta the angular distance to the nearest contact
     direction, at the root configuration and on random configurations;
  3  the layer-cake rewriting of that integral, against the direct form;
  4  the closed form (pi m / 3) tan^3 r_m against numerical quadrature of
     the same estimate;
  5  the table tab:covering-bound for m = 5 .. 24, including the monotonicity
     in m used in the proof of cor:m22;
  6  the bound is never violated: at the root configuration, at every
     subset of the roots of size 20 .. 24, and on perturbed and random
     packing-valid configurations;
  7  the two quoted constants, 8.046376 at m = 22 and 7.798989 at
     m = 24, and the density 0.632749 the second implies;
  8  the per-cell form of the bound and its Jensen collapse
     (prop:area-optimal): phi(s) = tan^3(C^{-1}(s)) has derivative
     3 sec^4(C^{-1}(s)) / (4 pi), is strictly convex, and is therefore
     minimised at equal cell areas, where the per-cell bound reproduces
     the global one exactly;
  9  the covering-radius bound arccos sqrt(5/8) = 37.7612 degrees
     (prop:covering-radius), the averaging step behind it, and the fact that
     it moves the volume bound only in the fifth decimal;
 10  where the shortfall at m = 24 goes: about 0.057 to the overlaps
     below r_24 and about 0.140 to the tail above it, and the total
     pairwise overlap of about 2.0 contributed at r_24 by the 96 pairs
     of the root system at exactly 60 degrees. These are Monte Carlo
     estimates and are quoted to two significant figures.

Run:  python3 covering_bound.py
Exits nonzero if any check fails.
"""

import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

TOT = 2.0 * np.pi ** 2          # surface measure of S^3
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("[%s] %s" % ("PASS" if ok else "FAIL", name))
    if detail:
        for line in detail.splitlines():
            print("       " + line)


def cap_area(r):
    """Surface measure of a geodesic cap of angular radius r in S^3."""
    return np.pi * (2.0 * r - np.sin(2.0 * r))


def r_of(m):
    """The unique root in (0, pi/2) of 2r - sin 2r = 2 pi / m."""
    return brentq(lambda r: 2.0 * r - np.sin(2.0 * r) - 2.0 * np.pi / m,
                  1e-14, np.pi / 2.0, xtol=1e-15, rtol=8.9e-16)


def bound(m):
    return np.pi * m / 3.0 * np.tan(r_of(m)) ** 3


# ---------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------

def d4_directions():
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0.0] * 4
                    v[i] = si
                    v[j] = sj
                    out.append(v)
    return np.array(out) / np.sqrt(2.0)


def separate(W, margin=1e-6, sweeps=4000, step=0.11):
    """Pairwise repulsion, used only to generate test configurations."""
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    m = len(W)
    tgt = 0.5 - margin
    for _ in range(sweeps):
        G = W @ W.T
        np.fill_diagonal(G, -1.0)
        if G.max() <= tgt:
            return W
        mv = np.zeros_like(W)
        for i in range(m):
            for j in range(i + 1, m):
                if G[i, j] > tgt:
                    d = G[i, j] - tgt
                    mv[i] += step * d * (W[i] - W[j])
                    mv[j] += step * d * (W[j] - W[i])
        W = W + mv
        W /= np.linalg.norm(W, axis=1, keepdims=True)
    G = W @ W.T
    np.fill_diagonal(G, -1.0)
    return W if G.max() <= 0.5 + 1e-12 else None


def cell_volume(W):
    m = len(W)
    for k in range(4):
        for s in (1.0, -1.0):
            c = np.zeros(4)
            c[k] = -s
            r = linprog(c, A_ub=W, b_ub=np.ones(m),
                        bounds=[(None, None)] * 4, method="highs")
            if r.status != 0:
                return None
    V = HalfspaceIntersection(np.hstack([W, -np.ones((m, 1))]),
                              np.zeros(4)).intersections
    return ConvexHull(V).volume


# ---------------------------------------------------------------------
# 1. the cap-area formula
# ---------------------------------------------------------------------

def check_cap_area():
    full = cap_area(np.pi)
    rng = np.random.default_rng(101)
    N = 2_000_000
    X = rng.normal(size=(N, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    e = np.zeros(4)
    e[0] = 1.0
    ang = np.arccos(np.clip(X @ e, -1, 1))
    ok = abs(full - TOT) < 1e-12
    detail = "C(pi) = %.12f   |S^3| = %.12f" % (full, TOT)
    for r in (0.3, 0.6, 0.9, 1.2):
        mc = np.mean(ang <= r) * TOT
        detail += "\n  r = %.1f:  formula %.6f   Monte Carlo %.6f" % (
            r, cap_area(r), mc)
        ok = ok and abs(cap_area(r) - mc) < 0.01
    record("cap-area formula C(r) = pi(2r - sin 2r)", ok, detail)


# ---------------------------------------------------------------------
# 2-3. radial identity and layer cake
# ---------------------------------------------------------------------

def radial_integral(W, rng, N=2_000_000):
    X = rng.normal(size=(N, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    h = (X @ W.T).max(axis=1)
    return TOT * np.mean(h ** -4.0), np.arccos(np.clip(h, -1, 1))


def check_radial():
    rng = np.random.default_rng(20260912)
    U = d4_directions()
    integral, delta = radial_integral(U, rng)
    v = cell_volume(U)
    ok = abs(integral / 4.0 - 8.0) < 5e-3 and abs(v - 8.0) < 1e-7
    record("radial identity vol = (1/4) int sec^4(delta) at the roots", ok,
           "(1/4) int sec^4 delta = %.6f   direct volume = %.8f\n"
           "covering radius = %.4f deg (exact value 45)  [Monte Carlo]"
           % (integral / 4.0, v, np.degrees(delta.max())))

    # layer cake, same data
    grid = np.linspace(0.0, np.pi / 2 - 1e-6, 1200)
    surv = np.array([np.mean(delta > r) for r in grid]) * TOT
    integrand = surv * 4.0 * (1.0 / np.cos(grid)) ** 4 * np.tan(grid)
    layer = TOT + np.trapezoid(integrand, grid)
    record("layer-cake rewriting of the same integral",
           abs(layer - integral) < 5e-2,
           "direct %.6f   layer cake %.6f  [Monte Carlo]"
           % (integral, layer))

    bad = 0
    tested = 0
    for m in (8, 12, 16, 20):
        W = separate(rng.normal(size=(m, 4)))
        if W is None:
            continue
        v = cell_volume(W)
        if v is None:
            continue
        integral, _ = radial_integral(W, rng, N=1_000_000)
        tested += 1
        if abs(integral / 4.0 - v) > 0.03 * v:
            bad += 1
    record("radial identity on random configurations", tested > 0 and bad == 0,
           "configurations tested: %d  [Monte Carlo, 1e6 samples each]"
           % tested)


# ---------------------------------------------------------------------
# 4-5. closed form and table
# ---------------------------------------------------------------------

def check_closed_form():
    worst = 0.0
    for m in (5, 9, 14, 18, 22, 24):
        rm = r_of(m)
        f = lambda r: max(0.0, TOT - m * cap_area(r)) * 4.0 * \
            (1.0 / np.cos(r)) ** 4 * np.tan(r)
        val, _ = quad(f, 0.0, rm, limit=800)
        num = (TOT + val) / 4.0
        worst = max(worst, abs(num - bound(m)))
    record("closed form (pi m/3) tan^3 r_m equals the quadrature",
           worst < 1e-8,
           "largest discrepancy over m in {5,9,14,18,22,24}: %.3e" % worst)

    vals = [bound(m) for m in range(5, 25)]
    mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    lines = []
    for m in range(5, 25):
        lines.append("  m = %2d   r_m = %9.6f deg   bound = %9.6f%s"
                     % (m, np.degrees(r_of(m)), bound(m),
                        "   >= 8" if bound(m) >= 8.0 else ""))
    record("bound is strictly decreasing in m over 5..24", mono,
           "\n".join(lines))

    mmax = max(m for m in range(5, 25) if bound(m) >= 8.0)
    record("bound exceeds 8 exactly for m <= 22", mmax == 22,
           "largest m with bound >= 8: %d   bound(22) = %.6f   "
           "bound(23) = %.6f" % (mmax, bound(22), bound(23)))


# ---------------------------------------------------------------------
# 6. the bound is never violated
# ---------------------------------------------------------------------

def check_no_violation():
    U = d4_directions()
    v = cell_volume(U)
    record("root configuration respects the bound",
           v >= bound(24) - 1e-9,
           "true volume 8 (computed %.8f)   bound at m = 24: %.6f   "
           "slack %.6f" % (v, bound(24), v - bound(24)))

    rng = np.random.default_rng(31337)
    viol = 0
    tested = 0
    lines = []
    for m in (20, 21, 22, 23):
        smallest = np.inf
        for _ in range(25):
            idx = rng.choice(24, size=m, replace=False)
            vv = cell_volume(U[idx])
            if vv is None:
                continue
            tested += 1
            smallest = min(smallest, vv)
            if vv < bound(m) - 1e-9:
                viol += 1
        lines.append("  m = %2d  smallest volume over root subsets %.6f   "
                     "bound %.6f" % (m, smallest, bound(m)))
    record("subsets of the root system respect the bound",
           viol == 0 and tested > 0,
           ("configurations tested: %d\n" % tested) + "\n".join(lines))

    viol = 0
    tested = 0
    smallest_ratio = np.inf
    for m in (9, 12, 15, 18, 20):
        for _ in range(10):
            W = separate(rng.normal(size=(m, 4)))
            if W is None:
                continue
            vv = cell_volume(W)
            if vv is None:
                continue
            tested += 1
            smallest_ratio = min(smallest_ratio, vv / bound(m))
            if vv < bound(m) - 1e-9:
                viol += 1
    record("random packing-valid configurations respect the bound",
           viol == 0 and tested > 0,
           "configurations tested: %d   smallest true/bound ratio %.4f  "
           "[double precision]" % (tested, smallest_ratio))


# ---------------------------------------------------------------------
# 7. the quoted constants
# ---------------------------------------------------------------------

def check_constants():
    b22, b24 = bound(22), bound(24)
    dens = (np.pi ** 2 / 2.0) / b24
    ok = (abs(b22 - 8.046376) < 1e-6 and abs(b24 - 7.798989) < 1e-6
          and abs(dens - 0.632749) < 1e-6)
    record("the constants quoted in the text", ok,
           "m = 22: %.6f (text 8.046376)\n"
           "m = 24: %.6f (text 7.798989), shortfall from 8: %.6f\n"
           "m = 23: %.6f, shortfall from 8: %.6f\n"
           "implied density (pi^2/2)/%.6f = %.6f (text 0.632749), "
           "against pi^2/16 = %.6f"
           % (b22, b24, 8.0 - b24, bound(23), 8.0 - bound(23), b24, dens,
              np.pi ** 2 / 16.0))


def check_area_optimality():
    """
    prop:area-optimal: the per-cell bound, its derivative in closed form,
    its convexity, and the fact that equal cell areas reproduce exactly
    the global bound of thm:covering-bound.
    """
    def rho_of_area(a):
        return brentq(lambda r: cap_area(r) - a, 1e-14, np.pi / 2.0,
                      xtol=1e-15, rtol=8.9e-16)

    worst = 0.0
    lines = []
    for a in (0.2, 0.5, 0.8225, 1.5, 3.0):
        rho = rho_of_area(a)
        h = 1e-6
        num = (np.tan(rho_of_area(a + h)) ** 3
               - np.tan(rho_of_area(a - h)) ** 3) / (2 * h)
        closed = 3.0 * (1.0 / np.cos(rho)) ** 4 / (4.0 * np.pi)
        worst = max(worst, abs(num - closed))
        lines.append("  area %.4f: numerical %.8f  closed form %.8f"
                     % (a, num, closed))
    record("phi'(s) = 3 sec^4(C^-1(s)) / (4 pi)", worst < 1e-6,
           "\n".join(lines))

    ss = np.linspace(0.05, 6.0, 500)
    ph = np.array([np.tan(rho_of_area(x)) ** 3 for x in ss])
    record("phi is strictly convex", bool((np.diff(ph, 2) > 0).all()),
           "all %d second differences positive" % (len(ph) - 2))

    m = 24
    equal = np.pi / 3.0 * m * np.tan(rho_of_area(TOT / m)) ** 3
    rng = np.random.default_rng(3)
    smallest = np.inf
    for _ in range(1500):
        a = rng.dirichlet(np.ones(m)) * TOT
        smallest = min(smallest,
                       np.pi / 3.0 * sum(np.tan(rho_of_area(x)) ** 3
                                         for x in a))
    record("equal cell areas reproduce the global bound and minimise it",
           abs(equal - bound(24)) < 1e-9 and smallest > equal - 1e-9,
           "per-cell bound at equal areas %.6f   global bound %.6f\n"
           "smallest value over 1500 random area splittings: %.6f"
           % (equal, bound(24), smallest))


def check_covering_radius():
    """
    prop:covering-radius. Every vertex of a spherical Voronoi cell is at
    angular distance at least arccos sqrt(5/8) from its centre.
    """
    R0 = np.arccos(np.sqrt(5.0 / 8.0))
    rng = np.random.default_rng(17)

    # the averaging step: four unit vectors in R^3 cannot have all six
    # pairwise inner products below -1/3
    bad = 0
    for _ in range(200000):
        V = rng.normal(size=(4, 3))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        G = V @ V.T
        np.fill_diagonal(G, -2.0)
        if G.max() < -1.0 / 3.0 - 1e-12:
            bad += 1
    record("four unit vectors in R^3 always have a pair with inner "
           "product at least -1/3", bad == 0,
           "counterexamples in 200000 random quadruples: %d" % bad)

    N = 400000
    X = rng.normal(size=(N, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    U = d4_directions()
    d = np.arccos(np.clip((X @ U.T).max(axis=1), -1, 1))
    smallest = d.max()
    tested = 1
    for m in (10, 14, 18, 22, 24):
        W = separate(rng.normal(size=(m, 4)))
        if W is None:
            continue
        dd = np.arccos(np.clip((X @ W.T).max(axis=1), -1, 1))
        smallest = min(smallest, dd.max())
        tested += 1
    record("covering radius never below arccos sqrt(5/8)",
           smallest > R0 - 0.02,
           "bound %.5f deg   smallest covering radius over %d "
           "configurations %.5f deg\n(D4 exact value 45; sampling "
           "underestimates a maximum slightly)  [Monte Carlo]"
           % (np.degrees(R0), tested, np.degrees(smallest)))

    grid = np.linspace(1e-6, np.pi / 2 - 1e-6, 4000)
    wgt = 4.0 * (1.0 / np.cos(grid)) ** 4 * np.tan(grid)
    ub = np.maximum(0.0, TOT - 24.0 * cap_area(grid))
    hole = np.where(grid < R0, cap_area(np.maximum(R0 - grid, 0.0)), 0.0)
    one = (TOT + np.trapezoid(np.maximum(ub, hole) * wgt, grid)) / 4.0
    many = np.minimum(np.where(grid < R0,
                               24.0 * cap_area(np.maximum(R0 - grid, 0.0)),
                               0.0), TOT)
    allh = (TOT + np.trapezoid(np.maximum(ub, many) * wgt, grid)) / 4.0
    record("the covering-radius bound does not move the volume bound",
           abs(one - 7.79902) < 5e-4 and abs(allh - 7.79966) < 5e-4,
           "plain %.5f   with one deep hole %.5f (text 7.79902)\n"
           "with one deep hole per cell, treated as disjoint: %.5f "
           "(text 7.79966)" % (bound(24), one, allh))


def check_shortfall():
    """
    Where the 0.2011 at m = 24 goes. Both parts are Monte Carlo.
    """
    rng = np.random.default_rng(2026)
    U = d4_directions()
    N = 3_000_000
    X = rng.normal(size=(N, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    h = (X @ U.T).max(axis=1)
    d = np.arccos(np.clip(h, -1, 1))

    grid = np.linspace(1e-6, np.pi / 2 - 1e-6, 1500)
    true_surv = np.array([np.mean(d > r) for r in grid]) * TOT
    ub_surv = np.maximum(0.0, TOT - 24.0 * cap_area(grid))
    wgt = 4.0 * (1.0 / np.cos(grid)) ** 4 * np.tan(grid)
    rm = r_of(24)
    i0 = int(np.searchsorted(grid, rm))
    below = np.trapezoid((true_surv[:i0] - ub_surv[:i0]) * wgt[:i0],
                         grid[:i0]) / 4.0
    above = np.trapezoid(true_surv[i0:] * wgt[i0:], grid[i0:]) / 4.0
    ok = (abs(below - 0.057) < 0.008 and abs(above - 0.140) < 0.014
          and abs(below + above - (8.0 - bound(24))) < 0.02)
    record("the shortfall at m = 24 splits as about 0.057 + 0.140", ok,
           "overlaps below r_24 = %.4f (text: about 0.057)\n"
           "tail above r_24     = %.4f (text: about 0.140)\n"
           "sum %.4f against the shortfall 8 - %.6f = %.4f  [Monte Carlo]"
           % (below, above, below + above, bound(24), 8.0 - bound(24)))

    # total pairwise overlap at r_24 contributed by the 60-degree pairs
    G = U @ U.T
    np.fill_diagonal(G, -2.0)
    npairs = int((np.abs(G - 0.5) < 1e-9).sum() // 2)
    a = np.zeros(4)
    a[0] = 1.0
    b = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0, 0.0])
    Y = rng.normal(size=(400000, 4))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    lens = np.mean((Y @ a >= np.cos(rm)) & (Y @ b >= np.cos(rm))) * TOT
    total = npairs * lens
    record("96 pairs at 60 degrees, total overlap about 2.0 at r_24",
           npairs == 96 and abs(total - 2.0) < 0.12,
           "pairs at exactly 60 degrees: %d\n"
           "one lens at r_24: %.6f   total: %.4f (text: about 2.0)  "
           "[Monte Carlo]" % (npairs, lens, total))


def main():
    print("The covering bound")
    print("=" * 62)
    check_cap_area()
    check_radial()
    check_closed_form()
    check_no_violation()
    check_constants()
    check_area_optimality()
    check_covering_radius()
    check_shortfall()
    print("=" * 62)
    npass = sum(1 for _, ok in RESULTS if ok)
    print("%d of %d checks passed" % (npass, len(RESULTS)))
    if npass != len(RESULTS):
        for name, ok in RESULTS:
            if not ok:
                print("  FAILED: " + name)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
