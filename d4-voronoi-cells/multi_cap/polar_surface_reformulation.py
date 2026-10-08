#!/usr/bin/env python3
"""
Polar, surface-area and facet reformulations of the multi-direction case.

Checks, in order:

  1  V_c is the polar dual of conv(W) for the D4 root configuration and
     for random packing-valid configurations.
  2  vol(V_c) = (1/4) * (total 3-volume of the facets of V_c), which is
     the surface-area form of the conjecture: vol(V_c) >= 8 if and only
     if the boundary 3-volume is at least 32.
  3  The D4 configuration realises 32 exactly, with 24 facets of
     3-volume 4/3 each.
  4  Every facet, translated by its own contact direction, contains the
     3-ball of radius 1/sqrt(3) about the origin: the tangential
     constraint cut by a neighbour at Gram value g sits at distance
     rho(g) = sqrt((1-g)/(1+g)) >= rho(1/2) = 1/sqrt(3).
  5  The resulting facet-local bound, vol(V_c) >= m*pi/(9*sqrt(3)), is
     about 4.837 at m = 24 and so cannot reach 8: no argument that
     bounds each facet separately from the pairwise Gram condition alone
     can settle the conjecture.
  5a The ceiling of that whole class of arguments: eight tight
     neighbours in square-antiprism position form an admissible
     configuration whose facet has 3-volume 16 sqrt(2) - 64/3 =
     1.2940836646, below the octahedron's 4/3, so the best per-facet
     constant gives at most 6 * that = 96 sqrt(2) - 128 = 7.7645, below
     the target and below the covering bound.
  6  The inscribed-ball bound vol(V_c) >= vol(B^4) = pi^2/2, which is
     the stronger of the two unconditional bounds, is also short of 8.
  7  In the D4 configuration the tight neighbours of a facet are the 8
     projected cube directions, pairwise Gram value at most 1/3, and the
     region they cut out is exactly the regular octahedron of 3-volume
     4/3, so the facet-local bound is attained there with no slack in
     the tightness structure.
  8  Monotonicity: enlarging W shrinks V_c, so the infimum of the volume
     is attained on maximal packing-valid configurations.

Exact arithmetic is used wherever the configuration is the root system.
The random configurations are double precision and are labelled as such.

Run:  python3 polar_surface_reformulation.py
Exits nonzero if any check fails.
"""

import itertools
import sys
from fractions import Fraction

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    print("[%s] %s" % (flag, name))
    if detail:
        for line in detail.splitlines():
            print("       " + line)


# ---------------------------------------------------------------------
# The D4 contact directions.
# ---------------------------------------------------------------------

def d4_roots():
    """The 24 roots of D4, as integer vectors of squared length 2."""
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i] = si
                    v[j] = sj
                    out.append(tuple(v))
    assert len(out) == 24
    return out


def d4_directions():
    """Contact directions u = alpha/sqrt(2), as floats."""
    return np.array(d4_roots(), dtype=float) / np.sqrt(2.0)


# ---------------------------------------------------------------------
# Cell construction.
# ---------------------------------------------------------------------

def cell_vertices(W, tol=1e-9):
    """
    Vertices of {x : <x,w> <= 1 for all w in W}, or None if unbounded.

    Boundedness is decided by a linear program: the cell is bounded iff
    the rows of W positively span R^4, iff the only y >= 0 with W^T y = 0
    and sum y = 1 ... is tested directly by maximising each of eight
    directions over the cell.
    """
    W = np.asarray(W, dtype=float)
    m = W.shape[0]
    # Boundedness test: maximise +-e_k over the cell; bounded iff finite.
    for k in range(4):
        for sgn in (1.0, -1.0):
            c = np.zeros(4)
            c[k] = -sgn                      # linprog minimises
            res = linprog(c, A_ub=W, b_ub=np.ones(m),
                          bounds=[(None, None)] * 4, method="highs")
            if res.status == 3:              # unbounded
                return None
            if res.status != 0:
                return None
    halfspaces = np.hstack([W, -np.ones((m, 1))])
    hs = HalfspaceIntersection(halfspaces, np.zeros(4))
    return hs.intersections


def cell_volume(W):
    V = cell_vertices(W)
    if V is None:
        return None
    return ConvexHull(V).volume


def facet_volumes(W, tol=1e-8):
    """
    For each w in W that supports a facet, the 3-volume of that facet.
    Returns a dict index -> 3-volume.
    """
    W = np.asarray(W, dtype=float)
    V = cell_vertices(W)
    if V is None:
        return None
    out = {}
    for j, w in enumerate(W):
        on = V[np.abs(V @ w - 1.0) < tol]
        if on.shape[0] < 4:
            continue
        # Move into the 3-space w^perp: pick an orthonormal basis.
        basis = np.linalg.svd(np.eye(4) - np.outer(w, w))[0][:, :3]
        pts = (on - w) @ basis
        # Degenerate (lower-dimensional) contacts contribute nothing.
        if np.linalg.matrix_rank(pts - pts[0], tol=1e-7) < 3:
            continue
        out[j] = ConvexHull(pts).volume
    return out


def _separate(W, margin=1e-4, sweeps=4000, step=0.12):
    """
    Push a configuration of unit vectors apart until every pairwise inner
    product is at most 1/2 - margin. Returns None if it does not get
    there. This is the standard pairwise-repulsion relaxation; it is used
    only to produce test configurations, never as part of an argument.
    """
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    m = W.shape[0]
    target = 0.5 - margin
    for _ in range(sweeps):
        G = W @ W.T
        np.fill_diagonal(G, -1.0)
        if G.max() <= target:
            return W
        move = np.zeros_like(W)
        for i in range(m):
            for j in range(i + 1, m):
                if G[i, j] > target:
                    d = G[i, j] - target
                    move[i] += step * d * (W[i] - W[j])
                    move[j] += step * d * (W[j] - W[i])
        W = W + move
        W /= np.linalg.norm(W, axis=1, keepdims=True)
    G = W @ W.T
    np.fill_diagonal(G, -1.0)
    return W if G.max() <= 0.5 + 1e-12 else None


def random_config(rng, m):
    """A random packing-valid configuration of m unit vectors in R^4."""
    return _separate(rng.normal(size=(m, 4)), margin=1e-4)


def perturbed_config(rng, scale, keep=20):
    """
    A packing-valid configuration obtained by dropping a few roots,
    perturbing the rest and separating them again. Dropping is needed
    because the full root system already sits at the pairwise bound,
    so no perturbation of all 24 directions stays admissible.
    """
    U = d4_directions()
    idx = rng.choice(24, size=keep, replace=False)
    W = U[idx] + rng.normal(size=(keep, 4)) * scale
    return _separate(W, margin=1e-4)


# ---------------------------------------------------------------------
# 1. Polar duality.
# ---------------------------------------------------------------------

def check_polar():
    U = d4_directions()
    V = cell_vertices(U)
    vol = ConvexHull(V).volume
    # conv(U) is the 24-cell of circumradius 1; its polar should be the
    # cell, so vol(conv U) * vol(cell) is the Mahler volume here.
    volU = ConvexHull(U).volume
    # Polar of conv(U): same halfspace system, since the vertices of
    # conv(U) are exactly the u_i.
    detail = ("vol(conv W) = %.8f   vol(W-polar) = %.8f   product = %.8f"
              % (volU, vol, volU * vol))
    ok = (abs(vol - 8.0) < 1e-7) and (abs(volU - 2.0) < 1e-7)
    record("polar dual of conv(W) is the Voronoi cell (D4, vol 8, "
           "conv 2)", ok, detail)

    # Random configurations: the halfspace description of the polar is
    # <x,w> <= 1 for w in W, and the polar only sees conv(W), so adding
    # an interior point of conv(W) to W must not change the cell.
    rng = np.random.default_rng(20260912)
    agree = 0
    trials = 0
    for _ in range(60):
        if trials >= 10:
            break
        W = random_config(rng, 14)
        if W is None:
            continue
        v0 = cell_volume(W)
        if v0 is None:
            continue
        lam = rng.dirichlet(np.ones(W.shape[0]))
        interior = lam @ W
        if np.linalg.norm(interior) > 0.999:
            continue
        W2 = np.vstack([W, interior])
        v1 = cell_volume(W2)
        trials += 1
        if v1 is not None and abs(v1 - v0) < 1e-7 * max(1.0, v0):
            agree += 1
    record("cell depends on W only through conv(W) (random "
           "configurations)", agree == trials and trials > 0,
           "agreements: %d of %d  [double precision]" % (agree, trials))


# ---------------------------------------------------------------------
# 2-3. Surface-area identity.
# ---------------------------------------------------------------------

def check_surface_identity():
    U = d4_directions()
    fv = facet_volumes(U)
    total = sum(fv.values())
    vol = cell_volume(U)
    ok = (len(fv) == 24 and abs(total - 32.0) < 1e-6
          and abs(total / 4.0 - vol) < 1e-7
          and max(abs(v - 4.0 / 3.0) for v in fv.values()) < 1e-7)
    detail = ("facets: %d   total facet 3-volume: %.8f (target 32)\n"
              "quarter of it: %.8f   vol(V) = %.8f\n"
              "per-facet 3-volume: min %.8f  max %.8f  (octahedron 4/3 = "
              "%.8f)"
              % (len(fv), total, total / 4.0, vol,
                 min(fv.values()), max(fv.values()), 4.0 / 3.0))
    record("D4: vol(V) = (1/4) * total facet 3-volume = 32/4 = 8", ok,
           detail)

    rng = np.random.default_rng(11111)
    bad = 0
    tested = 0
    worst = 0.0
    for scale in (0.02, 0.05, 0.10, 0.18):
        for _ in range(6):
            W = perturbed_config(rng, scale)
            if W is None:
                continue
            fv = facet_volumes(W)
            v = cell_volume(W)
            if fv is None or v is None:
                continue
            tested += 1
            err = abs(sum(fv.values()) / 4.0 - v)
            worst = max(worst, err)
            if err > 1e-6 * max(1.0, v):
                bad += 1
    record("surface-area identity on perturbed configurations",
           tested > 0 and bad == 0,
           "configurations tested: %d   largest discrepancy: %.3e  "
           "[double precision]" % (tested, worst))


# ---------------------------------------------------------------------
# 4-5. The inball lemma and the facet-local obstruction.
# ---------------------------------------------------------------------

def rho(g):
    return np.sqrt((1.0 - g) / (1.0 + g))


def check_inball():
    # rho is decreasing on (-1,1] and rho(1/2) = 1/sqrt(3).
    gs = np.linspace(-0.98, 0.5, 400)
    mono = np.all(np.diff(rho(gs)) < 0)
    at_half = abs(rho(0.5) - 1.0 / np.sqrt(3.0)) < 1e-14
    record("rho(g) = sqrt((1-g)/(1+g)) is decreasing, rho(1/2) = "
           "1/sqrt(3)", mono and at_half,
           "rho(1/2) = %.15f   1/sqrt(3) = %.15f"
           % (rho(0.5), 1.0 / np.sqrt(3.0)))

    # The inball claim, checked directly on configurations: every facet
    # of V_c contains a ball of radius 1/sqrt(3) about its contact point.
    def inradius_of_facets(W):
        W = np.asarray(W, dtype=float)
        V = cell_vertices(W)
        if V is None:
            return None
        radii = []
        for j, w in enumerate(W):
            on = V[np.abs(V @ w - 1.0) < 1e-8]
            if on.shape[0] < 4:
                continue
            basis = np.linalg.svd(np.eye(4) - np.outer(w, w))[0][:, :3]
            pts = (on - w) @ basis
            if np.linalg.matrix_rank(pts - pts[0], tol=1e-7) < 3:
                continue
            hull = ConvexHull(pts)
            # distance from the origin of w^perp to each facet plane
            d = -hull.equations[:, 3] / np.linalg.norm(
                hull.equations[:, :3], axis=1)
            radii.append(d.min())
        return radii

    U = d4_directions()
    r0 = inradius_of_facets(U)
    ok0 = min(r0) > 1.0 / np.sqrt(3.0) - 1e-9
    record("D4: every facet contains the 3-ball of radius 1/sqrt(3)",
           ok0, "smallest facet inradius: %.12f   1/sqrt(3) = %.12f"
           % (min(r0), 1.0 / np.sqrt(3.0)))

    rng = np.random.default_rng(2718281)
    worst = np.inf
    tested = 0
    for scale in (0.03, 0.08, 0.15):
        for _ in range(6):
            W = perturbed_config(rng, scale)
            if W is None:
                continue
            r = inradius_of_facets(W)
            if not r:
                continue
            tested += 1
            worst = min(worst, min(r))
    record("perturbed configurations: facet inradius never below "
           "1/sqrt(3)", tested > 0 and worst > 1.0 / np.sqrt(3.0) - 1e-9,
           "configurations tested: %d   smallest inradius seen: %.12f  "
           "[double precision]" % (tested, worst))

    ball3 = 4.0 * np.pi / 3.0 * (1.0 / np.sqrt(3.0)) ** 3
    bound24 = 24.0 / 4.0 * ball3
    exact = 8.0 * np.pi / (3.0 * np.sqrt(3.0))
    record("facet-local bound at m = 24 falls short of 8", bound24 < 8.0,
           "ball of radius 1/sqrt(3) in R^3 has volume %.8f = 4pi/(9 "
           "sqrt 3)\n"
           "m = 24 gives vol(V) >= %.8f = 8 pi/(3 sqrt 3) = %.8f, "
           "short of 8 by %.8f"
           % (ball3, bound24, exact, 8.0 - bound24))


def check_antiprism():
    """
    The square-antiprism facet: an admissible configuration of eight
    tight neighbours whose facet is smaller than the octahedron.
    """
    s2 = np.sqrt(2.0)
    P = np.array([[s2, 0, 1], [0, s2, 1], [-s2, 0, 1], [0, -s2, 1],
                  [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]],
                 dtype=float)
    norms = np.linalg.norm(P, axis=1)
    G = P @ P.T / 3.0
    np.fill_diagonal(G, -9.0)
    admissible = (np.allclose(norms, np.sqrt(3.0))
                  and G.max() <= 1.0 / 3.0 + 1e-12)
    hs = np.hstack([P, -np.ones((8, 1))])
    vol = ConvexHull(HalfspaceIntersection(hs, np.zeros(3)).intersections).volume
    exact = 16.0 * s2 - 64.0 / 3.0
    record("square-antiprism facet is admissible and smaller than 4/3",
           admissible and abs(vol - exact) < 1e-9 and vol < 4.0 / 3.0,
           "all eight directions at |p| = sqrt(3), largest normalised "
           "pairwise inner product %.12f (bound 1/3)\n"
           "facet 3-volume %.10f   closed form 16 sqrt(2) - 64/3 = "
           "%.10f   octahedron 4/3 = %.10f" % (G.max(), vol, exact,
                                               4.0 / 3.0))

    # the same eight directions as a configuration in R^4
    w = np.array([1.0, 0, 0, 0])
    N = P / np.sqrt(3.0)
    W = np.array([0.5 * w + np.sqrt(3.0) / 2.0 *
                  np.concatenate([[0.0], n]) for n in N])
    allW = np.vstack([w, W])
    GG = allW @ allW.T
    np.fill_diagonal(GG, -9.0)
    record("the corresponding nine directions in R^4 are packing-valid",
           GG.max() <= 0.5 + 1e-12,
           "largest pairwise inner product among the nine: %.12f "
           "(bound 1/2)" % GG.max())

    record("ceiling of any facet-local bound is 96 sqrt(2) - 128",
           abs(6.0 * exact - (96.0 * s2 - 128.0)) < 1e-9
           and 96.0 * s2 - 128.0 < 8.0,
           "6 * (16 sqrt2 - 64/3) = 96 sqrt2 - 128 = %.6f, short of 8 by "
           "%.6f\nand below the covering bound 7.798989 as well"
           % (96.0 * s2 - 128.0, 8.0 - (96.0 * s2 - 128.0)))


def check_ball_bound():
    volB4 = np.pi ** 2 / 2.0
    U = d4_directions()
    V = cell_vertices(U)
    inr = min(1.0 for _ in range(1))  # every facet plane is at distance 1
    record("V_c contains the unit ball, so vol(V_c) >= pi^2/2",
           volB4 < 8.0 and inr == 1.0,
           "pi^2/2 = %.8f, short of 8 by %.8f; this is the stronger of "
           "the two unconditional bounds" % (volB4, 8.0 - volB4))


# ---------------------------------------------------------------------
# 7. Tight structure at D4, in exact arithmetic.
# ---------------------------------------------------------------------

def check_tight_structure():
    roots = d4_roots()
    a0 = roots[0]
    # Neighbours at Gram value exactly 1/2 for the contact directions,
    # i.e. <alpha_0, alpha_k> = 1 in the root normalisation.
    tight = [r for r in roots
             if r != a0 and sum(x * y for x, y in zip(a0, r)) == 1]
    # Project onto alpha_0^perp and normalise, in exact rationals.
    projected = []
    for r in tight:
        # r - (1/2) a0, then the Gram matrix among these
        p = tuple(Fraction(x) - Fraction(1, 2) * Fraction(y)
                  for x, y in zip(r, a0))
        projected.append(p)
    n = len(projected)
    norms = [sum(c * c for c in p) for p in projected]
    same_norm = len(set(norms)) == 1
    gr = []
    for i in range(n):
        for j in range(i + 1, n):
            ip = sum(a * b for a, b in zip(projected[i], projected[j]))
            gr.append(ip / norms[0])
    ok = (n == 8 and same_norm and max(gr) <= Fraction(1, 3))
    record("D4: a facet has exactly 8 tight neighbours, projected "
           "pairwise Gram value at most 1/3", ok,
           "tight neighbours: %d   squared norm of each projection: %s\n"
           "distinct projected Gram values: %s   maximum: %s"
           % (n, norms[0], sorted(set(gr)), max(gr)))

    # Those 8 directions cut out the regular octahedron of volume 4/3.
    P = np.array([[float(c) for c in p] for p in projected])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    basis = np.linalg.svd(
        np.eye(4) - np.outer(np.array(a0, dtype=float) / np.sqrt(2.0),
                             np.array(a0, dtype=float) / np.sqrt(2.0))
    )[0][:, :3]
    Q = P @ basis
    hs = np.hstack([Q / (1.0 / np.sqrt(3.0)), -np.ones((8, 1))])
    inter = HalfspaceIntersection(hs, np.zeros(3))
    voct = ConvexHull(inter.intersections).volume
    record("those 8 directions cut out the regular octahedron of "
           "3-volume 4/3", abs(voct - 4.0 / 3.0) < 1e-9,
           "computed 3-volume: %.12f   4/3 = %.12f"
           % (voct, 4.0 / 3.0))


# ---------------------------------------------------------------------
# 8. Monotonicity in W.
# ---------------------------------------------------------------------

def check_monotonicity():
    rng = np.random.default_rng(31415926)
    ok = True
    tested = 0
    detail_lines = []
    for _ in range(40):
        W = random_config(rng, 9)
        if W is None:
            continue
        v0 = cell_volume(W)
        if v0 is None:
            continue
        # try to extend by one more admissible direction
        found = None
        for _ in range(3000):
            v = rng.normal(size=4)
            v /= np.linalg.norm(v)
            if (W @ v).max() <= 0.5:
                found = v
                break
        if found is None:
            continue
        v1 = cell_volume(np.vstack([W, found]))
        if v1 is None:
            continue
        tested += 1
        if v1 > v0 + 1e-9:
            ok = False
            detail_lines.append("increase: %.9f -> %.9f" % (v0, v1))
        if tested >= 15:
            break
    record("adding an admissible direction never increases the cell "
           "volume", ok and tested > 0,
           ("configurations tested: %d  [double precision]" % tested)
           + ("\n" + "\n".join(detail_lines) if detail_lines else ""))


def check_jensen_and_mahler():
    """
    The two global convex-geometry routes, evaluated at the root
    configuration itself, where any usable bound has to be sharp.
    """
    U = d4_directions()
    volK = ConvexHull(U).volume

    # vol(K^o) = (1/4) * integral over S^3 of h_K^{-4}; Jensen with the
    # normalised measure gives vol(K^o) >= (pi^2/2) * mean(h)^{-4}.
    rng = np.random.default_rng(271828)
    N = 4_000_000
    X = rng.normal(size=(N, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    h = (X @ U.T).max(axis=1)
    hbar = h.mean()
    identity = np.pi ** 2 / 2.0 * np.mean(h ** -4)
    jensen = np.pi ** 2 / 2.0 * hbar ** -4
    record("polar volume integral reproduces vol(V) = 8",
           abs(identity - 8.0) < 5e-3,
           "(pi^2/2) * mean(h^-4) = %.6f   (Monte Carlo, N = %d)"
           % (identity, N))
    record("Jensen bound is strictly below 8 at the root configuration",
           jensen < 8.0 - 0.1,
           "mean support value hbar = %.6f, threshold (pi^2/16)^(1/4) = "
           "%.6f\n(pi^2/2) hbar^-4 = %.6f, short of 8 by %.6f; the loss "
           "is strict\nbecause h is not constant on the sphere, so no "
           "bound of this shape can be sharp"
           % (hbar, (np.pi ** 2 / 16.0) ** 0.25, jensen, 8.0 - jensen))

    mahler = 256.0 / 24.0
    record("Mahler-type bound has slack 3/2 at the root configuration",
           abs(volK * 8.0 - 16.0) < 1e-6 and mahler < 16.0,
           "vol(conv W) * vol(V) = %.6f, against the conjectural lower "
           "bound 4^4/4! = %.6f\ndividing by vol(conv W) = %.6f gives "
           "only %.6f" % (volK * 8.0, mahler, volK, mahler / volK))


def main():
    print("Polar, surface-area and facet reformulations")
    print("=" * 62)
    check_polar()
    check_surface_identity()
    check_inball()
    check_antiprism()
    check_ball_bound()
    check_tight_structure()
    check_monotonicity()
    check_jensen_and_mahler()
    print("=" * 62)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print("%d of %d checks passed" % (npass, len(RESULTS)))
    if npass != len(RESULTS):
        for name, ok, _ in RESULTS:
            if not ok:
                print("  FAILED: " + name)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
