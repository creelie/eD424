#!/usr/bin/env python3
"""
Why the shape deficit of prop:area-optimal cannot be harvested cell by
cell.  Supports rem:no-local-cell.

The radial identity of lem:radial-form is 4 vol(V_c(W)) = int_{S^3} sec^4
delta, and the target is 32.  Two decompositions of S^3 come with the
configuration, and neither supports a local inequality.

  A. The Voronoi decomposition localises exactly: on Omega_w the
     integrand is sec^4 d(theta,w), so the target would follow from

         J(Omega_w) >= (16/pi^2) sigma(Omega_w)                  (LOCAL)

     for every cell, that is from the mean of sec^4 over each cell being
     at least 16/pi^2 = 1.6211389...  At the root configuration every
     cell gives exactly that, so (LOCAL) is sharp.  It is false: a
     direction can carry nine contacts at 60 degrees, and its cell then
     has mean 1.5727...  Pushing the nine out to 61 degrees, so that
     every inner product is strictly below 1/2, still gives 1.5953...

  B. The Delaunay decomposition does not localise at all.  Inside a
     Delaunay cell the nearest contact direction need not be a vertex of
     that cell, so int_T sec^4 delta is not a functional of T.  Even the
     functional it would have to be, the distance to T's own vertices,
     fails: at the regular spherical simplex of edge 60 degrees its mean
     is 1.5436..., short of 16/pi^2 by 4.78 per cent, and at edge 62
     degrees, strictly inside the admissible range, it is 1.5958...

Checks, in order:

  1  at the root configuration the mean of sec^4 over every spherical
     Voronoi cell is 16/pi^2, so (LOCAL) is sharp;
  2  nine directions of R^3 with pairwise inner products at most 1/3
     exist, so nine contacts at 60 degrees are admissible, and the
     resulting cell violates (LOCAL); the violation survives at 61
     degrees, where every inner product is strictly below 1/2;
  3  the Delaunay decomposition does not localise: sampled directions
     whose nearest contact lies outside their own Delaunay cell;
  4  the idealised Delaunay functional at the regular simplex, in the
     closed form 4 E[(1+M)^-4] / E[(1+q)^-2] with E[(1+M)^-4] = 1/5;
  5  the Delaunay cells of D4 are 24 congruent spherical octahedra of
     circumradius 45 degrees, and no four roots are pairwise at 60
     degrees, so the regular simplex occurs nowhere in D4, though it does
     occur as a Delaunay cell of a contact configuration.

Every integral over S^3 here is Monte Carlo and is reported as such; the
margins are percentages, not last digits.

Run:  python3 local_cell_obstruction.py
Exits nonzero if any check fails.
"""

import sys
from math import comb

import numpy as np
from scipy.spatial import ConvexHull

RESULTS = []
TARGET = 16.0 / np.pi ** 2
RNG = np.random.default_rng(20260912)


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("[%s] %s" % ("PASS" if ok else "FAIL", name))
    if detail:
        for line in detail.splitlines():
            print("       " + line)


def nz(X):
    X = np.atleast_2d(X)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def d4():
    o = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0.0] * 4
                    v[i], v[j] = si, sj
                    o.append(v)
    return nz(np.array(o))


def cell_mean(W, i, ns=3000000):
    """(mean of sec^4 over Omega_{w_i}, measure of Omega_{w_i})."""
    S = nz(RNG.normal(size=(ns, 4)))
    own = (S @ W.T).argmax(axis=1) == i
    cosd = (S[own] @ W.T).max(axis=1)
    return float(np.mean(cosd ** -4.0)), float(own.mean() * 2 * np.pi ** 2)


def delaunay_cells(W, tol=1e-7):
    """Delaunay cells on S^3: hull facets, merged when coplanar."""
    h = ConvexHull(W)
    raw = h.equations[:, :4]
    nrm = np.linalg.norm(raw, axis=1, keepdims=True)
    N = raw / nrm
    off = -h.equations[:, 4] / nrm[:, 0]
    reps, offs, cells = [], [], []
    for k, n in enumerate(N):
        hit = None
        for idx, r in enumerate(reps):
            if n @ r > 1 - tol and abs(off[k] - offs[idx]) < tol:
                hit = idx
                break
        if hit is None:
            reps.append(n)
            offs.append(off[k])
            cells.append(set(h.simplices[k]))
        else:
            cells[hit] |= set(h.simplices[k])
    return [sorted(c) for c in cells], np.array(reps), np.array(offs)


def simplex_mean(V, ns=2000000):
    """Mean of sec^4(distance to the rows of V) over the spherical
    simplex they span, in the barycentric parametrisation."""
    lam = RNG.dirichlet(np.ones(len(V)), size=ns)
    X = lam @ V
    r = np.linalg.norm(X, axis=1)
    w = r ** -4.0
    cosd = ((X / r[:, None]) @ V.T).max(axis=1)
    return float((w * cosd ** -4.0).sum() / w.sum())


def regular(e_deg):
    c = np.cos(np.radians(e_deg))
    G = np.full((4, 4), c)
    np.fill_diagonal(G, 1.0)
    return np.linalg.cholesky(G)


def circumradius(V):
    a = np.linalg.solve(V @ V.T, np.ones(len(V)))
    z = a @ V
    z /= np.linalg.norm(z)
    return float(np.degrees(np.arccos((z @ V.T).max()))), z


def main():
    U = d4()

    # ---- 1. (LOCAL) is sharp at the root configuration ---------------
    m0, s0 = cell_mean(U, 0)
    record("at the root configuration the mean of sec^4 over a Voronoi "
           "cell is 16/pi^2",
           abs(m0 - TARGET) < 3e-3 and abs(s0 - 2 * np.pi ** 2 / 24) < 3e-3,
           "mean %.6f against 16/pi^2 = %.6f\ncell measure %.6f against "
           "2 pi^2 / 24 = %.6f\n(Monte Carlo over S^3)"
           % (m0, TARGET, s0, 2 * np.pi ** 2 / 24))

    # ---- 2. nine contacts at 60 degrees break (LOCAL) ----------------
    P, maxip3 = None, None
    for _ in range(2000):
        Q = nz(RNG.normal(size=(9, 3)))
        for _ in range(20000):
            G = Q @ Q.T
            np.fill_diagonal(G, -2.0)
            if G.max() <= 1.0 / 3.0 + 1e-14:
                break
            D = np.maximum(G - 1.0 / 3.0, 0.0)
            np.fill_diagonal(D, 0.0)
            Q = nz(Q + 0.2 * (D.sum(axis=1)[:, None] * Q - D @ Q))
        G = Q @ Q.T
        np.fill_diagonal(G, -2.0)
        if G.max() <= 1.0 / 3.0 + 1e-12:
            P, maxip3 = Q, float(G.max())
            break
    w = np.array([1.0, 0.0, 0.0, 0.0])
    rows = []
    if P is not None:
        for ang in (60.0, 61.0):
            a = np.radians(ang)
            nb = np.hstack([np.full((9, 1), np.cos(a)), np.sin(a) * P])
            W9 = np.vstack([w, nb])
            G = W9 @ W9.T
            np.fill_diagonal(G, -2.0)
            mm, ss = cell_mean(W9, 0)
            rows.append((ang, float(G.max()), mm, ss))
    ok2 = (P is not None and all(r[2] < TARGET for r in rows)
           and rows[1][1] < 0.5 - 1e-3)
    record("nine contacts at 60 degrees give a cell that violates (LOCAL), "
           "and so do nine at 61 degrees",
           ok2,
           ("nine directions of R^3 with pairwise inner product at most "
            "1/3: largest is %.15f\n" % maxip3)
           + "\n".join(
               "contacts at %.0f deg: largest inner product %.6f, cell "
               "measure %.5f, mean of sec^4 %.6f  (target %.6f)"
               % (r[0], r[1], r[3], r[2], TARGET) for r in rows)
           + "\nadding further directions only shrinks the cell and lowers "
             "the mean")

    # ---- 3. the Delaunay decomposition does not localise -------------
    cells, reps, offs = delaunay_cells(U)
    Wp = nz(U[RNG.permutation(24)[:22]] + RNG.normal(size=(22, 4)) * 0.18)
    for _ in range(6000):
        G = Wp @ Wp.T
        np.fill_diagonal(G, -2.0)
        if G.max() <= 0.5:
            break
        D = np.maximum(G - 0.5, 0.0)
        np.fill_diagonal(D, 0.0)
        Wp = nz(Wp + 0.3 * (D.sum(axis=1)[:, None] * Wp - D @ Wp))
    cellsp, repsp, offsp = delaunay_cells(Wp)
    S = nz(RNG.normal(size=(200000, 4)))
    M = S @ repsp.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(M > 1e-12, offsp[None, :] / M, np.inf)
    which = t.argmin(axis=1)
    owner = (S @ Wp.T).argmax(axis=1)
    best = (S @ Wp.T).max(axis=1)
    nbad, worst = 0, 0.0
    for k in range(len(cellsp)):
        sel = np.where(which == k)[0]
        if not len(sel):
            continue
        inc = np.isin(owner[sel], cellsp[k])
        if inc.all():
            continue
        bad = sel[~inc]
        gap = best[bad] - (S[bad] @ Wp[cellsp[k]].T).max(axis=1)
        nbad += int((gap > 1e-9).sum())
        worst = max(worst, float(gap.max()))
    record("inside a Delaunay cell the nearest contact direction need not "
           "be a vertex of that cell",
           nbad > 0 and worst > 1e-3,
           "configuration of %d directions with %d Delaunay cells\n"
           "%d of %d sampled directions have their nearest contact outside "
           "the Delaunay cell containing them\n"
           "largest gap in cosine between the true nearest contact and the "
           "nearest vertex of the cell: %.6f\n"
           "so the integral over a Delaunay cell is not a functional of "
           "that cell alone" % (len(Wp), len(cellsp), nbad, len(S), worst))

    # ---- 4. the idealised Delaunay functional at the regular simplex --
    def pdf_M(x):
        s = 0.0
        for k in range(1, 5):
            u = 1 - k * x
            if u > 0:
                s += (-1) ** k * comb(4, k) * 3 * u ** 2 * (-k)
        return s

    ts = np.linspace(0.25, 1.0, 400001)
    fs = np.array([pdf_M(x) for x in ts])
    EA = float(np.trapezoid(fs * (1 + ts) ** -4, ts))
    EA1 = float(np.trapezoid(fs, ts))
    lam = RNG.dirichlet(np.ones(4), size=8000000)
    EB = float(np.mean((1 + (lam ** 2).sum(axis=1)) ** -2))
    closed = 4 * EA / EB
    r60 = simplex_mean(regular(60.0))
    r62 = simplex_mean(regular(62.0))
    Rr, _ = circumradius(regular(60.0))
    record("the regular simplex of edge 60 degrees falls 4.78 per cent "
           "short, and edge 62 degrees still falls short",
           abs(EA - 0.2) < 1e-6 and abs(EA1 - 1.0) < 1e-6
           and abs(closed - r60) < 4e-3 and closed < TARGET
           and r62 < TARGET,
           "E[(1+M)^-4] = %.9f, that is 1/5 exactly\n"
           "E[(1+q)^-2] = %.9f\n"
           "closed form 4 E[(1+M)^-4] / E[(1+q)^-2] = %.6f\n"
           "direct sampling, edge 60 deg                = %.6f\n"
           "direct sampling, edge 62 deg                = %.6f\n"
           "target 16/pi^2                              = %.6f\n"
           "shortfall at edge 60 deg: %.4f, that is %.2f per cent\n"
           "circumradius at edge 60 deg: %.4f deg, below 60, so saturation "
           "does not exclude it"
           % (EA, EB, closed, r60, r62, TARGET, TARGET - closed,
              100 * (TARGET - closed) / TARGET, Rr))

    # ---- 5. where the regular simplex does and does not occur --------
    sizes = sorted({len(c) for c in cells})
    circ = [np.degrees(np.arccos(abs(U[c] @ n).max()))
            for c, n in zip(cells, reps)]
    cliq = 0
    for i in range(24):
        for j in range(i + 1, 24):
            if abs(U[i] @ U[j] - 0.5) > 1e-9:
                continue
            for k in range(j + 1, 24):
                if (abs(U[i] @ U[k] - 0.5) > 1e-9
                        or abs(U[j] @ U[k] - 0.5) > 1e-9):
                    continue
                cliq = max(cliq, 3)
                for l in range(k + 1, 24):
                    if (abs(U[i] @ U[l] - 0.5) < 1e-9
                            and abs(U[j] @ U[l] - 0.5) < 1e-9
                            and abs(U[k] @ U[l] - 0.5) < 1e-9):
                        cliq = 4
    base = regular(60.0)
    Wc = [row for row in base]
    for p in nz(RNG.normal(size=(600000, 4))):
        if max(float(p @ q) for q in Wc) <= 0.5:
            Wc.append(p)
    Wc = np.array(Wc)
    _, zc = circumradius(base)
    near = np.sort(Wc @ zc)[::-1]
    cosR = float(base[0] @ zc)
    empty = abs(near[:4] - cosR).max() < 1e-9 and near[4] < cosR - 1e-6
    record("the Delaunay cells of D4 are octahedra, and the regular "
           "simplex occurs elsewhere but never in D4",
           len(cells) == 24 and sizes == [6]
           and max(abs(np.array(circ) - 45.0)) < 1e-6
           and cliq == 3 and empty and len(Wc) >= 12,
           "D4: %d Delaunay cells, %s vertices each, circumradius %.4f deg\n"
           "largest set of roots pairwise at 60 degrees: %d, so no Delaunay "
           "cell of D4 is a regular simplex of edge 60 degrees\n"
           "greedy completion of one such simplex: %d directions, all "
           "pairwise inner products at most 1/2; its circumcentre is at "
           "cos %.6f from each of the four and the next direction is at "
           "cos %.6f, so the circumsphere is empty"
           % (len(cells), sizes, circ[0], cliq, len(Wc), cosR, near[4]))

    print("=" * 62)
    n = sum(1 for _, o in RESULTS if o)
    print("%d of %d checks passed" % (n, len(RESULTS)))
    if n == len(RESULTS):
        print("The shape deficit of prop:area-optimal is not additive over")
        print("cells: no inequality applied one cell at a time, in either")
        print("decomposition, can reach the bound.")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
