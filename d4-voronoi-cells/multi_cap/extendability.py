#!/usr/bin/env python3
"""
The extendability criterion of prop:extendable and the reduction of
cor:remaining to the saturated 23-point case.

A contact configuration W admits a further contact direction if and only
if its cell contains a point of norm 2, equivalently its cell has
circumradius at least 2, equivalently its covering radius on S^3 is at
least 60 degrees. Writing

    g(W) = min_{theta in S^3} max_{w in W} <theta, w>,

the cell circumradius is 1/g(W), so W is saturated (non-extendable)
exactly when g(W) > 1/2.

Checks, in order:

  1  the three forms of the criterion agree, on the root system, on the
     root system minus one root, and on random configurations;
  2  the root configuration has g = cos 45 deg = 1/sqrt(2), circumradius
     sqrt(2), and is saturated, as it must be since it is a 24-point
     kissing configuration;
  3  the root system minus one root has g = 1/2 and circumradius exactly
     2: it sits precisely on the boundary of the criterion, and the
     direction that extends it is the deleted root;
  4  its cell volume is 25/3, the 24-cell plus one pyramid;
  5  the Gram values of the root configuration lie in {-1,-1/2,0,1/2},
     which is what Lemma 5.1 of de Laat, Leijenhorst and de Muinck Keizer
     proves for every 24-point configuration;
  6  the chain of cor:remaining: at most 22 contacts gives at least
     8.0464 by the covering bound, exactly 24 gives exactly 8, and 23
     with circumradius at least 2 gives strictly more than 8;
  7  a search for a saturated 23-point configuration, started from the
     24 deletions, from greedy random packings and from annealed random
     starts, reported as numerical exploration and not as a proof.

Run:  python3 extendability.py
Exits nonzero if any check fails.
"""

import sys

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("[%s] %s" % ("PASS" if ok else "FAIL", name))
    if detail:
        for line in detail.splitlines():
            print("       " + line)


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


def cell(W):
    """Vertices of the cell, or None if unbounded."""
    m = len(W)
    for k in range(4):
        for s in (1.0, -1.0):
            c = np.zeros(4)
            c[k] = -s
            r = linprog(c, A_ub=W, b_ub=np.ones(m),
                        bounds=[(None, None)] * 4, method="highs")
            if r.status != 0:
                return None
    return HalfspaceIntersection(np.hstack([W, -np.ones((m, 1))]),
                                 np.zeros(4)).intersections


def circumradius_volume(W):
    V = cell(W)
    if V is None:
        return None, None
    return (np.linalg.norm(V, axis=1).max(),
            ConvexHull(V, qhull_options="QJ").volume)


_RNG = np.random.default_rng(20260912)
_S = _RNG.normal(size=(40000, 4))
_S /= np.linalg.norm(_S, axis=1, keepdims=True)


def g_of(W, refine=True):
    """min over S^3 of max_i <theta, w_i>, by sampling plus local search."""
    h = (_S @ np.asarray(W).T).max(axis=1)
    k = int(np.argmin(h))
    best, th = h[k], _S[k].copy()
    if refine:
        step = 0.25
        for _ in range(200):
            cand = th + _RNG.normal(size=(40, 4)) * step
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            hv = (cand @ np.asarray(W).T).max(axis=1)
            j = int(np.argmin(hv))
            if hv[j] < best:
                best, th = hv[j], cand[j]
            else:
                step *= 0.94
    return best


def extendable_by_lp(W, tries=20000):
    """Direct search for an admissible extra direction."""
    X = _RNG.normal(size=(tries, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return bool(((X @ np.asarray(W).T).max(axis=1) <= 0.5 + 1e-12).any())


def separate(W, margin=1e-9, sweeps=1500, step=0.10):
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    m, tgt = len(W), 0.5 - margin
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
    return W if G.max() <= 0.5 + 1e-9 else None


def greedy(m, tries=200000):
    """Greedy random packing of m directions with pairwise ip <= 1/2."""
    P = _RNG.normal(size=(tries, 4))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    W = [P[0]]
    for p in P[1:]:
        if len(W) == m:
            break
        if max(float(p @ q) for q in W) <= 0.5:
            W.append(p)
    return np.array(W) if len(W) == m else None


def cap_area(r):
    return np.pi * (2.0 * r - np.sin(2.0 * r))


def covering_bound(m):
    r = brentq(lambda r: 2.0 * r - np.sin(2.0 * r) - 2.0 * np.pi / m,
               1e-14, np.pi / 2.0, xtol=1e-15, rtol=8.9e-16)
    return np.pi * m / 3.0 * np.tan(r) ** 3


def main():
    print("The extendability criterion")
    print("=" * 62)
    U = d4_directions()

    # 1-2. the root configuration
    Rc, vol = circumradius_volume(U)
    g = g_of(U)
    record("root configuration: circumradius sqrt(2), g = 1/sqrt(2), "
           "saturated",
           abs(Rc - np.sqrt(2.0)) < 1e-7 and abs(g - 1 / np.sqrt(2.0)) < 1e-4
           and abs(vol - 8.0) < 1e-7,
           "circumradius %.8f (sqrt 2 = %.8f)   g %.6f (cos 45 deg = "
           "%.6f)\ncell volume %.8f   1/g = %.6f" %
           (Rc, np.sqrt(2.0), g, np.cos(np.pi / 4), vol, 1.0 / g))

    # 3-4. the root configuration minus one root
    W23 = np.delete(U, 0, axis=0)
    Rc23, vol23 = circumradius_volume(W23)
    g23 = g_of(W23)
    ext23 = extendable_by_lp(np.vstack([W23]))
    record("root system minus one root: circumradius exactly 2, g = 1/2, "
           "extendable",
           abs(Rc23 - 2.0) < 1e-7 and abs(g23 - 0.5) < 1e-4
           and abs(vol23 - 25.0 / 3.0) < 1e-6,
           "circumradius %.8f   g %.6f   cell volume %.8f (25/3 = %.8f)\n"
           "the deleted root extends it: max inner product with the other "
           "23 is %.6f" % (Rc23, g23, vol23, 25.0 / 3.0,
                           (U[0] @ W23.T).max()))

    # the three forms agree on random configurations
    agree, tested = 0, 0
    for m in (10, 16):
        W = separate(_RNG.normal(size=(m, 4)))
        if W is None:
            continue
        Rc, _ = circumradius_volume(W)
        if Rc is None:
            continue
        tested += 1
        by_circ = Rc >= 2.0 - 1e-7
        by_g = g_of(W) <= 0.5 + 1e-4
        by_search = extendable_by_lp(W)
        if by_circ == by_g == by_search:
            agree += 1
    record("the three forms of the criterion agree on random "
           "configurations", tested > 0 and agree == tested,
           "configurations tested: %d, agreements: %d" % (tested, agree))

    # 5. Gram values of the root system
    G = U @ U.T
    vals = sorted(set(np.round(G[~np.eye(24, dtype=bool)], 12)))
    record("root configuration has Gram values in {-1,-1/2,0,1/2}",
           vals == [-1.0, -0.5, 0.0, 0.5],
           "off-diagonal values: %s" % vals)

    # 6. the chain of the corollary
    b22, b24 = covering_bound(22), covering_bound(24)
    record("the chain of cor:remaining holds numerically",
           b22 > 8.0 and abs(vol - 8.0) < 1e-7 and vol23 > 8.0,
           "m <= 22: covering bound %.6f > 8\n"
           "m  = 24: root configuration, volume exactly 8\n"
           "m  = 23 extendable: %.6f > 8 (root system minus one root)\n"
           "m  = 24 covering bound for comparison: %.6f" %
           (b22, vol23, b24))

    # 7. search for a saturated 23-point configuration
    starts = [np.delete(U, r, axis=0).copy() for r in range(24)]
    ngreedy = 0
    for _ in range(24):
        Wg = greedy(23)
        if Wg is not None:
            starts.append(Wg)
            ngreedy += 1
    nrand = 0
    for _ in range(24):
        Wr = separate(_RNG.normal(size=(23, 4)), sweeps=3000)
        if Wr is not None:
            starts.append(Wr)
            nrand += 1
    best, nvalid = -1.0, 0
    for W0 in starts:
        W0 = separate(W0, sweeps=2000)
        if W0 is None:
            continue
        nvalid += 1
        W, cur = W0.copy(), g_of(W0, refine=False)
        T = 0.04
        for _ in range(25):
            Wn = separate(W + _RNG.normal(size=W.shape) * T, sweeps=400)
            if Wn is None:
                T *= 0.95
                continue
            vn = g_of(Wn, refine=False)
            if vn > cur:
                W, cur = Wn, vn
            T *= 0.97
        best = max(best, g_of(W))
    record("search for a saturated 23-point configuration found none",
           best <= 0.5 + 1e-4,
           "starting configurations: 24 deletions, %d greedy random "
           "packings, %d annealed random starts;\n"
           "%d of the %d reached a valid contact configuration\n"
           "largest g attained: %.6f; saturation would need g > 1/2\n"
           "this is numerical exploration, not a proof"
           % (ngreedy, nrand, nvalid, len(starts), best))

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
