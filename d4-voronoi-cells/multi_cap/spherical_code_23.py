"""
spherical_code_23.py

Contact configurations of a given size, viewed as spherical codes.

A contact configuration of m directions is a set of m unit vectors in R^4
with pairwise inner products at most 1/2, that is an m-point code on S^3
of minimal angle at least 60 degrees. This script measures how much room
such a code has at m = 22, 23, 24, 25, and looks for a 23-point contact
configuration whose Gram matrix differs from that of a deletion of one
root of D_4.

The search is Riesz continuation. Minimising sum_{i<j} |w_i - w_j|^{-s}
with s increasing drives a configuration towards the best packing; the
result is then polished by pushing the largest inner product down
directly. Calibration is at m = 24, where the answer is exactly 1/2, and
at m = 25, where no contact configuration exists.

Output of a run is reproduced in sec:m23-codes of the paper.
"""
import numpy as np
from itertools import combinations
from collections import Counter
from scipy.spatial import ConvexHull, HalfspaceIntersection

SEED = 90210
NSTART = 1500
SCHEDULE = (4, 16, 64, 256, 1024)


def normalise(W):
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def d4_roots():
    """The 24 normalised roots of D_4."""
    V = []
    for i, j in combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si
                v[j] = sj
                V.append(v / np.sqrt(2.0))
    V = np.array(V)
    assert V.shape == (24, 4)
    return V


def maxdot(W):
    G = W @ W.T
    np.fill_diagonal(G, -np.inf)
    return G.max()


def fingerprint(W, ndig=2):
    """The multiset of Gram entries, rounded; invariant under O(4)."""
    G = W @ W.T
    iu = np.triu_indices(len(W), 1)
    vals, counts = np.unique(np.round(G[iu], ndig), return_counts=True)
    return tuple(zip(vals.tolist(), counts.tolist()))


def riesz(W, schedule=SCHEDULE, iters=250):
    W = normalise(W.copy())
    best, bestW = maxdot(W), W.copy()
    for s in schedule:
        lr = 0.08
        for _ in range(iters):
            D = W[:, None, :] - W[None, :, :]
            r2 = np.einsum("ijk,ijk->ij", D, D)
            np.fill_diagonal(r2, np.inf)
            coef = -s * r2 ** (-s / 2.0 - 1.0)
            G = 2.0 * np.einsum("ij,ijk->ik", coef, D)
            G = G - np.sum(G * W, axis=1, keepdims=True) * W
            gn = np.linalg.norm(G)
            if gn > 0:
                G /= gn
            W = normalise(W - lr * G)
            lr *= 0.995
            v = maxdot(W)
            if v < best:
                best, bestW = v, W.copy()
    return bestW, best


def polish(W, iters=8000):
    W = normalise(W.copy())
    best, bestW = maxdot(W), W.copy()
    lr, beta = 0.01, 200.0
    for it in range(iters):
        G = W @ W.T
        np.fill_diagonal(G, -np.inf)
        mx = G.max()
        if mx < best:
            best, bestW = mx, W.copy()
        Wt = np.exp(beta * (G - mx))
        np.fill_diagonal(Wt, 0.0)
        Wt /= Wt.sum()
        Gr = 2.0 * (Wt @ W)
        Gr = Gr - np.sum(Gr * W, axis=1, keepdims=True) * W
        gn = np.linalg.norm(Gr)
        if gn > 0:
            Gr /= gn
        W = normalise(W - lr * Gr)
        if it % 800 == 799:
            beta = min(beta * 2.0, 2.0e5)
            lr *= 0.55
    return bestW, best


def run(m, starts, rng):
    best, bestW = np.inf, None
    census = Counter()
    below = 0
    for _ in range(starts):
        W0 = normalise(rng.normal(size=(m, 4)))
        W, v = riesz(W0)
        W, v = polish(W)
        if v < best:
            best, bestW = v, W.copy()
        if v < 0.5 - 1e-9:
            below += 1
        if v < 0.5 + 5e-3:
            census[fingerprint(W)] += 1
    return bestW, best, census, below


# ------------------------------------------------------------ cell data

def cell_volume(W):
    A = np.hstack([W, -np.ones((len(W), 1))])
    hs = HalfspaceIntersection(A, np.zeros(4))
    return ConvexHull(hs.intersections).volume


def g_of(W, rng, nsample=200000):
    """min over u in S^3 of max_i <u,w_i>, from hull circumcentres and probes."""
    best = np.inf
    try:
        hull = ConvexHull(W)
        for simplex in hull.simplices:
            idx = np.asarray(simplex)
            if len(idx) != 4:
                continue
            try:
                y = np.linalg.solve(W[idx], np.ones(4))
            except np.linalg.LinAlgError:
                continue
            ny = np.linalg.norm(y)
            if ny > 1e-12:
                u = y / ny
                best = min(best, np.max(W @ u), np.max(W @ (-u)))
    except Exception:
        pass
    U = normalise(rng.default_rng(0).normal(size=(nsample, 4))) \
        if hasattr(rng, "default_rng") else normalise(rng.normal(size=(nsample, 4)))
    best = min(best, (U @ W.T).max(axis=1).min())
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    R = d4_roots()
    Rm = R[:23]

    print("the two reference configurations")
    print(f"  24 roots         largest inner product {maxdot(R):.12f}")
    print(f"                   cell volume           {cell_volume(R):.10f}")
    print(f"  delete one root  largest inner product {maxdot(Rm):.12f}")
    print(f"                   cell volume           {cell_volume(Rm):.10f}"
          f"   (25/3 = {25/3:.10f})")
    print(f"                   g(W)                  {g_of(Rm, rng):.10f}")
    print(f"                   Gram multiset         {fingerprint(Rm)}")
    print()

    print("calibration and search")
    print(f"{'m':>3}  {'starts':>7}  {'smallest largest inner product':>32}"
          f"  {'minimal angle':>15}")
    results = {}
    for m, starts in ((22, 120), (24, 120), (25, 60), (23, NSTART)):
        W, v, census, below = run(m, starts, rng)
        results[m] = (W, v, census, below)
        print(f"{m:>3}  {starts:>7}  {v:>32.12f}"
              f"  {np.degrees(np.arccos(v)):>13.7f} deg", flush=True)
    print()

    W23, v23, census23, below23 = results[23]
    print(f"at m = 23, over {NSTART} random starts")
    print(f"  smallest largest inner product found : {v23:.12f}")
    print(f"  starts that got below 1/2            : {below23}")
    print(f"  outcomes within 5e-3 of feasibility  : {sum(census23.values())}")
    print(f"  distinct Gram multisets among them   : {len(census23)}")
    target = fingerprint(Rm)
    for fp, c in census23.most_common():
        print(f"    count {c:5d}   matches a deletion: {fp == target}")
        if fp != target:
            print(f"      {fp}")
