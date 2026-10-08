#!/usr/bin/env python3
"""
inradius_search.py -- a direct search for a saturated 23-point contact
configuration, by maximising the inradius of the convex hull.

For unit vectors W = {w_1, ..., w_23} with the origin inside conv(W), the
quantity
    g(W) = min over theta in S^3 of max_i <theta, w_i>
of Proposition "Extendability and the circumradius" is the inradius of
conv(W): the distance from the origin to the nearest facet hyperplane.  W
admits a twenty-fourth contact direction exactly when g(W) <= 1/2 and is
saturated when g(W) > 1/2.  The deletion of a root from D4 has g = 1/2
exactly, the deleted root being its unique deepest hole.

The search maximises g(W) subject to the contact constraints
<w_i, w_j> <= 1/2 and |w_i| = 1 by sequential quadratic programming on the
epigraph form
    maximise t  subject to  offset_F(W) >= t for every facet F of conv(W),
with analytic gradients, the facet list being refreshed from the convex
hull after every solve until it stops changing.  Starts are of three
kinds, taken in rotation: uniformly random points pushed apart until they
satisfy the contact constraints; a deletion of a root with each direction
rotated through a random angle of up to 0.05, 0.15, 0.4 or 0.8 radians;
and twenty random roots completed by three random directions.  Every start
that ends with all constraints satisfied to 1e-9 is recorded with its
final g and the multiset of its inner products rounded to 1e-6, which at a
deletion is -1 (11 times), -1/2 (88), 0 (66), 1/2 (88).

Usage:  python3 inradius_search.py [starts] [seed]
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

M = 23
IU = np.triu_indices(M, 1)

def roots():
    R = []
    for i in range(4):
        for j in range(i+1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4); v[i] = si; v[j] = sj; R.append(v / np.sqrt(2))
    return np.array(R)
ROOTS = roots()

def normalise(W):
    return W / np.linalg.norm(W, axis=1, keepdims=True)

def push_apart(W, slack=0.0, iters=5000):
    """Iteratively separate the points until every inner product is at most
    1/2 + slack; aims a hair inside so that the target is met exactly."""
    W = normalise(W)
    target = 0.5 + slack - 1e-7
    for _ in range(iters):
        G = W @ W.T; np.fill_diagonal(G, -1)
        if G.max() <= target: break
        E = np.maximum(G - target, 0.0)
        W = normalise(W - 0.8 * (E @ W))
    return W

# ---------- constraints with analytic Jacobians
def norm_c(x):
    W = x[:4*M].reshape(M, 4)
    return np.sum(W*W, axis=1) - 1.0
def norm_j(x):
    W = x[:4*M].reshape(M, 4)
    J = np.zeros((M, 4*M+1))
    for i in range(M): J[i, 4*i:4*i+4] = 2*W[i]
    return J

def pair_c(x, P, slack=0.0):
    W = x[:4*M].reshape(M, 4)
    return 0.5 + slack - np.einsum('ij,ij->i', W[P[0]], W[P[1]])
def pair_j(x, P):
    W = x[:4*M].reshape(M, 4)
    n = len(P[0]); J = np.zeros((n, 4*M+1))
    for k, (i, j) in enumerate(zip(*P)):
        J[k, 4*i:4*i+4] = -W[j]; J[k, 4*j:4*j+4] = -W[i]
    return J

def _solve(P, b):
    try:
        return np.linalg.solve(P, b[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        # a facet hyperplane passing through the origin has infinite normal;
        # regularise, which only affects starts that are far from feasible
        return np.linalg.solve(P + 1e-9 * np.eye(4)[None], b[:, :, None])[:, :, 0]

def facet_c(x, Fs):
    W = x[:4*M].reshape(M, 4)
    P = W[Fs]                                  # (nf, 4, 4)
    n = _solve(P, np.ones((len(Fs), 4)))
    off = 1.0 / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    return off - x[-1]
def facet_j(x, Fs):
    W = x[:4*M].reshape(M, 4)
    nf = len(Fs)
    P = W[Fs]
    n = _solve(P, np.ones((nf, 4)))
    a = _solve(np.transpose(P, (0, 2, 1)), n)          # a = P^{-T} n
    nn = np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    J = np.zeros((nf, 4*M+1))
    for f in range(nf):
        for k in range(4):
            i = Fs[f, k]
            J[f, 4*i:4*i+4] += a[f, k] * n[f] / nn[f]**3
    J[:, -1] = -1.0
    return J

def restore(W, slack=0.0, iters=300, step0=0.05):
    """Sequential linear programming for feasibility: raise the smallest
    value of 1/2 + slack - <w_i, w_j> until it is nonnegative."""
    from scipy.optimize import linprog
    W = normalise(W); step = step0; n = 4*M
    def worst(W):
        G = W @ W.T; np.fill_diagonal(G, -1)
        return 0.5 + slack - G.max()
    w = worst(W)
    for _ in range(iters):
        if w >= 0: break
        x = np.r_[W.ravel(), 0.0]
        G = W @ W.T
        near = G[IU] > 0.5 + slack - 4*step - max(0, -w)
        P = (IU[0][near], IU[1][near])
        Pc = pair_c(x, P, slack); Pj = pair_j(x, P)[:, :n]
        c = np.zeros(n+1); c[-1] = -1.0
        A_ub = np.hstack([-Pj, np.ones((len(P[0]), 1))]); b_ub = Pc
        A_eq = np.zeros((M, n+1))
        for i in range(M): A_eq[i, 4*i:4*i+4] = W[i]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(M),
                      bounds=[(-step, step)]*n + [(None, None)], method='highs')
        if res.status != 0:
            step *= 0.5
            if step < 1e-10: break
            continue
        Wn = normalise(W + res.x[:n].reshape(M, 4)); wn = worst(Wn)
        if wn > w:
            W, w = Wn, wn; step = min(step*1.5, 0.2)
        else:
            step *= 0.5
            if step < 1e-10: break
    return W, w

def merit(W):
    """(g, largest inner product, facet list) at a unit-vector configuration.
    g is read from Qhull's facet equations, which are robust when a facet is
    not a simplex (at a deletion the cube of eight neighbours of the deleted
    root is one facet); the simplex list kept for the linearisation is
    restricted to simplices whose four vertices are affinely independent."""
    hull = ConvexHull(W)
    g = float((-hull.equations[:, -1]).min())
    Fs = hull.simplices
    dets = np.abs(np.linalg.det(W[Fs]))
    Fs = Fs[dets > 1e-7]
    G = W @ W.T; np.fill_diagonal(G, -1)
    return g, G.max(), Fs

def solve_epigraph(W, slack=0.0, iters=400, step0=0.05):
    """Sequential linear programming with a trust region: at each step
    maximise t over tangent moves d of norm at most `step` such that every
    linearised facet offset is at least t and every linearised pair inner
    product is at most 1/2; accept the move if the true g improves without
    violating the contact constraints, otherwise shrink the step."""
    from scipy.optimize import linprog
    W = normalise(W)
    step = step0
    try:
        g, gmax, Fs = merit(W)
    except Exception:
        return None
    # infeasible start: cannot happen after push_apart, but guard anyway
    if gmax > 0.5 + slack + 1e-9:
        return None
    n = 4*M
    for it in range(iters):
        x = np.r_[W.ravel(), 0.0]
        Fc = facet_c(x, Fs); Fj = facet_j(x, Fs)[:, :n]
        G = W @ W.T
        near = G[IU] > 0.5 + slack - 4*step
        P = (IU[0][near], IU[1][near])
        Pc = pair_c(x, P, slack); Pj = pair_j(x, P)[:, :n]
        # variables: d (n), t (1).  maximise t.
        c = np.zeros(n+1); c[-1] = -1.0
        # -Fj d + t <= Fc      (offset + Fj d >= t)
        A1 = np.hstack([-Fj, np.ones((len(Fs), 1))]); b1 = Fc
        # -Pj d <= Pc          (0.5 - G - Pj... : pair_c + Pj d >= 0)
        A2 = np.hstack([-Pj, np.zeros((len(P[0]), 1))]); b2 = Pc
        A_ub = np.vstack([A1, A2]); b_ub = np.r_[b1, b2]
        # tangency: w_i . d_i = 0
        A_eq = np.zeros((M, n+1))
        for i in range(M): A_eq[i, 4*i:4*i+4] = W[i]
        b_eq = np.zeros(M)
        bounds = [(-step, step)]*n + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status != 0:
            step *= 0.5
            if step < 1e-9: break
            continue
        d = res.x[:n].reshape(M, 4)
        pred = res.x[-1] - g
        Wn = normalise(W + d)
        try:
            gn, gmaxn, Fsn = merit(Wn)
        except Exception:
            step *= 0.5; continue
        if gmaxn <= 0.5 + slack + 1e-10 and gn > g + 1e-13:
            W, g, Fs = Wn, gn, Fsn
            if pred > 0 and (gn - g) > 0.5 * pred: step = min(step*1.5, 0.2)
        else:
            step *= 0.5
        if step < 1e-9 or pred < 1e-12: break
    G = W @ W.T; np.fill_diagonal(G, -1)
    return W, g, max(0.0, G.max() - 0.5 - slack), G

def random_start(kind, rng, pert):
    if kind == 'random':
        return push_apart(rng.normal(size=(M, 4)), slack=0.05)
    if kind == 'deletion':
        W = np.delete(ROOTS, rng.integers(24), axis=0)
        ang = rng.uniform(0, pert, size=M)
        D = rng.normal(size=(M, 4))
        D -= np.sum(D * W, axis=1, keepdims=True) * W
        D = normalise(D)
        return normalise(np.cos(ang)[:, None] * W + np.sin(ang)[:, None] * D)
    if kind == 'partial':
        idx = rng.choice(24, 20, replace=False)
        return normalise(np.vstack([ROOTS[idx], normalise(rng.normal(size=(3, 4)))]))
    raise ValueError(kind)

def multiset(G):
    u, c = np.unique(np.round(G[IU], 6), return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))

DELETION_MULTISET = {-1.0: 11, -0.5: 88, 0.0: 66, 0.5: 88}

def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng = np.random.default_rng(seed)
    kinds = ['random', 'deletion', 'partial']
    t0 = time.time()
    best = -1.0; best_W = None; n_ok = 0; n_del = 0; n_sat = 0
    hist = {}; by_kind = {k: [0, 0] for k in kinds}; path_hist = []
    for s in range(starts):
        kind = kinds[s % 3]
        W0 = random_start(kind, rng, pert=rng.choice([0.05, 0.15, 0.4, 0.8]))
        if kind == 'random':
            # continuation in the slack: optimise with relaxed contact
            # constraints, then tighten them in stages down to zero
            out = None; W = W0
            for slack in (0.05, 0.02, 0.01, 0.005, 0.002, 0.0):
                W, w = restore(W, slack=slack)
                if w < 0: out = None; break
                out = solve_epigraph(W, slack=slack)
                if out is None: break
                W = out[0]
                if slack == 0.05: g_relaxed = out[1]
            if out is None: continue
            W, g, viol, G = out
            path_hist.append((g_relaxed, g))
        else:
            W0, w = restore(W0, slack=0.0)
            if w < -1e-10: continue
            out = solve_epigraph(W0)
            if out is None: continue
            W, g, viol, G = out
        if viol > 1e-9: continue
        n_ok += 1; by_kind[kind][0] += 1
        ms = multiset(G)
        if ms == DELETION_MULTISET: n_del += 1; by_kind[kind][1] += 1
        b = round(g, 3); hist[b] = hist.get(b, 0) + 1
        if g > 0.5 + 1e-7:
            n_sat += 1
            print(f"  !! start {s} ({kind}): g = {g:.9f} > 1/2, violation {viol:.1e}, inner products {ms}")
            np.save(f"saturated_candidate_{s}.npy", W)
        if g > best: best, best_W = g, W
        if (s+1) % 200 == 0:
            print(f"[{s+1}/{starts}] feasible {n_ok}, deletion-type {n_del}, best g {best:.9f}, "
                  f"{time.time()-t0:.0f}s", flush=True)
    print()
    print(f"starts: {starts} (seed {seed}); feasible to 1e-9: {n_ok}; "
          f"with the inner-product multiset of a deletion: {n_del}")
    for k in kinds: print(f"   {k:9s}: feasible {by_kind[k][0]}, deletion-type {by_kind[k][1]}")
    print(f"largest g reached: {best:.9f}  (1/2 at every deletion)")
    print(f"starts with g > 1/2 + 1e-7: {n_sat}")
    print("final g, rounded to 1e-3, over the feasible endpoints:")
    for k in sorted(hist): print(f"   {k:.3f}: {hist[k]}")
    if path_hist:
        gr = np.array([p[0] for p in path_hist]); gf = np.array([p[1] for p in path_hist])
        print(f"random starts through the slack continuation: {len(path_hist)}; at slack 0.05 the "
              f"largest g was {gr.max():.6f}, at slack 0 the largest was {gf.max():.9f}")
    if best_W is not None:
        G = best_W @ best_W.T; np.fill_diagonal(G, -1)
        print(f"best endpoint: largest inner product {G.max():.9f}, multiset {multiset(G)}")
    print("RESULT: " + ("no saturated configuration found; every feasible endpoint has g <= 1/2."
          if n_sat == 0 else "candidates saved; verify them exactly before drawing any conclusion."))
    print(f"total time {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
