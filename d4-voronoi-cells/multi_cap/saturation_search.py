#!/usr/bin/env python3
"""
An independent search for a saturated 23-point contact configuration.

This complements spherical_code_23.py and extendability.py.  Those two
search for a code of large minimal angle and then read off the covering
radius.  Here the covering radius is the objective from the start, which
is the quantity the open case is actually about.

Write W = {w_1,...,w_23} on S^3 with <w_i,w_j> <= 1/2 for i != j, and

    g(W) = min_{theta in S^3} max_i <theta,w_i>.

Because max_i <theta,w_i> is the support function of conv(W), the inner
minimum is the distance from the origin to the nearest facet hyperplane
of conv(W), so g(W) is the inradius of conv(W) about the origin and is
computed exactly from a convex hull.  W is saturated exactly when
g(W) > 1/2.  Equivalently, writing z_I for the cell vertex carried by an
active set I, |z_I|^2 = 1^T G_I^{-1} 1 and g(W) = 1/max_I |z_I|.

Three things are run:

  1  calibration.  The 24 roots give g = 1/sqrt(2), the circumradius
     sqrt(2) of the 24-cell; each of the 24 deletions gives g = 1/2
     exactly, the boundary of the criterion.

  2  cyclic orbits.  Every element of SO(4) of order 23 is conjugate to a
     rotation by 2*pi*a/23 and 2*pi*b/23 in two orthogonal planes, and a
     single orbit of it is determined up to congruence by how the base
     point splits between those planes.  The splitting enters the 22
     constraints <w_0,w_k> <= 1/2 linearly, so each of the 528 pairs
     (a,b) is decided by one linear program.  None of them is feasible:
     no 23-point code of minimal angle 60 degrees is a single cyclic
     orbit, and therefore none has a transitive symmetry group.  The
     smallest violation over the 528 types is also computed, at 60
     digits, so that the sign of the verdict is not in question.

  3  direct optimisation.  Minimise max_I |z_I|^2 by projected gradient
     with the packing condition as a penalty, from perturbed deletions,
     from random starts and from partial root systems completed at
     random; then restore feasibility exactly and evaluate g by hull.
     The gradient is closed form: with a = G_I^{-1} 1 and z_I = sum_i
     a_i w_i, d|z_I|^2 / dw_i = -2 a_i z_I.

Run:  python3 saturation_search.py
Exits nonzero if the calibration fails.  A saturated configuration, if
one were found, is written to saturated23.npy and reported loudly; no
run so far has produced one.
"""

import sys

import mpmath as mp
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

RNG = np.random.default_rng(20260913)
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
                    v = np.zeros(4)
                    v[i] = si
                    v[j] = sj
                    out.append(v / np.sqrt(2.0))
    return np.array(out)


def normalise(W):
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def largest_inner_product(W):
    G = W @ W.T
    np.fill_diagonal(G, -9.0)
    return float(G.max())


def g_of(W):
    """g(W) as the inradius of conv(W) about the origin."""
    try:
        hull = ConvexHull(W)
    except Exception:
        return 0.0
    return float(np.min(np.abs(hull.equations[:, -1])))


def cell_vertices(W):
    try:
        hull = ConvexHull(W, qhull_options="Qt")
    except Exception:
        return None, None, None
    simp = hull.simplices
    G = np.einsum("fia,fja->fij", W[simp], W[simp])
    keep = np.abs(np.linalg.det(G)) > 1e-12
    simp, G = simp[keep], G[keep]
    if len(simp) == 0:
        return None, None, None
    a = np.linalg.solve(G, np.ones((G.shape[0], 4, 1)))[:, :, 0]
    return simp, a, a.sum(axis=1)


def objective(W, beta, lam):
    simp, a, r2 = cell_vertices(W)
    if simp is None:
        return None, None
    peak = r2.max()
    wt = np.exp(beta * (r2 - peak))
    val = peak + np.log(wt.sum()) / beta
    wt /= wt.sum()

    grad = np.zeros_like(W)
    z = np.einsum("fi,fia->fa", a, W[simp])
    term = (-2.0 * a[:, :, None] * z[:, None, :]) * wt[:, None, None]
    np.add.at(grad, simp.ravel(), term.reshape(-1, 4))

    G = W @ W.T
    np.fill_diagonal(G, -9.0)
    slack = np.maximum(G - 0.5, 0.0)
    grad += lam * (slack @ W)
    return val + lam * float((slack ** 2).sum()) / 2.0, grad


def restore_feasibility(W, iters=40000):
    step = 0.05
    for t in range(iters):
        G = W @ W.T
        np.fill_diagonal(G, -9.0)
        slack = np.maximum(G - 0.5, 0.0)
        if slack.max() < 1e-14:
            break
        W = normalise(W - step * (slack @ W))
        if t % 4000 == 3999:
            step *= 0.7
    return W


def descend(W0, steps=900, lam=800.0):
    W = normalise(W0.copy())
    mom = np.zeros_like(W)
    for t in range(steps):
        beta = 60.0 + 540.0 * t / steps
        rate = 0.03 * (1.0 - 0.92 * t / steps)
        val, grad = objective(W, beta, lam)
        if val is None:
            return None
        grad -= np.einsum("ia,ia->i", grad, W)[:, None] * W
        mom = 0.9 * mom + grad
        W = normalise(W - rate * mom)
    return restore_feasibility(W)


roots = d4_directions()

g24 = g_of(roots)
record("root system: g = 1/sqrt(2), circumradius sqrt(2)",
       abs(g24 - 1.0 / np.sqrt(2.0)) < 1e-12,
       "g = %.15f   1/g = %.15f" % (g24, 1.0 / g24))

gdel = np.array([g_of(np.delete(roots, k, axis=0)) for k in range(24)])
record("all 24 deletions: g = 1/2, circumradius 2",
       np.all(np.abs(gdel - 0.5) < 1e-12),
       "spread over the 24 deletions: %.3e" % float(np.ptp(gdel)))

feasible_cyclic = []
for a in range(23):
    for b in range(23):
        if a == 0 and b == 0:
            continue
        t1, t2 = 2 * np.pi * a / 23, 2 * np.pi * b / 23
        A = np.array([[np.cos(k * t1) - np.cos(k * t2)] for k in range(1, 23)])
        rhs = np.array([0.5 - np.cos(k * t2) for k in range(1, 23)])
        out = linprog(c=[0.0], A_ub=A, b_ub=rhs, bounds=[(0.0, 1.0)])
        if out.status == 0:
            feasible_cyclic.append((a, b))
mp.mp.dps = 60


def worst_violation(a, b):
    """min over c in [0,1] of the largest amount by which <w_0,w_k> beats 1/2."""
    t1, t2 = 2 * mp.pi * a / 23, 2 * mp.pi * b / 23
    A = [mp.cos(k * t1) - mp.cos(k * t2) for k in range(1, 23)]
    B = [mp.mpf(1) / 2 - mp.cos(k * t2) for k in range(1, 23)]
    f = lambda c: max(A[i] * c - B[i] for i in range(22))
    cand = [mp.mpf(0), mp.mpf(1)]
    for i in range(22):
        for j in range(i + 1, 22):
            d = A[i] - A[j]
            if abs(d) > mp.mpf("1e-40"):
                c = (B[i] - B[j]) / d
                if 0 <= c <= 1:
                    cand.append(c)
    return min(f(c) for c in cand)


margin, at = None, None
for a in range(23):
    for b in range(23):
        if a == 0 and b == 0:
            continue
        d = worst_violation(a, b)
        if margin is None or d < margin:
            margin, at = d, (a, b)

record("no 23-point code is a single cyclic orbit",
       len(feasible_cyclic) == 0 and margin > 0,
       "528 rotation types tested by linear programming; feasible: %d\n"
       "smallest violation over all 528, at 60 digits: %s, at (a,b) = %s\n"
       "so no contact configuration of 23 directions has a transitive "
       "symmetry group" % (len(feasible_cyclic), mp.nstr(margin, 12), at))

starts = []
for k in range(24):
    for spread in (0.15, 0.30, 0.55):
        starts.append(normalise(np.delete(roots, k, axis=0)
                                + spread * RNG.standard_normal((23, 4))))
for _ in range(90):
    starts.append(normalise(RNG.standard_normal((23, 4))))
for _ in range(40):
    part = np.vstack([roots[RNG.permutation(24)[:14]],
                      normalise(RNG.standard_normal((9, 4)))])
    starts.append(normalise(part + 0.2 * RNG.standard_normal((23, 4))))

best_g, best_W, feasible = 0.0, None, 0
for W0 in starts:
    W = descend(W0)
    if W is None or largest_inner_product(W) > 0.5 + 1e-11:
        continue
    feasible += 1
    g = g_of(W)
    if g > best_g:
        best_g, best_W = g, W.copy()

detail = ["starts %d, codes reached %d" % (len(starts), feasible),
          "largest g attained: %.12f" % best_g]
if best_W is not None:
    ip = (best_W @ best_W.T)[np.triu_indices(23, 1)]
    vals, cnt = np.unique(np.round(ip, 3), return_counts=True)
    detail.append("inner products of the best outcome: "
                  + ", ".join("%+.1f x%d" % (v, c) for v, c in zip(vals, cnt)))
record("no saturated configuration found", best_g <= 0.5 + 1e-10,
       "\n".join(detail))

if best_g > 0.5 + 1e-10:
    np.save("saturated23.npy", best_W)
    print("\n*** a configuration with g > 1/2 was reached; "
          "written to saturated23.npy ***")

print()
bad = [n for n, ok in RESULTS if not ok]
print("%d checks, %d passed" % (len(RESULTS), len(RESULTS) - len(bad)))
print("This is exploration. It is not evidence that no saturated "
      "configuration exists.")
sys.exit(1 if bad else 0)
