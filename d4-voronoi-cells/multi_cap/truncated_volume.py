#!/usr/bin/env python3
"""
truncated_volume.py -- the cell volume truncated at radius sqrt(8/5), the
three-point lower bound for the cell volume at twenty-three contacts.

For a contact configuration W the cell is V = {x : <x, w_i> <= 1} and
    vol(V) = (1/4) int_{S^3} sec^4 delta(theta) dtheta,
delta(theta) the angular distance from theta to the nearest w_i.  For any
R >= 1 the volume of V inside the ball of radius R is
    vol(V cap B(R)) = (1/4) int_{S^3} min(sec delta, R)^4 dtheta,
a lower bound for vol(V).  At R = sqrt(8/5) = sec(37.7612...) the bound
is a function of the pairwise and triple angles alone: four contact
directions never lie in a cap of angular radius below arcsin sqrt(3/8) =
37.7612 degrees, so for r below that radius no point of S^3 lies within r
of four of them, and the measure of the set of points farther than r from
all of them is 2 pi^2 - S1(r) + S2(r) - S3(r) exactly, with S1, S2, S3
the sums of the measures of the caps of radius r, of their pairwise
intersections and of their triple intersections.

This script evaluates vol(V cap B(sqrt(8/5))) by a fixed quasi-random
quadrature on S^3 (deterministic, so that it can be optimised), checks
the quadrature against the exact volume at the root system and at a
deletion, and then MINIMISES the truncated volume over 23-point
configurations with pairwise inner products at most 1/2 + delta, for a
decreasing schedule of delta, by the trust-region sequential linear
programme of inradius_search.py with the analytic gradient of the
quadrature.  Whether the truncated volume stays above 8 on the way down to
delta = 0 is the question; the deletion gives 8.1400 and the root system
itself, with twenty-four contacts, gives 7.968.

Usage: python3 truncated_volume.py [starts per level] [seed] [quadrature points] [truncation radius]
"""
import sys, time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import inradius_search as S

M = 23
IU = np.triu_indices(M, 1)
R3 = np.sqrt(8.0/5.0)

def quadrature(n, seed=12345):
    """A fixed set of n points on S^3: a scrambled Halton-type sequence pushed
    through the Gaussian map, so that the quadrature is deterministic."""
    from scipy.stats import qmc
    h = qmc.Halton(d=4, scramble=True, seed=seed).random(n)
    from scipy.special import erfinv
    Z = np.sqrt(2) * erfinv(2*h - 1)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)

class Truncated:
    def __init__(self, T, R=R3):
        self.T = T; self.R = R; self.w = 2*np.pi**2 / len(T)
    def value_grad(self, W):
        P = self.T @ W.T                       # (N, M) inner products
        idx = P.argmax(axis=1); m = P[np.arange(len(P)), idx]
        rad = np.minimum(1.0/m, self.R)
        val = self.w * np.sum(rad**4) / 4.0
        # gradient: d/dw_i (1/m)^4 = -4 m^-5 theta on the directions where i
        # is the nearest and the radius is not truncated
        active = (1.0/m) < self.R
        G = np.zeros_like(W)
        coef = np.where(active, -m**-5, 0.0) * self.w
        np.add.at(G, idx, coef[:, None] * self.T)
        return val, G
    def value(self, W):
        return self.value_grad(W)[0]

def exact_volume(W):
    hs = np.hstack([W, -np.ones((len(W), 1))]); hsi = HalfspaceIntersection(hs, np.zeros(4))
    return ConvexHull(hsi.intersections).volume

def minimise(W, Q, slack, iters=300, step0=0.03):
    """Trust-region SLP: minimise the truncated volume over tangent moves."""
    W = S.normalise(W); n = 4*M; step = step0
    val, Gd = Q.value_grad(W)
    for _ in range(iters):
        G = W @ W.T
        near = G[IU] > 0.5 + slack - 4*step
        P = (IU[0][near], IU[1][near])
        x = np.r_[W.ravel(), 0.0]
        Pc = S.pair_c(x, P, slack); Pj = S.pair_j(x, P)[:, :n]
        c = Gd.ravel()
        A_ub = -Pj; b_ub = Pc
        A_eq = np.zeros((M, n))
        for i in range(M): A_eq[i, 4*i:4*i+4] = W[i]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(M),
                      bounds=[(-step, step)]*n, method='highs')
        if res.status != 0:
            step *= 0.5
            if step < 1e-9: break
            continue
        d = res.x.reshape(M, 4)
        Wn = S.normalise(W + d)
        Gn = Wn @ Wn.T; np.fill_diagonal(Gn, -1)
        valn, Gdn = Q.value_grad(Wn)
        if Gn.max() <= 0.5 + slack + 1e-10 and valn < val - 1e-12:
            pred = float(c @ res.x)
            W, val, Gd = Wn, valn, Gdn
            if pred < 0 and (valn - val) < 0.5*pred: step = min(step*1.5, 0.1)
        else:
            step *= 0.5
            if step < 1e-9: break
    return W, val

def main():
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    nq = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
    Rtr = float(sys.argv[4]) if len(sys.argv) > 4 else R3       # truncation radius, default sqrt(8/5)
    rng = np.random.default_rng(seed)
    T = quadrature(nq); Q = Truncated(T, R=Rtr)
    Qfull = Truncated(T, R=1e9)
    t0 = time.time()
    Wdel = np.delete(S.ROOTS, 0, axis=0)
    print(f"quadrature with {nq} points: full volume at the root system {Qfull.value(S.ROOTS):.5f} (exact 8), "
          f"at a deletion {Qfull.value(Wdel):.5f} (exact 25/3 = 8.33333)")
    print(f"truncated at R = {Rtr:.6f} (angle {np.degrees(np.arccos(1/Rtr)):.4f} deg): root system {Q.value(S.ROOTS):.5f}, deletion {Q.value(Wdel):.5f}")
    SCHEDULE = [0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003, 0.001, 0.0]
    print(f"{'delta':>8} {'min angle':>10} {'min trunc. vol':>15} {'exact vol there':>16} {'endpoints':>9} {'deletions':>9}")
    pool = [Wdel]
    for slack in SCHEDULE:
        results = []
        for W in pool:
            W, w = S.restore(W, slack=slack)
            if w < -1e-10: continue
            Wm, v = minimise(W, Q, slack)
            results.append((v, Wm))
        for _ in range(starts):
            W0 = S.push_apart(rng.normal(size=(M, 4)), slack=max(slack, 0.02))
            W0, w = S.restore(W0, slack=slack)
            if w < -1e-10: continue
            Wm, v = minimise(W0, Q, slack)
            results.append((v, Wm))
        if not results:
            print(f"{slack:8.5f}  no feasible endpoint"); continue
        results.sort(key=lambda r: r[0])
        v, Wm = results[0]
        G = Wm @ Wm.T; np.fill_diagonal(G, -1)
        ndel = sum(1 for r in results if S.multiset((r[1] @ r[1].T)) == S.DELETION_MULTISET)
        print(f"{slack:8.5f} {np.degrees(np.arccos(G.max())):10.4f} {v:15.5f} {exact_volume(Wm):16.5f} "
              f"{len(results):9d} {ndel:9d}   [{time.time()-t0:.0f}s]", flush=True)
        np.save(f"continuation_out/trunc_min_{seed}_{Rtr:.4f}_{slack:g}.npy", Wm)
        pool = [r[1] for r in results[:8]]
    print(f"total time {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
