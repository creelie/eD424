#!/usr/bin/env python3
"""
multidir_nullspace_broad_sample_m20.py
=====================================
Extends multidir_nullspace_broad_sample.py's broad directional sampling
(done at one specific dense m=18 configuration) to a SECOND, genuinely
different dense configuration: m=20, obtained by the same greedy-density
growth as before (starting from root 0, always adding the remaining root
with the most already-active neighbours), continued two steps further
than the m=18 set already studied:
    active (m=20) = [0,2,4,5,6,7,8,9,10,11,12,13,15,16,17,19,20,21,22,23]
(This is a strict superset of the m=18 set: roots 15 and 19 added.)

Two further independently-grown m=18 configurations (from starting roots
5 and 12 instead of 0) were also tried and found to reproduce the
SAME double-precision eigenvalue spectrum as the original m=18 set to 4
decimal places (e.g. the 5th eigenvalue ~0.0753 in all three) -- almost
certainly the same geometric configuration up to the D4 Weyl group's
symmetry (order 1152) rather than new information, so they are not
pursued further here; m=20 is a genuinely different, larger and denser
configuration and is the one studied in this script.

STEP 0: double-precision Hessian spectrum at h=0.02,0.01,0.005 finds the
near-null space is (at least) 5-DIMENSIONAL here (one dimension larger
than at m=18): eigenvalues 0-4 all shrink by the same ~3.1-3.8x-per-
halving signature used throughout this line of work to flag an exactly
(or near-exactly) singular direction; eigenvalue 5 (~0.035, notably
smaller than m=18's own non-shrinking floor of ~0.075) does not shrink.

STEP 1-2: build the 5D near-null basis (5 eigenvectors at h=0.01) and
sample it broadly: 5 basis vectors, all 10 pairwise normalized sums, and
15 random directions spanning the full 5D span = 30 directions total,
each evaluated at s=+-0.015 (prec=35) via hp_volume.py to estimate the
quartic coefficient a4_est=(F(s)+F(-s))/(2s^4). A subsample is
re-checked at a second step size for the same quartic-scaling honesty
check as before.

Reported exactly as found, whatever the outcome.
"""
import sys
import time
import numpy as np
import mpmath as mp

from multidir_chain_hessian_extended import (
    build_roots, tangent_basis, u_of_v, make_F, numeric_hessian,
)
from hp_volume import hp_volume


def greedy_dense_growth(roots, adj, start, target):
    active = [start] + [k for k in range(24) if adj[start, k]]
    remaining = [k for k in range(24) if k not in active]
    while len(active) < target and remaining:
        scores = [(sum(adj[k, a] for a in active), k) for k in remaining]
        scores.sort(reverse=True)
        best_k = scores[0][1]
        active.append(best_k)
        remaining.remove(best_k)
    return sorted(active)


def hp_F_along(roots, active, bases, vec, s, prec):
    mp.mp.dps = prec
    dirs = roots.copy()
    for i, k in enumerate(active):
        v = s * vec[3 * i:3 * i + 3] @ bases[i]
        dirs[k] = u_of_v(roots[k], v)
    dirs_mp = [[mp.mpf(str(x)) for x in dirs[j]] for j in range(24)]
    return hp_volume(dirs_mp, prec=prec) - 8


def main():
    t_start = time.time()
    roots = build_roots()
    gram = roots @ roots.T
    adj = np.abs(gram - 0.5) < 1e-9
    active = greedy_dense_growth(roots, adj, 0, 20)
    print("Active set (m=20):", active, flush=True)

    F, dim = make_F(roots, active)
    print(f"dim = {dim}", flush=True)

    print("=" * 72, flush=True)
    print("STEP 0: double-precision spectrum at 3 step sizes", flush=True)
    print("=" * 72, flush=True)
    spectra = {}
    for h in (0.02, 0.01, 0.005):
        H, F0 = numeric_hessian(F, dim, h)
        eigvals, eigvecs = np.linalg.eigh(H)
        spectra[h] = (eigvals, eigvecs)
        print(f"  h={h}: smallest 7 eigenvalues = {np.round(eigvals[:7], 6)}", flush=True)
    for i in range(6):
        e02 = spectra[0.02][0][i]
        e01 = spectra[0.01][0][i]
        e005 = spectra[0.005][0][i]
        r1 = e02 / e01 if abs(e01) > 1e-12 else float('nan')
        r2 = e01 / e005 if abs(e005) > 1e-12 else float('nan')
        print(f"  eigenvalue[{i}] shrink ratios: h=.02/.01={r1:.3f}  h=.01/.005={r2:.3f}",
              flush=True)
    print(flush=True)

    eigvals, eigvecs = spectra[0.01]
    K = 5
    w = [eigvecs[:, i] for i in range(K)]

    rng = np.random.default_rng(20260911)
    test_dirs = {}
    for i in range(K):
        test_dirs[f"basis eigenvector w{i}"] = w[i]
    for i in range(K):
        for j in range(i + 1, K):
            test_dirs[f"(w{i}+w{j})/sqrt2"] = (w[i] + w[j]) / np.sqrt(2)
    for r in range(15):
        c = rng.normal(size=K)
        c = c / np.linalg.norm(c)
        vec = sum(c[i] * w[i] for i in range(K))
        test_dirs[f"random#{r} coeffs={np.round(c,3).tolist()}"] = vec

    print(f"Total test directions: {len(test_dirs)}", flush=True)
    print(flush=True)

    bases = [tangent_basis(roots[k]) for k in active]
    s_main = 0.015
    prec_main = 35

    results = []
    for name, vec in test_dirs.items():
        fp = hp_F_along(roots, active, bases, vec, s_main, prec_main)
        fm = hp_F_along(roots, active, bases, vec, -s_main, prec_main)
        a4_est = (fp + fm) / (2 * mp.mpf(s_main) ** 4)
        results.append((name, float(a4_est), float(fp), float(fm)))
        print(f"  [{name}]  F(+s)={float(fp):.4e}  F(-s)={float(fm):.4e}  "
              f"a4_est={float(a4_est):.6f}  [t={time.time()-t_start:.0f}s]", flush=True)

    print(flush=True)
    print("=" * 72, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 72, flush=True)
    a4vals = [r[1] for r in results]
    print(f"min a4_est = {min(a4vals):.6f}   max a4_est = {max(a4vals):.6f}", flush=True)
    n_neg = sum(1 for v in a4vals if v < 0)
    print(f"directions with negative a4_est: {n_neg} / {len(a4vals)}", flush=True)
    worst = min(results, key=lambda r: r[1])
    print(f"smallest a4_est found: {worst[1]:.6f}  at direction [{worst[0]}]", flush=True)
    print(flush=True)

    # re-check quartic scaling on 4 directions spanning the range
    results_sorted = sorted(results, key=lambda r: r[1])
    idxs = sorted(set([0, len(results_sorted)//2, len(results_sorted)-2, len(results_sorted)-1]))
    print("=" * 72, flush=True)
    print("Re-confirm quartic scaling (s=0.0075 vs s=0.015) on a subsample", flush=True)
    print("=" * 72, flush=True)
    for idx in idxs:
        name, a4_15, _, _ = results_sorted[idx]
        vec = test_dirs[name]
        s2 = 0.0075
        prec2 = 40
        fp2 = hp_F_along(roots, active, bases, vec, s2, prec2)
        fm2 = hp_F_along(roots, active, bases, vec, -s2, prec2)
        a4_075 = float((fp2 + fm2) / (2 * mp.mpf(s2) ** 4))
        ratio = a4_15 / a4_075 if abs(a4_075) > 1e-12 else float('nan')
        print(f"  [{name}]  a4_est(s=0.015)={a4_15:.6f}  a4_est(s=0.0075)={a4_075:.6f}"
              f"  ratio={ratio:.3f} (expect ~1.0)  [t={time.time()-t_start:.0f}s]", flush=True)

    print(flush=True)
    print("DONE. Total time: %.0f s" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    sys.exit(main())
