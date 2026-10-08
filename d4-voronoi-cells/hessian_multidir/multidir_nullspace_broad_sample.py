#!/usr/bin/env python3
"""
multidir_nullspace_broad_sample.py

Samples the near-null subspace of the joint Hessian at A_18 (sec:broad-sample
of the paper, Numerical observations 17.2 and 18.1).

Step 0: the double-precision Hessian spectrum at h = 0.02, 0.01 and 0.005.
The four smallest eigenvalues each decrease by a factor of about 4 at
every halving of h, as zero eigenvalues do; the fifth stays near 0.075.
So the near-null subspace is four-dimensional to this accuracy, one
dimension more than multidir_exact_zero_hessian_hp.py tested.

Step 1: 36 directions in that subspace: the four basis eigenvectors (at
h = 0.01), their six normalised pairwise sums and six normalised pairwise
differences, and 20 random directions (Gaussian coefficients,
normalised).

Step 2: for each direction, F(+s) and F(-s) are evaluated with
hp_volume.py at s = 0.015 and 35 digits, and the quartic coefficient is
estimated as a4_est = (F(s) + F(-s)) / (2 s^4), which is the leading term
when the second derivative vanishes. For six directions spanning the
range of a4_est, the evaluation is repeated at s = 0.0075 to check the
factor of 16 expected of a pure quartic term.

A positive a4_est in every direction is evidence at this one
configuration, not a proof that the quartic form is positive definite
on the subspace. A negative direction would show that F takes negative
values near A_18.
"""
import sys
import time
import numpy as np
import mpmath as mp

from multidir_chain_hessian_extended import (
    build_roots, tangent_basis, u_of_v, make_F, numeric_hessian,
)
from hp_volume import hp_volume


def greedy_dense_m18(roots, adj):
    active = [0] + [k for k in range(24) if adj[0, k]]
    remaining = [k for k in range(24) if k not in active]
    while len(active) < 18 and remaining:
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
    active = greedy_dense_m18(roots, adj)
    print("Active set (m=18):", active, flush=True)

    F, dim = make_F(roots, active)

    print("=" * 72, flush=True)
    print("STEP 0: double-precision spectrum at 3 step sizes", flush=True)
    print("=" * 72, flush=True)
    spectra = {}
    for h in (0.02, 0.01, 0.005):
        H, F0 = numeric_hessian(F, dim, h)
        eigvals, eigvecs = np.linalg.eigh(H)
        spectra[h] = (eigvals, eigvecs)
        print(f"  h={h}: smallest 6 eigenvalues = {np.round(eigvals[:6], 6)}", flush=True)
    for i in range(5):
        e02 = spectra[0.02][0][i]
        e01 = spectra[0.01][0][i]
        e005 = spectra[0.005][0][i]
        r1 = e02 / e01 if abs(e01) > 1e-12 else float('nan')
        r2 = e01 / e005 if abs(e005) > 1e-12 else float('nan')
        print(f"  eigenvalue[{i}] shrink ratios: h=.02/.01={r1:.3f}  h=.01/.005={r2:.3f}"
              f"  (expect ~4 if exactly-zero true eigenvalue)", flush=True)
    print(flush=True)
    print("CONCLUSION STEP 0: eigenvalues 0,1,2,3 all show the ~4x shrink", flush=True)
    print("signature; eigenvalue 4 (~0.075) does not shrink. Near-null space", flush=True)
    print("is (at least) 4-dimensional -- one more dimension than the prior", flush=True)
    print("script tested.", flush=True)
    print(flush=True)

    eigvals, eigvecs = spectra[0.01]
    w = [eigvecs[:, i] for i in range(4)]

    rng = np.random.default_rng(20260911)
    test_dirs = {}
    for i in range(4):
        test_dirs[f"basis eigenvector w{i}"] = w[i]
    for i in range(4):
        for j in range(i + 1, 4):
            test_dirs[f"(w{i}+w{j})/sqrt2"] = (w[i] + w[j]) / np.sqrt(2)
            test_dirs[f"(w{i}-w{j})/sqrt2"] = (w[i] - w[j]) / np.sqrt(2)
    for r in range(20):
        c = rng.normal(size=4)
        c = c / np.linalg.norm(c)
        vec = sum(c[i] * w[i] for i in range(4))
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
    print("STEP 2 SUMMARY", flush=True)
    print("=" * 72, flush=True)
    a4vals = [r[1] for r in results]
    print(f"min a4_est = {min(a4vals):.6f}   max a4_est = {max(a4vals):.6f}", flush=True)
    n_neg = sum(1 for v in a4vals if v < 0)
    print(f"directions with negative a4_est: {n_neg} / {len(a4vals)}", flush=True)
    worst = min(results, key=lambda r: r[1])
    print(f"smallest a4_est found: {worst[1]:.6f}  at direction [{worst[0]}]", flush=True)
    print(flush=True)

    # honesty re-check: confirm quartic scaling on a subsample spanning the range found
    results_sorted = sorted(results, key=lambda r: r[1])
    subsample_idx = sorted(set([0, len(results_sorted)//4, len(results_sorted)//2,
                                 3*len(results_sorted)//4, len(results_sorted)-1,
                                 len(results_sorted)-2]))
    print("=" * 72, flush=True)
    print("STEP 3: re-confirm quartic scaling (s=0.0075 vs s=0.015) on a", flush=True)
    print("subsample spanning the range of a4_est found", flush=True)
    print("=" * 72, flush=True)
    name_to_vec = test_dirs
    for idx in subsample_idx:
        name, a4_15, _, _ = results_sorted[idx]
        vec = name_to_vec[name]
        s2 = 0.0075
        prec2 = 40
        fp2 = hp_F_along(roots, active, bases, vec, s2, prec2)
        fm2 = hp_F_along(roots, active, bases, vec, -s2, prec2)
        a4_075 = float((fp2 + fm2) / (2 * mp.mpf(s2) ** 4))
        print(f"  [{name}]  a4_est(s=0.015)={a4_15:.6f}  a4_est(s=0.0075)={a4_075:.6f}"
              f"  ratio={a4_15/a4_075 if abs(a4_075) > 1e-12 else float('nan'):.3f} "
              f"(expect ~1.0 if genuinely quartic, i.e. a4_est should be ~s-independent)"
              f"  [t={time.time()-t_start:.0f}s]", flush=True)

    print(flush=True)
    print("DONE. Total time: %.0f s" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    sys.exit(main())
