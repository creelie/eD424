#!/usr/bin/env python3
"""
quartic_sample_dense.py
========================
Genuine next step beyond multidir_nullspace_broad_sample.py: instead of
just sampling a4_est(v) at scattered directions v in the 4D near-null
subspace at A_18 and checking every value is positive, gather ENOUGH
samples to actually fit the full quartic form Q(x0,x1,x2,x3) (35
coefficients, the general degree-4 homogeneous polynomial in 4
variables) by least squares, so that a subsequent script can test
whether the FITTED quartic admits a sum-of-squares / PSD-Gram-matrix
certificate -- a check of positivity on the ENTIRE subspace, not just at
sampled rays.

RESUMABLE (added after this session's cloud sandbox container was
reclaimed mid-run twice, killing the background process each time
without warning): on start, this script reads any existing rows in
DATA_OUT (matched by exact direction name) and skips recomputing them,
appending only the missing ones. The active-set, eigenbasis-construction
method, RNG seed, and per-direction naming are all fixed and
deterministic, so a direction named "rand17" always means the exact same
unit vector across runs -- resuming never mixes data from a different
configuration.

HONESTY NOTE, up front: the fit is from noisy finite-difference/floating
point estimates of a4_est(v), not from an exact symbolic derivation of
the quartic tensor. Whatever comes out of this is numerical evidence
about the fitted approximant, not an exact certificate about the true
quartic form. This script exists to get real overdetermined data; the
follow-up script (quartic_gram_check.py) does the fit and SDP check and
states plainly what is and is not established by whatever it finds.

Design: same active set, same eigenbasis-construction method, same
estimator (a4_est = (F(s)+F(-s))/(2 s^4), s=0.015, 35 digits) as
multidir_nullspace_broad_sample.py, for direct comparability. N_RANDOM
new random unit directions are drawn (seed distinct from the earlier
36-direction run to avoid duplicating it), giving N_RANDOM equations in
35 unknowns once combined with the 4 basis vectors -- a real
overdetermined least-squares problem, not merely enough equations to
solve exactly (which would fit noise).
"""
import os
import sys
import time
import numpy as np
import mpmath as mp

from multidir_chain_hessian_extended import (
    build_roots, tangent_basis, u_of_v, make_F, numeric_hessian,
)
from hp_volume import hp_volume

N_RANDOM = 150
SEED = 987654321
WORKDIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORKDIR, "quartic_dense_A18_run.log")
DATA_OUT = os.path.join(WORKDIR, "quartic_dense_A18_data.tsv")


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


def load_done_names():
    """Names already computed with a real (non-PENDING) value, from any prior run."""
    done = set()
    if not os.path.exists(DATA_OUT):
        return done
    with open(DATA_OUT) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            name, _, _, _, _, val = parts
            if val != "PENDING":
                done.add(name)
    return done


def main():
    t_start = time.time()
    already_done = load_done_names()
    if already_done:
        print(f"RESUME: found {len(already_done)} already-computed directions in "
              f"{DATA_OUT}; skipping those, appending the rest.", flush=True)

    log = open(OUT, "a")
    data = open(DATA_OUT, "a")

    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    roots = build_roots()
    gram = roots @ roots.T
    adj = np.abs(gram - 0.5) < 1e-9
    active = greedy_dense_m18(roots, adj)
    p("Active set (m=18):", active)

    F, dim = make_F(roots, active)
    H, F0 = numeric_hessian(F, dim, 0.01)
    eigvals, eigvecs = np.linalg.eigh(H)
    p("smallest 6 eigenvalues at h=0.01:", np.round(eigvals[:6], 6).tolist())
    w = [eigvecs[:, i] for i in range(4)]

    rng = np.random.default_rng(SEED)
    test_dirs = {}
    coeff_record = {}
    for i in range(4):
        c = np.zeros(4); c[i] = 1.0
        test_dirs[f"basis w{i}"] = w[i]
        coeff_record[f"basis w{i}"] = c
    for i in range(4):
        for j in range(i + 1, 4):
            cs = np.zeros(4); cs[i] = 1/np.sqrt(2); cs[j] = 1/np.sqrt(2)
            cd = np.zeros(4); cd[i] = 1/np.sqrt(2); cd[j] = -1/np.sqrt(2)
            test_dirs[f"(w{i}+w{j})/sqrt2"] = cs[0]*w[0]+cs[1]*w[1]+cs[2]*w[2]+cs[3]*w[3]
            test_dirs[f"(w{i}-w{j})/sqrt2"] = cd[0]*w[0]+cd[1]*w[1]+cd[2]*w[2]+cd[3]*w[3]
            coeff_record[f"(w{i}+w{j})/sqrt2"] = cs.copy()
            coeff_record[f"(w{i}-w{j})/sqrt2"] = cd.copy()

    # IMPORTANT for resume-correctness: draw all N_RANDOM directions from the
    # RNG in the same fixed order every run, regardless of how many are
    # already done, so "rand17" is always the same unit vector.
    for r in range(N_RANDOM):
        c = rng.normal(size=4)
        c = c / np.linalg.norm(c)
        vec = sum(c[i] * w[i] for i in range(4))
        name = f"rand{r}"
        test_dirs[name] = vec
        coeff_record[name] = c

    todo = {name: vec for name, vec in test_dirs.items() if name not in already_done}
    p(f"Total directions: {len(test_dirs)} | already done: {len(already_done)} | "
      f"remaining this run: {len(todo)} "
      f"(target overdetermination vs 35 unknowns: {len(test_dirs)/35:.1f}x)")

    if not todo:
        p("Nothing left to do -- all directions already computed.")
        log.close(); data.close()
        return

    bases = [tangent_basis(roots[k]) for k in active]
    s_main = 0.015
    prec_main = 35

    for name, vec in todo.items():
        fp = hp_F_along(roots, active, bases, vec, s_main, prec_main)
        fm = hp_F_along(roots, active, bases, vec, -s_main, prec_main)
        a4_est = float((fp + fm) / (2 * mp.mpf(s_main) ** 4))
        c = coeff_record[name]
        p(f"  [{name}] coeffs={np.round(c,6).tolist()} a4_est={a4_est:.8f} "
          f"F+={float(fp):.4e} F-={float(fm):.4e} [t={time.time()-t_start:.0f}s]")
        data.write(f"{name}\t{c[0]:.8f}\t{c[1]:.8f}\t{c[2]:.8f}\t{c[3]:.8f}\t{a4_est:.10f}\n")
        data.flush()

    p("DONE. Total time this run: %.0f s" % (time.time() - t_start))
    log.close()
    data.close()


if __name__ == "__main__":
    sys.exit(main())
