#!/usr/bin/env python3
"""
slack_continuation.py -- how the largest inradius of a 23-point code
behaves as its minimal angle is forced up to 60 degrees.

For delta >= 0 let K(delta) be the set of 23-point configurations on S^3
with all pairwise inner products at most 1/2 + delta, and let
    h(delta) = max { g(W) : W in K(delta) },
g(W) the inradius of conv(W), the cosine of the covering radius.  The sets
K(delta) are compact and decrease to K(0) as delta decreases to 0, and g is
continuous, so h(delta) decreases to h(0); the saturated 23-point case is
exactly the statement h(0) = 1/2, since g = 1/2 at every deletion of a root
and g > 1/2 is saturation.

This script estimates h on a decreasing schedule of delta by local
optimisation from many starts at each level: the best configurations of
the previous level, restored to the new constraint and re-optimised, and a
batch of fresh random starts.  The optimiser is the trust-region sequential
linear programme of inradius_search.py.  For each level it reports the
largest g found, the multiset of inner products of the configuration that
attains it, and how many of the level's endpoints are deletions of a root.

The best configuration of every level is saved as
continuation_out/best_<seed>_<delta>.npy.

Usage: python3 slack_continuation.py [fresh starts per level] [seed]
"""
import sys, time, os
import numpy as np
import inradius_search as S

SCHEDULE = [0.05, 0.03, 0.02, 0.015, 0.01, 0.009, 0.008, 0.007, 0.005, 0.003,
            0.002, 0.0015, 0.001, 0.0007, 0.0005, 0.0003, 0.0002, 0.0001, 0.00005, 0.0]

def optimise_at(W, slack):
    W, w = S.restore(W, slack=slack)
    if w < -1e-10:
        return None
    return S.solve_epigraph(W, slack=slack)

def main():
    fresh = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    keep = 20
    rng = np.random.default_rng(seed)
    t0 = time.time()
    os.makedirs("continuation_out", exist_ok=True)
    pool = []
    print(f"{'delta':>8} {'min angle':>10} {'h(delta)':>10} {'cov. radius':>12} "
          f"{'endpoints':>9} {'deletions':>9}  inner products of the best")
    for slack in SCHEDULE:
        results = []
        for W in pool:
            out = optimise_at(W, slack)
            if out is not None and out[2] <= 1e-9: results.append(out)
        for _ in range(fresh):
            W0 = S.push_apart(rng.normal(size=(S.M, 4)), slack=max(slack, 0.02))
            out = optimise_at(W0, slack)
            if out is not None and out[2] <= 1e-9: results.append(out)
        if not results:
            print(f"{slack:8.5f}  no feasible endpoint")
            pool = []
            continue
        results.sort(key=lambda r: -r[1])
        best = results[0]
        ndel = sum(1 for r in results if S.multiset(r[3]) == S.DELETION_MULTISET)
        ms = S.multiset(best[3])
        top = sorted(ms.items(), key=lambda kv: -kv[0])[:3]
        print(f"{slack:8.5f} {np.degrees(np.arccos(0.5+slack)):10.4f} {best[1]:10.6f} "
              f"{np.degrees(np.arccos(best[1])):12.4f} {len(results):9d} {ndel:9d}  "
              f"largest values {top}   [{time.time()-t0:.0f}s]", flush=True)
        np.save(f"continuation_out/best_{seed}_{slack:g}.npy", best[0])
        if slack == 0.0 and best[1] > 0.5 + 1e-7:
            np.save("saturated_candidate_continuation.npy", best[0])
            print("  !! g > 1/2 at slack 0: candidate saved, verify exactly")
        pool = [r[0] for r in results[:keep]]
    print(f"total time {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
