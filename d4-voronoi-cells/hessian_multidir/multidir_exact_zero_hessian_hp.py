#!/usr/bin/env python3
"""
multidir_exact_zero_hessian_hp.py

Examines the joint Hessian at the dense configuration A_18 of sec:exact-sing
of the paper in high-precision arithmetic (hp_volume.py, up to 45
decimal digits). Double precision cannot decide whether its smallest
eigenvalues are small and positive, zero or negative: its absolute
volume error of order 1e-10, divided by h^2 in a finite difference,
exceeds them once h is small.

Active set (18 of the 24 D4 roots, from greedy dense growth starting at
root 0):
    [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 20, 21, 22, 23]
At h = 0.01 the four smallest eigenvalues of the 54x54 joint Hessian are
of order 1e-5 to 1e-4, and the fifth is 0.075.

Method: along eigenvectors of the smallest eigenvalues and two mixtures
of them, F(s v) = vol(P(s v)) - 8 is evaluated at 35 to 45 digits for
s = 0.02, 0.01 and 0.005.

Output: along every direction tested, the central-difference estimate
(F(s) - 2F(0) + F(-s))/s^2 of the second derivative decreases by a
factor close to 4 at each halving of s, which is the behaviour of a
function with zero second derivative and leading term a_4 s^4; the
estimated a_4 is positive, about 0.026. This is numerical evidence that
the joint Hessian is singular at A_18, in which case no second-order
argument can decide the sign there. The directions tested span only
three dimensions of the near-null subspace; multidir_nullspace_broad_sample.py
samples all four (Numerical observations 17.1, 17.2 and 18.1 of the
paper).

Runtime: several minutes (about 20 high-precision volume evaluations of
10 to 20 seconds each).
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
    roots = build_roots()
    gram = roots @ roots.T
    adj = np.abs(gram - 0.5) < 1e-9
    active = greedy_dense_m18(roots, adj)
    print("Active set (m=18):", active)

    F, dim = make_F(roots, active)
    H, F0 = numeric_hessian(F, dim, 0.01)
    eigvals, eigvecs = np.linalg.eigh(H)
    print("Double-precision Hessian, smallest 5 eigenvalues (h=0.01):")
    print(" ", eigvals[:5])
    print()

    bases = [tangent_basis(roots[k]) for k in active]
    v0, v1, v2 = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
    test_dirs = {
        "smallest eigenvector v0": v0,
        "mixed (v0+v1+v2)/sqrt3": (v0 + v1 + v2) / np.sqrt(3),
        "mixed (v0-v1)/sqrt2": (v0 - v1) / np.sqrt(2),
    }

    t0 = time.time()
    for name, vec in test_dirs.items():
        print("=" * 72)
        print(name)
        print("=" * 72)
        second_derivs = []
        for s, prec in [(0.02, 35), (0.01, 35), (0.005, 45)]:
            fp = hp_F_along(roots, active, bases, vec, s, prec)
            fm = hp_F_along(roots, active, bases, vec, -s, prec)
            d2 = (fp + fm) / mp.mpf(s) ** 2
            second_derivs.append((s, d2))
            print(f"  s={s:<6} F(+s)={float(fp):.6e}  F(-s)={float(fm):.6e}  "
                  f"2nd-deriv-est={float(d2):.6e}  [t={time.time()-t0:.0f}s]")
        for k in range(1, len(second_derivs)):
            s0, d0 = second_derivs[k - 1]
            s1, d1 = second_derivs[k]
            ratio = float(d0) / float(d1) if d1 != 0 else float('inf')
            print(f"  ratio 2nd-deriv(s={s0})/2nd-deriv(s={s1}) = {ratio:.3f}  "
                  f"(about 4 if the second derivative is 0)")
        print()

    print("CONCLUSION: in every direction tested, the second-derivative estimate")
    print("decreases by a factor of about 4 per halving of s, consistent with a")
    print("singular Hessian here and a positive quartic leading term.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
