#!/usr/bin/env python3
"""
third_edge_symmetry.py
=======================
Resolves the open question left at the end of sec:why-boundary-arcs ("Where the
transition window's minimiser actually sits"): is the fundamental
triangle's third edge (v1-v2, the one edge that is neither of this
paper's two named boundary arcs) the exact image of an already-certified
arc under some isometry?

ANSWER (proved here, both symbolically-exact and as a numerical sanity
check): yes. S = diag(1,1,1,-1) -- the SAME symmetry already used
elsewhere in this paper (arc2_w1v2/w1v2_explore.py) to swap w1<->v2 and
reduce the second arc's own range to [0,pi/4] -- also satisfies
S(v1)=v1 and S(v2)=w1 exactly. Since S is a genuine automorphism of the
D4 root system fixing u0 (it permutes the 24 roots bijectively: a root
+-e_i+-e_j maps to another root of the same index pair, with the sign
at index 4 flipped when 4 in {i,j} and left alone otherwise), it carries
the entire packing configuration at any deviation eta to a CONGRUENT
configuration at deviation S(eta) -- hence identical Voronoi volume,
hence identical defect -- for every angle theta and every eta, not just
the third edge.

Applying this with eta = dev_{v1v2}(s) = cos(s) v1 + sin(s) v2 and using
S v1 = v1, S v2 = w1 gives S(dev_{v1v2}(s)) = dev_{v1w1}(s) EXACTLY, for
every s (not merely sampled values) -- so defect on the third edge
equals defect on the FIRST arc, point for point. Since the first arc
already carries a complete positivity certificate across its entire
domain (Theorems arc1-breakpoints-new, arc1-certs-new: all 8 regions,
7 exact zero-margin + 1 with an explicit thin margin), the third edge
is certified for free: no new certificate computation is needed or
performed here.

WHAT THIS DOES NOT ESTABLISH, stated plainly: it does not complete the
second arc (w1-v2), which still carries only 3 of its own ~8-9
region-and-sub-part units (see hessian_multidir/ and the paper's
sec:arc2-cert for the count there). It says nothing about the
two-dimensional INTERIOR of the fundamental triangle, which remains
completely untouched by any certificate or symmetry argument anywhere
in this paper. Conjecture (Direction-of-Deviation Positivity) remains
open, in both its original and restated forms.
"""
import sympy as sp
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

sqrt2 = sp.sqrt(2)


def exact_check():
    u0 = sp.Matrix([1, 1, 0, 0]) / sqrt2
    v1 = sp.Matrix([1, -1, 0, 0]) / sqrt2
    w1 = sp.Matrix([0, 0, 1, 1]) / sqrt2
    v2 = sp.Matrix([0, 0, 1, -1]) / sqrt2
    S = sp.diag(1, 1, 1, -1)

    checks = {
        "S*u0 == u0": sp.simplify(S * u0 - u0) == sp.zeros(4, 1),
        "S*v1 == v1": sp.simplify(S * v1 - v1) == sp.zeros(4, 1),
        "S*v2 == w1": sp.simplify(S * v2 - w1) == sp.zeros(4, 1),
        "S*S == Id": sp.simplify(S * S - sp.eye(4)) == sp.zeros(4, 4),
    }

    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i] = si
                    v[j] = sj
                    roots.append(sp.Matrix(v))
    roots_set = set(tuple(r) for r in roots)
    image_set = set(tuple(S * r) for r in roots)
    checks["S permutes the 24 D4 roots bijectively (exact)"] = (roots_set == image_set)

    s = sp.symbols('s', real=True)
    dev_v1v2 = sp.cos(s) * v1 + sp.sin(s) * v2
    dev_v1w1 = sp.cos(s) * v1 + sp.sin(s) * w1
    Sdev = sp.simplify(S * dev_v1v2)
    checks["S(dev_v1v2(s)) == dev_v1w1(s), symbolic s"] = (sp.simplify(Sdev - dev_v1w1) == sp.zeros(4, 1))

    print("=== Exact symbolic verification (sympy, Z[sqrt2] arithmetic) ===")
    all_ok = True
    for name, ok in checks.items():
        print(f"  {name}: {ok}")
        all_ok = all_ok and bool(ok)
    print(f"ALL EXACT CHECKS PASS: {all_ok}")
    return all_ok


def numeric_sanity_check():
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i] = si
                    v[j] = sj
                    roots.append(v / np.sqrt(2))
    roots = np.array(roots)

    def find(vec):
        v = np.array(vec, dtype=float)
        v /= np.linalg.norm(v)
        d = roots @ v
        idx = np.argmax(d)
        assert d[idx] > 1 - 1e-9
        return idx

    idx0 = find([1, 1, 0, 0])
    u0 = roots[idx0]
    v1 = roots[find([1, -1, 0, 0])]
    w1 = roots[find([0, 0, 1, 1])]
    v2 = roots[find([0, 0, 1, -1])]

    def F_direct(theta, e_perp):
        e_perp = e_perp / np.linalg.norm(e_perp)
        u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
        dirs = roots.copy()
        dirs[idx0] = u1
        Am = dirs
        bm = -np.ones(len(dirs))
        hs = np.hstack([Am, bm.reshape(-1, 1)])
        hi = HalfspaceIntersection(hs, np.zeros(4))
        hull = ConvexHull(hi.intersections, qhull_options='QJ')
        return hull.volume - 8.0

    def dev_v1v2(s):
        return np.cos(s) * v1 + np.sin(s) * v2

    def dev_v1w1(s):
        return np.cos(s) * v1 + np.sin(s) * w1

    print()
    print("=== Numerical sanity check: defect(theta, third edge(s)) vs defect(theta, first arc(s)) ===")
    maxdiff = 0.0
    for s_val in [0.05, 0.15, 0.30, 0.45, 0.60, 0.75]:
        for theta in [0.2, 0.5, 0.9, 1.2, 1.5]:
            F1 = F_direct(theta, dev_v1v2(s_val))
            F2 = F_direct(theta, dev_v1w1(s_val))
            d = abs(F1 - F2)
            maxdiff = max(maxdiff, d)
    print(f"  max |defect(third edge) - defect(first arc)| over 30 sampled (theta,s): {maxdiff:.2e}")
    print("  (nonzero only at the level of convex-hull-volume floating point noise;")
    print("   the exact symbolic check above is what actually proves the identity.)")


if __name__ == "__main__":
    ok = exact_check()
    numeric_sanity_check()
    print()
    if ok:
        print("CONCLUSION: the third edge (v1-v2) is exactly certified by Arc 1's own")
        print("certificate, via the symmetry S. No new certificate was computed or is")
        print("needed for this edge. This does NOT complete the second arc (w1-v2) and")
        print("does NOT address the fundamental triangle's interior -- both remain open.")
