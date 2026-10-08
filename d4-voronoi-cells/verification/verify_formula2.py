#!/usr/bin/env python3
"""
verify_formula2.py

The finer-grid companion to verify_formula.py: the same check of the first arc's breakpoint classification
(thm:arc1-breakpoints-new): that the four curves

    universal :  phi = pi/3
    W         :  phi = 2 arctan(sin s)
    A         :  phi = 2 arctan(cos s)
    Y         :  sin(phi) (sin s + cos s) = 1

account for every change in the combinatorial structure of the perturbed
cell along the arc, and that no further transition occurs.

The structure tracked is the incidence pattern of the moving facet: the
set of roots whose hyperplane shares a vertex with the tilted half-space.
That is what a breakpoint is.  Tracking instead whether a given facet
carries a vertex at all is not the same thing and does not change along
this arc, so it detects nothing.

The check is a dense scan in phi at each sampled s, with bisection
refinement at every detected change, and a comparison of the refined
values against the four closed forms.  Double precision throughout: this
is a cross-check of the exact symbolic derivation in
arc1_v1w1/breakpoint_curves.py, not a substitute for it.

run on a grid of 2000 points in phi and at 16 values of s rather than
900 and 8.
"""

import math

import numpy as np
from scipy.spatial import HalfspaceIntersection

GRID = 2000
S_VALUES = np.linspace(0.04, math.pi / 4 - 0.005, 16)
TOL = 2e-5


def build():
    vecs, labs = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i], v[j] = si, sj
                    vecs.append(v / math.sqrt(2))
                    labs.append("%se%d%se%d" % ("+" if si == 1 else "-", i + 1,
                                                "+" if sj == 1 else "-", j + 1))
    return np.array(vecs), labs


U, LABELS = build()
IDX0 = LABELS.index("+e1+e2")
U0 = U[IDX0]
V1 = np.array([1, -1, 0, 0]) / math.sqrt(2)
W1 = np.array([0, 0, 1, 1]) / math.sqrt(2)


def incidence(phi, s):
    """Roots whose hyperplane shares a vertex with the moving facet."""
    u1 = math.cos(phi) * U0 + math.sin(phi) * (math.cos(s) * V1 + math.sin(s) * W1)
    N = U.copy()
    N[IDX0] = u1
    hs = HalfspaceIntersection(np.hstack([N, -np.ones((24, 1))]), np.zeros(4))
    P = hs.intersections
    on_cap = P[np.abs(P @ u1 - 1) < 1e-7]
    if len(on_cap) == 0:
        return frozenset()
    out = set()
    for k in range(24):
        if k == IDX0:
            continue
        if np.any(np.abs(on_cap @ N[k] - 1) < 1e-7):
            out.add(LABELS[k])
    return frozenset(out)


def breakpoints(s, grid=GRID):
    xs = np.linspace(1e-4, math.pi / 2 - 1e-4, grid)
    prev = incidence(xs[0], s)
    step = xs[1] - xs[0]
    found = []
    for x in xs[1:]:
        cur = incidence(x, s)
        if cur != prev:
            lo, hi = x - step, x
            for _ in range(50):
                mid = (lo + hi) / 2
                if incidence(mid, s) == prev:
                    lo = mid
                else:
                    hi = mid
            found.append((lo + hi) / 2)
            prev = cur
    # the tilted facet degenerates as phi -> 0; drop anything in that
    # numerical boundary layer
    return [b for b in found if b > 1e-3]


def curves(s):
    return {
        "universal": math.pi / 3,
        "W": 2 * math.atan(math.sin(s)),
        "A": 2 * math.atan(math.cos(s)),
        "Y": math.asin(min(1.0, 1.0 / (math.sin(s) + math.cos(s)))),
    }


def main():
    print("Checking that every detected breakpoint matches one of the four curves.")
    print()
    print("%9s  %-46s  %s" % ("s", "breakpoints found", "matched"))
    unmatched = 0
    total = 0
    for s in S_VALUES:
        br = breakpoints(s)
        cs = curves(s)
        names = []
        for b in br:
            total += 1
            hit = [nm for nm, val in cs.items() if abs(val - b) < TOL]
            if hit:
                names.append(hit[0])
            else:
                names.append("UNMATCHED(%.6f)" % b)
                unmatched += 1
        print("%9.5f  %-46s  %s"
              % (s, ", ".join("%.6f" % b for b in br), ", ".join(names)))
    print()
    print("  breakpoints detected: %d" % total)
    print("  unmatched:            %d" % unmatched)
    print()
    print("  RESULT: %s" % ("PASS" if unmatched == 0 else "**FAIL**"))
    raise SystemExit(0 if unmatched == 0 else 1)


if __name__ == "__main__":
    main()
