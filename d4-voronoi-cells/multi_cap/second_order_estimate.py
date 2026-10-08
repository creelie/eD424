#!/usr/bin/env python3
"""
Putting the pairwise overlaps back into the covering estimate,
lem:no-triples, prop:second-order and equation eq:second-order.

Below r_m the caps of radius r about a contact configuration meet only
in pairs: three contact directions have circumradius at least
arccos(sqrt(2/3)) = 35.2644 deg, above r_23 = 34.6106 and
r_24 = 34.0987.  So on [0, r_m] inclusion-exclusion is exact,

    sigma({delta <= r}) = m C(r) - sum_{i<j} Lambda(r, gamma_ij),

with the lens measure

    Lambda(r, gamma) = 4 pi int_{gamma/2}^{r} sin^2 t (1 - tan(gamma/2) cot t) dt,

and the layer-cake step of thm:covering-bound gives the strongest bound any
argument can extract from cap measures and pairwise intersections.

Checks, in order:

  1  the circumradius bound of lem:no-triples, by the Rayleigh-quotient
     argument and against direct computation on random admissible
     triples;
  2  r_23 and r_24 both lie below arccos(sqrt(2/3)), so the hypothesis
     of prop:second-order holds at both counts;
  3  the closed form for Lambda against direct integration on S^3;
  4  the value of the second-order bound at the root system and at the
     root system minus one root, in high-precision quadrature.

Run:  python3 second_order_estimate.py
Exits nonzero if any check fails.
"""

import itertools
import sys

import numpy as np
from mpmath import mp, mpf, quad, sin, tan, sec, pi, findroot, sqrt, acos, degrees

mp.dps = 30
RESULTS = []
RNG = np.random.default_rng(20260912)


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("[%s] %s" % ("PASS" if ok else "FAIL", name))
    if detail:
        for line in detail.splitlines():
            print("       " + line)


def cap_area(r):
    return pi * (2 * r - sin(2 * r))


def r_of(m):
    return findroot(lambda r: 2 * r - sin(2 * r) - 2 * pi / m, mpf(0.6))


def lens(r, gamma):
    if r <= gamma / 2:
        return mpf(0)
    return 4 * pi * quad(lambda t: sin(t) ** 2
                         * (1 - tan(gamma / 2) / tan(t)), [gamma / 2, r])


def dsec4(r):
    return 4 * sec(r) ** 4 * tan(r)


def nz(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def d4():
    o = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0.0] * 4
                    v[i], v[j] = si, sj
                    o.append(v)
    return nz(np.array(o))


def main():
    R0 = acos(sqrt(mpf(2) / 3))

    # ---- 1. the circumradius of a triple ----------------------------
    worst = 90.0
    bad = 0
    for _ in range(40000):
        g = RNG.uniform(-1.0, 0.5, size=3)
        G = np.array([[1.0, g[0], g[1]],
                      [g[0], 1.0, g[2]],
                      [g[1], g[2], 1.0]])
        if np.linalg.eigvalsh(G).min() < 1e-9:
            continue
        q = float(np.ones(3) @ np.linalg.solve(G, np.ones(3)))
        if q <= 0:
            continue
        Rdeg = np.degrees(np.arccos(min(1.0, 1.0 / np.sqrt(q))))
        worst = min(worst, Rdeg)
        if Rdeg < float(degrees(R0)) - 1e-9:
            bad += 1
    record("three contact directions have circumradius at least "
           "arccos sqrt(2/3)",
           bad == 0,
           "arccos sqrt(2/3) = %.6f deg\nsmallest circumradius over 40000 "
           "random admissible triples: %.6f deg\nviolations: %d"
           % (float(degrees(R0)), worst, bad))

    # ---- 2. r_m sits below it at m = 23 and m = 24 ------------------
    r23, r24 = r_of(23), r_of(24)
    record("r_23 and r_24 lie below arccos sqrt(2/3), so the caps meet "
           "only in pairs",
           r23 < R0 and r24 < R0,
           "r_23 = %.6f deg,  r_24 = %.6f deg,  threshold %.6f deg"
           % (float(degrees(r23)), float(degrees(r24)), float(degrees(R0))))

    # ---- 3. the lens formula against direct sampling ----------------
    ok3 = True
    lines = []
    S = nz(RNG.normal(size=(4000000, 4)))
    for gdeg, rdeg in ((60.0, 34.0), (60.0, 40.0), (90.0, 50.0)):
        g, r = mpf(gdeg) * pi / 180, mpf(rdeg) * pi / 180
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([float(np.cos(np.radians(gdeg))),
                      float(np.sin(np.radians(gdeg))), 0.0, 0.0])
        c = float(np.cos(np.radians(rdeg)))
        hit = ((S @ a >= c) & (S @ b >= c)).mean() * 2 * np.pi ** 2
        cf = float(lens(r, g))
        lines.append("gamma=%.0f deg, r=%.0f deg: closed form %.6f, "
                     "sampled %.6f" % (gdeg, rdeg, cf, hit))
        if abs(cf - hit) > 4e-3 + 0.02 * cf:
            ok3 = False
    record("the lens formula agrees with direct sampling on S^3", ok3,
           "\n".join(lines))

    # ---- 4. the second-order bound at both counts -------------------
    U = d4()
    lines = []
    vals = {}
    for W, label in ((U, "root system"),
                     (np.delete(U, 0, axis=0), "root system minus one root")):
        m = len(W)
        rm = r_of(m)
        G = W @ W.T
        np.fill_diagonal(G, -2.0)
        angs = [np.degrees(np.arccos(np.clip(G[i, j], -1, 1)))
                for i, j in itertools.combinations(range(m), 2)]
        close = [a for a in angs if a < 2 * float(degrees(rm)) - 1e-9]
        n60 = sum(1 for a in close if abs(a - 60.0) < 1e-9)
        base = 2 * pi ** 2 + quad(lambda r: (2 * pi ** 2 - m * cap_area(r))
                                  * dsec4(r), [0, rm])
        extra = n60 * quad(lambda r: lens(r, pi / 3) * dsec4(r),
                           [pi / 6, rm])
        vals[m] = float((base + extra) / 4)
        lines.append("%-26s m=%2d  r_m=%.6f deg  close pairs %d (all at 60 "
                     "deg: %s)" % (label, m, float(degrees(rm)),
                                   len(close), n60 == len(close)))
        lines.append("    covering estimate      %.6f" % float(base / 4))
        lines.append("    with overlaps put back %.6f" % vals[m])
    record("the second-order bound is 7.997885 at m = 23 and 7.858738 at "
           "m = 24, both below 8",
           abs(vals[23] - 7.997885) < 5e-6 and abs(vals[24] - 7.858738) < 5e-6
           and vals[23] < 8 and vals[24] < 8,
           "\n".join(lines)
           + "\n    at m = 23 the shortfall left is %.6f" % (8 - vals[23]))

    print("=" * 62)
    n = sum(1 for _, o in RESULTS if o)
    print("%d of %d checks passed" % (n, len(RESULTS)))
    if n == len(RESULTS):
        print("Counting every pairwise overlap exactly still returns less")
        print("than 8 at twenty-three contacts.")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
