"""
pair_budget.py

What the pairwise estimate would need at twenty-three contacts.

prop:second-order adds the exact pairwise cap overlaps to the covering
estimate. Evaluated at the deletion configuration, where the 88 close pairs
sit at exactly 60 degrees, it returns 7.997885, short of 8 by 0.002115.
That number is a property of the deletion configuration, and the deletion
configuration is extendable: its cell has circumradius exactly 2, so it is
not one of the configurations left open by Problem 7.33. This script
computes what the same estimate needs from a configuration that is
saturated, and what the contact graph can supply.

Checks:
  (a) the per-pair weight w(gamma) and its collapse as gamma leaves 60 deg;
  (b) the deletion value 7.997885 recovered from 88 pairs;
  (c) the number of pairs at 60 degrees that would carry the estimate to 8;
  (d) an elementary upper bound on the degree of the 60-degree graph, hence
      on the number of tight pairs a configuration of 23 directions can have;
  (e) that the target of (c) lies strictly between (b) and (d).
"""
import numpy as np
from itertools import combinations
from scipy.optimize import brentq
from scipy.integrate import quad

PI = np.pi
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def cap_area_S3(r):
    return PI * (2 * r - np.sin(2 * r))


def r_m(m):
    return brentq(lambda r: 2 * r - np.sin(2 * r) - 2 * PI / m, 1e-12, PI)


def covering_bound(m):
    return PI * m / 3.0 * np.tan(r_m(m)) ** 3


def lens(r, gamma):
    if r <= gamma / 2:
        return 0.0
    f = lambda t: np.sin(t) ** 2 * (1 - np.tan(gamma / 2) / np.tan(t))
    val, _ = quad(f, gamma / 2, r, limit=300)
    return 4 * PI * val


def pair_weight(gamma, m=23):
    """(1/4) int_0^{r_m} Lambda(r, gamma) d(sec^4 r): what one pair adds."""
    R = r_m(m)
    if gamma >= 2 * R:
        return 0.0
    f = lambda r: lens(r, gamma) * 4 * np.tan(r) / np.cos(r) ** 4
    val, _ = quad(f, gamma / 2, R, limit=400)
    return 0.25 * val


def d4_roots_int():
    V = []
    for i, j in combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4, dtype=int)
                v[i], v[j] = si, sj
                V.append(v)
    return np.array(V)


if __name__ == "__main__":
    m = 23
    R = r_m(m)
    base = covering_bound(m)
    w60 = pair_weight(PI / 3, m)
    short = 8.0 - base

    print("=" * 62)
    print("The pairwise estimate at twenty-three contacts")
    print("=" * 62)
    print(f"  r_23                              {np.degrees(R):.6f} deg")
    print(f"  covering estimate                 {base:.9f}")
    print(f"  shortfall to 8                    {short:.9f}")
    print(f"  weight of one pair at 60 degrees  {w60:.12f}")
    print(f"  a pair at gamma >= 2 r_23 = {np.degrees(2*R):.4f} deg contributes nothing")
    print()

    print("(a) the weight collapses as the pair opens up")
    tbl = [(d, pair_weight(np.radians(d), m)) for d in
           (60.0, 60.5, 61.0, 62.0, 64.0, 66.0, 68.0, 69.0)]
    for d, w in tbl:
        print(f"      gamma = {d:5.1f} deg    w = {w:.12f}    "
              f"w/w(60) = {w/w60:.5f}")
    check("the weight is strictly decreasing in the pair angle",
          all(tbl[i][1] > tbl[i + 1][1] for i in range(len(tbl) - 1)))
    check("half the weight is gone by 62 degrees",
          tbl[3][1] / w60 < 0.51,
          f"w(62)/w(60) = {tbl[3][1]/w60:.5f}")
    print()

    print("(b) the deletion configuration")
    A = d4_roots_int()
    G = A @ A.T
    tight = [(i, j) for i in range(23) for j in range(i + 1, 23) if G[i, j] == 1]
    val88 = base + len(tight) * w60
    print(f"      tight pairs among 23 of the roots   {len(tight)}")
    print(f"      second-order value                  {val88:.9f}")
    check("88 tight pairs, and the estimate returns the quoted 7.997885",
          len(tight) == 88 and abs(val88 - 7.997885) < 5e-7,
          f"computed {val88:.9f}")
    print()

    print("(c) what the same estimate would need")
    need = short / w60
    n_need = int(np.ceil(need))
    print(f"      pairs at 60 degrees required        {need:.4f}, so {n_need}")
    print(f"      value at {n_need} pairs                   "
          f"{base + n_need * w60:.9f}")
    print(f"      value at {n_need-1} pairs                   "
          f"{base + (n_need-1) * w60:.9f}")
    check("91 pairs at 60 degrees carry the estimate past 8",
          base + 91 * w60 > 8.0 > base + 90 * w60,
          f"90 pairs give {base + 90*w60:.9f}, 91 give {base + 91*w60:.9f}")
    print()

    print("(d) how many tight pairs the contact graph can carry")
    # directions at exactly 60 degrees from w, written in w-perp, are unit
    # vectors of R^3 with pairwise inner product at most 1/3
    alpha = np.arccos(1.0 / 3.0)
    cap_S2 = 2 * PI * (1 - np.cos(alpha / 2))
    dmax = int(np.floor(4 * PI / cap_S2))
    print(f"      link condition on S^2: pairwise angle at least "
          f"{np.degrees(alpha):.6f} deg")
    print(f"      disjoint caps of radius {np.degrees(alpha/2):.6f} deg, "
          f"area {cap_S2:.6f} of 4 pi = {4*PI:.6f}")
    print(f"      so the degree is at most             {dmax}")
    print(f"      and 23 directions carry at most      {(23 * dmax) // 2} tight pairs")
    check("the elementary degree bound is 10, so at most 115 tight pairs",
          dmax == 10 and (23 * dmax) // 2 == 115)
    print()

    print("(e) the window")
    print(f"      deletion configuration              {len(tight)} pairs")
    print(f"      needed by the pairwise estimate     {n_need} pairs")
    print(f"      elementary ceiling                  {(23 * dmax) // 2} pairs")
    check("the target lies strictly above the deletion and strictly below "
          "the ceiling", len(tight) < n_need < (23 * dmax) // 2)

    print()
    print("=" * 62)
    print(f"{len(PASS)} of {len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    print("=" * 62)
    print("The estimate of prop:second-order falls short at the deletion")
    print("configuration by three pairs. The deletion configuration is not")
    print("saturated, so this says nothing about the case left open.")
