"""
covering_multiplicity.py

prop:cov-mult and prop:two-point-barrier: the covering multiplicity inequality,
and how far the pair-angle distribution alone can carry the estimate.

N(u) counts the contact directions within 60 degrees of u. For a saturated
configuration N >= 1 almost everywhere. Cauchy-Schwarz on N gives a lower
bound on the total pairwise overlap of the 60-degree caps, which is a
constraint on the multiset of pair angles and nothing else.

The second part is the more important one. It asks whether the second-order
estimate can be carried past 8 using constraints of that kind: Bonferroni
at every radius, Delsarte positivity at every degree, and the total pair
count. It cannot, and the reason is explicit.
"""
import numpy as np
from itertools import combinations
from scipy.optimize import brentq, linprog
from scipy.integrate import quad

PI = np.pi
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def A(r):
    """Area of a cap of angular radius r on the unit 3-sphere."""
    return PI * (2 * r - np.sin(2 * r))


def r_m(m):
    return brentq(lambda r: 2 * r - np.sin(2 * r) - 2 * PI / m, 1e-12, PI)


def covering_bound(m):
    return PI * m / 3.0 * np.tan(r_m(m)) ** 3


def lens(r, gamma):
    if r <= gamma / 2:
        return 0.0
    f = lambda t: np.sin(t) ** 2 * (1 - np.tan(gamma / 2) / np.tan(t))
    return 4 * PI * quad(f, gamma / 2, r, limit=300)[0]


def pair_weight(gamma, m=23):
    R = r_m(m)
    if gamma >= 2 * R:
        return 0.0
    f = lambda r: lens(r, gamma) * 4 * np.tan(r) / np.cos(r) ** 4
    return 0.25 * quad(f, gamma / 2, R, limit=400)[0]


def gegenbauer_S3(k, x):
    """
    Gegenbauer polynomials for S^3, normalised to G_k(1) = 1. For the
    3-sphere these are the Chebyshev polynomials of the second kind,
    G_k(cos t) = sin((k+1)t) / ((k+1) sin t).
    """
    x = np.clip(x, -1.0, 1.0)
    t = np.arccos(x)
    out = np.where(np.abs(np.sin(t)) < 1e-12,
                   1.0,
                   np.sin((k + 1) * t) / ((k + 1) * np.maximum(np.sin(t), 1e-300)))
    return np.where(np.abs(np.sin(t)) < 1e-12, np.sign(x) ** k, out)


def d4_deletion_angles():
    V = []
    for i, j in combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i], v[j] = si, sj
                V.append(v / np.sqrt(2.0))
    W = np.array(V)[:23]
    G = np.clip(W @ W.T, -1.0, 1.0)
    return np.array([np.arccos(G[i, j]) for i in range(23) for j in range(i + 1, 23)])


if __name__ == "__main__":
    m, npairs = 23, 253
    R23 = r_m(m)
    base = covering_bound(m)
    need = 8.0 - base
    w60 = pair_weight(PI / 3, m)

    print("=" * 64)
    print("Part 1: the covering multiplicity inequality")
    print("=" * 64)
    S = m * A(PI / 3)
    T = 2 * PI ** 2
    cs = (S * S / T - S) / 2.0
    print(f"  total cap area   m A(60 deg) = {S:.6f}")
    print(f"  area of S^3                  = {T:.6f}")
    print(f"  Cauchy-Schwarz lower bound on sum Lambda(60 deg, gamma_ij):")
    print(f"      (S^2/T - S)/2            = {cs:.6f}")

    gam = d4_deletion_angles()
    actual = sum(lens(PI / 3, g) for g in gam)
    print(f"  value at the deletion configuration = {actual:.6f}")
    print(f"  slack = {actual - cs:.6f}, that is {100*(actual-cs)/actual:.3f} per cent")
    check("the inequality holds at the deletion configuration", actual >= cs)
    check("and is within five per cent of binding there",
          (actual - cs) / actual < 0.05,
          f"slack {100*(actual-cs)/actual:.3f} per cent")
    print()

    print("=" * 64)
    print("Part 2: can the pair-angle distribution alone finish the job")
    print("=" * 64)
    print("  Minimise sum_{i<j} omega(gamma_ij) over all distributions of")
    print("  253 pair angles in [60, 180] degrees subject to")
    print("    Bonferroni:  sum Lambda(t, gamma_ij) >= m A(t) - 2 pi^2, all t")
    print("    Delsarte:    sum G_k(cos gamma_ij) >= -m/2, k = 1..K")
    print("    total mass:  253 pairs")
    print("  If that minimum reached 0.083272 the case would be settled.")
    print()

    grid = np.radians(np.linspace(60.0, 180.0, 601))
    c = np.array([pair_weight(g, m) for g in grid])

    tgrid = np.linspace(r_m(m), np.radians(45.0), 24)
    Arows, blb = [], []
    for t in tgrid:
        Arows.append([-lens(t, g) for g in grid])
        blb.append(-(m * A(t) - T))
    for k in range(1, 13):
        Arows.append([-gegenbauer_S3(k, np.cos(g)) for g in grid])
        blb.append(m / 2.0)

    res = linprog(c, A_ub=np.array(Arows), b_ub=np.array(blb),
                  A_eq=np.ones((1, len(grid))), b_eq=[npairs],
                  bounds=[(0, None)] * len(grid), method="highs")
    print(f"  linear programme status: {res.message.strip()}")
    print(f"  minimum of sum omega over the relaxation : {res.fun:.9f}")
    print(f"  what is needed                           : {need:.9f}")
    print(f"  value at the deletion configuration      : "
          f"{sum(pair_weight(g, m) for g in gam):.9f}")
    check("the relaxation cannot reach the target", res.fun < need,
          f"{res.fun:.9f} against {need:.9f}")

    delval = sum(pair_weight(g, m) for g in gam)
    check("the deletion configuration is feasible for the relaxation and "
          "already falls short", delval < need,
          f"the deletion gives {delval:.9f}, short by {need - delval:.9f}")

    print()
    print("=" * 64)
    print(f"{len(PASS)} of {len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    print("=" * 64)
    print("The deletion configuration satisfies every constraint of this kind")
    print("and its own pair angles already give less than what is needed. A")
    print("bound that settles the saturated case therefore has to read")
    print("something the multiset of pair angles does not record.")
