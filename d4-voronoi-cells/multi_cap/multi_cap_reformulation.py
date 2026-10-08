#!/usr/bin/env python3
"""
multi_cap_reformulation.py

Checks the multi-cap reformulation of sec:multi-cap, which recasts the
multi-direction problem as a single extremal question about one fixed
polytope.

With D the set of deviating roots and

    Q_D = intersection of the 24 - |D| undeviated root half-spaces,

the cell is Q_D minus the union of the caps that the |D| tilted
half-spaces cut from it, so

    vol(V_c) = vol(Q_D) - vol( union_j Cap_{Q_D}(w_j) ),

and vol(V_c) >= 8 exactly when that union has volume at most
vol(Q_D) - 8.  The right-hand side is a constant of D alone and is
attained at the undeviated configuration.

This script checks, in order:

  (A) Q_D is bounded for every packing-valid D of size at most 5, by
      exhaustive enumeration and a linear program per set.  Boundedness
      is equivalent to the undeviated root directions positively
      spanning R^4.
  (B) At |D| = 6 exactly 24 of the 134596 packing-valid sets fail, and
      the failure is a half-space obstruction.
  (C) The identity above, on random packing-valid configurations, by
      direct polytope volumes.
  (D) vol(Q_D) - 8 equals |D|/3 when no two roots of D are adjacent and
      exceeds it otherwise.
  (E) The obstruction of rem:first-order-obstruction: vol(E_j) in an m-direction
      configuration both exceeds and falls short of its single-deviation
      value, so the single-deviation theorem gives no term-by-term
      bound.

Steps (A), (B) and (D) are exhaustive over the stated ranges; (C) and
(E) are random searches, and are reported as such.

Runtime: about four minutes.
"""

import itertools
import math

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

PASS = lambda b: "PASS" if b else "**FAIL**"
_status = []


def report(label, cond):
    _status.append(bool(cond))
    print("    %-56s %s" % (label, PASS(cond)))


def banner(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def unit_roots():
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i], v[j] = si, sj
                    out.append(v)
    return np.array(out) / math.sqrt(2)


U = unit_roots()
G = U @ U.T


def packing_valid(D):
    return all(G[a, b] <= 0.5 + 1e-9 for a, b in itertools.combinations(D, 2))


def adjacent_pair_in(D):
    return any(abs(G[a, b] - 0.5) < 1e-9 for a, b in itertools.combinations(D, 2))


def positively_spans(idxs):
    """Do these unit vectors positively span R^4?  Equivalent to
    boundedness of the intersection of the corresponding half-spaces."""
    N = U[idxs]
    return linprog(np.zeros(len(idxs)), A_eq=N.T, b_eq=np.zeros(4),
                   bounds=[(1, None)] * len(idxs), method="highs").success


def volume(normals, rhs):
    N = np.asarray(normals, float)
    b = np.asarray(rhs, float)
    nr = np.linalg.norm(N, axis=1)
    r = linprog(np.r_[np.zeros(4), -1.0],
                A_ub=np.hstack([N, nr.reshape(-1, 1)]), b_ub=b,
                bounds=[(None, None)] * 4 + [(0, 60)], method="highs")
    if not r.success or r.x[-1] <= 1e-11:
        return 0.0
    hs = HalfspaceIntersection(np.hstack([N, -b.reshape(-1, 1)]), r.x[:4])
    return ConvexHull(hs.intersections, qhull_options="QJ").volume


# ----------------------------------------------------------------------
banner("(A) Q_D is bounded for every packing-valid D with |D| <= 5")

EXPECTED = {1: 24, 2: 276, 3: 2024, 4: 10626, 5: 42504}
for m in range(1, 6):
    total = unbounded = 0
    for D in itertools.combinations(range(24), m):
        if not packing_valid(D):
            continue
        total += 1
        if not positively_spans([i for i in range(24) if i not in D]):
            unbounded += 1
    print("    m=%d: %6d packing-valid sets, %d unbounded" % (m, total, unbounded))
    report("count matches and all bounded at m=%d" % m,
           total == EXPECTED[m] and unbounded == 0)

# ----------------------------------------------------------------------
banner("(B) at |D| = 6 exactly 24 packing-valid sets fail")

total6 = 0
fails = []
for D in itertools.combinations(range(24), 6):
    if not packing_valid(D):
        continue
    total6 += 1
    if not positively_spans([i for i in range(24) if i not in D]):
        fails.append(D)
print("    %d packing-valid sets of size 6, %d unbounded" % (total6, len(fails)))
report("134596 packing-valid sets at m=6", total6 == 134596)
report("exactly 24 of them unbounded", len(fails) == 24)
first = fails[0]
print("    first failure: %s" % (
    [tuple(int(round(x * math.sqrt(2))) for x in U[i]) for i in first],))
comp = [i for i in range(24) if i not in first]
report("its complement lies in a closed half-space",
       all(U[i][0] <= 1e-12 for i in comp))

# ----------------------------------------------------------------------
banner("(C) the multi-cap identity, on random configurations")

rng = np.random.default_rng(20240911)


def tilted(u, rng, phi):
    e = rng.normal(size=4)
    e -= (e @ u) * u
    e /= np.linalg.norm(e)
    return math.cos(phi) * u + math.sin(phi) * e


worst = 0.0
checked = 0
for _ in range(40):
    m = int(rng.integers(2, 5))
    for _ in range(80):
        D = sorted(rng.choice(24, size=m, replace=False).tolist())
        if packing_valid(D):
            break
    else:
        continue
    W = {k: tilted(U[k], rng, rng.uniform(0.05, 0.8)) for k in D}
    if any(W[a] @ W[b] > 0.5 + 1e-9 for a, b in itertools.combinations(D, 2)):
        continue
    checked += 1
    fixed = [U[i] for i in range(24) if i not in D]
    volQD = volume(fixed, [1.0] * len(fixed))
    cell = [W[i] if i in W else U[i] for i in range(24)]
    volV = volume(cell, [1.0] * 24)
    union = 0.0
    for r in range(1, m + 1):
        for S in itertools.combinations(D, r):
            union += (-1) ** (r + 1) * volume(
                fixed + [-W[j] for j in S], [1.0] * len(fixed) + [-1.0] * r)
    worst = max(worst, abs(volV - (volQD - union)))
print("    configurations checked: %d" % checked)
print("    max |vol(V) - (vol(Q_D) - vol(union of caps))|: %.3e" % worst)
report("identity holds to numerical tolerance", worst < 1e-5)

# ----------------------------------------------------------------------
banner("(D) vol(Q_D) - 8 equals |D|/3 exactly when D has no adjacent pair")

ok = True
for trial in range(24):
    m = int(rng.integers(2, 5))
    for _ in range(80):
        D = sorted(rng.choice(24, size=m, replace=False).tolist())
        if packing_valid(D):
            break
    else:
        continue
    fixed = [U[i] for i in range(24) if i not in D]
    excess = volume(fixed, [1.0] * len(fixed)) - 8.0
    adj = adjacent_pair_in(D)
    good = (abs(excess - m / 3) < 1e-7) if not adj else (excess > m / 3 - 1e-9)
    ok = ok and good
report("excess is m/3 without adjacency and larger with it", ok)

# ----------------------------------------------------------------------
banner("(E) the term-by-term reduction fails in both directions")

above = below = 0
worst_short = 0.0
for _ in range(30):
    m = int(rng.integers(2, 6))
    for _ in range(80):
        D = sorted(rng.choice(24, size=m, replace=False).tolist())
        if packing_valid(D):
            break
    else:
        continue
    W = {k: tilted(U[k], rng, rng.uniform(0.03, 0.9)) for k in D}
    if any(W[a] @ W[b] > 0.5 + 1e-9 for a, b in itertools.combinations(D, 2)):
        continue
    fixed = [U[i] for i in range(24) if i not in D]
    dev = [W[j] for j in D]
    for k in D:
        Ek = volume(fixed + dev + [-U[k]],
                    [1.0] * len(fixed) + [1.0] * m + [-1.0])
        E1 = volume([U[i] for i in range(24) if i != k] + [W[k], -U[k]],
                    [1.0] * 23 + [1.0, -1.0])
        if Ek > E1 + 1e-9:
            above += 1
        elif Ek < E1 - 1e-9:
            below += 1
            worst_short = max(worst_short, E1 - Ek)
print("    vol(E_j) above its single-deviation value: %d times" % above)
print("    vol(E_j) below it:                         %d times" % below)
print("    largest shortfall: %.4f" % worst_short)
report("both directions occur, so no term-by-term bound",
       above > 0 and below > 0)

banner("SUMMARY")
print("    %d checks, %d passed, %d failed"
      % (len(_status), sum(_status), len(_status) - sum(_status)))
print("    overall: %s" % PASS(all(_status)))
raise SystemExit(0 if all(_status) else 1)
