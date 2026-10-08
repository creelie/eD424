#!/usr/bin/env python3
"""
margin_floor_check.py -- the proposition "The root system spends the margin".

The margin programme (las2_margin.jl) adds mu * w(u) to the two-point
constraint of the second level, with

    w(u) = (u + 1) (u + 1/2)^2 u^2 (1/2 + s - u),   nonnegative on [-1, 1/2 + s].

For any 24-point code C with inner products in that interval the chain of the
certificate gives mu * sum_{pairs of C} w <= N - 24.  The normalised roots of
D4 are such a code.  This script checks, exactly, that their 276 pairs have
inner products -1, -1/2, 0, 1/2 only, that 96 pairs sit at 1/2, and that the
sum of w over the pairs is 36 s; and, in ball arithmetic (python-flint), that
max w < 0.019779 on [-1, 0.508] at s = 0.008, so that the bound
w(u) <= (N - 24)/mu, whose right side is at least 36 s = 0.288, excludes no
inner product.  It also locates the slack below which the margin could pin
anything, where max w = 36 s (floating point, about 5.2e-4).
"""
from fractions import Fraction as Fr
from itertools import combinations, product

from flint import arb, ctx

ctx.prec = 128


def w(u, s):
    return (u + 1) * (u + Fr(1, 2)) ** 2 * u ** 2 * (Fr(1, 2) + s - u)


# the 24 roots +-e_i +- e_j of D4, of norm sqrt 2; inner products of the
# normalised roots are half those of the roots
roots = []
for i, j in combinations(range(4), 2):
    for a, b in product((1, -1), repeat=2):
        v = [0] * 4
        v[i], v[j] = a, b
        roots.append(v)
assert len(roots) == 24
ips = [Fr(sum(x * y for x, y in zip(p, q)), 2) for p, q in combinations(roots, 2)]
assert len(ips) == 276
values = sorted(set(ips))
assert values == [Fr(-1), Fr(-1, 2), Fr(0), Fr(1, 2)], values
n_half = sum(1 for t in ips if t == Fr(1, 2))
assert n_half == 96
s = Fr(1, 125)                                     # 0.008
total = sum(w(t, s) for t in ips)
assert total == 36 * s
print('[PASS] 276 pairs of normalised roots, inner products', [str(v) for v in values])
print('[PASS] 96 pairs at 1/2, each with w = 3s/8; sum of w over the pairs = 36 s = %s' % (36 * s))

# ball-arithmetic bound on max w over [-1, 1/2 + s] at s = 0.008
sa = arb(1) / 125
lo, hi = arb(-1), arb(1) / 2 + sa
n = 200000
best = arb(0)
bound = 0.0
for k in range(n):
    a = lo + (hi - lo) * k / n
    b = lo + (hi - lo) * (k + 1) / n
    u = arb.union(a, b)                            # a ball containing [a, b]
    val = (u + 1) * (u + arb(1) / 2) ** 2 * u ** 2 * (arb(1) / 2 + sa - u)
    ub = float(val.upper())
    if ub > bound:
        bound = ub
assert bound < 0.019779, bound
assert 0.019779 < float(36 * s)
print('[PASS] max w on [-1, 0.508] at s = 0.008 is below %.6f (ball arithmetic, %d subintervals)' % (bound, n))
print('[PASS] 36 s = 0.288 exceeds max w: the margin excludes no inner product at s = 0.008')


# floating point: where max w meets 36 s
def maxw(sf, m=20001):
    top = 0.0
    for k in range(m):
        u = -1 + (1.5 + sf) * k / (m - 1)
        top = max(top, (u + 1) * (u + 0.5) ** 2 * u ** 2 * (0.5 + sf - u))
    return top


a, b = 1e-4, 1e-3
for _ in range(40):
    mid = (a * b) ** 0.5
    if maxw(mid) > 36 * mid:
        a = mid
    else:
        b = mid
print('max w = 36 s near s = %.3e (floating point)' % a)
