#!/usr/bin/env python3
"""
code24_exact.py -- an exact 24-point code in S^3 far from the root system
(exact rational arithmetic).

Takes the code saved by code24_second.py, moves each point to a nearby point
of S^3 with rational coordinates (inverse stereographic projection of a
rational point, so that |w|^2 = 1 exactly), and checks in exact arithmetic:
  * every point has norm 1;
  * the largest inner product is at most 1/2 + s for the rational s printed;
  * the code is far from the root system: the sorted inner products of some
    point with the others differ from those of a root, {1, 1/2 x 8, 0 x 6,
    -1/2 x 8, -1}, by more than a printed rational amount; and some inner
    product lies at a printed distance from -1, -1/2, 0, 1/2, 1, which gives
    d(W) >= half that distance (if |w_pi(i) - R r_i| <= e_i then
    |<w_pi(i), w_pi(j)> - <r_i, r_j>| <= e_i + e_j <= 2 d(W));
and writes the points to code24_exact.txt for lean/D4SecondCode.lean.
So statement (i) of "What is left" (24 unit vectors with inner products at
most 1/2 + s lie near the root system) fails at that s.
Usage: python3 code24_exact.py [code24_second.npy]
"""
import sys
from fractions import Fraction as F
import numpy as np


def rational_point(x, den=10**6):
    """A point of S^3 with rational coordinates near the unit vector x."""
    # stereographic projection from the pole -e_k with the largest |x_k| away
    k = int(np.argmax(np.abs(x)))
    sgn = 1 if x[k] > 0 else -1
    idx = [i for i in range(4) if i != k]
    y = [F(round(den * x[i] / (1 + sgn * x[k])), den) for i in idx]
    n2 = sum(v * v for v in y)
    out = [F(0)] * 4
    for j, i in enumerate(idx):
        out[i] = 2 * y[j] / (1 + n2)
    out[k] = sgn * (1 - n2) / (1 + n2)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'code24_second.npy'
    W = np.load(path)
    P = [rational_point(w) for w in W]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert all(dot(p, p) == 1 for p in P)
    G = [[dot(P[i], P[j]) for j in range(24)] for i in range(24)]
    top = max(G[i][j] for i in range(24) for j in range(24) if i != j)
    s = top - F(1, 2)
    root_row = sorted([F(1)] + [F(1, 2)] * 8 + [F(0)] * 6 + [F(-1, 2)] * 8 + [F(-1)])
    dev = max(max(abs(a - b) for a, b in zip(sorted(G[i]), root_row)) for i in range(24))
    print('all 24 points on S^3 exactly; largest inner product %.8f, slack s = %.8f' % (top, s))
    print('largest deviation of a sorted Gram row from the root system row: %.6f' % dev)
    import math
    s_up = F(math.ceil(s * 10**6), 10**6); dev_lo = F(math.floor(dev * 10**4), 10**4)
    assert s <= s_up and dev >= dev_lo
    print('exact: slack <= %s = %.6f and deviation >= %s = %.4f' % (s_up, float(s_up), dev_lo, float(dev_lo)))
    # an inner product far from {-1, -1/2, 0, 1/2, 1}: within 2 d(W) of one of them otherwise
    vals = [F(-1), F(-1, 2), F(0), F(1, 2), F(1)]
    far, pair = max((min(abs(G[i][j] - v) for v in vals), (i, j)) for i in range(24) for j in range(i + 1, 24))
    far_lo = F(math.floor(far * 1000), 1000)
    assert far >= far_lo
    print('the inner product of points %d and %d, %.6f, is at least %s from -1, -1/2, 0, 1/2, 1;'
          % (pair[0], pair[1], G[pair[0]][pair[1]], far_lo))
    print('so d(W) >= %s / 2 = %.4f > 1/48' % (far_lo, float(far_lo) / 2))
    with open('code24_exact.txt', 'w') as f:
        f.write('# 24 points of S^3 with rational coordinates (numerator/denominator), one per line\n')
        for q in P:
            f.write(' '.join('%d/%d' % (x.numerator, x.denominator) for x in q) + '\n')
        f.write('# slack bound %s, far pair %d %d, distance bound %s\n' % (s_up, pair[0], pair[1], far_lo))


if __name__ == '__main__':
    main()
