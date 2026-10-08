#!/usr/bin/env python3
"""
pushout_check.py -- the arithmetic of thm:G-pushout: statement (G) for the push-outs
of the root system, 24 centres (2 + delta_i) u_i on the rays of the normalised roots.

The proof in the paper bounds, with eta_i = delta_i / 2 and eta_v the largest eta
over the six roots through a vertex v of the 24-cell Q,
    vol V(Y) >= 8 + (1/3) sum_i (1 - (1 - eta_i)^4)                 (lem:pure-push),
    vol (V(Y) \\ K(Y)) <= sum_v 2 h_v^4 <= 768 sum_i eta_i^4,
    h_v = sqrt2 ((1 + eta_v) - 1/(1 + eta_v)),
the second by 24 vertex pyramids, valid while (1 + eta_max)^2 - 1 < 1/3 (the
cross-sections stay inside the octahedral facets of the hull).  T only grows
when a centre moves out, so T(Y) <= 8 forces every delta_i below the crossing of
the one-centre ray, where
    T(delta) = 9 pi^2/8 - 23 S(2) - S(2 + delta) + 88 Pi(1, 1, 1/2) + 8 Pi(1, 1 + delta/2, 1/2).
This script checks, in Arb ball arithmetic and exact rationals:
  1. T(DELTA_B) > 8 at DELTA_B = 0.1971 (so T <= 8 gives delta_i < 0.1971);
  2. (1 + 0.09855)^2 - 1 < 1/3;
  3. g(eta)/eta = (1/3)(4 - 6 eta + 4 eta^2 - eta^3) - 768 eta^3 >= 2/5 on
     [0, 0.09855] (its derivative is at most -2 + (8/3) eta < 0, so its value at
     the right end is the least);
  4. the pyramid volume: int_0^h 8 s^3 ds = 2 h^4 and 2 h_v^4 <= 128 eta_v^4.
It then compares the bound 8 + sum g(eta_i) with the volume of V(Y) cap K(Y)
computed by qhull for random push-outs in the region (floating point, a check of
the bound, not part of the proof).
"""
import math
import sys
from fractions import Fraction as Fr

import numpy as np
from flint import arb, fmpq, ctx

ctx.prec = 128
DELTA_B = Fr(1971, 10000)
ETA_B = DELTA_B / 2


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)
    if not ok:
        sys.exit(1)


def A_(x):
    x = Fr(x)
    return arb(fmpq(x.numerator, x.denominator))


R2 = A_(Fr(3, 2))
R = R2.sqrt()
PI = arb.pi()


def S(d):
    h = A_(d) / 2
    anti = h / 8 * (5 * R2 - 2 * h * h) * (R2 - h * h).sqrt() + 3 * R2 ** 2 / 8 * (h / R).asin()
    return 4 * PI / 3 * (3 * R2 ** 2 / 8 * PI / 2 - anti)


def G(h, x):
    tx = x.tan()
    return R2 ** 2 * x - 2 * R2 * h * h * tx + h ** 4 * (tx + tx ** 3 / 3)


def Pi(h1, h2, u):
    """lem:pair-closed, as a ball; the branch decisions must be definite."""
    h1, h2, ua = A_(h1), A_(h2), A_(u)
    g = ua.acos()
    a1, a2 = (h1 / R).acos(), (h2 / R).acos()
    L = (-a1).max(g - a2)
    U = a1.min(g + a2)
    assert L < U
    pc = arb.atan2(h2 - h1 * g.cos(), h1 * g.sin())
    assert L < pc < U
    return PI / 4 * (G(h2, pc - g) - G(h2, L - g) + G(h1, U) - G(h1, pc))


def main():
    print('thm:G-pushout: (G) along the push-outs of the root system')
    VB = 9 * PI ** 2 / 8
    Tb = VB - 23 * S(2) - S(2 + DELTA_B) + 88 * Pi(1, 1, Fr(1, 2)) + 8 * Pi(1, 1 + DELTA_B / 2, Fr(1, 2))
    check('T on the one-centre ray at delta = %s exceeds 8' % float(DELTA_B), Tb > 8, 'T = %s' % Tb.str(10))
    T0 = VB - 24 * S(2) + 96 * Pi(1, 1, Fr(1, 2))
    check('T at the root system is 7.906940...', abs(T0 - arb('7.90694')) < arb('1e-5'), T0.str(10))
    lam1 = (1 + ETA_B) ** 2 - 1
    check('(1 + eta_B)^2 - 1 < 1/3 (the vertex pyramids stay inside the octahedral facets)', lam1 < Fr(1, 3), '%.5f' % float(lam1))

    def g_over(e):
        return Fr(1, 3) * (4 - 6 * e + 4 * e * e - e ** 3) - 768 * e ** 3
    # derivative of g/eta: (1/3)(-6 + 8 e - 3 e^2) - 2304 e^2 < 0 on [0, 1]
    # d/de [g(e)/e] = -2 + (8/3) e - 2305 e^2 <= -2 + (8/3) e, negative for e < 3/4
    check('g(eta)/eta decreases on [0, eta_B]: its derivative is at most -2 + (8/3) eta < 0', -2 + Fr(8, 3) * ETA_B < 0)
    check('g(eta)/eta >= 2/5 on [0, eta_B]', g_over(ETA_B) >= Fr(2, 5), '%.5f at eta_B = %s' % (float(g_over(ETA_B)), float(ETA_B)))
    # h = sqrt2 ((1 + e) - 1/(1 + e)) = sqrt2 e (2 + e)/(1 + e) <= 2 sqrt2 e for e >= 0, so 2 h^4 <= 128 e^4
    e = ETA_B
    check('pyramid: 2 h^4 <= 128 eta^4, since (2 + eta)/(1 + eta) <= 2 for eta >= 0', (2 + e) / (1 + e) <= 2)
    print('  so vol(V(Y) cap K(Y)) >= 8 + sum_i g(eta_i) >= 8 + (2/5) sum_i eta_i = 8 + sum_i delta_i / 5')

    # floating-point comparison with qhull
    from truncated_search import T, roots
    from cell_hull_search import cell_volume
    U = roots(); U = U / np.linalg.norm(U, axis=1)[:, None]
    rng = np.random.default_rng(0)
    worst, n = 1e9, 0
    for _ in range(400):
        k = rng.integers(1, 25); dl = np.zeros(24)
        dl[rng.choice(24, size=k, replace=False)] = rng.uniform(0, float(DELTA_B), size=k) * rng.random() ** 0.5
        Y = (2 + dl)[:, None] * U
        if T(Y) > 8:
            continue
        n += 1
        vol = cell_volume(Y)
        worst = min(worst, vol - 8 - sum(dl) / 5)
    check('qhull, %d random push-outs with T <= 8: vol(V cap K) - 8 - sum(delta)/5 >= 0' % n, worst >= -1e-9, 'least %.3e' % worst)
    print('PASS: every push-out of the root system with T <= 8 has vol(V(Y) cap K(Y)) >= 8 + sum(delta_i)/5')


if __name__ == '__main__':
    main()
