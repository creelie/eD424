#!/usr/bin/env python3
"""
twentyfour_close_check.py -- the constants of the lemma "Twenty-four centres
within 2.444", in ball arithmetic at 200 bits (python-flint).

S(d) is the volume of the cap {x in B(sqrt(3/2)) : <x, u> > d/2} cut off by the
half-space of a centre at distance d, and T(Y) = 9 pi^2 / 8 - U(Y).  Let Y have
M centres, 25 <= M <= 33, within sqrt 6, and T(Y) <= 8.

  (a) At most 22 centres within 2.444 is impossible:
      U(Y) <= 22 S(2) + (M - 22) S(2.444) <= 22 S(2) + 11 S(2.444) < 9 pi^2/8 - 8.
  (b) Exactly 23 within 2.444, forming X, is impossible: the twenty-three-centre
      theorem gives T(X) >= A* + min(s(D), B) with A* = 9 pi^2/8 - 23 S(2),
      D = 2.1648, s(D) = S(2) - S(D) and B = 0.0929000002 (the value of the
      certificate), and T(Y) >= T(X) - (M - 23) S(2.444) > 8 for M <= 33.
  (c) With 34 centres the room of (b) is not enough, and (C_M) at M = 34 is the
      theorem on thirty-one or more centres.
  (d) At thirty centres the 24 closest have T <= 8 + 6 S(2.4).
"""
from flint import arb, ctx

ctx.prec = 200
R2 = arb(3) / 2
R = R2.sqrt()
PI = arb.pi()


def S(d):
    h = arb(d) / 2
    anti = h / 8 * (5 * R2 - 2 * h * h) * (R2 - h * h).sqrt() + 3 * R2 * R2 / 8 * (h / R).asin()
    return 4 * PI / 3 * (3 * R2 * R2 / 8 * PI / 2 - anti)


def check(name, ok):
    print('[%s] %s' % ('PASS' if ok else 'FAIL', name))
    if not ok:
        raise SystemExit(1)


tau = 9 * PI ** 2 / 8 - 8
A = 9 * PI ** 2 / 8 - 23 * S(2)
sD = S(2) - S(arb('2.1648'))
B = arb('0.0929000002')
s2444 = S(arb('2.444'))
room = A + sD - 8
print('S(2) =', S(2).str(10), ' S(2.444) =', s2444.str(10), ' 9pi^2/8 - 8 =', tau.str(10))
check('(a) 22 S(2) + 11 S(2.444) = %s < 9 pi^2/8 - 8' % (22 * S(2) + 11 * s2444).str(10),
      22 * S(2) + 11 * s2444 < tau)
check('s(D) = %s < B = 0.0929000002, so min(s(D), B) = s(D)' % sD.str(10), sD < B)
check('T(X) - 8 >= A* + s(D) - 8 = %s > 0' % room.str(10), room > 0)
for M in range(25, 34):
    check('(b) M = %d: (M - 23) S(2.444) = %s < %s' % (M, ((M - 23) * s2444).str(8), room.str(8)),
          (M - 23) * s2444 < room)
check('(b) at M = 33: T(Y) - 8 >= %s' % (room - 10 * s2444).str(8), room - 10 * s2444 > arb('1.1e-6'))
check('(c) M = 34: 11 S(2.444) = %s exceeds the room' % (11 * s2444).str(8), 11 * s2444 > room)
check('(d) 6 S(2.4) = %s < 0.00368' % (6 * S(arb('2.4'))).str(10), 6 * S(arb('2.4')) < arb('0.00368'))
