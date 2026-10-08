#!/usr/bin/env python3
"""
robust_ceiling.py -- the ceiling of thm:m24-robust of the paper
(the robust reading of the kernel of de Laat, Leijenhorst and de Muinck Keizer).

Write p_2 for the two-point polynomial of llm24_out/llm24_p2.txt (sigma_2 in the
paper; nonnegative on [-1, 1/2]).  The proof pins every inner product u <= 1/2 of a
24-point code with slack kappa by
     p_2(u) <= E(24, kappa),     E(24, kappa) >= C(24,2) * L2 * kappa   (pairs above 1/2 alone)
and
     p_2(u) >= qmin * f(u),      f(u) = (u+1)(u+1/2)^2 u^2 (1/2-u),
so that |u - z| <= delta for z in {-1, -1/2, 0, 1/2} needs qmin * fmin(delta) > E.
Even with the triple and quadruple sums B_3, B_4 set to zero this gives

     kappa  <  qmin * fmin(delta) / (276 * L2).

Since fmin(delta) ~ delta^2 / 8 near the double zeros, kappa scales as delta^2,
and delta can never reach 1/4 (the four target values are 1/2 apart).  Floating
point at 60 digits; the certified values of the paper are those of explicit_eps0.py.
"""
import os
import sys
from fractions import Fraction as Fr
from mpmath import mp, mpf, polyval, diff, findroot
mp.dps = 60
sys.set_int_max_str_digits(0)
HERE = os.path.dirname(os.path.abspath(__file__))

coef = {}
for line in open(os.path.join(HERE, 'llm24_out', 'llm24_p2.txt')):
    if line.startswith('#') or not line.strip():
        continue
    i, c = line.split()
    coef[int(i)] = Fr(c)
deg = max(coef)
cs = [mpf(coef.get(i, 0).numerator) / mpf(coef.get(i, 0).denominator) for i in range(deg + 1)]
p2 = lambda u: polyval(cs[::-1], u)
dp2 = lambda u: polyval([c * i for i, c in enumerate(cs)][1:][::-1], u)

# L2: |p_2'| on [1/2, 1/2 + 1e-3]
L2 = max(abs(dp2(mpf(1) / 2 + mpf(k) / 1000 / 200)) for k in range(201))
# qmin: min of p_2 / f on [-1, 1/2] (fine grid; the paper's certified value is 2.62e-4)
f = lambda u: (u + 1) * (u + mpf(1) / 2) ** 2 * u ** 2 * (mpf(1) / 2 - u)
grid = [mpf(-1) + mpf(3) / 2 * k / 20000 for k in range(1, 20000)]
qmin = min(p2(u) / f(u) for u in grid if abs(f(u)) > mpf('1e-40'))
print(f"degree of p_2: {deg}")
print(f"L2 = max|p_2'| on [1/2, 1/2+1e-3]  = {float(L2):.6f}   (paper 0.226)")
print(f"qmin = min p_2/f on [-1,1/2]        = {float(qmin):.4e}  (paper 2.62e-4)")
print()
print("ceiling on kappa with B_3 = B_4 = 0 (pair terms alone), as a function of the")
print("rounding tolerance delta that the root-lattice step could conceivably allow:")
print(f"{'delta':>8} {'fmin(delta)':>14} {'kappa ceiling':>16}")
for d in ('3e-4', '1e-3', '1e-2', '0.1', '0.25'):
    d = mpf(d)
    fmin = min(f(-1 + d), f(-mpf(1)/2 - d), f(-mpf(1)/2 + d), f(-d), f(d), f(mpf(1)/2 - d))
    kap = qmin * fmin / (276 * L2)
    print(f"{float(d):8.1e} {float(fmin):14.4e} {float(kap):16.3e}")
print()
print("the paper's actual value, with B_3, B_4 included:  kappa = 2e-26")
print("the target:                                        kappa = 1e-2")
