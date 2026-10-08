#!/usr/bin/env python3
"""
omega_closed_form.py -- the simplified closed forms of the pair weight omega
that lean/certificate/D4Omega.lean evaluates, checked against the closed form
of certificate_check.py.

With u = cos(gamma), tau = tan(gamma/2) = ((1-u)/(1+u))^(1/2) and t* = tan r_* = 1/sqrt2,

    omega(u)  = 4 pi [ (9/32) arctan((t* - tau)/(1 + t* tau)) - (4 tau^3 - 24 tau + 11 sqrt2)/96 ],
    omega'(u) = (pi/16) (3u - 1)^2 / ((1 + u)^2 sqrt(1 - u^2)),

for 1/3 <= u < 1.  The script checks (1) symbolically, that the derivative of
the first is the second; (2) that the first vanishes at u = 1/3; (3) that the
algebraic part of the closed form of certificate_check.py, rewritten in tau,
is -(4 tau^3 - 24 tau + 11 sqrt2)/96 and its gamma-terms reduce to
(9/64)(2 r_* - gamma); and (4) numerically, at 40 digits, that both agree with
certificate_check.py (omega and d omega/du) at twelve points of [1/3, 0.5733].
Usage: python3 omega_closed_form.py
"""
import sympy as sp
from mpmath import mp, mpf, iv, acos
import certificate_check as C

u, t = sp.symbols('u t', positive=True)
ts = 1 / sp.sqrt(2)
tau = sp.sqrt((1 - u) / (1 + u))
omega = 4 * sp.pi * (sp.Rational(9, 32) * sp.atan((ts - tau) / (1 + ts * tau))
                     - (4 * tau ** 3 - 24 * tau + 11 * sp.sqrt(2)) / 96)
omega1 = sp.pi / 16 * (3 * u - 1) ** 2 / ((1 + u) ** 2 * sp.sqrt(1 - u ** 2))
ok = True

d = sp.simplify(sp.diff(omega, u) - omega1)
print('(1) d/du omega - omega\' simplifies to', d); ok &= d == 0
w13 = sp.simplify(omega.subs(u, sp.Rational(1, 3)))
print('(2) omega(1/3) =', w13); ok &= w13 == 0

# (3) the closed form of certificate_check.py in terms of tau
sg = (1 + u) * tau
Kp = sg / 4 + tau * (1 - u) / 4                                # K = -gamma/4 + Kp
pr_star = -(ts + ts ** 3 / 3) / 8 + sp.Rational(9, 16) * Kp - (2 * ts ** 3 / 3) / 4 - tau / 2 * sp.Rational(1, 16)
pr_half = -(tau + tau ** 3 / 3) / 8 + Kp / (1 + u) ** 2 - (2 * tau ** 3 / 3) / 4 - tau / 2 * tau ** 4 / 4
alg = sp.simplify((pr_star - pr_half).subs(u, (1 - t ** 2) / (1 + t ** 2)))
target = -(4 * t ** 3 - 24 * t + 11 * sp.sqrt(2)) / 96
print('(3) algebraic part in tau:', sp.factor(alg), '; equal to the stated form:', sp.simplify(alg - target) == 0)
ok &= sp.simplify(alg - target) == 0
print('    gamma-terms: (9/32) r_* from prim(r_*) and -(9/64) gamma from K there; those of prim(gamma/2) cancel')

mp.dps = 40
om0, om1, om2, A_star, r_star = C.closed_forms()
f0 = sp.lambdify(u, omega, 'mpmath'); f1 = sp.lambdify(u, omega1, 'mpmath')
worst0 = worst1 = mpf(0)
for k in range(12):
    uu = mpf(1) / 3 + (mpf('0.5733') - mpf(1) / 3) * (k + 1) / 12
    g = iv.mpf(acos(uu))
    a0, a1 = om0(g), om1(g)
    worst0 = max(worst0, abs(f0(uu) - mpf(a0.mid)))
    worst1 = max(worst1, abs(f1(uu) - mpf(a1.mid)))
print('(4) largest differences from certificate_check.py: omega %.2e, omega\' %.2e' % (float(worst0), float(worst1)))
ok &= worst0 < 1e-13 and worst1 < 1e-13
print('PASS' if ok else 'FAIL')
