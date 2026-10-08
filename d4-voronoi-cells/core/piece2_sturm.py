import sympy as sp
from fractions import Fraction
from math import comb

def poly_mul(p, q):
    r = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i+j] += a * b
    return r
def poly_add(p, q):
    n = max(len(p), len(q)); r = [Fraction(0)] * n
    for i, a in enumerate(p): r[i] += a
    for i, b in enumerate(q): r[i] += b
    return r
def poly_sub(p, q):
    return poly_add(p, [Fraction(-1) * x for x in q])
def poly_scale(c, p):
    return [Fraction(c) * x for x in p]
def pow_poly(p, n):
    r = [Fraction(1)]
    for _ in range(n): r = poly_mul(r, p)
    return r
def poly_eval(p, x):
    x = Fraction(x)
    return sum(a * x**i for i, a in enumerate(p))

p_   = [Fraction(1), Fraction(0), Fraction(1)]
cn_  = [Fraction(1), Fraction(0), Fraction(-1)]
sn_  = [Fraction(0), Fraction(2)]
p4_  = pow_poly(p_, 4)
cn2_ = pow_poly(cn_, 2)
cn3_ = pow_poly(cn_, 3)
sn4_ = pow_poly(sn_, 4)

N_p4 = [Fraction(0)]
for coeff, factors in [
    (Fraction(184),  [sn4_]),
    (Fraction(-200), [cn3_, sn_]),
    (Fraction(64),   [cn_, sn_]),
    (Fraction(-32),  [cn2_, sn_]),
    (Fraction(-32),  [sn_, pow_poly(p_, 3)]),
    (Fraction(384),  [cn2_, pow_poly(p_, 2)]),
    (Fraction(-64),  [cn_, pow_poly(p_, 3)]),
    (Fraction(32),   [cn3_, p_]),
    (Fraction(-152), [p4_]),
]:
    term = [coeff]
    for f in factors: term = poly_mul(term, f)
    N_p4 = poly_add(N_p4, term)

c_minus_s_n_ = [Fraction(1), Fraction(-2), Fraction(-1)]
D_p4 = poly_mul([Fraction(24)], poly_mul(cn3_, poly_mul(c_minus_s_n_, p_)))
Q_p4 = poly_sub(poly_scale(8, D_p4), N_p4)
while Q_p4 and Q_p4[-1] == 0: Q_p4.pop()
print("degree of Q_II*(1+t^2)^4:", len(Q_p4)-1)

# sanity: matches d4_poly_certificate.py values
print("Q(3/5) =", poly_eval(Q_p4, Fraction(3,5)), float(poly_eval(Q_p4, Fraction(3,5))))
print("Q(2/3) =", poly_eval(Q_p4, Fraction(2,3)))
print("Q(1)   =", poly_eval(Q_p4, Fraction(1)), float(poly_eval(Q_p4, Fraction(1))))
print("Q(1/2) [Piece I side, sanity] =", poly_eval(Q_p4, Fraction(1,2)))

t = sp.symbols('t')
Qsym = sum(sp.Rational(a.numerator, a.denominator) * t**i for i, a in enumerate(Q_p4))
Qsym = sp.expand(Qsym)
Qpoly = sp.Poly(Qsym, t)
print("\nsympy poly degree:", Qpoly.degree())

# 1/sqrt(3) ~ 0.5773502691896258
# choose rational t_a < 1/sqrt(3) close to it, and check roots in [t_a, 1]
t_a = sp.Rational(5773, 10000)   # 0.5773 < 1/sqrt(3)
t_b = sp.Rational(1)
print(f"\nt_a = {t_a} = {float(t_a)}, 1/sqrt(3) = {float(1/sp.sqrt(3))}")
assert float(t_a) < 1/3**0.5

nroots = Qpoly.count_roots(t_a, t_b)
print(f"Number of real roots of Q_II*(1+t^2)^4 in [{t_a}, {t_b}]: {nroots}")

# also check just above to bracket true interval (1/sqrt3, 1) tightly with two straddling checks
t_a2 = sp.Rational(57735, 100000)   # 0.57735, compare to 0.57735026...
print(f"t_a2 = {t_a2} = {float(t_a2)} vs 1/sqrt3={float(1/sp.sqrt(3))}  (t_a2 < 1/sqrt3: {float(t_a2) < 1/3**0.5})")
nroots2 = Qpoly.count_roots(t_a2, t_b)
print(f"roots in [{t_a2}, {t_b}]: {nroots2}")

# real roots of Q at all (to see the true picture / confirm none near this range)
allroots = sp.real_roots(Qpoly)
print("\nAll real roots (exact, as roots or intervals):")
for r in allroots:
    print("  ", r, "~", float(r))

print("\n--- refined check: exclude the known boundary root at t=1 ---")
t_a = sp.Rational(5773, 10000)
t_mid = sp.Rational(999, 1000)
nroots_main = Qpoly.count_roots(t_a, t_mid)
print(f"roots in [{t_a}, {t_mid}] (main range, excludes t=1): {nroots_main}")

nroots_upper = Qpoly.count_roots(t_mid, sp.Rational(1))
print(f"roots in [{t_mid}, 1] (upper sliver incl. boundary root at t=1): {nroots_upper}")

# check sign just left of 1
for tv in [sp.Rational(9995,10000), sp.Rational(99995,100000), sp.Rational(999995,1000000)]:
    val = Qpoly.eval(tv)
    print(f"Q({tv}) = {val} = {float(val):.6e}  sign={'+' if val>0 else ('0' if val==0 else '-')}")

# lower sliver: 1/sqrt(3) to t_a, using a rational just BELOW 1/sqrt3 to bracket properly
t_low = sp.Rational(577, 1000)  # 0.577 < 1/sqrt(3)
print(f"\nt_low={t_low}={float(t_low)}  (< 1/sqrt3={float(1/sp.sqrt(3))})")
nroots_lower = Qpoly.count_roots(t_low, t_a)
print(f"roots in [{t_low}, {t_a}] (brackets 1/sqrt(3)): {nroots_lower}")
for tv in [sp.Rational(5774,10000), sp.Rational(57736,100000), t_a]:
    val = Qpoly.eval(tv)
    print(f"Q({tv}) = {float(val):.6e}")

print("\n--- cleanest single-bracket version ---")
t_clean_lo = sp.Rational(1, 5)   # 0.2 -- below 1/sqrt(3)=0.57735 and safely above the root at ~0.178
t_clean_hi = sp.Rational(999, 1000)
print(f"1/sqrt(3) = {float(1/sp.sqrt(3))}  (must lie in ({float(t_clean_lo)}, {float(t_clean_hi)}))")
n_clean = Qpoly.count_roots(t_clean_lo, t_clean_hi)
print(f"roots of Q_II*(1+t^2)^4 in [{t_clean_lo}, {t_clean_hi}]: {n_clean}")
print(f"Q({t_clean_lo}) = {float(Qpoly.eval(t_clean_lo)):.6f}")
print(f"Q(3/5) = {float(Qpoly.eval(sp.Rational(3,5))):.6f}  (reference interior point, known positive)")

n_upper = Qpoly.count_roots(t_clean_hi, sp.Rational(1))
print(f"roots in [{t_clean_hi}, 1]: {n_upper}  (should be exactly the known root AT t=1)")
print(f"Q(1) exact = {Qpoly.eval(sp.Rational(1))}")
