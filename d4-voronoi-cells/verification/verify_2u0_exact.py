import numpy as np
from fractions import Fraction as Fr

# EXACT verification (rational arithmetic where possible) that Z*=2*u0
# lies on the moving-cap hyperplane <z,u1(theta,t)>=1 iff cos(theta)=1/2,
# independent of t, and lies exactly on 8 fixed root hyperplanes.

# u0=(1,1,0,0)/sqrt2. Work with Z*=2u0=(sqrt2,sqrt2,0,0) symbolically
# via sympy to keep sqrt2 exact.
import sympy as sp

sqrt2 = sp.sqrt(2)
u0 = sp.Matrix([1,1,0,0])/sqrt2
v1 = sp.Matrix([1,-1,0,0])/sqrt2
w1 = sp.Matrix([0,0,1,1])/sqrt2
Zstar = 2*u0
print(f"Z* = 2*u0 = {list(Zstar)}")

theta, t = sp.symbols('theta t', real=True)
e_perp = sp.cos(t)*v1 + sp.sin(t)*w1
u1 = sp.cos(theta)*u0 + sp.sin(theta)*e_perp

expr = sp.simplify((Zstar.T*u1)[0,0])
print(f"<Z*, u1(theta,t)> = {expr}")
print(f"Independent of t: {t not in expr.free_symbols}")
print(f"Solve <Z*,u1>=1 for theta: {sp.solve(sp.Eq(expr,1), theta)}")

print()
print("Exact check: Z* lies on <z,r>=1 for all 8 roots +-e1+-e3, +-e1+-e4,")
print("+-e2+-e3, +-e2+-e4:")
roots8 = []
for i in (0,1):  # e1 or e2
    for j in (2,3):  # e3 or e4
        for si in (1,-1):
            for sj in (1,-1):
                r = sp.zeros(4,1)
                r[i]=si; r[j]=sj
                r = r/sqrt2
                roots8.append((i,j,si,sj,r))
all_eq_1 = True
for i,j,si,sj,r in roots8:
    val = sp.simplify((Zstar.T*r)[0,0])
    ok = (val == 1)
    all_eq_1 = all_eq_1 and ok
print(f"  All 8 give <Z*,r>=1 exactly: {all_eq_1}")

print()
print("Debugging individual values:")
for i,j,si,sj,r in roots8:
    val = sp.nsimplify(sp.simplify((Zstar.T*r)[0,0]))
    print(f"  e{i+1}(sign{si:+d}) e{j+1}(sign{sj:+d}): <Z*,r> = {val}")
