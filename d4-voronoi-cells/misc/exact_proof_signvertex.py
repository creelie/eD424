import sympy as sp

sqrt2 = sp.sqrt(2)
u0 = sp.Matrix([1,1,0,0])/sqrt2
v1 = sp.Matrix([1,-1,0,0])/sqrt2
w1 = sp.Matrix([0,0,1,1])/sqrt2
W  = sp.Matrix([1,1,1,1])/sqrt2   # fixed sign vertex

theta, t = sp.symbols('theta t', real=True)
eperp = sp.cos(t)*v1 + sp.sin(t)*w1
u1 = sp.cos(theta)*u0 + sp.sin(theta)*eperp

# (a) W on 5 fixed root hyperplanes (excluding u0 = +e1+e2 itself)
roots_pp = {
 '+e1+e3': sp.Matrix([1,0,1,0])/sqrt2,
 '+e1+e4': sp.Matrix([1,0,0,1])/sqrt2,
 '+e2+e3': sp.Matrix([0,1,1,0])/sqrt2,
 '+e2+e4': sp.Matrix([0,1,0,1])/sqrt2,
 '+e3+e4': sp.Matrix([0,0,1,1])/sqrt2,
}
print("(a) <W,r> for the 5 fixed '++' hyperplanes (excluding u0 itself):")
for name,r in roots_pp.items():
    val = sp.simplify((W.T*r)[0,0])
    print(f"   {name}: {val}")
    assert val==1

# also check u0's own hyperplane value (should also be 1, but it's MOVING)
print("   (+e1+e2 = u0 itself, also gives", sp.simplify((W.T*u0)[0,0]), "but this facet moves)")

# (b) <W, u1(theta,t)>
expr = sp.simplify((W.T*u1)[0,0])
print()
print(f"(b) <W,u1(theta,t)> = {expr}")
assert sp.simplify(expr - (sp.cos(theta)+sp.sin(theta)*sp.sin(t))) == 0

# solve <W,u1>=1 for the locus
eq = sp.Eq(expr,1)
print(f"    Setting =1: {eq}")

# show equivalent to sin(t)=tan(theta/2) via half-angle
lhs = sp.cos(theta)+sp.sin(theta)*sp.sin(t) - 1
half = sp.simplify(sp.trigsimp(lhs.rewrite(sp.tan)))
print(f"    cos(theta)+sin(theta)sin(t)-1 rewritten via tan-half-angle: {half}")

# direct check: cos(theta)+sin(theta)*sin(t)=1  <=>  sin(t) = tan(theta/2)
th = sp.symbols('th', positive=True)
lhs2 = sp.cos(theta) + sp.sin(theta)*sp.tan(theta/2) - 1
# cos(th) + sin(th) tan(th/2) - 1 vanishes identically, since
# sin(th) tan(th/2) = 2 sin^2(th/2) = 1 - cos(th).  simplify() alone does
# not always reduce it, so rewrite the half-angle first.
lhs2s = sp.simplify(sp.trigsimp(sp.expand_trig(lhs2.rewrite(sp.sin))))
if lhs2s != 0:
    lhs2s = sp.simplify(lhs2.rewrite(sp.tan).subs(sp.tan(theta/2),
                        sp.sin(theta)/(1 + sp.cos(theta))))
print(f"    Direct substitution sin(t)->tan(theta/2): cos(theta)+sin(theta)tan(theta/2)-1 = {lhs2s}")
assert lhs2s == 0
print("    CONFIRMED: cos(theta)+sin(theta)*sin(t)=1  <=>  sin(t)=tan(theta/2), identically.")

print()
print("(c) Endpoint check: at t=pi/4, solve for theta in (0,pi/2):")
eq2 = sp.Eq(sp.sin(sp.pi/4), sp.tan(theta/2))
sol = sp.solve(eq2, theta)
print(f"    sin(pi/4)=tan(theta/2)  =>  theta = {sol}")
for s in sol:
    print(f"      theta = {s} = {sp.nsimplify(sp.simplify(s))},  numeric = {float(s):.10f}")
print(f"    Compare arccos(1/3) = {float(sp.acos(sp.Rational(1,3))):.10f}")
# verify exactly theta = acos(1/3)
cand = [s for s in sol if s.is_real and 0 < float(s) < sp.pi/2]
if cand:
    diff = sp.simplify(sp.cos(cand[0]) - sp.Rational(1,3))
    print(f"    cos(theta_sol) - 1/3 = {diff}  (should be 0)")

print()
print("(d) Endpoint check: as t->0, theta->0 (trivial line), matches formula sin(0)=tan(0)=0.")
