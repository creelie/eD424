import sympy as sp

sqrt2 = sp.sqrt(2)
u0 = sp.Matrix([1,1,0,0])/sqrt2
v1 = sp.Matrix([1,-1,0,0])/sqrt2
w1 = sp.Matrix([0,0,1,1])/sqrt2
W  = sp.Matrix([1,1,1,1])/sqrt2

theta, t = sp.symbols('theta t', real=True)
eperp = sp.cos(t)*v1 + sp.sin(t)*w1
u1 = sp.cos(theta)*u0 + sp.sin(theta)*eperp

roots_pp = {
 '+e1+e3': sp.Matrix([1,0,1,0])/sqrt2,
 '+e1+e4': sp.Matrix([1,0,0,1])/sqrt2,
 '+e2+e3': sp.Matrix([0,1,1,0])/sqrt2,
 '+e2+e4': sp.Matrix([0,1,0,1])/sqrt2,
 '+e3+e4': sp.Matrix([0,0,1,1])/sqrt2,
}
print("(a) <W,r>=1 exactly for the 5 fixed '++' hyperplanes (excluding u0=+e1+e2 itself):")
for name,r in roots_pp.items():
    val = sp.simplify((W.T*r)[0,0])
    assert val==1
    print(f"   {name}: {val}  OK")

expr = sp.simplify((W.T*u1)[0,0])
print()
print(f"(b) <W,u1(theta,t)> = {expr}")
assert sp.simplify(expr - (sp.cos(theta)+sp.sin(theta)*sp.sin(t))) == 0

print()
print("(c) Equivalence  cos(theta)+sin(theta)sin(t)=1  <=>  sin(t)=tan(theta/2):")
tan_half = sp.sin(theta)/(1+sp.cos(theta))   # = tan(theta/2), valid for theta in (0,pi)
check = sp.simplify(sp.cos(theta) + sp.sin(theta)*tan_half - 1)
print(f"    substituting sin(t)->tan(theta/2)=sin(theta)/(1+cos(theta)): "
      f"cos(theta)+sin(theta)sin(t)-1 -> {check}")
assert check == 0
print("    CONFIRMED exactly: <W,u1>=1  <=>  sin(t) = tan(theta/2), for theta in (0,pi).")

print()
print("(d) Endpoint t=pi/4 (Round 7's traced 'high branch' endpoint):")
eq2 = sp.Eq(sp.sin(sp.pi/4), sp.sin(theta)/(1+sp.cos(theta)))
sol = sp.solve(eq2, theta)
print(f"    Solutions of sin(pi/4) = tan(theta/2): {sol}")
for s in sol:
    sf = sp.simplify(s)
    print(f"      theta = {sf}, numeric = {float(sf):.12f}")
target = [s for s in sol if float(s) > 0 and float(s) < float(sp.pi/2)]
assert len(target)==1
theta_sol = target[0]
cosdiff = sp.simplify(sp.cos(theta_sol) - sp.Rational(1,3))
print(f"    cos(theta_sol) - 1/3 = {cosdiff}  (0 means theta_sol = arccos(1/3) EXACTLY)")
assert cosdiff == 0
print(f"    EXACT MATCH: theta = arccos(1/3), matching the numerically-traced 'high")
print(f"    branch' endpoint at t=pi/4 from eperp_v1w1_breakpoint_classification.py")
print(f"    to machine precision (was 1.2309594173... in both cases).")

print()
print("(e) Endpoint t->0: sin(0)=0=tan(theta/2) => theta=0 (trivial, matches (0,0)).")
print()
print("(f) Small-t asymptotics: sin(t)~t, tan(theta/2)~theta/2 => theta ~ 2t as t->0+,")
print("    matching the numerically observed ratio theta_c/t -> 2 as t->0 exactly.")
