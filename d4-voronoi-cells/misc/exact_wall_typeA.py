import sympy as sp
import numpy as np

# Recreate the same fixed-root indexing as identify_wall2.py to map index 19 (fixed array)
# and quadruple {3,7,11,15} (fixed array) back to actual root vectors, then redo everything
# exactly in sympy (Q(sqrt2)).

roots_num = []
roots_exact = []
sqrt2 = sp.sqrt(2)
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v = [0,0,0,0]; v[i]=si; v[j]=sj
                roots_num.append(np.array(v,dtype=float)/np.sqrt(2))
                ev = sp.Matrix([sp.Rational(si) if k==i else (sp.Rational(sj) if k==j else 0) for k in range(4)])/sqrt2
                roots_exact.append(ev)
roots_num = np.array(roots_num)

def find(vec):
    v = np.array(vec,dtype=float)/np.linalg.norm(vec)
    d = roots_num @ v
    k = np.argmax(d)
    assert d[k] > 1-1e-9
    return k

i0 = find([1,1,0,0])
fixed_idx = [k for k in range(24) if k != i0]   # maps "fixed array position" -> root index
fixed_exact = [roots_exact[k] for k in fixed_idx]

u0_exact = roots_exact[i0]
v1_exact = sp.Matrix([1,-1,0,0])/sqrt2
w1_exact = sp.Matrix([0,0,1,1])/sqrt2

# quadruple I = {3,7,11,15} in fixed-array indexing (0-based, matching identify_wall2.py output)
I = [3,7,11,15]
pts = [fixed_exact[k] for k in I]

# Gram matrix G_I and Cramer vertex z_I = G_I^{-1} 1 (coords in basis pts), then z_I as R^4 vector
G = sp.Matrix(4,4, lambda a,b: (pts[a].T*pts[b])[0,0])
G = sp.simplify(G)
print("Gram matrix G_I =")
sp.pprint(G)
detG = sp.simplify(G.det())
print("det(G_I) =", detG)

if detG == 0:
    print()
    print("=" * 70)
    print("THE QUADRUPLE IS LINEARLY DEPENDENT")
    print("=" * 70)
    print("    The Gram matrix of the four chosen facet normals is singular,")
    print("    so this quadruple determines no vertex by Cramer's rule: the")
    print("    four hyperplanes meet in a line rather than a point, or not at")
    print("    all.  That degeneracy is what this diagnostic was written to")
    print("    detect, and it is why the vertex solvers elsewhere in the")
    print("    package try every independent subset of the active facets")
    print("    rather than taking the first four (see")
    print("    arc1_v1w1/region4_exact_volume.py).")
    raise SystemExit(0)

alpha = G.solve(sp.ones(4, 1))
alpha = sp.simplify(alpha)
print("alpha =", alpha.T)

zI = sp.zeros(4,1)
for a in range(4):
    zI += alpha[a]*pts[a]
zI = sp.simplify(zI)
print("z_I =", zI.T)
print("|z_I|^2 =", sp.simplify((zI.T*zI)[0,0]))

# now the exact wall condition: <z_I, u1(theta,t)> = 1
# u1 = cos(theta) u0 + sin(theta)( cos(t) v1 + sin(t) w1 )
a_coef = sp.simplify((zI.T*u0_exact)[0,0])   # <z_I,u0>
b_coef = sp.simplify((zI.T*v1_exact)[0,0])   # <z_I,v1>
c_coef = sp.simplify((zI.T*w1_exact)[0,0])   # <z_I,w1>
print()
print("<z_I,u0> =", a_coef)
print("<z_I,v1> =", b_coef)
print("<z_I,w1> =", c_coef)

theta, t = sp.symbols('theta t', real=True)
lhs = sp.cos(theta)*a_coef + sp.sin(theta)*(sp.cos(t)*b_coef + sp.sin(t)*c_coef)
eq = sp.Eq(lhs, 1)
print()
print("Exact wall equation: <z_I,u1(theta,t)> = 1  i.e.")
print(sp.simplify(lhs), "= 1")

# Evaluate at theta=0.8 numerically and solve for t in [0,pi/2], compare to floating-point transition (0.40,0.50)
theta_val = sp.Float(0.8)
lhs_num = lhs.subs(theta, theta_val)
f = sp.lambdify(t, lhs_num - 1, 'numpy')
ts = np.linspace(0, np.pi/2, 2000)
vals = f(ts)
sign_changes = [(ts[k], ts[k+1]) for k in range(len(ts)-1) if vals[k]*vals[k+1] < 0]
print()
print("theta=0.8: sign changes (candidate exact roots) of <z_I,u1>-1 in t in [0,pi/2]:", sign_changes)
