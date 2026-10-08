import sympy as sp
import numpy as np
import itertools

sqrt2 = sp.sqrt(2)
roots_num = []
roots_exact = []
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
fixed_root_indices = [k for k in range(24) if k != i0]   # 23 fixed roots, actual root indices

u0_exact = roots_exact[i0]
v1_exact = sp.Matrix([1,-1,0,0])/sqrt2
w1_exact = sp.Matrix([0,0,1,1])/sqrt2

def cramer_vertex(idx4):
    pts = [roots_exact[k] for k in idx4]
    G = sp.Matrix(4,4, lambda a,b: (pts[a].T*pts[b])[0,0])
    detG = sp.nsimplify(G.det())
    if detG == 0:
        return None, 0
    alpha = G.solve(sp.ones(4,1))
    zI = sp.zeros(4,1)
    for a in range(4):
        zI += alpha[a]*pts[a]
    return sp.simplify(zI), detG

# search for a nondegenerate quadruple among the 23 fixed roots (not involving i0)
# that is ALSO a genuine vertex of the D4 24-cell itself (i.e. <zI, u_k> <= 1 for all other 22 fixed roots)
found = None
for idx4 in itertools.combinations(fixed_root_indices, 4):
    zI, detG = cramer_vertex(idx4)
    if zI is None:
        continue
    # check feasibility against ALL 24 roots (including u0) using floating point for speed
    zI_num = np.array([float(x) for x in zI], dtype=float)
    vals = roots_num @ zI_num
    if np.all(vals <= 1+1e-9):
        # genuine vertex of the original 24-cell not involving u0's root -> a "type-a" candidate
        b = sp.simplify((zI.T*v1_exact)[0,0])
        c = sp.simplify((zI.T*w1_exact)[0,0])
        if b != 0 or c != 0:
            found = (idx4, zI, detG, b, c)
            break

print("Found quadruple:", found[0])
idx4, zI, detG, b, c = found
a = sp.simplify((zI.T*u0_exact)[0,0])
print("z_I =", zI.T, "  det(G_I) =", detG)
print("<z_I,u0> =", a, "  <z_I,v1> =", b, "  <z_I,w1> =", c)

theta, t = sp.symbols('theta t', real=True, positive=True)
lhs = sp.cos(theta)*a + sp.sin(theta)*(sp.cos(t)*b + sp.sin(t)*c)
print()
print("Exact wall equation:  cos(theta)*(%s) + sin(theta)*[cos(t)*(%s) + sin(t)*(%s)] = 1" % (a,b,c))

amp = sp.sqrt(b**2+c**2)
print("Amplitude of the (cos t, sin t) part:", sp.simplify(amp))
print("=> for fixed theta, the equation is  sin(theta)*amp*cos(t - phi) = 1 - cos(theta)*a")
print("   which has AT MOST 2 solutions t in any period, by elementary trig (R cos(t-phi)=const).")

# numeric check at theta=0.8
theta_val = 0.8
lhs_num = sp.lambdify(t, lhs.subs(theta, theta_val), 'numpy')
ts = np.linspace(0, np.pi/2, 4000)
vals = lhs_num(ts) - 1
roots_found = [ (ts[k]+ts[k+1])/2 for k in range(len(ts)-1) if vals[k]*vals[k+1] < 0 ]
print()
print(f"theta={theta_val}: numeric sign-change locations in t in [0,pi/2]:", roots_found)
