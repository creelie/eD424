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
fixed_root_indices = [k for k in range(24) if k != i0]

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

candidates = []
for idx4 in itertools.combinations(fixed_root_indices, 4):
    zI, detG = cramer_vertex(idx4)
    if zI is None:
        continue
    zI_num = np.array([float(x) for x in zI], dtype=float)
    vals = roots_num @ zI_num
    if np.all(vals <= 1+1e-9):
        a = (zI.T*u0_exact)[0,0]
        b = (zI.T*v1_exact)[0,0]
        c = (zI.T*w1_exact)[0,0]
        a_n, b_n, c_n = float(a), float(b), float(c)
        amp = np.hypot(b_n, c_n)
        # for some theta in (0,pi/2), does sin(theta)*amp >= 1 - cos(theta)*a_n  (i.e. a real crossing exists)?
        thetas = np.linspace(0.05, np.pi/2-0.05, 50)
        active = False
        for th in thetas:
            rhs = 1 - np.cos(th)*a_n
            lhsmax = np.sin(th)*amp
            if abs(rhs) <= lhsmax:
                active = True
                break
        if active and amp > 1e-9:
            candidates.append((idx4, zI, detG, a, b, c))

print(f"Found {len(candidates)} candidate quadruples with an active wall somewhere in theta in (0,pi/2)")
idx4, zI, detG, a, b, c = candidates[0]
print("Using quadruple", idx4)
print("z_I =", zI.T, " det=", detG)
print(f"<z_I,u0>={a}  <z_I,v1>={b}  <z_I,w1>={c}")

theta, t = sp.symbols('theta t', real=True, positive=True)
lhs = sp.cos(theta)*a + sp.sin(theta)*(sp.cos(t)*b + sp.sin(t)*c)
eq = sp.Eq(lhs, 1)
print()
print("Exact wall equation:", sp.nsimplify(lhs), "= 1")

theta_val = sp.Rational(4,5)  # 0.8
lhs_th = lhs.subs(theta, theta_val)
sol = sp.solveset(sp.Eq(lhs_th,1), t, domain=sp.Interval(0, sp.pi/2))
print(f"\ntheta=4/5: exact solveset for t in [0,pi/2]: {sol}")
sol_n = [sp.N(s) for s in sol] if sol.is_FiniteSet else sol
print("numeric:", sol_n)
