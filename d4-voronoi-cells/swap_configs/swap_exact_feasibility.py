import sympy as sp
import numpy as np
import time

sqrt2 = sp.sqrt(2)
sqrt3 = sp.sqrt(3)

# exact roots
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
    return int(np.argmax(d))

ia = find([1,1,0,0]); ib = find([1,0,1,0])
alpha_e, beta_e = roots_exact[ia], roots_exact[ib]
s = sp.Rational(1,2)
beta_perp_raw = beta_e - s*alpha_e
beta_perp = sp.simplify(beta_perp_raw / sp.sqrt((beta_perp_raw.T*beta_perp_raw)[0]))
alpha_perp_raw = alpha_e - s*beta_e
alpha_perp = sp.simplify(alpha_perp_raw / sp.sqrt((alpha_perp_raw.T*alpha_perp_raw)[0]))
print("beta_perp =", beta_perp.T)
print("alpha_perp =", alpha_perp.T)

theta = sp.symbols('theta', real=True)
u1 = sp.cos(theta)*alpha_e + sp.sin(theta)*beta_perp
u2 = sp.cos(theta)*beta_e + sp.sin(theta)*alpha_perp
u1 = sp.simplify(u1)
u2 = sp.simplify(u2)
print("u1(theta) =", u1.T)
print("u2(theta) =", u2.T)

fixed_idx = [k for k in range(24) if k not in (ia,ib)]
active_exact = [roots_exact[k] for k in fixed_idx] + [u1, u2]  # index 22->u1, 23->u2

# find one concrete vertex quadruple (from earlier float run at theta=0.3, support with 5 elts)
# pick a degree-5 support and choose any 4 of the 5 indices as our quadruple, check consistency
t0 = time.time()
# use one support found earlier for illustration
test_support = [10,12,18,19,22]
quad = test_support[:4]
pts = [active_exact[k] for k in quad]
G = sp.Matrix(4,4, lambda a,b: sp.simplify((pts[a].T*pts[b])[0,0]))
print("Gram matrix (symbolic) built, t=", time.time()-t0)
print(G)

print()
print("=== now a quadruple involving the MOVING direction u1 (index 22) ===")
t0 = time.time()
quad2 = [0,7,12,22]
pts2 = [active_exact[k] for k in quad2]
G2 = sp.Matrix(4,4, lambda a,b: sp.trigsimp(sp.expand_trig(sp.simplify((pts2[a].T*pts2[b])[0,0]))))
print("Gram matrix (symbolic, with u1) built, t=", time.time()-t0)
sp.pprint(G2)

t0=time.time()
detG2 = sp.simplify(G2.det())
print("det(G2) =", detG2, "  time:", time.time()-t0)

t0=time.time()
alpha_vec = G2.solve(sp.ones(4,1))
alpha_vec = sp.simplify(alpha_vec)
print("alpha (Cramer coeffs) computed, time:", time.time()-t0)
print(alpha_vec.T)

t0=time.time()
zI = sp.zeros(4,1)
for a in range(4):
    zI += alpha_vec[a]*pts2[a]
zI = sp.simplify(sp.trigsimp(zI))
print("z_I computed, time:", time.time()-t0)
print("z_I =", zI.T)
