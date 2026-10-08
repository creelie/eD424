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
        return None
    alpha = G.solve(sp.ones(4,1))
    zI = sp.zeros(4,1)
    for a in range(4):
        zI += alpha[a]*pts[a]
    return sp.simplify(zI)

theta_val = 0.8
found_ts = []
n_valid_vertices = 0
for idx4 in itertools.combinations(fixed_root_indices, 4):
    zI = cramer_vertex(idx4)
    if zI is None:
        continue
    zI_num = np.array([float(x) for x in zI], dtype=float)
    vals = roots_num @ zI_num
    if not np.all(vals <= 1+1e-9):
        continue
    n_valid_vertices += 1
    a = float((zI.T*u0_exact)[0,0])
    b = float((zI.T*v1_exact)[0,0])
    c = float((zI.T*w1_exact)[0,0])
    amp = np.hypot(b,c)
    if amp < 1e-9:
        continue
    rhs = 1 - np.cos(theta_val)*a
    lhsmax = np.sin(theta_val)*amp
    if abs(rhs) > lhsmax:
        continue
    phi = np.arctan2(c,b)  # b*cos t + c*sin t = amp*cos(t-phi)
    val = rhs/(np.sin(theta_val)*amp)
    val = np.clip(val,-1,1)
    for sgn in (+1,-1):
        tt = phi + sgn*np.arccos(val)
        tt = tt % (2*np.pi)
        if 0 <= tt <= np.pi/2+1e-9:
            found_ts.append(round(tt,4))

found_ts = sorted(set(found_ts))
print(f"Total nondegenerate genuine 24-cell vertex-quadruples among fixed roots checked: {n_valid_vertices}")
print(f"Exact predicted wall t-values (theta=0.8) landing in [0,pi/2]: {found_ts}")
print()
print("Compare to floating-point-observed transitions at theta=0.8: [0.457, 0.641, 0.956, 1.140]")

print()
print("=== cross-check at more theta values ===")
observed = {0.3:[0.168,1.429], 0.5:[0.273,1.324], 0.8:[0.457,0.641,0.956,1.140],
            1.0:[0.220,0.588,1.009,1.377], 1.2:[0.089,0.772,0.825,1.508], 1.4:[0.036,0.588,1.009,1.561]}

def predict(theta_val):
    found_ts = []
    for idx4 in itertools.combinations(fixed_root_indices, 4):
        zI = cramer_vertex(idx4)
        if zI is None:
            continue
        zI_num = np.array([float(x) for x in zI], dtype=float)
        vals = roots_num @ zI_num
        if not np.all(vals <= 1+1e-9):
            continue
        a = float((zI.T*u0_exact)[0,0])
        b = float((zI.T*v1_exact)[0,0])
        c = float((zI.T*w1_exact)[0,0])
        amp = np.hypot(b,c)
        if amp < 1e-9:
            continue
        rhs = 1 - np.cos(theta_val)*a
        lhsmax = np.sin(theta_val)*amp
        if abs(rhs) > lhsmax:
            continue
        phi = np.arctan2(c,b)
        val = np.clip(rhs/(np.sin(theta_val)*amp), -1, 1)
        for sgn in (+1,-1):
            tt = (phi + sgn*np.arccos(val)) % (2*np.pi)
            if 0 <= tt <= np.pi/2+1e-9:
                found_ts.append(round(tt,4))
    return sorted(set(found_ts))

for th, obs in observed.items():
    pred = predict(th)
    print(f"theta={th}: predicted={pred}")
    print(f"           observed ={obs}")
