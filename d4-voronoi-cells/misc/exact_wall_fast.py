import numpy as np
import itertools

roots_num = []
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v = np.zeros(4); v[i]=si; v[j]=sj
                roots_num.append(v/np.sqrt(2))
roots_num = np.array(roots_num)

def find(vec):
    v = np.array(vec,dtype=float)/np.linalg.norm(vec)
    d = roots_num @ v
    k = np.argmax(d)
    assert d[k] > 1-1e-9
    return k

i0 = find([1,1,0,0])
fixed_root_indices = [k for k in range(24) if k != i0]
u0 = roots_num[i0]
v1 = np.array([1,-1,0,0])/np.sqrt(2)
w1 = np.array([0,0,1,1])/np.sqrt(2)

# Precompute (a,b,c) for every genuine nondegenerate 24-cell vertex quadruple among fixed roots
abc_list = []
for idx4 in itertools.combinations(fixed_root_indices, 4):
    pts = roots_num[list(idx4)]
    G = pts @ pts.T
    detG = np.linalg.det(G)
    if abs(detG) < 1e-9:
        continue
    alpha = np.linalg.solve(G, np.ones(4))
    zI = alpha @ pts
    vals = roots_num @ zI
    if np.all(vals <= 1+1e-9):
        a = zI @ u0
        b = zI @ v1
        c = zI @ w1
        amp = np.hypot(b,c)
        if amp > 1e-9:
            abc_list.append((a,b,c,amp))

print(f"Cached {len(abc_list)} candidate (a,b,c) triples")

def predict(theta_val):
    ts_found = []
    for a,b,c,amp in abc_list:
        rhs = 1 - np.cos(theta_val)*a
        lhsmax = np.sin(theta_val)*amp
        if abs(rhs) > lhsmax:
            continue
        phi = np.arctan2(c,b)
        val = np.clip(rhs/(np.sin(theta_val)*amp), -1, 1)
        for sgn in (+1,-1):
            tt = (phi + sgn*np.arccos(val)) % (2*np.pi)
            if -1e-9 <= tt <= np.pi/2+1e-9:
                ts_found.append(round(tt,4))
    return sorted(set(ts_found))

observed = {0.3:[0.168,1.429], 0.5:[0.273,1.324], 0.8:[0.457,0.641,0.956,1.140],
            1.0:[0.220,0.588,1.009,1.377], 1.2:[0.089,0.772,0.825,1.508], 1.4:[0.036,0.588,1.009,1.561]}
for th, obs in observed.items():
    pred = predict(th)
    print(f"theta={th}:")
    print(f"  predicted (exact, type-A) = {pred}")
    print(f"  observed  (fp grid)       = {obs}")
