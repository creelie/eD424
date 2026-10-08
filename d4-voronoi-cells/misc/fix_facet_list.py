import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

roots = []
labels = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si
                v[j] = sj
                roots.append(v / np.sqrt(2))
                sgn = lambda s: '+' if s == 1 else '-'
                labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
roots = np.array(roots)

def find(vec):
    v = np.array(vec, dtype=float)
    v /= np.linalg.norm(v)
    d = roots @ v
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]

t0 = 0.15
theta_c = 0.2966808040
e_perp = np.cos(t0)*v1+np.sin(t0)*w1
u1 = np.cos(theta_c)*u0+np.sin(theta_c)*e_perp
dirs = roots.copy(); dirs[idx0]=u1

v_test = np.array([0.37787,1.41421356,0,0])
print("Inner products of the transitioning vertex with all 24 directions:")
for j in range(24):
    val = np.dot(v_test, dirs[j])
    lbl = labels[j] if j != idx0 else "u1(moving)"
    if abs(val-1.0) < 1e-3:
        print(f"  {lbl}: <v,dir>={val:.8f}   (close to 1: {abs(val-1)<1e-6})")
