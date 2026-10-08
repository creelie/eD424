import numpy as np

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
    v = np.array(vec, dtype=float); v/=np.linalg.norm(v)
    d = roots@v; idx=np.argmax(d); assert d[idx]>1-1e-9
    return idx
idx0 = find([1,1,0,0])
u0 = roots[idx0]
v1 = roots[find([1,-1,0,0])]
w1 = roots[find([0,0,1,1])]

def V_vertex(theta, t):
    A = np.cos(theta)+np.sin(theta)*np.cos(t)
    B = np.cos(theta)-np.sin(theta)*np.cos(t)
    z1 = np.sqrt(2)*(1-B)/A
    return np.array([z1, np.sqrt(2), 0, 0])

t0 = 0.1
theta_c = 0.1990074415
for theta in [theta_c-0.01, theta_c-0.001, theta_c, theta_c+0.001, theta_c+0.01]:
    V = V_vertex(theta, t0)
    vals = [(labels[j], np.dot(V,roots[j])) for j in range(24) if j!=idx0]
    vals.append(("u1", np.dot(V, np.cos(theta)*u0+np.sin(theta)*(np.cos(t0)*v1+np.sin(t0)*w1))))
    vals_sorted = sorted(vals, key=lambda x: -x[1])
    print(f"theta={theta:.6f}: top 6 by value:")
    for lbl,val in vals_sorted[:6]:
        print(f"    {lbl}: {val:.8f}")
    print()
