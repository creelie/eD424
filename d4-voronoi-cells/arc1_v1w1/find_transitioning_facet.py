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

def V_vertex(theta, t):
    A = np.cos(theta)+np.sin(theta)*np.cos(t)
    B = np.cos(theta)-np.sin(theta)*np.cos(t)
    z1 = np.sqrt(2)*(1-B)/A
    return np.array([z1, np.sqrt(2), 0, 0])

for t0, theta_c in [(0.05,0.0998752538),(0.1,0.1990074415),(0.15,0.2966808040)]:
    V = V_vertex(theta_c, t0)
    print(f"t={t0}, theta_c={theta_c}: V={np.round(V,6)}")
    for j,r in enumerate(roots):
        val = np.dot(V,r)
        if abs(val-1.0)<1e-4:
            print(f"    {labels[j]}: <V,r>={val:.8f}")
    print()
