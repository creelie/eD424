import numpy as np
import sympy as sp

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

print("Candidate breakpoint curves: for each fixed root r_k (k != idx0),")
print("the tangency condition <r_k, u1(theta,t)> = 1, i.e.")
print("  cos(theta)*a_k + sin(theta)*(cos(t)*b_k + sin(t)*c_k) = 1")
print("where a_k=<r_k,u0>, b_k=<r_k,v1>, c_k=<r_k,w1>.")
print()

seen = {}
for k in range(24):
    if k == idx0:
        continue
    a = round(np.dot(roots[k], u0) * 2) / 2
    b = round(np.dot(roots[k], v1) * 2) / 2
    c = round(np.dot(roots[k], w1) * 2) / 2
    key = (a, b, c)
    seen.setdefault(key, []).append(labels[k])

print(f"{'(a_k, b_k, c_k)':<20} {'#roots':<8} example labels")
for key, labs in sorted(seen.items()):
    print(f"{str(key):<20} {len(labs):<8} {labs}")
