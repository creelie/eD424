#!/usr/bin/env python3
"""
Map the 2D combinatorial partition of the triangle's interior at a fixed
theta, to understand whether there's a single dominant ("generic") type
covering most of the interior, with lower-dimensional exceptional loci --
the usual polytope-theory pattern -- or something messier.
"""
import numpy as np
from collections import Counter
from scipy.spatial import HalfspaceIntersection, ConvexHull

roots = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si; v[j] = sj
                roots.append(v / np.sqrt(2))
roots = np.array(roots)

def find(vec):
    v = np.array(vec, dtype=float); v /= np.linalg.norm(v)
    d = roots @ v
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]
v2 = roots[find([0, 0, 1, -1])]

def n_vertices(theta, e_perp):
    e_perp = e_perp / np.linalg.norm(e_perp)
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy(); dirs[idx0] = u1
    hs = np.hstack([dirs, -np.ones((len(dirs),1))])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    hull = ConvexHull(hi.intersections, qhull_options='QJ')
    return len(hull.vertices)

def dev_abc(a,b,c):
    v = a*v1+b*w1+c*v2
    return v/np.linalg.norm(v)

# Use barycentric-style grid: a,b,c >=0 summing to 1 (linear), then project
N = 40
theta = 1.45
counts = Counter()
grid = {}
for i in range(N+1):
    for j in range(N+1-i):
        k = N - i - j
        a,b,c = i/N, j/N, k/N
        if a+b+c < 1e-9:
            continue
        nv = n_vertices(theta, dev_abc(a,b,c))
        counts[nv]+=1
        grid[(i,j)] = nv

total = sum(counts.values())
print(f"theta={theta}, grid points={total}")
for k,v in sorted(counts.items()):
    print(f"  n_vertices={k}: {v} pts ({100*v/total:.1f}%)")

# print a coarse ASCII map (edges of triangle at i=0, j=0, or k=0)
print()
print("ASCII map (rows = j from N down to 0, cols = i from 0 to N); each char")
print("codes n_vertices: 32->'.', 34->'a', 37->'b', 40->'c', other->'?':")
code = {32:'.', 34:'a', 37:'b', 40:'c'}
for j in range(N, -1, -1):
    row = []
    for i in range(N+1-j if False else N+1):
        if i+j> N:
            row.append(' ')
            continue
        nv = grid.get((i,j))
        if nv is None:
            row.append(' ')
        else:
            row.append(code.get(nv,'?'))
    print(''.join(row))
