import numpy as np
from itertools import combinations

roots = []
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v = np.zeros(4); v[i]=si; v[j]=sj
                roots.append(v/np.sqrt(2))
roots = np.array(roots)

def find(vec):
    v = np.array(vec,dtype=float)/np.linalg.norm(vec)
    d = roots @ v
    k = np.argmax(d)
    assert d[k] > 1-1e-9
    return k

ia = find([1,1,0,0]); ib = find([1,0,1,0])
print('alpha idx', ia, roots[ia], 'beta idx', ib, roots[ib], 'gram', roots[ia]@roots[ib])

n=24
verts=[]
vert_facets=[]
for combo in combinations(range(n),4):
    A = roots[list(combo)]
    if abs(np.linalg.det(A)) < 1e-9:
        continue
    x = np.linalg.solve(A, np.ones(4))
    vals = roots @ x
    if np.all(vals <= 1+1e-7):
        active = tuple(k for k in range(n) if vals[k] > 1-1e-7)
        dup=False
        for v2,f2 in zip(verts,vert_facets):
            if np.linalg.norm(x-v2)<1e-6:
                dup=True
                break
        if not dup:
            verts.append(x); vert_facets.append(active)
print('unique verts', len(verts))
on_alpha = [i for i,f in enumerate(vert_facets) if ia in f]
on_beta  = [i for i,f in enumerate(vert_facets) if ib in f]
both = set(on_alpha) & set(on_beta)
print('vertices on alpha facet:', len(on_alpha))
print('vertices on beta facet:', len(on_beta))
print('vertices on BOTH (shared edge/ridge vertices):', len(both))
for i in both:
    print(' shared vertex', np.round(verts[i],4), 'active facets:', vert_facets[i])
