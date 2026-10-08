import itertools, numpy as np
from scipy.spatial import ConvexHull
from fractions import Fraction as F

# 8 axis vertices: +-sqrt(2) e_i
axis = []
for i in range(4):
    for s in (1,-1):
        v = [0,0,0,0]
        v[i] = s*np.sqrt(2)
        axis.append(v)

# 16 sign vertices: (1/sqrt2)(eps1,eps2,eps3,eps4)
sign = []
for eps in itertools.product([1,-1], repeat=4):
    v = [e/np.sqrt(2) for e in eps]
    sign.append(v)

verts = np.array(axis+sign)
print("num vertices:", len(verts))
hull = ConvexHull(verts)
print("volume (scipy convex hull):", hull.volume)
print("num facets:", len(hull.simplices))

# norms
norms = np.linalg.norm(verts, axis=1)
print("all norms == sqrt(2)? ", np.allclose(norms, np.sqrt(2)))

# roots
roots = []
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v=[0,0,0,0]; v[i]=si; v[j]=sj
                roots.append(v)
roots = np.array(roots)
print("num roots:", len(roots))

# check support function h(alpha/sqrt2) = 1 for every root direction
dirs = roots/np.sqrt(2)
supp = np.max(verts @ dirs.T, axis=0)
print("support at root directions, min/max:", supp.min(), supp.max())

# check support function at coordinate directions e_i
coord_dirs = np.eye(4)
supp_coord = np.max(verts @ coord_dirs.T, axis=0)
print("support at coordinate directions:", supp_coord)
