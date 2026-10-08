import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
for _p in (_PKG_ROOT, _os.path.join(_PKG_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
def _data(_name):
    """Resolve a bundled data file, wherever the script is run from."""
    _c = _os.path.join(_DATA_DIR, _name)
    if _os.path.exists(_c):
        return _c
    if _os.path.exists(_name):
        return _name
    raise SystemExit(
        "required data file %r not found; expected it in %s. "
        "Run the matching *_derive.py step first, or fetch the bundled "
        "copy from the package's data/ directory." % (_name, _DATA_DIR))

import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

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
    return int(np.argmax(d))

ia = find([1,1,0,0]); ib = find([1,0,1,0])
alpha, beta = roots[ia], roots[ib]
fixed_idx = [k for k in range(24) if k not in (ia,ib)]
fixed = roots[fixed_idx]
s = alpha@beta
beta_perp = beta - s*alpha; beta_perp/=np.linalg.norm(beta_perp)
alpha_perp = alpha - s*beta; alpha_perp/=np.linalg.norm(alpha_perp)

def dirs_at(theta):
    u1 = np.cos(theta)*alpha + np.sin(theta)*beta_perp
    u2 = np.cos(theta)*beta + np.sin(theta)*alpha_perp
    return np.vstack([fixed, u1.reshape(1,-1), u2.reshape(1,-1)])

theta0 = 0.3  # inside (0, pi/6=0.5236)
dirs = dirs_at(theta0)
A = dirs; b=-np.ones(len(dirs))
hs = np.hstack([A,b.reshape(-1,1)])
hi = HalfspaceIntersection(hs, np.zeros(4))
verts = hi.intersections
print("n_verts:", len(verts))
hull = ConvexHull(verts, qhull_options='QJ')
print("n_simplices (boundary tetrahedra):", len(hull.simplices))
print("hull.volume:", hull.volume)

vals = dirs @ verts.T
tight = vals > 1-1e-6
supports = [frozenset(np.where(tight[:,k])[0].tolist()) for k in range(verts.shape[0])]
degs = [len(s) for s in supports]
print("degree histogram:", sorted(degs).count(5), "deg5,", sorted(degs).count(6),"deg6")

# find a valid defining quadruple for each vertex (any 4 of its tight indices w/ nonzero det)
import itertools
quad_for_vertex = []
for k in range(len(verts)):
    supp = sorted(supports[k])
    found = None
    for quad in itertools.combinations(supp, 4):
        pts = dirs[list(quad)]
        G = pts@pts.T
        if abs(np.linalg.det(G)) > 1e-6:
            found = quad
            break
    quad_for_vertex.append(found)
print("all vertices have a valid quadruple:", all(q is not None for q in quad_for_vertex))

# sanity: recompute each vertex via its quadruple's Cramer formula (floating point) and compare
maxerr = 0
for k, quad in enumerate(quad_for_vertex):
    pts = dirs[list(quad)]
    G = pts@pts.T
    alpha_c = np.linalg.solve(G, np.ones(4))
    zrecon = alpha_c @ pts
    err = np.linalg.norm(zrecon - verts[k])
    maxerr = max(maxerr, err)
print("max reconstruction error:", maxerr)

# save the data for the next script (vertex list, quads, simplices)
np.save(_os.path.join(_DATA_DIR, 'swap_verts_theta03.npy'), verts)
np.save(_os.path.join(_DATA_DIR, 'swap_simplices_theta03.npy'), hull.simplices)
import pickle
with open(_os.path.join(_DATA_DIR, 'swap_quads_theta03.pkl'), 'wb') as f:
    pickle.dump({'quad_for_vertex': quad_for_vertex, 'supports': supports}, f)
print("saved.")
