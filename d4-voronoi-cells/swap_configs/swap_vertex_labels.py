import numpy as np
from scipy.spatial import HalfspaceIntersection

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
alpha, beta = roots[ia], roots[ib]
fixed_idx = [k for k in range(24) if k not in (ia,ib)]
fixed = roots[fixed_idx]
s = alpha@beta
beta_perp = beta - s*alpha; beta_perp/=np.linalg.norm(beta_perp)
alpha_perp = alpha - s*beta; alpha_perp/=np.linalg.norm(alpha_perp)
thetastar = np.pi/3

def dirs_at(theta):
    u1 = np.cos(theta)*alpha + np.sin(theta)*beta_perp
    u2 = np.cos(theta)*beta + np.sin(theta)*alpha_perp
    return np.vstack([fixed, u1.reshape(1,-1), u2.reshape(1,-1)])
    # index len(fixed)=22 -> u1, 23 -> u2

def vertex_supports(theta, tol=1e-6):
    dirs = dirs_at(theta)
    A = dirs; b=-np.ones(len(dirs))
    hs = np.hstack([A,b.reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts = hi.intersections
    vals = dirs @ verts.T
    tight = vals > 1-tol
    supports = [frozenset(np.where(tight[:,k])[0].tolist()) for k in range(verts.shape[0])]
    return supports

thetas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
sets_of_supports = []
for th in thetas:
    supp = vertex_supports(th)
    degrees = sorted(len(s) for s in supp)
    print(f"theta={th}: n_verts={len(supp)}  degree-hist={degrees.count(4)}x4,{degrees.count(5)}x5,{degrees.count(6)}x6")
    sets_of_supports.append(set(supp))

# check if the SAME quadruples (as index sets, restricted to their 4-subsets) appear across all thetas
# reduce each support (which might be a k>4 set) to all its 4-subsets for comparison is complex;
# instead just check: are the degree-4 supports (genuine simple vertices) identical across theta?
deg4_sets = []
for th in thetas:
    supp = vertex_supports(th)
    deg4 = set(s for s in supp if len(s)==4)
    deg4_sets.append(deg4)
    print(f"theta={th}: {len(deg4)} degree-4 vertices")

common = deg4_sets[0]
for s in deg4_sets[1:]:
    common = common & s
print(f"\ncommon degree-4 supports across all {len(thetas)} thetas: {len(common)}")
all_union = set()
for s in deg4_sets:
    all_union |= s
print(f"union of degree-4 supports across all thetas: {len(all_union)}")

print()
print("=== comparing full support sets (any degree) across thetas ===")
supp_sets = []
for th in thetas:
    supp_sets.append(set(vertex_supports(th)))
common_all = supp_sets[0]
for s in supp_sets[1:]:
    common_all = common_all & s
print(f"common support-sets across all {len(thetas)} thetas: {len(common_all)} out of 33")
union_all = set()
for s in supp_sets:
    union_all |= s
print(f"union: {len(union_all)}")
