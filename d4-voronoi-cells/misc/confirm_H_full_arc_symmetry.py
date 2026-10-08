import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

roots = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si
                v[j] = sj
                roots.append(v / np.sqrt(2))
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

H = 0.5*np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])
print("Check H(v1)==w1, H(w1)==v1, H(u0)==u0:")
print(f"  {np.allclose(H@v1,w1)}  {np.allclose(H@w1,v1)}  {np.allclose(H@u0,u0)}")

def F_direct(theta, e_perp):
    e_perp = e_perp/np.linalg.norm(e_perp)
    u1 = np.cos(theta)*u0+np.sin(theta)*e_perp
    dirs = roots.copy(); dirs[idx0]=u1
    Am=dirs; bm=-np.ones(len(dirs))
    hs=np.hstack([Am,bm.reshape(-1,1)])
    hi=HalfspaceIntersection(hs,np.zeros(4))
    hull=ConvexHull(hi.intersections, qhull_options='QJ')
    return hull.volume-8.0

def e_perp_arc(t):
    return np.cos(t)*v1+np.sin(t)*w1

print()
print("Direct check: F(theta, e_perp(t)) == F(theta, e_perp(pi/2 - t))")
print("(this should follow exactly from Lemma lem:hadamardsymmetry, since")
print("H(e_perp(t)) = e_perp(pi/2-t) exactly -- verifying that identity")
print("and its F-consequence together):")
maxdiff=0
for t in [0.1, 0.3, 0.5, 0.7]:
    e_t = e_perp_arc(t)
    e_mirror_exact = H @ e_t
    e_mirror_formula = e_perp_arc(np.pi/2-t)
    diff_vec = np.max(np.abs(e_mirror_exact-e_mirror_formula))
    for theta in [0.3, 0.9, 1.3]:
        F1 = F_direct(theta, e_t)
        F2 = F_direct(theta, e_perp_arc(np.pi/2-t))
        d = abs(F1-F2)
        maxdiff = max(maxdiff,d)
        print(f"  t={t:.2f} theta={theta:.2f}: F(t)={F1:.8f} F(pi/2-t)={F2:.8f} diff={d:.2e}  [H(e_perp(t))==e_perp(pi/2-t) vec-diff={diff_vec:.2e}]")
print(f"max diff overall: {maxdiff:.2e}")
