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

def tangent_basis(root):
    # orthonormal basis of the 3-dim space orthogonal to `root`
    # start from standard basis, Gram-Schmidt against root
    B = []
    for e in np.eye(4):
        v = e - (e@root)*root
        for b in B:
            v = v - (v@b)*b
        n = np.linalg.norm(v)
        if n > 1e-8:
            B.append(v/n)
        if len(B)==3:
            break
    return np.array(B)  # 3x4

def u_of_v(root, v):
    # v is a small tangent vector (in R^4, orthogonal to root); exponential map
    t = np.linalg.norm(v)
    if t < 1e-14:
        return root.copy()
    ep = v/t
    return np.cos(t)*root + np.sin(t)*ep

def poly_volume(dirs):
    A = dirs; b = -np.ones(len(dirs))
    hs = np.hstack([A,b.reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    hull_pts = hi.intersections
    from scipy.spatial import ConvexHull
    hull = ConvexHull(hull_pts, qhull_options='QJ')
    return hull.volume

def make_F(active_idx):
    fixed_idx = [k for k in range(24) if k not in active_idx]
    fixed = roots[fixed_idx]
    active_roots = [roots[k] for k in active_idx]
    bases = [tangent_basis(r) for r in active_roots]
    m = len(active_idx)
    def F(coeffs):
        # coeffs: flat array length 3m, local coords in each tangent basis
        dirs_active = []
        for i in range(m):
            v = coeffs[3*i:3*i+3] @ bases[i]
            dirs_active.append(u_of_v(active_roots[i], v))
        dirs = np.vstack([fixed] + [d.reshape(1,-1) for d in dirs_active])
        return poly_volume(dirs) - 8.0
    return F, 3*m

def numeric_hessian(F, dim, h):
    H = np.zeros((dim,dim))
    F0 = F(np.zeros(dim))
    # diagonal
    for i in range(dim):
        ei = np.zeros(dim); ei[i]=h
        H[i,i] = (F(ei) - 2*F0 + F(-ei))/h**2
    # off-diagonal
    for i in range(dim):
        for j in range(i+1,dim):
            eij = np.zeros(dim); eij[i]=h; eij[j]=h
            eimj = np.zeros(dim); eimj[i]=h; eimj[j]=-h
            mij = np.zeros(dim); mij[i]=-h; mij[j]=h
            mimj = np.zeros(dim); mimj[i]=-h; mimj[j]=-h
            H[i,j]=H[j,i] = (F(eij)-F(eimj)-F(mij)+F(mimj))/(4*h**2)
    return H, F0

# sanity check m=1
i0 = find([1,1,0,0])
F1, dim1 = make_F([i0])
H1, F0 = numeric_hessian(F1, dim1, 0.02)
print("m=1 check: F(0)=",F0)
print("Hessian (should be ~ (2/3)*I_3):")
print(np.round(H1,4))
print("eigenvalues:", np.round(np.linalg.eigvalsh(H1),4))

print()
print("="*70)
print("m=2 (adjacent pair alpha=(1,1,0,0), beta=(1,0,1,0)):")
ia = find([1,1,0,0]); ib = find([1,0,1,0])
print("Gram(alpha,beta) =", roots[ia]@roots[ib])
F2, dim2 = make_F([ia, ib])
H2, F0_2 = numeric_hessian(F2, dim2, 0.02)
print("F(0) =", F0_2)
print("Full 6x6 Hessian:")
print(np.round(H2,4))
ev2 = np.linalg.eigvalsh(H2)
print("eigenvalues:", np.round(ev2,4))
print("min eigenvalue:", ev2.min())

print()
print("="*70)
print("m=3 (mutually-adjacent triple: (1,1,0,0),(1,0,1,0),(1,0,0,1)):")
ic = find([1,1,0,0]); idd = find([1,0,1,0]); ie = find([1,0,0,1])
G3 = np.array([[roots[a]@roots[b] for b in [ic,idd,ie]] for a in [ic,idd,ie]])
print("Gram matrix of the triple:")
print(np.round(G3,4))
F3, dim3 = make_F([ic, idd, ie])
H3, F0_3 = numeric_hessian(F3, dim3, 0.02)
print("F(0) =", F0_3)
ev3 = np.linalg.eigvalsh(H3)
print("eigenvalues (9 total):", np.round(ev3,4))
print("min eigenvalue:", ev3.min())

# cross-check with a smaller step for stability
H3b, _ = numeric_hessian(F3, dim3, 0.01)
ev3b = np.linalg.eigvalsh(H3b)
print()
print("cross-check at h=0.01, min eigenvalue:", ev3b.min())
print("difference:", abs(ev3.min()-ev3b.min()))

print()
print("="*70)
print("Non-adjacent pair check (orthogonal): alpha=(1,1,0,0), k=index1=(1,-1,0,0)/sqrt2:")
i_orth = 1
Fo, dimo = make_F([ia, i_orth])
Ho, F0o = numeric_hessian(Fo, dimo, 0.02)
print("cross block (rows 0-2, cols 3-5):")
print(np.round(Ho[0:3,3:6],4))
print("eigenvalues:", np.round(np.linalg.eigvalsh(Ho),4))

print()
print("Non-adjacent pair check (obtuse, Gram=-1/2): alpha=(1,1,0,0), k=index6:")
i_obt = 6
Fb, dimb = make_F([ia, i_obt])
Hb, F0b = numeric_hessian(Fb, dimb, 0.02)
print("cross block:")
print(np.round(Hb[0:3,3:6],4))
print("eigenvalues:", np.round(np.linalg.eigvalsh(Hb),4))

print()
print("="*70)
print("Path triple (a-b adjacent, b-k adjacent, a-k orthogonal), a=(1,1,0,0), b=(1,0,1,0), k=index1:")
Fp, dimp = make_F([ia, ib, i_orth])
Hp, F0p = numeric_hessian(Fp, dimp, 0.02)
evp = np.linalg.eigvalsh(Hp)
print("eigenvalues (9):", np.round(evp,4))
print("min eigenvalue:", evp.min())
