import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

roots, labels = [], []
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v = np.zeros(4)
                v[i]=si; v[j]=sj
                roots.append(v/np.sqrt(2))
                sgn=lambda s: '+' if s==1 else '-'
                labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
roots=np.array(roots)

def find(vec):
    v=np.array(vec,dtype=float); v/=np.linalg.norm(v)
    idx=np.argmax(roots@v)
    assert (roots@v)[idx]>1-1e-9
    return idx

idx0=find([1,1,0,0]); u0=roots[idx0]
v1=roots[find([1,-1,0,0])]; w1=roots[find([0,0,1,1])]

def u1_vec(theta,t):
    return np.cos(theta)*u0+np.sin(theta)*(np.cos(t)*v1+np.sin(t)*w1)

def polytope(theta,t):
    u1=u1_vec(theta,t)
    dirs=roots.copy(); dirs[idx0]=u1
    hs=np.hstack([dirs,(-np.ones(len(dirs))).reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    return hi.intersections, dirs

theta0,t0 = 0.05, 0.2
verts, dirs = polytope(theta0,t0)
print("num vertices:", len(verts))

hull = ConvexHull(verts, qhull_options='QJ')
print("hull volume (scipy):", hull.volume)
print("num simplices (facets of the triangulated boundary):", len(hull.simplices))

# check: does (1/4)*sum over facets of origin-distance*3Darea match?
# scipy's hull.volume already computes exact volume via origin-based
# simplex decomposition internally, so let's just verify our expected
# formula: since all halfspaces have unit-norm normals with offset 1,
# distance from origin to each facet hyperplane is exactly 1.
# So vol = (1/4)*sum of 3D "areas" of the facets -- but easier: just
# directly compute the volume via summing signed 4-simplex volumes
# from origin over the hull.simplices triangulation (which already
# triangulates the FULL boundary into 3-simplices).
vol_manual = 0.0
for simplex in hull.simplices:
    pts = verts[simplex]  # 4 points (3-simplex living in a hyperplane)
    # 4-simplex = origin + these 4 points; signed volume = det(pts)/4!
    M = np.array(pts)  # 4x4 matrix (since ambient dim 4, this is a square)
    vol_manual += abs(np.linalg.det(M))/24.0
print("manual origin-cone decomposition volume:", vol_manual)
