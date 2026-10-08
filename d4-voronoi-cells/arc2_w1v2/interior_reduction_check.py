import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import minimize

roots = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4); v[i]=si; v[j]=sj
                roots.append(v/np.sqrt(2))
roots = np.array(roots)

def find(vec):
    vec = np.array(vec,dtype=float); vec/=np.linalg.norm(vec)
    d = roots@vec; idx=np.argmax(d)
    assert d[idx]>1-1e-9
    return idx

idx0 = find([1,1,0,0])
u0 = roots[idx0]
v1 = roots[find([1,-1,0,0])]
w1 = roots[find([0,0,1,1])]
v2 = roots[find([0,0,1,-1])]

def F_direct(theta, e_perp):
    e_perp = e_perp/np.linalg.norm(e_perp)
    u1 = np.cos(theta)*u0 + np.sin(theta)*e_perp
    dirs = roots.copy(); dirs[idx0]=u1
    hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    hull = ConvexHull(hi.intersections, qhull_options='QJ')
    return hull.volume - 8.0

basis3 = np.column_stack([v1,w1,v2])  # orthonormal basis of the 3D transverse space

def e_perp_of(phi1, phi2):
    v = np.array([np.sin(phi1)*np.cos(phi2), np.sin(phi1)*np.sin(phi2), np.cos(phi1)])
    return basis3 @ v

def F_of(theta, phi1, phi2):
    return F_direct(theta, e_perp_of(phi1, phi2))

def dist_to_arc_v1w1(vec):
    # great circle through v1,w1 has normal n = v1 x w1 (in the abc-coordinates, n = v2-direction)
    # in (a,b,c) coords relative to (v1,w1,v2) basis, this arc is c=0 great circle (v2-component=0)
    coords = basis3.T @ vec  # (a,b,c)
    coords /= np.linalg.norm(coords)
    return abs(coords[2])  # |c| = angular sine-distance (approx) from the v1-w1 (c=0) great circle

def dist_to_arc_w1v2(vec):
    coords = basis3.T @ vec
    coords /= np.linalg.norm(coords)
    return abs(coords[0])  # |a| = distance from the w1-v2 (a=0) great circle

def abc_of(vec):
    coords = basis3.T @ vec
    coords /= np.linalg.norm(coords)
    return coords

rng = np.random.default_rng(2026)
print(f"{'theta':>8} {'F_best':>12} {'|a|(distW1V2)':>14} {'|b|':>10} {'|c|(distV1W1)':>14}  in_triangle_abc>=0?")
thetas = [1.05, 1.09, 1.12, 1.15, 1.18, 1.20, 1.22, 1.24, 1.26, 1.30]
for theta in thetas:
    best = 1e9; best_params=None
    for _ in range(30):
        x0 = [rng.uniform(0,np.pi), rng.uniform(0,2*np.pi)]
        res = minimize(lambda p: F_of(theta,*p), x0, method='Nelder-Mead',
                        options={'xatol':1e-9,'fatol':1e-12,'maxiter':800,'maxfev':800})
        if res.fun < best:
            best = res.fun; best_params = res.x
    e_star = e_perp_of(*best_params)
    a,b,c = abc_of(e_star)
    dW1V2 = abs(a)  # distance from w1-v2 arc (a=0 plane)
    dV1W1 = abs(c)  # distance from v1-w1 arc (c=0 plane)
    in_oct = (a>=-1e-6 and b>=-1e-6 and c>=-1e-6) or (a<=1e-6 and b<=1e-6 and c<=1e-6) # up to global sign
    print(f"{theta:8.4f} {best:12.8f} {dW1V2:14.6f} {b:10.6f} {dV1W1:14.6f}  a,b,c=({a:.4f},{b:.4f},{c:.4f})")
