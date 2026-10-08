"""
Probe: the hypothesis of Theorem 1.1 (thm:local-uncond) constrains only the
active neighbours (distance < 2 sqrt2).  Allow, besides 23 roots at distance 2
and one deviated neighbour at distance rho >= 2, one further neighbour y at
distance >= 2 sqrt2, all 26 centres pairwise >= 2 apart, and minimise the
volume of the true Voronoi cell of the centre.  Numerical exploration only.
"""
import itertools, math, sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
R = []
for i, j in itertools.combinations(range(4), 2):
    for si in (1, -1):
        for sj in (1, -1):
            v = np.zeros(4); v[i] = si; v[j] = sj; R.append(v / math.sqrt(2))
R = np.array(R); r0 = R[0]; W = R[1:]
Bperp = np.linalg.svd(r0[None])[2][1:]            # basis of r0-perp
S2 = 2 * math.sqrt(2)


def unpack(p):
    th, a, b, c, lr, y0, y1, y2, y3 = p
    e = Bperp.T @ np.array([a, b, c]); e /= np.linalg.norm(e) + 1e-300
    w = math.cos(th) * r0 + math.sin(th) * e
    rho = 2 + abs(lr)
    y = np.array([y0, y1, y2, y3])
    return w, rho, y


def volume(p):
    w, rho, y = unpack(p)
    centres = np.vstack([2 * W, rho * w, y])
    pen = 0.0
    ny = np.linalg.norm(y)
    pen += max(0.0, S2 - ny) ** 2
    Dm = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    iu = np.triu_indices(len(centres), 1)
    pen += (np.clip(2 - Dm[iu], 0, None) ** 2).sum()
    U = np.vstack([W, w, y / ny])
    h = np.r_[np.ones(23), rho / 2, ny / 2]
    try:
        P = HalfspaceIntersection(np.hstack([U, -h[:, None]]), np.zeros(4)).intersections
        v = ConvexHull(P).volume
    except Exception:
        return 1e3
    return v + 1e4 * pen, v, pen


def f(p):
    r = volume(p)
    return r if isinstance(r, float) else r[0]


best = None
for start in range(int(sys.argv[2]) if len(sys.argv) > 2 else 200):
    th = rng.uniform(0, 1.2)
    e = rng.normal(size=3)
    d = rng.normal(size=4); d /= np.linalg.norm(d)
    y = rng.uniform(S2, 3.3) * d
    p0 = np.r_[th, e, rng.uniform(0, 0.5), y]
    res = minimize(f, p0, method='Nelder-Mead',
                   options={'maxiter': 6000, 'xatol': 1e-9, 'fatol': 1e-12})
    r = volume(res.x)
    if isinstance(r, float):
        continue
    val, v, pen = r
    if pen < 1e-10 and (best is None or v < best[0]):
        best = (v, res.x)
        w, rho, yy = unpack(res.x)
        print('start %3d: true cell volume %.9f  (tilt %.4f rad, rho %.4f, |y| %.4f)'
              % (start, v, res.x[0], rho, np.linalg.norm(yy)), flush=True)
print('least true cell volume found: %.9f  (below 8: %s)' % (best[0], best[0] < 8 - 1e-9))
