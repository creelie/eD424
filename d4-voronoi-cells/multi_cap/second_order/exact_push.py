"""
An exact lower bound with no expansion in the push-outs.

For unit normals w_i (any tilts) and support numbers h_i = 1 + eta_i, eta_i >= 0:

  vol{<x,w_i> <= 1+eta_i}  >=  vol(Q_w) + sum_i A_i^w (s_i/4) [1 - (1 - eta_i/s_i)^4],   (eta_i <= s_i)

where Q_w = {<x,w_j> <= 1} is the contact cell of the directions, A_i^w the
3-volume of its i-th facet, and s_i the height above facet i of the apex of the
i-th deletion {<x,w_j> <= 1, j != i}.

Proof.  Along h(t) = 1 + t eta, d/dt vol = sum_i eta_i A_i(t) with A_i(t) the
facet area; A_i(t) >= a_i(t eta_i) := area of the section of the deletion at
height t eta_i above facet i (the other support numbers are >= 1, and facets
grow with the other support numbers); by Brunn-Minkowski a_i(s)^{1/3} is
concave on [0, s_i] and vanishes at s_i, so a_i(s) >= (1 - s/s_i)^3 a_i(0);
integrate.  At the root system: vol >= 8 + (1/3) sum_i [1 - (1 - eta_i)^4],
sharp on every one-centre ray, and >= 8 + (2/3) S - (1/2) S^2 for eta_i <= 1.

This script checks the inequality on random tilted packings, and the
pure-push corollary.
"""
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
sys.path.insert(0, '.')
import hessian as Hm

U = Hm.U
rng = np.random.default_rng(21)


def hull_of(normals, h):
    hs = np.hstack([normals, -h[:, None]])
    return HalfspaceIntersection(hs, np.zeros(4))


def vol(normals, h):
    return ConvexHull(hull_of(normals, h).intersections).volume


def facet_area(normals, h, i):
    """3-volume of facet i: the vertices of the polytope lying on hyperplane i, projected"""
    pts = hull_of(normals, h).intersections
    on = pts[np.abs(pts @ normals[i] - h[i]) < 1e-9]
    if len(on) < 4:
        return 0.0
    # orthonormal basis of the hyperplane
    n = normals[i]; Q, _ = np.linalg.qr(np.c_[n, np.eye(4)]); Bs = Q[:, 1:4]
    P = (on - on.mean(0)) @ Bs
    try:
        return ConvexHull(P).volume
    except Exception:
        return 0.0


def apex_height(normals, i):
    """s_i = max <x, w_i> - 1 over the deletion {<x,w_j> <= 1, j != i}"""
    others = [j for j in range(24) if j != i]
    res = linprog(-normals[i], A_ub=normals[others], b_ub=np.ones(23), bounds=[(None, None)] * 4, method='highs')
    return -res.fun - 1.0


def packing_ok(N_, d):
    Y = N_ * d[:, None]
    D = np.linalg.norm(Y[:, None] - Y[None], axis=2) + 10 * np.eye(24)
    return D.min() >= 2 - 1e-12


print("pure-push corollary at the root system:  vol(1+eta) >= 8 + (1/3) sum [1-(1-eta_i)^4]")
worst = np.inf
for trial in range(200):
    eta = np.abs(rng.standard_normal(24)) ** rng.uniform(1, 5)
    eta *= rng.uniform(0.002, 0.9) / eta.sum()
    v = vol(U, 1 + eta)
    bound = 8 + np.sum(1 - (1 - eta) ** 4) / 3
    worst = min(worst, v - bound)
print(f"   200 random pushes, sum eta up to 0.9:  min(vol - bound) = {worst:+.3e}  (>= 0 required; 0 on one-centre rays)")

print("\nthe exact inequality with tilts:  vol >= vol(Q_w) + sum_i A_i (s_i/4)[1-(1-eta_i/s_i)^4]")
worst = np.inf; n_ok = 0
for trial in range(120):
    S = 10 ** rng.uniform(-2, -0.4)
    eta = np.abs(rng.standard_normal(24)) ** rng.uniform(1, 4); eta *= S / (2 * eta.sum())
    xi = np.zeros(96); xi[:72] = rng.standard_normal(72); xi[72:] = eta
    d = 2 * (1 + eta)
    for lam in np.geomspace(1.0, 1e-4, 50):
        x = xi.copy(); x[:72] *= lam * S
        W, h = Hm.configuration(x, 1.0)
        if packing_ok(W, d):
            break
    else:
        continue
    v = vol(W, h)
    vQ = vol(W, np.ones(24))
    A = np.array([facet_area(W, np.ones(24), i) for i in range(24)])
    s = np.array([apex_height(W, i) for i in range(24)])
    if np.any(eta > s):
        continue
    bound = vQ + np.sum(A * s / 4 * (1 - (1 - eta / s) ** 4))
    n_ok += 1
    worst = min(worst, v - bound)
print(f"   {n_ok} tilted packings: min(vol - bound) = {worst:+.3e}  (>= 0 required)")
print(f"   typical apex heights s_i at the root system: 1;  here between {s.min():.4f} and {s.max():.4f} on the last sample")
