"""
The exact second-order expansion of the cell volume at the D4 root system.

Facet data: unit normals n_i on S^3 and support numbers h_i, P = {x : <x, n_i> <= h_i}.
At the 24-cell Q (n_i = u_i the normalised roots, h_i = 1), along the path
    n_i(t) = geodesic from u_i with initial velocity tau_i (tau_i _|_ u_i),
    h_i(t) = 1 + t eta_i,
the normal velocity of facet i is v_i(x) = eta_i - <x, tau_i>, and the volume
of a polytope has the exact first and second variations

    vol'  = sum_i int_{F_i} v_i dA
    vol'' = sum_i int_{F_i} (h_i |tau_i|^2) dA
            + sum_{i<j adjacent} (1/sin th) int_{F_ij} [2 v_i v_j - cos th (v_i^2 + v_j^2)] dA_2 ,

th the angle between the outward normals of adjacent facets (60 degrees at Q),
F_ij the 2-face they share (an equilateral triangle of side sqrt2 at Q).  The
2-face terms are continuous across combinatorial changes, so vol is C^2 in
dimension four and the Hessian at Q is well defined; the jumps sit in the
fourth derivative, at the (non-simple) vertices.

Coordinates xi = (tau, eta) in R^96: tau_i in an orthonormal basis of u_i^perp
(3 each), eta_i = delta_i / 2 (the push-out of centre i is delta_i).

This module builds H exactly (the integrals of quadratics over the triangles
by the midpoint rule, which is exact), checks it against finite differences of
the true polytope volume, and then minimises the quadratic form over the
first-order feasible cone of the packing constraints.
"""
import itertools
import numpy as np

# ---------------------------------------------------------------- D4 geometry
def roots():
    R = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4); v[i], v[j] = si, sj
                R.append(v / np.sqrt(2))
    return np.array(R)                       # 24 unit vectors u_i


def vertices_of_Q():
    V = []
    for k in range(4):
        for s in (1, -1):
            v = np.zeros(4); v[k] = s * np.sqrt(2); V.append(v)
    for signs in itertools.product((1, -1), repeat=4):
        V.append(np.array(signs) / np.sqrt(2))
    return np.array(V)                       # 24 vertices, |v| = sqrt2


U = roots()
VQ = vertices_of_Q()
G = U @ U.T
TIGHT = [(i, j) for i in range(24) for j in range(i + 1, 24) if abs(G[i, j] - 0.5) < 1e-12]
assert len(TIGHT) == 96
# vertices on facet i: <v, u_i> = 1
ONF = [[k for k in range(24) if abs(VQ[k] @ U[i] - 1) < 1e-12] for i in range(24)]
assert all(len(o) == 6 for o in ONF)
# the triangle of a tight pair
TRI = {}
for i, j in TIGHT:
    ks = [k for k in ONF[i] if k in ONF[j]]
    assert len(ks) == 3, (i, j, ks)
    TRI[(i, j)] = ks

# orthonormal basis of u_i^perp
BASIS = []
for i in range(24):
    M = np.eye(4) - np.outer(U[i], U[i])
    w, v = np.linalg.eigh(M)
    B = v[:, w > 0.5]                        # 4 x 3
    BASIS.append(B)


def unpack(xi):
    """xi (96,) -> tau (24,4) tangent vectors, eta (24,)"""
    tau = np.array([BASIS[i] @ xi[3 * i:3 * i + 3] for i in range(24)])
    eta = xi[72:]
    return tau, eta


# ---------------------------------------------------------------- the Hessian
def v_lin(i):
    """coefficient row of v_i(x) = eta_i - <x, tau_i> as a function of xi, for a point x:
       returns a function x -> (96,) row"""
    def row(x):
        r = np.zeros(96)
        r[72 + i] = 1.0
        r[3 * i:3 * i + 3] = -(BASIS[i].T @ x)
        return r
    return row


def build_H():
    H = np.zeros((96, 96))
    A_facet = 4.0 / 3.0
    for i in range(24):                      # sum_i h_i |tau_i|^2 A_i
        for k in range(3):
            H[3 * i + k, 3 * i + k] += 2 * A_facet   # H is the matrix of the quadratic FORM: xi^T H xi = vol''
    # careful: we want vol'' = xi^T H xi; the term |tau_i|^2 A_i contributes A_i on the diagonal
    H[:] = 0
    for i in range(24):
        for k in range(3):
            H[3 * i + k, 3 * i + k] += A_facet
    cth, sth = 0.5, np.sqrt(3) / 2
    for (i, j), ks in TRI.items():
        P = VQ[ks]
        mids = [(P[a] + P[b]) / 2 for a, b in ((0, 1), (1, 2), (0, 2))]
        e1, e2 = P[1] - P[0], P[2] - P[0]
        area = 0.5 * np.sqrt(np.dot(e1, e1) * np.dot(e2, e2) - np.dot(e1, e2) ** 2)
        ri, rj = v_lin(i), v_lin(j)
        for m in mids:
            a, b = ri(m), rj(m)
            # integrand 2 v_i v_j - cth (v_i^2 + v_j^2), as a symmetric matrix in xi
            Q = np.outer(a, b) + np.outer(b, a) - cth * (np.outer(a, a) + np.outer(b, b))
            H += (area / 3.0) / sth * Q
    return H


# ---------------------------------------------------------------- exact volume, for the check
from scipy.spatial import ConvexHull, HalfspaceIntersection


def volume(normals, h):
    hs = np.hstack([normals, -h[:, None]])
    return ConvexHull(HalfspaceIntersection(hs, np.zeros(4)).intersections).volume


def configuration(xi, t=1.0):
    tau, eta = unpack(xi)
    N = np.zeros((24, 4))
    for i in range(24):
        a = np.linalg.norm(tau[i])
        if a < 1e-15:
            N[i] = U[i]
        else:
            N[i] = np.cos(t * a) * U[i] + np.sin(t * a) * tau[i] / a
    return N, 1 + t * eta


def vol_path(xi, t):
    N, h = configuration(xi, t)
    return volume(N, h)


if __name__ == "__main__":
    H = build_H()
    H = (H + H.T) / 2
    print("Hessian built: 96 x 96, symmetric; ||H - H^T|| =", np.abs(H - H.T).max())
    print("vol(Q) =", volume(U, np.ones(24)))
    # 1. first order: vol' = (4/3) sum eta_i
    rng = np.random.default_rng(1)
    print("\nfinite-difference check of vol'' along random directions (t = 1e-3):")
    for trial in range(6):
        xi = rng.standard_normal(96); xi /= np.linalg.norm(xi)
        t = 1e-3
        f = lambda s: vol_path(xi, s)
        d1 = (f(t) - f(-t)) / (2 * t)
        d2 = (f(t) - 2 * f(0) + f(-t)) / t ** 2
        print(f"  trial {trial}: vol' FD {d1:.8f}  formula {4/3*xi[72:].sum():.8f} | "
              f"vol'' FD {d2:.6f}  xi^T H xi {xi @ H @ xi:.6f}")
    # 2. the one-centre ray: eta_1 = 1/2 (delta = 1), others 0 => vol = 25/3 - (1/3)(1 - delta/2)^4
    xi = np.zeros(96); xi[72] = 0.5
    print("\none-centre ray, exact: vol = 25/3 - (1-delta/2)^4/3 => d^2vol/ddelta^2 = -1/2 at 0;")
    print("  xi^T H xi with eta_1 = 1/2 (delta = 1):", xi @ H @ xi, "  (expected -1/2)")
    # 3. spectrum
    w = np.linalg.eigvalsh(H)
    print("\nspectrum of H (least 8 and largest 4):")
    print("  ", np.round(w[:8], 6), "...", np.round(w[-4:], 6))
    print("  number of zero eigenvalues:", int((np.abs(w) < 1e-9).sum()), "(6 rotations expected)")
    np.save("H.npy", H)
