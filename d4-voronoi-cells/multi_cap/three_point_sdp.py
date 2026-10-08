#!/usr/bin/env python3
"""
three_point_sdp.py -- the three-point (Bachoc-Vallentin) relaxation of the
pair inequality of thm:strict-reduction, and the certificate of thm:certificate.

The question.  For twenty-three contact directions w_1, ..., w_23 (pairwise
inner products at most 1/2) write u_ij = <w_i, w_j> and
    E(W) = sum_{i<j} omega(u_ij),
where omega(u) = Pi(1, 1, u) is the pair term of two contacts (eq:omega of
the paper): omega(1/2) = 0.00144541 at sixty degrees, decreasing to zero at
u = 1/3 (70.53 degrees), and zero below.  By eq:Astar the m = 23 case
follows from
    E(W) >= 8 - A_* = 0.092855570   for every W,
and the pair-angle relaxation reaches only 0.0738, about four fifths of
this target (three_point_reduction.py).  This script computes the next relaxation in the
hierarchy, the one that sees the triples, and produces the certificate
that certificate_check.py verifies (thm:certificate, lem:three-point-psd).

The certificate.  Let G_k be the Gegenbauer polynomials of S^3, G_k(cos t)
= sin((k+1)t) / ((k+1) sin t), and let S_k(u, v, t) be the matrices of
Bachoc and Vallentin [BV08] for n = 4 (Legendre polynomials in the
inner argument, and the Chebyshev basis T_i(u) T_j(v) in place of
u^i v^j, which changes nothing but the conditioning), symmetrised over
the six orderings of (u, v, t).  For
f(u) = sum_k f_k G_k(u) with f_k >= 0 for k >= 1 and F(u, v, t) = sum_k
<F_k, S_k(u, v, t)> with F_k positive semidefinite, every finite W on S^3
satisfies sum_{i,j} f(u_ij) >= N^2 f_0 and sum_{(i,j,k)} F(u_ij, u_ik,
u_jk) >= 0, the second sum over all ordered triples, coincidences
included.  Splitting the triple sum by coincidence pattern, if
    omega(u) + omega(v) + omega(t) - f(u) - f(v) - f(t) - F(u, v, t)
      - (F(1,u,u) + F(1,v,v) + F(1,t,t)) / (N - 2)  >=  0                (cert)
for every admissible triple (u, v, t) with u, v, t in [-1, 1/2] (that is,
1 + 2uvt - u^2 - v^2 - t^2 >= 0), then
    E(W) >= N (N f_0 - f(1)) / 2 - N F(1,1,1) / (6 (N - 2)).
In its first mode the script maximises this bound.  The condition (cert)
is imposed on a grid of admissible triples and then checked on a far
finer set; the worst points of the check are added to the grid and the
programme is solved again, for the given number of rounds; the reported
bound is reduced by (N choose 3) / (N - 2) times the largest violation
found, so that it is a bound over the finer set.  Nothing in this mode
is a proof: the conditions are sampled, not certified.  What it measures
is whether the three-point relaxation has room to reach the target,
and at degree 8 it has (0.09523 against 0.09286).

In its second mode, with a fifth argument giving the bound, the bound
is fixed there and the least slack of (cert) over the sample is maximised,
with F_k - 1e-7 I and f_k - 1e-7 positive semidefinite, and the result
is written to continuation_out/certificate_d<degree>.npz.  That file is
what certificate_check.py verifies in exact and interval arithmetic,
and what lean/gen_certificate_lean.py turns into a Lean file.

Usage: python3 three_point_sdp.py [degree] [grid per axis] [solver] [rounds] [bound]
  e.g. python3 three_point_sdp.py 8 30 CLARABEL 5          (relaxation values)
       python3 three_point_sdp.py 8 30 CLARABEL 5 0.0929   (the certificate)
"""
import sys, time
import numpy as np
from itertools import permutations
from numpy.polynomial import legendre as LEG
import cvxpy as cp
import three_point_reduction as TPR

N = 23
SCALE = 1000.0                      # omega is of order 1e-3; the solver sees SCALE * omega
R_STAR = np.arcsin(np.sqrt(1.0 / 3.0))
U_MIN = np.cos(2 * R_STAR)          # 1/3: omega vanishes below

def omega_table(n=400):
    gam = np.linspace(np.pi / 3, 2 * R_STAR, n)
    vals = np.array([TPR.omega(g, R_STAR) for g in gam])
    return np.cos(gam)[::-1], vals[::-1]

class Omega:
    def __init__(self):
        self.u, self.v = omega_table()
    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        out = np.interp(u, self.u, self.v, left=0.0, right=self.v[-1])
        return np.where(u < U_MIN, 0.0, out)

def gegen(k, u):
    """G_k on S^3: U_k(u)/(k+1) by the Chebyshev recurrence."""
    u = np.asarray(u, dtype=float)
    a, b = np.ones_like(u), 2 * u
    if k == 0: return a
    for _ in range(k - 1):
        a, b = b, 2 * u * b - a
    return b / (k + 1)

def leg_coeffs(k):
    c = np.zeros(k + 1); c[k] = 1.0
    return LEG.leg2poly(c)             # monomial coefficients of P_k

def phi(k, u, v, t, LC):
    """((1-u^2)(1-v^2))^{k/2} P_k((t - uv)/sqrt((1-u^2)(1-v^2))), as a polynomial."""
    a = LC[k]; x = t - u * v; s2 = (1 - u * u) * (1 - v * v)
    out = np.zeros_like(x)
    for m in range(k % 2, k + 1, 2):
        out = out + a[m] * x ** m * s2 ** ((k - m) // 2)
    return out

def cheb(u, n):
    """T_0(u), ..., T_{n-1}(u): the basis in which the matrices Y_k are written
    (a congruence away from the monomial basis u^i of [BV08], and far better
    conditioned on [-1, 1/2])."""
    u = np.asarray(u, dtype=float)
    out = [np.ones_like(u), u]
    for _ in range(n - 2): out.append(2 * u * out[-1] - out[-2])
    return np.stack(out[:n], axis=-1)

def Ymat(k, d, u, v, t, LC):
    """Y_k(u,v,t)[i,j] = T_i(u) T_j(v) phi_k, i,j = 0..d-k, for arrays of points."""
    n = d - k + 1
    ui = cheb(u, n); vj = cheb(v, n)
    return phi(k, u, v, t, LC)[..., None, None] * ui[..., :, None] * vj[..., None, :]

def Smat(k, d, u, v, t, LC):
    acc = 0
    for p in permutations((u, v, t)):
        acc = acc + Ymat(k, d, *p, LC)
    return acc / 6.0

def sample_grid(n_uv, n_t):
    """Admissible triples with u <= v <= t <= 1/2, u,v on a grid denser near 1/2."""
    s = np.linspace(0, 1, n_uv)
    g = 0.5 - 1.5 * (1 - s) ** 1.5        # from -1 to 1/2, denser near 1/2
    U, V, T = [], [], []
    for i, u in enumerate(g):
        for v in g[i:]:
            lo = max(u * v - np.sqrt((1 - u * u) * (1 - v * v)), v)
            hi = min(u * v + np.sqrt((1 - u * u) * (1 - v * v)), 0.5)
            if hi < lo: continue
            for t in np.linspace(lo, hi, n_t):
                U.append(u); V.append(v); T.append(t)
    return np.array(U), np.array(V), np.array(T)

def sample_random(n, rng):
    """Random admissible triples: three random points of S^3 conditioned on
    pairwise inner products at most 1/2, plus random boundary (coplanar) ones."""
    X = rng.normal(size=(3 * n, 3, 4)); X /= np.linalg.norm(X, axis=2, keepdims=True)
    u = np.einsum('ni,ni->n', X[:, 0], X[:, 1]); v = np.einsum('ni,ni->n', X[:, 0], X[:, 2])
    t = np.einsum('ni,ni->n', X[:, 1], X[:, 2])
    ok = (u <= .5) & (v <= .5) & (t <= .5)
    Y = rng.normal(size=(n, 3, 3)); Y /= np.linalg.norm(Y, axis=2, keepdims=True)
    ub = np.einsum('ni,ni->n', Y[:, 0], Y[:, 1]); vb = np.einsum('ni,ni->n', Y[:, 0], Y[:, 2])
    tb = np.einsum('ni,ni->n', Y[:, 1], Y[:, 2])
    okb = (ub <= .5) & (vb <= .5) & (tb <= .5)
    return (np.r_[u[ok], ub[okb]], np.r_[v[ok], vb[okb]], np.r_[t[ok], tb[okb]])

def constraint_rows(d, u, v, t, LC, om):
    """Rows of (cert): the omega part, the coefficient vectors of f_k, and of vec(F_k)."""
    rhs = SCALE * (om(u) + om(v) + om(t))
    Af = np.stack([gegen(k, u) + gegen(k, v) + gegen(k, t) for k in range(d + 1)], axis=1)
    AF = []
    one = np.ones_like(u)
    for k in range(d + 1):
        M = Smat(k, d, u, v, t, LC)
        M = M + (Smat(k, d, one, u, u, LC) + Smat(k, d, one, v, v, LC) + Smat(k, d, one, t, t, LC)) / (N - 2)
        AF.append(M.reshape(len(u), -1))
    return rhs, Af, AF

def solve(d, u, v, t, solver, om, LC, three_point=True, margin_at=None, eps=1e-7):
    """Maximise the bound; or, with margin_at = B given, fix the bound at B
    (in the units of the solver) and maximise the least slack of (cert) over
    the sample, with F_k - eps I and f_k - eps (k >= 1) positive semidefinite
    so that the rounded certificate stays strictly feasible."""
    rhs, Af, AF = constraint_rows(d, u, v, t, LC, om)
    f = cp.Variable(d + 1)
    Fs = [cp.Variable((d - k + 1, d - k + 1), symmetric=True) for k in range(d + 1)]
    expr = Af @ f
    if three_point:
        for k in range(d + 1):
            expr = expr + AF[k] @ cp.vec(Fs[k], order='C')
    S111 = [Smat(k, d, np.array([1.0]), np.array([1.0]), np.array([1.0]), LC)[0] for k in range(d + 1)]
    F111 = sum(cp.sum(cp.multiply(S111[k], Fs[k])) for k in range(d + 1)) if three_point else 0
    bound = N * (N * f[0] - cp.sum(f)) / 2 - N * F111 / (6 * (N - 2))
    if margin_at is None:
        cons = [expr <= rhs, f[1:] >= 0] + [F >> 0 for F in Fs]
        prob = cp.Problem(cp.Maximize(bound), cons)
    else:
        s = cp.Variable()
        cons = [expr + s <= rhs, f[1:] >= eps, bound >= margin_at] + \
               [F - eps * np.eye(F.shape[0]) >> 0 for F in Fs]
        prob = cp.Problem(cp.Maximize(s), cons)
    t0 = time.time()
    opts = {}
    if solver == 'CLARABEL' and margin_at is not None:
        # the maximin problem is degenerate; a little static regularisation keeps the interior-point method on course
        opts = {'tol_gap_abs': 1e-5, 'tol_gap_rel': 1e-4, 'tol_feas': 1e-7, 'max_iter': 500,
                'static_regularization_constant': 1e-6, 'equilibrate_max_iter': 50}
    try:
        prob.solve(solver=solver, verbose=False, **opts)
    except cp.error.SolverError:
        print(f"  ({solver} failed; solving with SCS)", flush=True)
        prob.solve(solver='SCS', eps=1e-9, max_iters=200000, verbose=False)
    el = time.time() - t0
    fv = f.value; Fv = [F.value for F in Fs] if three_point else None
    return prob.value, fv, Fv, el

def check(d, fv, Fv, LC, om, rng, n_random=400000, n_uv=120, n_t=60, n_keep=3000):
    """Largest violation of (cert) over a fine grid and a random sample, and the
    n_keep worst points, to be added to the constraint set."""
    worst = -np.inf; keep = []
    for (u, v, t) in (sample_grid(n_uv, n_t), sample_random(n_random, rng)):
        for lo in range(0, len(u), 100000):
            uu, vv, tt = u[lo:lo + 100000], v[lo:lo + 100000], t[lo:lo + 100000]
            rhs, Af, AF = constraint_rows(d, uu, vv, tt, LC, om)
            val = Af @ fv
            if Fv is not None:
                for k in range(d + 1):
                    val = val + AF[k] @ Fv[k].reshape(-1)
            viol = val - rhs
            worst = max(worst, float(np.max(viol)))
            idx = np.argsort(viol)[-n_keep:]
            keep.append((viol[idx], uu[idx], vv[idx], tt[idx]))
    vi = np.concatenate([k[0] for k in keep]); order = np.argsort(vi)[-n_keep:]
    U = np.concatenate([k[1] for k in keep])[order]; V = np.concatenate([k[2] for k in keep])[order]
    T = np.concatenate([k[3] for k in keep])[order]
    return worst, (U, V, T)

def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_uv = int(sys.argv[2]) if len(sys.argv) > 2 else 36
    solver = sys.argv[3] if len(sys.argv) > 3 else 'CLARABEL'
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    margin = float(sys.argv[5]) if len(sys.argv) > 5 else None
    n_t = 24
    om = Omega(); rng = np.random.default_rng(1)
    A_star = TPR.Aconst(N, R_STAR); target = 8 - A_star
    LC = [leg_coeffs(k) for k in range(d + 1)]
    print(f"target 8 - A_* = {target:.9f}; omega(1/2) = {om(0.5):.8f}; degree {d}; grid {n_uv} x {n_t}; "
          f"solver {solver}; {rounds} rounds of refinement")
    if margin is not None:
        # certificate mode: the bound is fixed at `margin` and the least slack of (cert) is maximised
        print(f"certificate mode: bound fixed at {margin:.9f}, the least slack of (cert) over the sample is maximised")
        u, v, t = sample_grid(n_uv, n_t)
        for r in range(rounds):
            val, fv, Fv, el = solve(d, u, v, t, solver, om, LC, three_point=True, margin_at=margin * SCALE)
            worst, (U, V, T) = check(d, fv, Fv, LC, om, rng)
            print(f"  round {r + 1}: least slack on the sample {val / SCALE:.3e} ({len(u)} triples, {el:.0f}s); "
                  f"least slack on the fine check {-worst / SCALE:.3e}", flush=True)
            np.savez(f"continuation_out/certificate_d{d}.npz", f=fv, bound=margin, slack=val / SCALE,
                     **{f"F{k}": Fv[k] for k in range(d + 1)})
            u, v, t = np.r_[u, U], np.r_[v, V], np.r_[t, T]
        print("certificate saved; certificate_check.py verifies it in exact and interval arithmetic")
        return
    for three in (False, True):
        kind = "three-point" if three else "two-point"
        u, v, t = sample_grid(n_uv, n_t)
        for r in range(rounds):
            val, fv, Fv, el = solve(d, u, v, t, solver, om, LC, three_point=three)
            worst, (U, V, T) = check(d, fv, Fv, LC, om, rng)
            corr = worst * (N * (N - 1) * (N - 2) / 6) / (N - 2)
            b = (val - corr) / SCALE
            print(f"  {kind}, round {r + 1}: sampled bound {val / SCALE:.6f} on {len(u)} triples ({el:.0f}s); "
                  f"largest violation on the fine check {worst / SCALE:.2e}; "
                  f"bound over the fine set {b:.6f} = {100 * b / target:.1f} per cent of the target", flush=True)
            u, v, t = np.r_[u, U], np.r_[v, V], np.r_[t, T]
        if three:
            np.savez(f"continuation_out/three_point_sdp_d{d}.npz", f=fv, **{f"F{k}": Fv[k] for k in range(d + 1)})
    print("(sampled conditions, refined by the worst points of each check; an indication of the strength "
          "of the relaxation, not a proof)")

if __name__ == "__main__":
    main()
