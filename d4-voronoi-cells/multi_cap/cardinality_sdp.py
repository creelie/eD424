#!/usr/bin/env python3
"""
cardinality_sdp.py -- the Bachoc-Vallentin three-point bound for A(4, t), the
largest number of points of S^3 with pairwise inner products at most t, on the
ENLARGED domain t = 1/2 + s.  Floating point with sampled constraints and
refinement rounds (exploration); certificate mode fixes the bound and maximises
the least slack, for certify_cardinality.py to verify rigorously.

The bound.  With f(u) = sum_{k>=1} f_k G_k(u), f_k >= 0, G_k the Gegenbauer
polynomials of S^3, and F(u,v,w) = sum_k <F_k, S_k(u,v,w)>, F_k PSD, S_k the
symmetrised Bachoc-Vallentin matrices for n = 4 in the Chebyshev basis: if
    (i)   f(u) + 3 F(u,u,1) <= -1              for u in [-1, t],
    (ii)  F(u,v,w)          <= 0               for admissible (u,v,w) in [-1,t]^3,
then every code C on S^3 with inner products at most t has
    |C| <= 1 + f(1) + F(1,1,1).
(Sum the two positivity identities over C and split by coincidence pattern.)
If (i) holds only up to +e1 and (ii) only up to +e2, the bound becomes
    |C| <= 1 + f(1) + F(1,1,1) + (|C|-1) e1 + (|C|-1)(|C|-2) e2 ,
which, evaluated at |C| = 25, is what has to stay below 25.

Usage: python3 cardinality_sdp.py d t [rounds] [bound_for_certificate]
       python3 cardinality_sdp.py sweep d rounds t1 t2 ...     (the bound at several thresholds)
Certificates are written to cardinality_certificates/cert_d<d>_t<t>.npz; certify_cardinality.py
verifies them.  Exploration: floating point, sampled constraints.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvxpy as cp
from three_point_sdp import gegen, leg_coeffs, Smat

rng = np.random.default_rng(7)


def grid_1d(t, n):
    s = np.linspace(0, 1, n)
    return -1 + (t + 1) * (1 - (1 - s) ** 1.5)      # denser near t


def grid_3d(t, n_uv, n_w):
    s = np.linspace(0, 1, n_uv)
    g = t - (t + 1) * (1 - s) ** 1.5
    U, V, W = [], [], []
    for i, u in enumerate(g):
        for v in g[i:]:
            r = np.sqrt(max((1 - u * u) * (1 - v * v), 0.0))
            lo = max(u * v - r, v)
            hi = min(u * v + r, t)
            if hi < lo:
                continue
            for w in np.linspace(lo, hi, n_w):
                U.append(u); V.append(v); W.append(w)
    return np.array(U), np.array(V), np.array(W)


def random_3d(t, n):
    X = rng.normal(size=(3 * n, 3, 4)); X /= np.linalg.norm(X, axis=2, keepdims=True)
    u = np.einsum('ni,ni->n', X[:, 0], X[:, 1]); v = np.einsum('ni,ni->n', X[:, 0], X[:, 2])
    w = np.einsum('ni,ni->n', X[:, 1], X[:, 2])
    ok = (u <= t) & (v <= t) & (w <= t)
    Y = rng.normal(size=(n, 3, 3)); Y /= np.linalg.norm(Y, axis=2, keepdims=True)
    ub = np.einsum('ni,ni->n', Y[:, 0], Y[:, 1]); vb = np.einsum('ni,ni->n', Y[:, 0], Y[:, 2])
    wb = np.einsum('ni,ni->n', Y[:, 1], Y[:, 2])
    okb = (ub <= t) & (vb <= t) & (wb <= t)
    return np.r_[u[ok], ub[okb]], np.r_[v[ok], vb[okb]], np.r_[w[ok], wb[okb]]


class Rows:
    """coefficient rows of (i) and (ii) for given sample points"""
    def __init__(self, d, LC):
        self.d, self.LC = d, LC

    def rows_i(self, u):
        one = np.ones_like(u)
        Af = np.stack([gegen(k, u) for k in range(1, self.d + 1)], 1)
        AF = [3 * Smat(k, self.d, u, u, one, self.LC).reshape(len(u), -1) for k in range(self.d + 1)]
        return Af, AF

    def rows_ii(self, u, v, w):
        AF = [Smat(k, self.d, u, v, w, self.LC).reshape(len(u), -1) for k in range(self.d + 1)]
        return AF


def solve(d, t, U1, P3, margin_at=None, eps=1e-7, verbose=False):
    LC = [leg_coeffs(k) for k in range(d + 1)]
    R = Rows(d, LC)
    f = cp.Variable(d)                                   # f_1 .. f_d
    Fs = [cp.Variable((d - k + 1, d - k + 1), symmetric=True) for k in range(d + 1)]
    Af, AF1 = R.rows_i(U1)
    e1 = Af @ f + sum(AF1[k] @ cp.vec(Fs[k], order='C') for k in range(d + 1))
    AF2 = R.rows_ii(*P3)
    e2 = sum(AF2[k] @ cp.vec(Fs[k], order='C') for k in range(d + 1))
    one = np.array([1.0])
    S111 = [Smat(k, d, one, one, one, LC)[0] for k in range(d + 1)]
    F111 = sum(cp.sum(cp.multiply(S111[k], Fs[k])) for k in range(d + 1))
    bound = 1 + cp.sum(f) + F111                         # G_k(1) = 1
    if margin_at is None:
        cons = [e1 <= -1, e2 <= 0, f >= 0] + [F >> 0 for F in Fs]
        prob = cp.Problem(cp.Minimize(bound), cons)
    else:
        s = cp.Variable()
        cons = [e1 + s <= -1, e2 + s <= 0, f >= eps, bound <= margin_at] + \
               [F - eps * np.eye(F.shape[0]) >> 0 for F in Fs]
        prob = cp.Problem(cp.Maximize(s), cons)
    t0 = time.time()
    opts = dict(tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7, max_iter=500,
                static_regularization_constant=1e-6, equilibrate_max_iter=50, tol_ktratio=1e-6)
    if margin_at is not None:
        opts.update(static_regularization_constant=1e-7, equilibrate_max_iter=50,
                    tol_gap_abs=1e-6, tol_gap_rel=1e-6)
    try:
        prob.solve(solver='CLARABEL', verbose=verbose, **opts)
    except cp.error.SolverError:
        print('  (Clarabel failed, trying SCS)', flush=True)
        prob.solve(solver='SCS', eps=1e-9, max_iters=200000, verbose=verbose)
    fv, Fv = f.value, [F.value for F in Fs]
    return prob.status, prob.value, fv, Fv, time.time() - t0, R


def check(R, t, fv, Fv, n_rand=300000, n_keep=(1500, 3000)):
    """largest violations of (i) and (ii) on fine sets; worst points to add"""
    d = R.d
    u = grid_1d(t, 20001)
    Af, AF1 = R.rows_i(u)
    v1 = Af @ fv + sum(AF1[k] @ Fv[k].reshape(-1) for k in range(d + 1)) + 1
    w1 = float(v1.max()); keep1 = u[np.argsort(v1)[-n_keep[0]:]]
    worst2 = -np.inf; keep2 = []
    for (uu, vv, ww) in (grid_3d(t, 120, 60), random_3d(t, n_rand)):
        for lo in range(0, len(uu), 100000):
            a, b, c = uu[lo:lo + 100000], vv[lo:lo + 100000], ww[lo:lo + 100000]
            AF2 = R.rows_ii(a, b, c)
            v2 = sum(AF2[k] @ Fv[k].reshape(-1) for k in range(d + 1))
            worst2 = max(worst2, float(v2.max()))
            idx = np.argsort(v2)[-n_keep[1]:]
            keep2.append((v2[idx], a[idx], b[idx], c[idx]))
    vi = np.concatenate([k[0] for k in keep2]); order = np.argsort(vi)[-n_keep[1]:]
    K2 = tuple(np.concatenate([k[j] for k in keep2])[order] for j in (1, 2, 3))
    return w1, keep1, worst2, K2


def run(d, t, rounds=4, margin=None, n1=400, n_uv=30, n_w=20, quiet=False):
    U1 = grid_1d(t, n1)
    P3 = grid_3d(t, n_uv, n_w)
    out = None
    for r in range(rounds):
        status, val, fv, Fv, el, R = solve(d, t, U1, P3, margin_at=margin)
        w1, K1, w2, K2 = check(R, t, fv, Fv)
        if margin is None:
            # bound valid over the fine sets, evaluated at |C| = 25
            b25 = val + 24 * max(w1, 0) + 24 * 23 * max(w2, 0)
            if not quiet:
                print(f"  d={d} t={t:.5f} round {r+1}: sampled bound {val:.6f} ({len(U1)}+{len(P3[0])} pts, {el:.0f}s) "
                      f"viol (i) {w1:.1e} (ii) {w2:.1e}  -> bound at |C|=25 corrected: {b25:.6f}", flush=True)
            out = (val, w1, w2, b25, fv, Fv)
        else:
            if not quiet:
                print(f"  d={d} t={t:.5f} certificate round {r+1}: least slack on sample {val:.3e} ({el:.0f}s); "
                      f"on fine check: (i) {-w1-1:.3e}  (ii) {-w2:.3e}", flush=True)
            out = (val, w1, w2, fv, Fv)
        U1 = np.r_[U1, K1]
        P3 = tuple(np.r_[P3[j], K2[j]] for j in range(3))
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'sweep':
        d = int(sys.argv[2]); rounds = int(sys.argv[3])
        for t in [float(x) for x in sys.argv[4:]]:
            val, w1, w2, b25, fv, Fv = run(d, t, rounds, n1=300, n_uv=26, n_w=16)
            print(f"RESULT d={d} t={t:.4f} s={t-0.5:.4f}  sampled {val:.5f}  corrected@25 {b25:.5f}", flush=True)
        sys.exit(0)
    d = int(sys.argv[1]); t = float(sys.argv[2])
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    margin = float(sys.argv[4]) if len(sys.argv) > 4 else None
    out = run(d, t, rounds, margin)
    if margin is not None:
        val, w1, w2, fv, Fv = out
        np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cardinality_certificates', f'cert_d{d}_t{t:.5f}.npz'),
                 f=fv, t=t, bound=margin, slack=val,
                 **{f'F{k}': Fv[k] for k in range(d + 1)})
        print('saved')
