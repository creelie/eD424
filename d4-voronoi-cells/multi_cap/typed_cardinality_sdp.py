#!/usr/bin/env python3
"""
typed_cardinality_sdp.py -- a three-point bound for codes of two types on S^3:
n points of type 1 with pairwise inner products at most t1, and one further
point of type 2 with inner product at most t2 >= t1 with each of them.  The
programme looks for a certificate that no such code exists (exploration,
floating point, sampled constraints refined in rounds; typed_cardinality_check.py
proves the certificate it writes).

The identity.  With f(u) = sum_{k>=1} f_k G_k(u), f_k >= 0, and
F(u,v,w) = sum_k <F_k, S_k(u,v,w)>, F_k positive semidefinite (the notation of
cardinality_sdp.py), every finite C on S^3 satisfies

    |C| A + sum_{x != y} g(u_xy) + sum_{x,y,z distinct} F(u_xy, u_xz, u_yz) >= 0,

A = f(1) + F(1,1,1), g(u) = f(u) + 3 F(u,u,1), the sums over ordered pairs and
triples.  For a code of n + 1 points of the two types, split the pairs and the
triples by how many of their points are of type 2:

    n(n-1) ordered pairs of type 11, inner product in [-1, t1];
    2n                      of type 12, inner product in [-1, t2];
    n(n-1)(n-2) ordered triples of type 111, all three inner products <= t1;
    3n(n-1)                 of type 112, two inner products <= t2 (those at
                            the type-2 point) and the third <= t1.

If g <= -1 on [-1, t1], g <= a2 on [-1, t2], F <= b1 on the admissible triples
of type 111 and F <= b2 on those of type 112, then

    Z = (n+1) A - n(n-1) + 2n a2 + n(n-1)(n-2) b1 + 3n(n-1) b2 >= 0

for every such code; a certificate with Z < 0 proves that none exists.

Usage: python3 typed_cardinality_sdp.py d t1 t2 [rounds] [n] [save.npz]
       python3 typed_cardinality_sdp.py sweep d rounds t1 t2a t2b ...
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cvxpy as cp
from three_point_sdp import leg_coeffs, Smat, cheb, phi, gegen
from cardinality_sdp import Rows, grid_1d

rng = np.random.default_rng(11)


def grid_112(t1, t2, n_uv, n_w):
    """admissible (u, v, w) with u <= v <= t2 (at the type-2 point) and w <= t1."""
    s = np.linspace(0, 1, n_uv)
    g = t2 - (t2 + 1) * (1 - s) ** 1.5
    U, V, W = [], [], []
    for i, u in enumerate(g):
        for v in g[i:]:
            r = np.sqrt(max((1 - u * u) * (1 - v * v), 0.0))
            lo, hi = max(u * v - r, -1.0), min(u * v + r, t1)
            if hi < lo:
                continue
            for w in np.linspace(lo, hi, n_w):
                U.append(u); V.append(v); W.append(w)
    return np.array(U), np.array(V), np.array(W)


def grid_111(t1, n_uv, n_w):
    s = np.linspace(0, 1, n_uv)
    g = t1 - (t1 + 1) * (1 - s) ** 1.5
    U, V, W = [], [], []
    for i, u in enumerate(g):
        for v in g[i:]:
            r = np.sqrt(max((1 - u * u) * (1 - v * v), 0.0))
            lo, hi = max(u * v - r, v), min(u * v + r, t1)
            if hi < lo:
                continue
            for w in np.linspace(lo, hi, n_w):
                U.append(u); V.append(v); W.append(w)
    return np.array(U), np.array(V), np.array(W)


def random_triples(n, t_uv, t_w):
    """random admissible triples with u, v <= t_uv and w <= t_w: points of S^3 and of S^2
    (the coplanar boundary), and triples near the corner u = v = t_uv."""
    out = [[], [], []]
    for dim in (4, 3):
        X = rng.normal(size=(n, 3, dim)); X /= np.linalg.norm(X, axis=2, keepdims=True)
        u = np.einsum('ni,ni->n', X[:, 0], X[:, 1]); v = np.einsum('ni,ni->n', X[:, 0], X[:, 2])
        w = np.einsum('ni,ni->n', X[:, 1], X[:, 2])
        ok = (u <= t_uv) & (v <= t_uv) & (w <= t_w)
        for j, a in enumerate((u, v, w)):
            out[j].append(a[ok])
    a = t_uv - 0.05 * rng.random(n) ** 2; b = t_uv - 0.05 * rng.random(n) ** 2
    ph = rng.uniform(0, np.pi, n)
    w = a * b + np.sqrt((1 - a * a) * (1 - b * b)) * np.cos(ph)
    ok = w <= t_w
    for j, x in enumerate((a, b, w)):
        out[j].append(x[ok])
    return tuple(np.concatenate(o) for o in out)


def F_values(Fv, d, LC, u, v, w):
    """F(u,v,w) = sum_k <F_k, S_k(u,v,w)> at arrays of points, without forming the rows:
    with F_k symmetric the six permutations of S_k pair off by the third argument."""
    out = np.zeros_like(u)
    for (a, b, c) in ((u, v, w), (u, w, v), (v, w, u)):
        for k in range(d + 1):
            n = d - k + 1
            out += phi(k, a, b, c, LC) * np.einsum('ni,ij,nj->n', cheb(a, n), Fv[k], cheb(b, n))
    return out / 3.0


def g_values(fv, Fv, d, LC, u):
    """g(u) = f(u) + 3 F(u,u,1)."""
    f = sum(fv[k - 1] * gegen(k, u) for k in range(1, d + 1))
    return f + 3 * F_values(Fv, d, LC, u, u, np.ones_like(u))


def climb(Fv, d, LC, u, v, w, t_uv, t_w, steps=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4), tries=16, reps=4):
    """local maxima of F over the admissible triples with u, v <= t_uv and w <= t_w, by random
    hill climbing from the given points in the coordinates (u, v, phi), w = uv + r cos phi,
    r = sqrt((1-u^2)(1-v^2)); a move that leaves the domain is clipped onto its boundary."""
    def to_w(a, b, ph):
        return a * b + np.sqrt(np.clip((1 - a * a) * (1 - b * b), 0, None)) * np.cos(ph)

    def project(a, b, ph):
        a = np.clip(a, -1, t_uv); b = np.clip(b, -1, t_uv); ph = np.clip(ph, 0, np.pi)
        r = np.sqrt(np.clip((1 - a * a) * (1 - b * b), 0, None))
        ww = a * b + r * np.cos(ph)
        over = ww > t_w
        if over.any():                       # move phi up to the plane w = t_w where possible
            c = np.where(r[over] > 1e-12, (t_w - a[over] * b[over]) / np.maximum(r[over], 1e-12), -1.0)
            ph = ph.copy(); ph[over] = np.arccos(np.clip(c, -1, 1))
            ww = to_w(a, b, ph)
        return a, b, ph, ww

    r = np.sqrt(np.clip((1 - u * u) * (1 - v * v), 1e-300, None))
    ph = np.arccos(np.clip((w - u * v) / r, -1, 1))
    a, b, ph, ww = project(u.copy(), v.copy(), ph)
    best = F_values(Fv, d, LC, a, b, ww)
    m = len(a)
    for st in steps:
        for _ in range(reps):
            D = rng.normal(size=(tries, m, 3)) * st
            ca, cb, cph, cw = project((a + D[..., 0]).ravel(), (b + D[..., 1]).ravel(), (ph + D[..., 2]).ravel())
            ok = cw <= t_w + 1e-15
            val = np.where(ok, F_values(Fv, d, LC, ca, cb, cw), -np.inf).reshape(tries, m)
            j = val.argmax(0); vbest = val[j, np.arange(m)]
            up = vbest > best
            idx = j * m + np.arange(m)
            a = np.where(up, ca[idx], a); b = np.where(up, cb[idx], b)
            ph = np.where(up, cph[idx], ph); best = np.where(up, vbest, best)
    return best, (a, b, to_w(a, b, ph))


def solve(d, t1, t2, n, U1, U2, P111, P112):
    LC = [leg_coeffs(k) for k in range(d + 1)]
    R = Rows(d, LC)
    f = cp.Variable(d)
    Fs = [cp.Variable((d - k + 1, d - k + 1), symmetric=True) for k in range(d + 1)]
    a2, b1, b2 = cp.Variable(), cp.Variable(), cp.Variable()
    Af, AF = R.rows_i(U1)
    g1 = Af @ f + sum(AF[k] @ cp.vec(Fs[k], order='C') for k in range(d + 1))
    Af, AF = R.rows_i(U2)
    g2 = Af @ f + sum(AF[k] @ cp.vec(Fs[k], order='C') for k in range(d + 1))
    AF = R.rows_ii(*P111)
    F1 = sum(AF[k] @ cp.vec(Fs[k], order='C') for k in range(d + 1))
    AF = R.rows_ii(*P112)
    F2 = sum(AF[k] @ cp.vec(Fs[k], order='C') for k in range(d + 1))
    one = np.array([1.0])
    S111 = [Smat(k, d, one, one, one, LC)[0] for k in range(d + 1)]
    A = cp.sum(f) + sum(cp.sum(cp.multiply(S111[k], Fs[k])) for k in range(d + 1))
    Z = (n + 1) * A - n * (n - 1) + 2 * n * a2 + n * (n - 1) * (n - 2) * b1 + 3 * n * (n - 1) * b2
    cons = [g1 <= -1, g2 <= a2, F1 <= b1, F2 <= b2, f >= 0, A <= 30] + [F >> 0 for F in Fs]
    prob = cp.Problem(cp.Minimize(Z), cons)
    t0 = time.time()
    opts = dict(tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7, max_iter=500,
                static_regularization_constant=1e-6, equilibrate_max_iter=50, tol_ktratio=1e-6)
    try:
        prob.solve(solver='CLARABEL', **opts)
    except cp.error.SolverError:
        print('  (Clarabel failed, trying SCS)', flush=True)
        prob.solve(solver='SCS', eps=1e-9, max_iters=200000)
    return prob, f.value, [F.value for F in Fs], (a2.value, b1.value, b2.value), R, time.time() - t0


def evaluate(R, fv, Fv, t1, t2, n, n_climb=600):
    """the largest values of g and F on fine sets and by local climbing, the corrected Z, and the
    points to add to the samples."""
    d = R.d; LC = R.LC
    out = {}
    for name, lo, hi in (('g1', -1, t1), ('g2', -1, t2)):
        u = np.r_[np.linspace(lo, hi, 80001), hi]
        val = g_values(fv, Fv, d, LC, u)
        out[name] = (float(val.max()), u[np.argsort(val)[-800:]])
    for name, t_uv, t_w, P in (('F1', t1, t1, (grid_111(t1, 90, 40), random_triples(250000, t1, t1))),
                               ('F2', t2, t1, (grid_112(t1, t2, 90, 40), random_triples(250000, t2, t1)))):
        vals, pts = [], [[], [], []]
        for (uu, vv, ww) in P:
            for s0 in range(0, len(uu), 50000):
                a, b, c = uu[s0:s0 + 50000], vv[s0:s0 + 50000], ww[s0:s0 + 50000]
                val = F_values(Fv, d, LC, a, b, c)
                idx = np.argsort(val)[-3000:]
                vals.append(val[idx]); pts[0].append(a[idx]); pts[1].append(b[idx]); pts[2].append(c[idx])
        vals = np.concatenate(vals); pts = [np.concatenate(x) for x in pts]
        order = np.argsort(vals)[::-1]
        top = order[:n_climb]
        cv, cp_ = climb(Fv, d, LC, pts[0][top], pts[1][top], pts[2][top], t_uv, t_w)
        best = max(float(vals.max()), float(cv.max()))
        keep = order[:3000]
        add = tuple(np.r_[pts[j][keep], cp_[j]] for j in range(3))
        out[name] = (best, add)
    one = np.array([1.0])
    A = float(np.sum(fv)) + sum(float(np.sum(Smat(k, d, one, one, one, LC)[0] * Fv[k])) for k in range(d + 1))
    g1 = out['g1'][0]
    # Z with the fine maxima, g1 entering through its excess over -1
    Z = (n + 1) * A + n * (n - 1) * g1 + 2 * n * out['g2'][0] + n * (n - 1) * (n - 2) * out['F1'][0] \
        + 3 * n * (n - 1) * out['F2'][0]
    return Z, A, out


KEEP1 = int(os.environ.get('KEEP1', '0'))   # samples kept per one-variable set between rounds (0: all)
KEEP3 = int(os.environ.get('KEEP3', '0'))   # samples kept per triple set between rounds (0: all)


def prune(R, fv, Fv, U1, U2, P111, P112):
    """keep, of the samples so far, those on which g and F come closest to their
    largest values at the current solution, so that the programme stays small; the
    points the new round adds are appended after this."""
    d, LC = R.d, R.LC
    def top(vals, k):
        return np.argsort(vals)[-k:] if k and len(vals) > k else np.arange(len(vals))
    i1 = top(g_values(fv, Fv, d, LC, U1), KEEP1); i2 = top(g_values(fv, Fv, d, LC, U2), KEEP1)
    j1 = top(F_values(Fv, d, LC, *P111), KEEP3); j2 = top(F_values(Fv, d, LC, *P112), KEEP3)
    return U1[i1], U2[i2], tuple(a[j1] for a in P111), tuple(a[j2] for a in P112)


def run(d, t1, t2, rounds=12, n=24, quiet=False, save=None):
    """refine until the corrected Z is below 0 (the code is excluded, in floating point) or the
    sampled Z is above 0 (then no certificate of degree d exists: the sampled programme is a
    relaxation of the exact one), or the rounds run out."""
    U1, U2 = grid_1d(t1, 300), grid_1d(t2, 300)
    P111, P112 = grid_111(t1, 24, 14), grid_112(t1, t2, 24, 14)
    res = None
    for r in range(rounds):
        prob, fv, Fv, abv, R, el = solve(d, t1, t2, n, U1, U2, P111, P112)
        if fv is None:
            print('  solver status %s' % prob.status, flush=True)
            return None
        Z, A, out = evaluate(R, fv, Fv, t1, t2, n)
        if not quiet:
            print('  d=%d t1=%.5f t2=%.5f n=%d round %d: %s, sampled Z %.5f, corrected Z %.5f; A %.4f, '
                  'max g on [-1,t1] %.2e (+1), g on [-1,t2] %.3e (a2 %.3e), F111 %.2e (b1 %.2e), F112 %.2e (b2 %.2e) '
                  '[%d+%d+%d+%d samples, %.0f s]'
                  % (d, t1, t2, n, r + 1, prob.status, prob.value, Z, A, out['g1'][0] + 1, out['g2'][0], abv[0],
                     out['F1'][0], abv[1], out['F2'][0], abv[2], len(U1), len(U2), len(P111[0]), len(P112[0]), el),
                  flush=True)
        res = (prob.value, Z, fv, Fv)
        if save:
            np.savez(save, d=d, t1=t1, t2=t2, n=n, f=fv, a2=abv[0], b1=abv[1], b2=abv[2],
                     **{'F%d' % k: Fv[k] for k in range(d + 1)})
        if Z < 0:
            print('  corrected Z below 0: the code is excluded in floating point', flush=True)
            break
        if prob.value > 1e-4:
            print('  sampled Z above 0: no certificate of degree %d' % d, flush=True)
            break
        if KEEP1 or KEEP3:
            U1, U2, P111, P112 = prune(R, fv, Fv, U1, U2, P111, P112)
        U1 = np.r_[U1, out['g1'][1]]; U2 = np.r_[U2, out['g2'][1]]
        P111 = tuple(np.r_[P111[j], out['F1'][1][j]] for j in range(3))
        P112 = tuple(np.r_[P112[j], out['F2'][1][j]] for j in range(3))
    return res


if __name__ == '__main__':
    if sys.argv[1] == 'sweep':
        d, rounds, t1 = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
        for t2 in map(float, sys.argv[5:]):
            res = run(d, t1, t2, rounds, quiet=True)
            if res is not None:
                print('RESULT d=%d t1=%.5f t2=%.5f: sampled Z %.5f, corrected Z %.5f (below 0 excludes the code)'
                      % (d, t1, t2, res[0], res[1]), flush=True)
        sys.exit(0)
    d, t1, t2 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    run(d, t1, t2, rounds, n, save=sys.argv[6] if len(sys.argv) > 6 else None)
