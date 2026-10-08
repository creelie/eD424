#!/usr/bin/env python3
"""
cap_probe.py -- can a labelled three-point certificate exclude a further
centre within sqrt6 beside 24 centres within 2 + delta?  (floating point,
sampled constraints; exploration, not proof.)

24 points of label 0 with inner products at most 1/2 + kappa, and one point of
label 1 whose inner products with them are at most tau.  For 24 centres within
2.0161 (thm:kissing-stable) kappa = a(2.0161, 2.0161) - 1/2 = 0.0080 and a further
centre within sqrt6 has tau = a(sqrt6, 2.0161) = 0.6141.  The kernels are those
of three_point_probes.py (Bachoc and Vallentin, with labels); normalised to
total trace 1, a negative least value of 24 (24 b_0 + b_1) + 24 s_0 + s_1
means a certificate excludes the configuration.  hole24.py puts the true
threshold for kappa = 0.008 at tau = 0.678.
Usage: python3 cap_probe.py degree tau kappa [rounds]
"""
import sys
import time
import numpy as np
import cvxpy as cp
import three_point_sdp as TPS
from three_point_probes import Kernels, solve

rng = np.random.default_rng(11)


def sample_grid_k(n_uv, n_t, top):
    s = np.linspace(0, 1, n_uv)
    g = top - (1 + top) * (1 - s) ** 1.5
    U, V, T = [], [], []
    for i, u in enumerate(g):
        for v in g[i:]:
            lo = max(u * v - np.sqrt((1 - u * u) * (1 - v * v)), v)
            hi = min(u * v + np.sqrt((1 - u * u) * (1 - v * v)), top)
            if hi < lo:
                continue
            for t in np.linspace(lo, hi, n_t):
                U.append(u); V.append(v); T.append(t)
    return np.array(U), np.array(V), np.array(T)


def cap(d, tau, kap, rounds):
    K = Kernels(d, labelled=True)

    def lim(za, zb):
        return np.where((za + zb) > 0.5, tau, 0.5 + kap)

    def rand(n):
        out_g, out_z, got = [], [], 0
        while got < n:
            m = 4 * n
            z = np.zeros((m, 3))
            z[rng.random(m) < 0.5, 0] = 1.0
            A = np.stack([lim(z[:, 0], z[:, 1]), lim(z[:, 0], z[:, 2]), lim(z[:, 1], z[:, 2])], 1)
            g = A - (A + 1) * rng.random((m, 3)) ** 2.5
            g = np.where(rng.random((m, 3)) < 0.35, A, g)
            ok = 1 + 2 * g[:, 0] * g[:, 1] * g[:, 2] - (g ** 2).sum(1) >= 0
            out_g.append(g[ok]); out_z.append(z[ok]); got += ok.sum()
        return np.concatenate(out_g)[:n], np.concatenate(out_z)[:n]

    g3, z3 = rand(8000)
    gu, gv, gt = sample_grid_k(22, 10, 0.5 + kap)
    g3 = np.r_[g3, np.stack([gu, gv, gt], 1)]
    z3 = np.r_[z3, np.zeros((len(gu), 3))]
    mg = []
    for a in np.linspace(-1, tau, 18):
        for b in np.linspace(-1, tau, 18):
            if b < a:
                continue
            s = np.sqrt(max((1 - a * a) * (1 - b * b), 0))
            lo, hi = a * b - s, min(a * b + s, 0.5 + kap)
            if hi < lo:
                continue
            for c in np.linspace(lo, hi, 8):
                mg += [(a, b, c), (b, a, c)]
    mg = np.array(mg)
    g3 = np.r_[g3, mg]
    z3 = np.r_[z3, np.tile([1.0, 0.0, 0.0], (len(mg), 1))]
    W, P, Q = [], [], []
    for zp, zq in ((0.0, 0.0), (1.0, 0.0)):
        A = float(lim(np.array(zp), np.array(zq)))
        w = np.r_[A - (A + 1) * np.linspace(0, 1, 400) ** 2, A]
        W.append(w); P.append(np.full(len(w), zp)); Q.append(np.full(len(w), zq))
    wp, zp, zq = np.concatenate(W), np.concatenate(P), np.concatenate(Q)
    b0, b1 = cp.Variable(), cp.Variable()
    for r in range(rounds):
        tr, _ = K.triples(g3, z3)
        pi = K.pairs(wp, zp, zq)
        sig = K.points(np.array([0.0, 1.0]))
        bp = cp.multiply(1 - zp, b0) + cp.multiply(zp, b1)
        bq = cp.multiply(1 - zq, b0) + cp.multiply(zq, b1)
        val = 24 * (24 * b0 + b1) + 24 * sig[0] + sig[1]
        norm = sum(cp.trace(F) for F in K.Fm) + sum(cp.trace(A) for A in K.Am)
        prob = cp.Problem(cp.Minimize(val), K.link + [tr <= 0, pi <= bp + bq, norm <= 1])
        t0 = time.time()
        solve(prob)
        gn, zn = rand(32000)
        vt = K.triple_values(gn, zn)
        print('cap: degree %d, tau %.4f, kappa %.4f, round %d: %s; least value %.5f (negative: 24 + 1 excluded); worst fresh triple %.1e [%.0f s]'
              % (d, tau, kap, r + 1, prob.status, prob.value, vt.max(), time.time() - t0), flush=True)
        g3 = np.r_[g3, gn[np.argsort(vt)[-2000:]]]
        z3 = np.r_[z3, zn[np.argsort(vt)[-2000:]]]



if __name__ == '__main__':
    cap(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 3)
