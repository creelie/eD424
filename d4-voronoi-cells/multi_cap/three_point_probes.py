#!/usr/bin/env python3
"""
three_point_probes.py -- two probes of what three-point certificates can see
about twenty-four points of S^3 (floating point, sampled constraints; exploration,
not proof).  They support the paragraph "What is left" of sec:remains.

Both use the kernels of Bachoc and Vallentin with labels: for a fixed point e of a
finite set C, (x, y) -> phi_k(<e,x>, <e,y>, <x,y>) a(x) a(y)^T is a positive
semidefinite matrix kernel when a(x) = T(<e,x>) (x) b(label(x), label(e)), with T the
Chebyshev polynomials and b = (1, label(x), label(e)); summed over the triples of C,
and with a two-point part added, it splits into distinct triples, pairs and points.

  windows d kappa win [rounds]
      24-point codes with inner products at most 1/2 + kappa.  Asks for a certificate
      whose pair constraint is -1 inside the windows of half-width win around -1, -1/2,
      0, 1/2 and -1 - gamma outside them.  Summing, (number of pairs outside) * gamma
      <= 24 sigma - 276, so the margin gamma - (24 sigma - 276), maximised, is positive
      exactly when such a certificate forces every pair into the windows.  It also
      prints the count bound 1 + 2 sigma.

  cap d tau [rounds]
      24 points of label 0 with inner products at most 1/2 and one point of label 1
      whose inner products with them are at most tau (a centre at distance d from c
      has tau = d/4 against contacts, so tau = sqrt6/4 = 0.612 at the end of the
      shell).  With the kernels normalised to total trace 1, the least value of
      24 (24 b_0 + b_1) + 24 s_0 + s_1 is negative exactly when a certificate excludes
      that configuration; 0 means none does.

Every round adds the most violated of fresh random constraints and prints the worst
violation, so a value is to be read against it.  Needs numpy, cvxpy (Clarabel, SCS).
"""
import sys
import time

import cvxpy as cp
import numpy as np

import three_point_sdp as TPS

rng = np.random.default_rng(11)


class Kernels:
    def __init__(self, d, labelled):
        self.d = d
        self.nb = 3 if labelled else 1
        self.LC = [TPS.leg_coeffs(k) for k in range(d + 1)]
        self.M = [self.nb * (d - k + 1) for k in range(d + 1)]
        self.IU = [np.triu_indices(m) for m in self.M]
        self.PSI = 3 if labelled else 1
        self.IU2 = np.triu_indices(self.PSI)
        self.Fu = [cp.Variable(m * (m + 1) // 2) for m in self.M]
        self.Fm = [cp.Variable((m, m), PSD=True) for m in self.M]
        self.Au = [cp.Variable(self.PSI * (self.PSI + 1) // 2) for _ in range(d + 1)]
        self.Am = [cp.Variable((self.PSI, self.PSI), PSD=True) for _ in range(d + 1)]
        self.link = []
        for k in range(d + 1):
            i, j = self.IU[k]
            self.link.append(self.Fu[k] == self.Fm[k][i, j])
            i, j = self.IU2
            self.link.append(self.Au[k] == self.Am[k][i, j])

    def b(self, zx, ze):
        one = np.ones_like(zx)
        return np.stack([one, zx, ze], 1) if self.nb == 3 else one[:, None]

    def psi(self, z):
        return np.stack([np.ones_like(z), z, z * z], 1) if self.PSI == 3 else np.ones_like(z)[:, None]

    def feat(self, k, w, zx, ze):
        T = TPS.cheb(w, self.d - k + 1)
        return (T[:, :, None] * self.b(zx, ze)[:, None, :]).reshape(len(w), -1)

    def sym(self, A, iu):
        As = A + np.transpose(A, (0, 2, 1))
        i, j = iu
        return np.where(i == j, As[:, i, j] / 2, As[:, i, j])

    def triples(self, g, z):
        """rows of the symmetrised triple term; g = (ab, ac, bc), z = labels (a, b, c)"""
        G = {(0, 1): g[:, 0], (1, 0): g[:, 0], (0, 2): g[:, 1], (2, 0): g[:, 1], (1, 2): g[:, 2], (2, 1): g[:, 2]}
        rows = []
        for k in range(self.d + 1):
            acc = 0
            for (e, x, y) in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
                u, v, t = G[(e, x)], G[(e, y)], G[(x, y)]
                fx = self.feat(k, u, z[:, x], z[:, e])
                fy = self.feat(k, v, z[:, y], z[:, e])
                acc = acc + TPS.phi(k, u, v, t, self.LC)[:, None, None] * fx[:, :, None] * fy[:, None, :]
            rows.append(self.sym(acc, self.IU[k]))
        return sum(rows[k] @ self.Fu[k] for k in range(self.d + 1)), rows

    def pairs(self, w, zp, zq):
        one = np.ones_like(w)
        expr = 0
        for k in range(self.d + 1):
            acc = 0
            for (ze, zx) in ((zp, zq), (zq, zp)):
                if k == 0:
                    f1 = self.feat(0, one, ze, ze)
                    fw = self.feat(0, w, zx, ze)
                    acc = acc + f1[:, :, None] * fw[:, None, :]
                fw = self.feat(k, w, zx, ze)
                acc = acc + ((1 - w * w) ** k / 2)[:, None, None] * fw[:, :, None] * fw[:, None, :]
            expr = expr + self.sym(acc, self.IU[k]) @ self.Fu[k]
            pp, pq = self.psi(zp), self.psi(zq)
            B = TPS.gegen(k, w)[:, None, None] * pp[:, :, None] * pq[:, None, :]
            expr = expr + self.sym(B, self.IU2) @ self.Au[k]
        return expr

    def points(self, z):
        one = np.ones_like(z)
        f1 = self.feat(0, one, z, z)
        expr = self.sym(f1[:, :, None] * f1[:, None, :] / 2, self.IU[0]) @ self.Fu[0]
        pz = self.psi(z)
        for k in range(self.d + 1):
            expr = expr + self.sym(TPS.gegen(k, one)[:, None, None] * pz[:, :, None] * pz[:, None, :] / 2, self.IU2) @ self.Au[k]
        return expr

    def triple_values(self, g, z):
        _, rows = self.triples(g, z)
        return sum(rows[k] @ self.Fu[k].value for k in range(self.d + 1))


def solve(prob):
    try:
        prob.solve(solver='CLARABEL', max_iter=400)
    except Exception:
        prob.solve(solver='SCS', eps=1e-7, max_iters=60000)


def random_triples(n, top):
    X = rng.normal(size=(8 * n, 3, 4))
    X /= np.linalg.norm(X, axis=2, keepdims=True)
    g = np.stack([np.einsum('ni,ni->n', X[:, 0], X[:, 1]), np.einsum('ni,ni->n', X[:, 0], X[:, 2]),
                  np.einsum('ni,ni->n', X[:, 1], X[:, 2])], 1)
    return g[(g <= top).all(1)][:n]


def windows(d, kap, win, rounds):
    top = 0.5 + kap
    K = Kernels(d, labelled=False)
    gu, gv, gt = TPS.sample_grid(24, 12)
    G3 = np.r_[np.stack([gu, gv, gt], 1), random_triples(6000, top)]
    G3 = G3[(G3 <= top).all(1)]
    wp = np.r_[np.linspace(-1, top, 3000), -1, -0.5, 0, 0.5, top]
    inside = (wp <= -1 + win) | (np.abs(wp + 0.5) <= win) | (np.abs(wp) <= win) | (wp >= 0.5 - win)
    gam = cp.Variable(nonneg=True)
    z0 = lambda n: np.zeros(n)
    for r in range(rounds):
        tau, _ = K.triples(G3, np.zeros((len(G3), 3)))
        pi = K.pairs(wp, z0(len(wp)), z0(len(wp)))
        sig = K.points(np.zeros(1))
        margin = gam - (24 * sig[0] - 276)
        prob = cp.Problem(cp.Maximize(margin), K.link + [tau <= 0, pi[inside] <= -1, pi[~inside] <= -1 - gam, gam <= 1000])
        t0 = time.time()
        solve(prob)
        Gn = random_triples(40000, top)
        vt = K.triple_values(Gn, np.zeros((len(Gn), 3)))
        print('windows: degree %d, kappa %.4f, half-width %.3f, round %d: %s; count bound 1 + 2 sigma = %.4f, gamma = %.4f, '
              'margin = %.4f; worst fresh triple %.1e [%.0f s]'
              % (d, kap, win, r + 1, prob.status, 1 + 2 * float(sig.value[0]), gam.value, prob.value, vt.max(), time.time() - t0),
              flush=True)
        G3 = np.r_[G3, Gn[np.argsort(vt)[-1500:]]]


def cap(d, tau, rounds):
    K = Kernels(d, labelled=True)

    def lim(za, zb):
        return np.where((za + zb) > 0.5, tau, 0.5)

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
    gu, gv, gt = TPS.sample_grid(22, 10)
    g3 = np.r_[g3, np.stack([gu, gv, gt], 1)]
    z3 = np.r_[z3, np.zeros((len(gu), 3))]
    mg = []
    for a in np.linspace(-1, tau, 18):
        for b in np.linspace(-1, tau, 18):
            if b < a:
                continue
            s = np.sqrt(max((1 - a * a) * (1 - b * b), 0))
            lo, hi = a * b - s, min(a * b + s, 0.5)
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
        print('cap: degree %d, tau %.4f, round %d: %s; least value %.5f (negative: 24 + 1 excluded); worst fresh triple %.1e [%.0f s]'
              % (d, tau, r + 1, prob.status, prob.value, vt.max(), time.time() - t0), flush=True)
        g3 = np.r_[g3, gn[np.argsort(vt)[-2000:]]]
        z3 = np.r_[z3, zn[np.argsort(vt)[-2000:]]]


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == 'windows':
        windows(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 3)
    elif sys.argv[1] == 'cap':
        cap(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 4)
