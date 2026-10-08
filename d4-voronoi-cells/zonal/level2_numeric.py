#!/usr/bin/env python3
"""
level2_numeric.py -- the second-level kernel of the certificate of de Laat,
Leijenhorst and de Muinck Keizer, evaluated numerically (float64), and its
validation.

A_2K(Q) is linear in the kernel blocks X_lambda (M_lambda = T X T^T, with the
deposited transforms T).  For configurations Q of 0 to 4 points of S^3, given
by their Gram matrices, `Blocks.rows` returns matrices Y_lambda with
A_2K(Q) = sum_lambda <X_lambda, Y_lambda>, built from the zonal matrices of this
directory (the reduced output of psker, as verify45.py reads it).  That is the
machinery a sampled second-level programme needs: 60 blocks, 5298 unknowns,
one row per sampled configuration.

Run as a script, it checks the evaluator against the deposited certificate:
A_2K(empty) = 24 and A_2K({x}) = -1; A_2K({x,y}) = -sigma_2(u) against the exact
coefficients in multi_cap/llm24_out/llm24_p2.txt; A_2K = 0 at triples and
quadruples of the root system (the equality case); A_2K <= 0 at random
admissible triples and quadruples.  Floating point; exploration, not proof.

Usage: python3 level2_numeric.py DATA PSPICKLE
  DATA      the folder proofs/4_24 of the certificate
  PSPICKLE  ps.txt.reduced.pkl, which verify45.py writes next to psker's ps.txt
"""
import itertools
import os
import pickle
import sys
import time
from fractions import Fraction

import numpy as np

sys.set_int_max_str_digits(0)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify45 as V                     # noqa: E402  (irreps, admissible tuples, number parsing)
from gl2 import countels                 # noqa: E402

D2 = 16


class Zonal:
    """the zonal matrix entries, prepared for evaluation at arrays of inner products"""

    def __init__(self, ps_path):
        ps = pickle.load(open(ps_path, 'rb'))
        self.prep = {}
        for (lam, i, j), p in ps.items():
            rs = [countels(lam, 1, i), countels(lam, 2, i)]
            cs = [countels(lam, 1, j), countels(lam, 2, j)]
            EV, C, FA = [], [], []
            for ev, c in p.items():
                cr = [ev[0] + ev[2], ev[1] + ev[3]]
                cc = [ev[0] + ev[1], ev[2] + ev[3]]
                fa = [rs[0] - cr[0], rs[1] - cr[1], cs[0] - cc[0], cs[1] - cc[1]]
                assert all(x % 2 == 0 and x >= 0 for x in fa)
                EV.append(ev)
                C.append(float(c))
                FA.append([x // 2 for x in fa])
            self.prep[(lam, i, j)] = (np.array(EV, int), np.array(C), np.array(FA, int))

    def value(self, lam, w1, w2, U):
        """Z_lambda[w1, w2] at the inner products U (n x 6), or None if the entry is absent"""
        i, j = max(w1, w2), min(w1, w2)
        if (lam, i, j) not in self.prep:
            return None
        if i != w1:
            U = U[:, [1, 0, 2, 4, 3, 5]]
        a, b, c, d = U[:, 2], U[:, 3], U[:, 4], U[:, 5]
        Ss = np.stack([a + b + c + d, a + b - c - d, a - b + c - d, a - b - c + d], 1)
        pl = np.stack([2 * (1 + U[:, 0]), 2 * (1 - U[:, 0]), 2 * (1 + U[:, 1]), 2 * (1 - U[:, 1])], 1)
        EV, C, FA = self.prep[(lam, i, j)]
        out = np.zeros(len(U))
        for t in range(len(C)):
            term = np.full(len(U), C[t])
            for k in range(4):
                if EV[t, k]:
                    term = term * Ss[:, k] ** EV[t, k]
                if FA[t, k]:
                    term = term * pl[:, k] ** FA[t, k]
            out += term
        return out


def load_q(path):
    with open(path) as f:
        nr, nc = [int(x) for x in f.readline().split()]
        return [[V.parse_q(t) for t in f.readline().split()] for _ in range(nr)]


class Blocks:
    def __init__(self, data, ps_path, with_X=True):
        self.Z = Zonal(ps_path)
        self.lams, self.info = [], {}
        for lam in V.irreps():
            fn = os.path.join(data, 'dense_%d_%d.txt' % lam)
            if not os.path.exists(fn):
                continue
            ws = V.weven(lam)
            dt = (D2 - sum(lam)) // 2
            order = [(i, jk[0], jk[1]) for i in (0, 1, 2) for jk in V.admissible_tuples(lam, i, ws, dt)]
            T = np.array([[float(x) for x in row] for row in load_q(os.path.join(data, 'transform_%d_%d.txt' % lam))])
            X = np.array([[float(x) for x in row] for row in load_q(fn)]) if with_X else None
            self.lams.append(lam)
            self.info[lam] = dict(ws=ws, dt=dt, pos={t: n for n, t in enumerate(order)}, T=T, X=X,
                                  m=T.shape[1], n=T.shape[0])
        self.nvar = sum(v['m'] * (v['m'] + 1) // 2 for v in self.info.values())

    def rows(self, G):
        """G: (ns, k, k) Gram matrices of k-point configurations (k = 0..4); the matrices
        Y_lambda (ns, m, m) with A_2K(Q) = sum_lambda <X_lambda, Y_lambda>."""
        ns, k = G.shape[0], G.shape[1]
        subs = [s for l in (0, 1, 2) for s in itertools.combinations(range(k), l)]
        out = {lam: np.zeros((ns, self.info[lam]['m'], self.info[lam]['m'])) for lam in self.lams}
        one = np.ones(ns)
        for a1 in range(len(subs)):
            for a2 in range(a1, len(subs)):
                J1, J2 = subs[a1], subs[a2]
                if tuple(sorted(set(J1) | set(J2))) != tuple(range(k)):
                    continue
                n1, n2 = len(J1), len(J2)
                f = lambda Ja, Jb, a, b: G[:, Ja[min(a, len(Ja) - 1)], Jb[min(b, len(Jb) - 1)]]
                if n1 > 0 and n2 > 0:
                    ips = [f(J1, J1, 0, 1), f(J2, J2, 0, 1), f(J1, J2, 0, 0), f(J1, J2, 0, 1),
                           f(J1, J2, 1, 0), f(J1, J2, 1, 1)]
                else:
                    ips = [one] * 6
                    if n1 > 0:
                        ips[0] = f(J1, J1, 0, 1)
                    elif n2 > 0:
                        ips[1] = f(J2, J2, 0, 1)
                U = np.stack(ips, 1)
                fac = 0.5 * (2 if J1 != J2 else 1)
                p0 = ips[0] if n1 == 2 else one
                p1 = ips[1] if n2 == 2 else one
                for lam in self.lams:
                    inf = self.info[lam]
                    at1 = V.admissible_tuples(lam, n1, inf['ws'], inf['dt'])
                    at2 = V.admissible_tuples(lam, n2, inf['ws'], inf['dt'])
                    if not at1 or not at2:
                        continue
                    C = np.zeros((ns, inf['n'], inf['n']))
                    zc = {}
                    for (j1, k1) in at1:
                        za = inf['pos'][(n1, j1, k1)]
                        for (j2, k2) in at2:
                            zb = inf['pos'][(n2, j2, k2)]
                            if (k1, k2) not in zc:
                                zc[(k1, k2)] = self.Z.value(lam, k1, k2, U)
                            z = zc[(k1, k2)]
                            if z is None:
                                continue
                            v = fac * p0 ** j1 * p1 ** j2 * z
                            C[:, za, zb] += v
                            C[:, zb, za] += v
                    out[lam] += np.einsum('ai,sab,bj->sij', inf['T'], C, inf['T'])
        return out

    def value(self, Y):
        """A_2K at the samples, for the deposited X"""
        return sum(np.einsum('sij,ij->s', Y[lam], self.info[lam]['X']) for lam in self.lams)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    t0 = time.time()
    B = Blocks(sys.argv[1], sys.argv[2])
    print('%d blocks, %d unknowns [%.0f s]' % (len(B.lams), B.nvar, time.time() - t0))
    v0 = B.value(B.rows(np.zeros((1, 0, 0))))[0]
    v1 = B.value(B.rows(np.ones((1, 1, 1))))[0]
    print('A_2K(empty) = %.12f (24);  A_2K({x}) = %.12f (-1 - sigma_1)' % (v0, v1))
    c = [Fraction(0)] * 17
    for line in open(os.path.join(HERE, '..', 'multi_cap', 'llm24_out', 'llm24_p2.txt')):
        if line.startswith('#') or not line.strip():
            continue
        i, v = line.split(None, 1)
        v = v.strip()
        c[int(i)] = Fraction(*[int(x) for x in v.split('/')]) if '/' in v else Fraction(int(v))
    cf = np.array([float(x) for x in c])
    u = np.linspace(-1, 0.5, 31)
    v2 = B.value(B.rows(np.stack([np.array([[1, x], [x, 1]]) for x in u])))
    s2 = np.polyval(cf[::-1], u)
    print('pairs, 31 inner products in [-1, 1/2]: max |A_2K + sigma_2| = %.1e (sigma_2 up to %.1e)'
          % (np.abs(v2 + s2).max(), s2.max()))
    R = []
    for a, b in itertools.combinations(range(4), 2):
        for sa in (1, -1):
            for sb in (1, -1):
                w = np.zeros(4); w[a] = sa; w[b] = sb; R.append(w / np.sqrt(2))
    R = np.array(R)
    rng = np.random.default_rng(0)
    for k in (3, 4):
        G = np.stack([R[i] @ R[i].T for i in [rng.choice(24, k, replace=False) for _ in range(40)]])
        print('%d-point subsets of the root system (40): max |A_2K| = %.1e' % (k, np.abs(B.value(B.rows(G))).max()))
    for k in (3, 4):
        out = []
        while len(out) < 300:
            X = rng.normal(size=(k, 4)); X /= np.linalg.norm(X, axis=1, keepdims=True)
            Gm = X @ X.T
            if (Gm[np.triu_indices(k, 1)] <= 0.5).all():
                out.append(Gm)
        print('random admissible %d-point configurations (300): max A_2K = %.1e (<= 0)' % (k, B.value(B.rows(np.stack(out))).max()))
    print('[%.0f s]' % (time.time() - t0))


if __name__ == '__main__':
    main()
