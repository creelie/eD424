#!/usr/bin/env python3
"""
typed3pt.py -- a three-point (Bachoc-Vallentin) bound for spherical codes on S^3
whose points carry TYPES, each pair of types with its own largest inner product.
Floating point exploration (sampled constraints, refinement rounds); a negative
value is evidence that no code with the given counts exists, to be certified
separately.

Kernels (positive for every finite set C of typed points of S^3):
  two-point   sum_{x,y} sum_{k=1..d} G_k(<x,y>) e(x)^T f_k e(y),        f_k (m x m) psd
  three-point for each pole type P and k = 0..d,
              sum_{e of type P} sum_{x,y} phi_k(<e,x>,<e,y>,<x,y>) a_k(x;e)^T F^P_k a_k(y;e),
              a_k(x;e) = e(x) (x) T_{d-k+1}(<e,x>),  F^P_k psd,
  where e(x) is the one-hot type vector, G_k = U_k/(k+1), phi_k(u,v,t) =
  ((1-u^2)(1-v^2))^{k/2} P_k((t-uv)/sqrt((1-u^2)(1-v^2))) with P_k Legendre.
Splitting the sum by coincidence pattern gives
  0 <= sum_types n_s p_s + sum_{type pairs} N_st PAIR_st + sum_{type triples} N_str TRIPLE_str
so Val = sum n_s p_s + sum N_st c_st + sum N_str c_str < 0, with PAIR <= c and TRIPLE <= c
on the admissible inner products, shows that no such code exists.
"""
import sys, time, os, itertools, json
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from numpy.polynomial import legendre as LEG
from math import comb

rng = np.random.default_rng(int(os.environ.get('SEED', '3')))


def cheb(u, n):
    u = np.asarray(u, float)
    out = [np.ones_like(u), u]
    for _ in range(n - 2):
        out.append(2 * u * out[-1] - out[-2])
    return np.stack(out[:n], -1)


def gegen_all(u, d):
    u = np.asarray(u, float)
    out = [np.ones_like(u), 2 * u]
    for _ in range(d - 1):
        out.append(2 * u * out[-1] - out[-2])
    return np.stack([out[k] / (k + 1) for k in range(d + 1)], -1)


LCACHE = {}


def phi(k, u, v, t):
    if k not in LCACHE:
        c = np.zeros(k + 1); c[k] = 1.0
        LCACHE[k] = LEG.leg2poly(c)
    a = LCACHE[k]
    x = t - u * v
    s2 = np.maximum((1 - u * u) * (1 - v * v), 0)
    out = np.zeros_like(x)
    for m in range(k % 2, k + 1, 2):
        out = out + a[m] * x ** m * s2 ** ((k - m) // 2)
    return out


class Layout:
    """index layout of all matrix variables, each stored as diag + strict upper (row-major)."""
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.blocks = {}      # name -> (offset, size)
        off = 0
        for k in range(1, d + 1):
            self.blocks[('f', k)] = (off, m); off += m * (m + 1) // 2
        for P in range(m):
            for k in range(d + 1):
                s = m * (d - k + 1)
                self.blocks[('F', P, k)] = (off, s); off += s * (s + 1) // 2
        self.n = off
        self.tri = {}

    def idx(self, name):
        off, s = self.blocks[name]
        if s not in self.tri:
            M = np.zeros((s, s), dtype=np.int64)
            M[np.arange(s), np.arange(s)] = np.arange(s)
            iu = np.triu_indices(s, 1)
            M[iu] = s + np.arange(len(iu[0]))
            M[(iu[1], iu[0])] = s + np.arange(len(iu[0]))
            self.tri[s] = M
        return off + self.tri[s]

    def rank1(self, rows, name, A, B, w):
        """coefficient of <X, (A_n B_n^T + B_n A_n^T)/2> * w_n for each sample n (row offsets rows),
        A, B: lists of (index array into the s x s matrix, value array (N, len))"""
        raise NotImplementedError


class Builder:
    """accumulates sparse rows: row r gets sum over entries of coefficient * variable."""
    def __init__(self, L):
        self.L = L
        self.r, self.c, self.v = [], [], []

    def add_bilinear(self, rows, name, ia, va, ib, vb, w):
        """adds w_n * a_n^T X b_n for symmetric X (block `name`), where a_n is supported on
        indices ia (array) with values va (N, la) and b_n on ib with vb (N, lb)."""
        I = self.L.idx(name)
        sub = I[np.ix_(ia, ib)]                                  # (la, lb) variable indices
        coef = (va[:, :, None] * vb[:, None, :]) * w[:, None, None]  # (N, la, lb)
        # <X, a b^T> = sum_ij a_i b_j X_ij ; with X_ij = X_ji stored once, coefficients add up
        N = coef.shape[0]
        self.r.append(np.repeat(rows, sub.size))
        self.c.append(np.tile(sub.ravel(), N))
        self.v.append(coef.reshape(N, -1).ravel())

    def matrix(self, nrows):
        r = np.concatenate(self.r); c = np.concatenate(self.c); v = np.concatenate(self.v)
        return sp.csr_matrix((v, (r, c)), shape=(nrows, self.L.n))


def feat(L, typ, k, u):
    """a_k(x;e) for x of type typ at <e,x> = u: (indices, values)"""
    mk = L.d - k + 1
    return np.arange(typ * mk, (typ + 1) * mk), cheb(u, mk)


def add_point(B, rows, s):
    L = B.L
    one = np.ones(len(rows))
    for k in range(1, L.d + 1):
        B.add_bilinear(rows, ('f', k), np.array([s]), one[:, None], np.array([s]), one[:, None], one)
    ia, va = feat(L, s, 0, one)
    B.add_bilinear(rows, ('F', s, 0), ia, va, ia, va, one)


def add_pair(B, rows, s, t, u):
    L = B.L
    N = len(u); one = np.ones(N)
    G = gegen_all(u, L.d)
    for k in range(1, L.d + 1):
        B.add_bilinear(rows, ('f', k), np.array([s]), one[:, None], np.array([t]), one[:, None], 2 * G[:, k])
    # pattern 2: pole z (type s) with the other point w (type t), and the reverse
    for (P, Q) in ((s, t), (t, s)):
        ia, va = feat(L, P, 0, one)
        ib, vb = feat(L, Q, 0, u)
        B.add_bilinear(rows, ('F', P, 0), ia, va, ib, vb, 2 * one)
    # pattern 3: pole z, x = y = w
    for (P, Q) in ((s, t), (t, s)):
        for k in range(L.d + 1):
            ib, vb = feat(L, Q, k, u)
            B.add_bilinear(rows, ('F', P, k), ib, vb, ib, vb, (1 - u * u) ** k)


def add_triple(B, rows, types, u12, u13, u23):
    L = B.L
    s1, s2, s3 = types
    for (P, (Qa, ua), (Qb, ub), tt) in ((s1, (s2, u12), (s3, u13), u23),
                                         (s2, (s1, u12), (s3, u23), u13),
                                         (s3, (s1, u13), (s2, u23), u12)):
        for k in range(L.d + 1):
            ia, va = feat(L, Qa, k, ua)
            ib, vb = feat(L, Qb, k, ub)
            B.add_bilinear(rows, ('F', P, k), ia, va, ib, vb, 2 * phi(k, ua, ub, tt))


def grid1(hi, n):
    s = np.linspace(0, 1, n)
    return hi - (hi + 1) * (1 - s) ** 1.6


def triple_grid(T12, T13, T23, n1, n2, n3):
    out = []
    for u in grid1(T12, n1):
        for v in grid1(T13, n2):
            r = np.sqrt(max((1 - u * u) * (1 - v * v), 0))
            lo, hi = max(u * v - r, -1.0), min(u * v + r, T23)
            if hi < lo:
                continue
            ws = np.unique(np.r_[np.linspace(lo, hi, n3), hi])
            for w in ws:
                out.append((u, v, w))
    return np.array(out)


def random_triples(T12, T13, T23, n):
    outs = []
    for dim in (4, 3, 2):
        X = rng.normal(size=(30 * n, 3, dim)); X /= np.linalg.norm(X, axis=2, keepdims=True)
        g = np.stack([np.einsum('ni,ni->n', X[:, 0], X[:, 1]), np.einsum('ni,ni->n', X[:, 0], X[:, 2]),
                      np.einsum('ni,ni->n', X[:, 1], X[:, 2])], 1)
        ok = (g[:, 0] <= T12) & (g[:, 1] <= T13) & (g[:, 2] <= T23)
        outs.append(g[ok][:n])
    return np.concatenate(outs)


def ncount(counts, combo):
    out = 1
    for s in set(combo):
        out *= comb(counts[s], combo.count(s))
    return out


def solve(counts, Tmat, d, rounds=3, sizes=(14, 14, 7), nrand=1500, verbose=True):
    m = len(counts)
    L = Layout(m, d)
    pair_list = [(s, t) for s in range(m) for t in range(s, m) if (comb(counts[s], 2) if s == t else counts[s] * counts[t]) > 0]
    triple_list = [c for c in itertools.combinations_with_replacement(range(m), 3) if ncount(counts, list(c)) > 0]
    psamp = {p: np.unique(np.r_[grid1(Tmat[p[0]][p[1]], 600), Tmat[p[0]][p[1]]]) for p in pair_list}
    tsamp = {}
    for c in triple_list:
        T12, T13, T23 = Tmat[c[0]][c[1]], Tmat[c[0]][c[2]], Tmat[c[1]][c[2]]
        tsamp[c] = np.r_[triple_grid(T12, T13, T23, *sizes), random_triples(T12, T13, T23, nrand)]
    for rnd in range(rounds):
        t0 = time.time()
        X = {}
        for name, (off, s) in L.blocks.items():
            X[name] = cp.Variable((s, s), PSD=True)
        def vec(name):
            M = X[name]; s = M.shape[0]
            iu = np.triu_indices(s, 1)
            ii = np.r_[np.arange(s), iu[0]]; jj = np.r_[np.arange(s), iu[1]]
            Sel = sp.csr_matrix((np.ones(len(ii)), (np.arange(len(ii)), ii + jj * s)), shape=(len(ii), s * s))
            return Sel @ cp.vec(M, order='F')
        xall = cp.hstack([vec(name) for name in L.blocks])
        cons = []
        # points
        B = Builder(L); rows = np.arange(m)
        for s in range(m):
            add_point(B, np.array([s]), s)
        Ppt = B.matrix(m)
        p = Ppt @ xall
        cvars = {}
        val = sum(counts[s] * p[s] for s in range(m))
        for pr in pair_list:
            u = psamp[pr]
            B = Builder(L); add_pair(B, np.arange(len(u)), pr[0], pr[1], u)
            cv = cp.Variable(); cvars[pr] = cv
            cons.append(B.matrix(len(u)) @ xall <= cv)
            val = val + (comb(counts[pr[0]], 2) if pr[0] == pr[1] else counts[pr[0]] * counts[pr[1]]) * cv
        for c in triple_list:
            g = tsamp[c]
            B = Builder(L); add_triple(B, np.arange(len(g)), c, g[:, 0], g[:, 1], g[:, 2])
            cv = cp.Variable(); cvars[c] = cv
            cons.append(B.matrix(len(g)) @ xall <= cv)
            val = val + ncount(counts, list(c)) * cv
        tr = sum(cp.trace(X[name]) for name in L.blocks)
        cons.append(tr <= 1)
        prob = cp.Problem(cp.Minimize(val), cons)
        try:
            prob.solve(solver='CLARABEL', max_iter=400)
        except cp.error.SolverError:
            print('   CLARABEL failed; SCS', flush=True)
            prob.solve(solver='SCS', eps=1e-7, max_iters=200000)
        # check on finer samples
        xv = np.concatenate([np.r_[np.diag(X[n].value), X[n].value[np.triu_indices(X[n].shape[0], 1)]] for n in L.blocks])
        corr = 0.0; viol = {}
        newp, newt = {}, {}
        for pr in pair_list:
            T = Tmat[pr[0]][pr[1]]
            u = np.r_[np.linspace(-1, T, 20001)]
            B = Builder(L); add_pair(B, np.arange(len(u)), pr[0], pr[1], u)
            vv = B.matrix(len(u)) @ xv - cvars[pr].value
            viol[pr] = vv.max(); corr += (comb(counts[pr[0]], 2) if pr[0] == pr[1] else counts[pr[0]] * counts[pr[1]]) * max(vv.max(), 0)
            newp[pr] = u[np.argsort(vv)[-60:]]
        for c in triple_list:
            T12, T13, T23 = Tmat[c[0]][c[1]], Tmat[c[0]][c[2]], Tmat[c[1]][c[2]]
            g = np.r_[random_triples(T12, T13, T23, 12000), triple_grid(T12, T13, T23, 30, 30, 12)]
            vv = np.concatenate([(lambda gg: (lambda B: (add_triple(B, np.arange(len(gg)), c, gg[:, 0], gg[:, 1], gg[:, 2]), B.matrix(len(gg)) @ xv)[1])(Builder(L)))(g[i:i + 5000]) for i in range(0, len(g), 5000)]) - cvars[c].value
            viol[c] = vv.max(); corr += ncount(counts, list(c)) * max(vv.max(), 0)
            newt[c] = g[np.argsort(vv)[-400:]]
        if verbose:
            print('d=%d round %d: %s Val=%.4e corrected=%.4e  maxviol=%.1e  [%.0f s]' % (
                d, rnd + 1, prob.status, prob.value, prob.value + corr, max(viol.values()), time.time() - t0), flush=True)
        for pr in pair_list:
            psamp[pr] = np.r_[psamp[pr], newp[pr]]
        for c in triple_list:
            tsamp[c] = np.r_[tsamp[c], newt[c]]
    return prob.value, prob.value + corr, X, cvars


def a_(d, e):
    return (d * d + e * e - 4) / (2 * d * e)


if __name__ == '__main__':
    mode = sys.argv[1]
    d = int(sys.argv[2]); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    if mode == 'kiss':        # test: n points with inner products <= 1/2
        n = int(sys.argv[4])
        solve([n], [[0.5]], d, rounds)
    elif mode == 'kisshole':   # 24 points with inner products <= 1/2 and nF holes at <= tau, holes <= 2/3 mutually
        nF = int(sys.argv[4]); tau = float(sys.argv[5])
        solve([24, nF], [[0.5, tau], [tau, 2 / 3]], d, rounds)
    elif mode == 'slackhole':  # 24 points within distance dd (slack 1/2 - 2/dd^2) and nF holes at a(sqrt6, dd)
        nF = int(sys.argv[4]); dd = float(sys.argv[5])
        R6 = 6 ** .5
        T = [[a_(dd, dd), a_(dd, R6)], [a_(dd, R6), a_(R6, R6)]]
        print('T =', np.round(T, 5), flush=True)
        solve([24, nF], T, d, rounds)
    elif mode == 'res30':     # A: 22 within 2.05, B: 2 within 2.25, F: nF in [2.4, sqrt6)
        nF = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        R6 = 6 ** .5
        r = [2.05, 2.25, R6]
        T = [[a_(r[i], r[j]) for j in range(3)] for i in range(3)]
        print('T =', np.round(T, 5))
        solve([22, 2, nF], T, d, rounds)
