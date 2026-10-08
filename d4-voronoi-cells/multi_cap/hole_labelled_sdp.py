#!/usr/bin/env python3
"""
hole_labelled_sdp.py -- the labelled three-point programmes of the Remark "What (C) needs from
twenty-five to thirty centres" (floating point; evidence, not proof).

   n directions A with pairwise inner products <= t = 1/2 + kappa, and one further direction B
   with <B, a> <= tau for every a in A.  Can a certificate exclude n = 24 ?

A centre within sqrt 6 of c beside 24 centres within 2.0161 needs tau = a(2.0161, sqrt 6) = 0.6141,
with kappa = 0.008.  Result (runs/hole_labelled_*.log): no certificate of degree 10 or 12 excludes
n = 24, or even n = 24.5; without B (mode 'plain') none of degree 10 excludes n = 24.9 and none of
degree 12 excludes n = 24.7.  The values at tau = 0.6141 fall with the degree as fast as those at
tau = 0.71, where the root system with a direction in one of its deep holes is feasible.

Kernels (all positivity is Schoenberg / Bachoc-Vallentin, valid for ANY finite set):
  two-point  sum_{x,y} sum_{k>=1} G_k(<x,y>) <f_k, e_l(x) e_l(y)^T>,  f_k 2x2 PSD
  three-point, one PSD matrix per POLE TYPE P in {A, B}:
     sum_{e of type P} sum_{x,y} phi_k(<e,x>,<e,y>,<x,y>) (e_l(x) (x) T(<e,x>))^T F^P_k (e_l(y) (x) T(<e,y>))
  (e_l = one-hot label vector, T = Chebyshev T_0..T_{d-k}); F^A_k is 2m x 2m, F^B_0 is 2m x 2m,
  F^B_k (k>=1) only needs its A-block (the B-rows only meet phi_k(1,.,.) = 0 for k >= 1).
The sum of all of these is >= 0 and splits by coincidence pattern into
  n pA + pB + C(n,2) gAA + n gAB + C(n,3) hAAA + C(n,2) hAAB.
With gAA <= cAA on [-1,t], gAB <= cAB on [-1,tau], hAAA <= cAAA, hAAB <= cAAB on the admissible
domains, Val(n) = n pA + pB + C(n,2) cAA + n cAB + C(n,3) cAAA + C(n,2) cAAB < 0 excludes the
configuration.  We minimise Val(n) with cAA = -1 and total trace <= 1000 (TMAX).
Mode 'plain' drops B (the fixed-cardinality three-point bound for n directions).

Floating point, sampled constraints with refinement rounds: a NEGATIVE value is only evidence
(it would need rigorous verification); a value >= 0 (nothing excluded) is robust, because
sampling only makes the programme optimistic.
Requires numpy, scipy, cvxpy with the Clarabel solver.
Usage: python3 hole_labelled_sdp.py mode d kappa tau n [rounds]
  e.g. python3 hole_labelled_sdp.py hole 12 0.008 0.6141 24.5 2
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from numpy.polynomial import legendre as LEG
from math import comb

rng = np.random.default_rng(5)
LAST = {}
NORM = 'pair'
import os
TMAX = float(os.environ.get('TMAX', '1e3'))
# accept Clarabel's last iterate on numerical trouble, and keep its primal/dual objective values
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL as _CL
from cvxpy.settings import OPTIMAL_INACCURATE as _OI
for _k in ('NumericalError', 'InsufficientProgress', 'MaxIterations'):
    _CL.STATUS_MAP[_k] = _OI
SOLINFO = {}
_inv = _CL.invert
def _invert(self, solution, inverse_data):
    SOLINFO.update(status=str(solution.status), pobj=solution.obj_val, dobj=solution.obj_val_dual, iters=solution.iterations)
    return _inv(self, solution, inverse_data)
_CL.invert = _invert


def cheb(u, n):
    u = np.asarray(u, dtype=float)
    out = [np.ones_like(u), u]
    for _ in range(n - 2):
        out.append(2 * u * out[-1] - out[-2])
    return np.stack(out[:n], axis=-1)


def gegen(k, u):
    u = np.asarray(u, dtype=float)
    a, b = np.ones_like(u), 2 * u
    if k == 0:
        return a
    for _ in range(k - 1):
        a, b = b, 2 * u * b - a
    return b / (k + 1)


def leg_coeffs(k):
    c = np.zeros(k + 1); c[k] = 1.0
    return LEG.leg2poly(c)


def phi(k, u, v, t, LC):
    a = LC[k]; x = t - u * v; s2 = np.maximum((1 - u * u) * (1 - v * v), 0)
    out = np.zeros_like(x)
    for m in range(k % 2, k + 1, 2):
        out = out + a[m] * x ** m * s2 ** ((k - m) // 2)
    return out


def symvec(X):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    i = np.r_[np.arange(n), iu[0]]; j = np.r_[np.arange(n), iu[1]]
    return X[i, j]


def sym_coef_rank1(a, b, w):
    """coefficients of <X, a b^T> (X symmetric) in symvec(X) order; a, b (N, n), w (N,)"""
    n = a.shape[1]
    iu = np.triu_indices(n, 1)
    dg = a * b
    up = a[:, iu[0]] * b[:, iu[1]] + a[:, iu[1]] * b[:, iu[0]]
    return w[:, None] * np.hstack([dg, up])


def sym_coef_full(S):
    """coefficients of <X, S> for a batch of (not nec. symmetric) S (N, n, n)"""
    n = S.shape[1]
    iu = np.triu_indices(n, 1)
    dg = np.einsum('nii->ni', S)
    up = S[:, iu[0], iu[1]] + S[:, iu[1], iu[0]]
    return np.hstack([dg, up])


def embed(Tv, lab, nl=2):
    """one-hot label (x) T : (N, m) -> (N, nl*m)"""
    N, m = Tv.shape
    out = np.zeros((N, nl * m))
    out[:, lab * m:(lab + 1) * m] = Tv
    return out


class Cert:
    def __init__(self, d, withB):
        self.d, self.B = d, withB
        self.LC = [leg_coeffs(k) for k in range(d + 1)]
        self.m = [d - k + 1 for k in range(d + 1)]
        nl = 2 if withB else 1
        self.nl = nl
        self.f = [cp.Variable((nl, nl), PSD=True) for k in range(d + 1)]      # f[0] unused
        self.FA = [cp.Variable((nl * m, nl * m), PSD=True) for m in self.m]
        if withB:
            self.FB = [cp.Variable((2 * self.m[0], 2 * self.m[0]), PSD=True)] + \
                      [cp.Variable((m, m), PSD=True) for m in self.m[1:]]
        self.vf = [symvec(x) for x in self.f]
        self.vFA = [symvec(x) for x in self.FA]
        if withB:
            self.vFB = [symvec(x) for x in self.FB]

    def trace(self):
        tr = sum(cp.trace(x) for x in self.f[1:]) + sum(cp.trace(x) for x in self.FA)
        if self.B:
            tr = tr + sum(cp.trace(x) for x in self.FB)
        return tr

    # ---- coefficient blocks: dict var-list-name -> list over k of (N, len) arrays (or None)
    def _zero(self, N):
        blocks = {'f': [None] * (self.d + 1), 'FA': [None] * (self.d + 1)}
        if self.B:
            blocks['FB'] = [None] * (self.d + 1)
        return blocks

    def _add(self, blocks, name, k, C):
        blocks[name][k] = C if blocks[name][k] is None else blocks[name][k] + C

    def expr(self, blocks):
        """return (sparse matrix over stacked variable vector, cvxpy vector)"""
        mats, vecs = [], []
        for name, vl in (('f', self.vf), ('FA', self.vFA)) + ((('FB', self.vFB),) if self.B else ()):
            for k, C in enumerate(blocks[name]):
                if C is None:
                    continue
                mats.append(C); vecs.append(vl[k])
        return mats, vecs

    def lin(self, blocks):
        mats, vecs = self.expr(blocks)
        return sum(sp.csr_matrix(M) @ v for M, v in zip(mats, vecs))

    def lin_value(self, blocks):
        mats, vecs = self.expr(blocks)
        return sum(M @ v.value for M, v in zip(mats, vecs))

    # ---- the terms
    def point_A(self):
        b = self._zero(1)
        one = np.ones(1)
        eA = np.zeros((1, self.nl)); eA[0, 0] = 1
        for k in range(1, self.d + 1):
            self._add(b, 'f', k, sym_coef_rank1(eA, eA, one))
        T1 = embed(cheb(one, self.m[0]), 0, self.nl)
        self._add(b, 'FA', 0, sym_coef_rank1(T1, T1, one))
        return b

    def point_B(self):
        b = self._zero(1)
        one = np.ones(1)
        eB = np.zeros((1, 2)); eB[0, 1] = 1
        for k in range(1, self.d + 1):
            self._add(b, 'f', k, sym_coef_rank1(eB, eB, one))
        T1 = embed(cheb(one, self.m[0]), 1, 2)
        self._add(b, 'FB', 0, sym_coef_rank1(T1, T1, one))
        return b

    def pair_AA(self, u):
        N = len(u); b = self._zero(N); one = np.ones(N)
        eA = np.zeros((N, self.nl)); eA[:, 0] = 1
        for k in range(1, self.d + 1):
            self._add(b, 'f', k, sym_coef_rank1(eA, eA, 2 * gegen(k, u)))
        T1 = embed(cheb(one, self.m[0]), 0, self.nl)
        Tu0 = embed(cheb(u, self.m[0]), 0, self.nl)
        self._add(b, 'FA', 0, sym_coef_rank1(T1, Tu0, 4 * one))
        for k in range(self.d + 1):
            Tu = embed(cheb(u, self.m[k]), 0, self.nl)
            self._add(b, 'FA', k, sym_coef_rank1(Tu, Tu, 2 * (1 - u * u) ** k))
        return b

    def pair_AB(self, u):
        N = len(u); b = self._zero(N); one = np.ones(N)
        eA = np.zeros((N, 2)); eA[:, 0] = 1
        eB = np.zeros((N, 2)); eB[:, 1] = 1
        for k in range(1, self.d + 1):
            self._add(b, 'f', k, sym_coef_rank1(eA, eB, 2 * gegen(k, u)))
        m0 = self.m[0]
        T1A = embed(cheb(one, m0), 0); T1B = embed(cheb(one, m0), 1)
        TuA = embed(cheb(u, m0), 0); TuB = embed(cheb(u, m0), 1)
        self._add(b, 'FA', 0, sym_coef_rank1(T1A, TuB, 2 * one))           # (a;a,B),(a;B,a)
        self._add(b, 'FB', 0, sym_coef_rank1(T1B, TuA, 2 * one))           # (B;B,a),(B;a,B)
        for k in range(self.d + 1):
            TuBk = embed(cheb(u, self.m[k]), 1)
            self._add(b, 'FA', k, sym_coef_rank1(TuBk, TuBk, (1 - u * u) ** k))   # (a;B,B)
            if k == 0:
                self._add(b, 'FB', 0, sym_coef_rank1(TuA, TuA, one))            # (B;a,a)
            else:
                Tuk = cheb(u, self.m[k])
                self._add(b, 'FB', k, sym_coef_rank1(Tuk, Tuk, (1 - u * u) ** k))
        return b

    def triple_AAA(self, a, bb, c):
        """sum over the 6 orderings; pole type A"""
        N = len(a); b = self._zero(N)
        for k in range(self.d + 1):
            m = self.m[k]
            acc = np.zeros((N, m, m))
            for (u, v, w) in ((a, bb, c), (bb, a, c), (a, c, bb), (c, a, bb), (bb, c, a), (c, bb, a)):
                acc += phi(k, u, v, w, self.LC)[:, None, None] * cheb(u, m)[:, :, None] * cheb(v, m)[:, None, :]
            if self.nl == 2:
                big = np.zeros((N, 2 * m, 2 * m)); big[:, :m, :m] = acc
                acc = big
            self._add(b, 'FA', k, sym_coef_full(acc))
        return b

    def triple_AAB(self, u, v, w):
        """u = <a1,a2>, v = <a1,B>, w = <a2,B>"""
        N = len(u); b = self._zero(N)
        for k in range(self.d + 1):
            m = self.m[k]
            TuA = embed(cheb(u, m), 0)
            self._add(b, 'FA', k, sym_coef_rank1(TuA, embed(cheb(v, m), 1), 2 * phi(k, u, v, w, self.LC)))
            self._add(b, 'FA', k, sym_coef_rank1(TuA, embed(cheb(w, m), 1), 2 * phi(k, u, w, v, self.LC)))
            if k == 0:
                self._add(b, 'FB', 0, sym_coef_rank1(embed(cheb(v, m), 0), embed(cheb(w, m), 0), 2 * phi(0, v, w, u, self.LC)))
            else:
                self._add(b, 'FB', k, sym_coef_rank1(cheb(v, m), cheb(w, m), 2 * phi(k, v, w, u, self.LC)))
        return b


# ---------------- samples
def grid1(lo, hi, n):
    s = np.linspace(0, 1, n)
    return hi - (hi + 1) * (1 - s) ** 1.6 if lo == -1 else np.linspace(lo, hi, n)


def adm(u, v, w):
    return 1 + 2 * u * v * w - u * u - v * v - w * w >= -1e-12


def grid_AAA(t, n_uv, n_w):
    g = grid1(-1, t, n_uv)
    out = []
    for i, u in enumerate(g):
        for v in g[i:]:
            r = np.sqrt(max((1 - u * u) * (1 - v * v), 0))
            lo, hi = max(u * v - r, v), min(u * v + r, t)
            if hi < lo:
                continue
            for w in np.linspace(lo, hi, n_w):
                out.append((u, v, w))
    return np.array(out)


def grid_AAB(t, tau, n_u, n_v, n_w):
    gu = grid1(-1, t, n_u); gv = grid1(-1, tau, n_v)
    out = []
    for u in gu:
        for i, v in enumerate(gv):
            # w in [v, tau], admissible with (u, v): w in [uv - r, uv + r]
            r = np.sqrt(max((1 - u * u) * (1 - v * v), 0))
            lo, hi = max(u * v - r, v), min(u * v + r, tau)
            if hi < lo:
                continue
            for w in np.linspace(lo, hi, n_w):
                out.append((u, v, w))
    return np.array(out)


def rand_triples(n, lims):
    """random (x,y,z) on S^3 and on great 2-spheres (boundary), inner products (xy, xz, yz) <= lims"""
    outs = []
    for dim in (4, 3):
        X = rng.normal(size=(20 * n, 3, dim)); X /= np.linalg.norm(X, axis=2, keepdims=True)
        g = np.stack([np.einsum('ni,ni->n', X[:, 0], X[:, 1]), np.einsum('ni,ni->n', X[:, 0], X[:, 2]),
                      np.einsum('ni,ni->n', X[:, 1], X[:, 2])], 1)
        outs.append(g[(g <= np.array(lims)).all(1)][:n])
    # plus points pushed to the upper limits
    g = np.concatenate(outs)
    return g


def solve(mode, d, kap, tau, n, rounds=4, verbose=True, sizes=None):
    t = 0.5 + kap
    withB = (mode == 'hole')
    C = Cert(d, withB)
    sizes = sizes or dict(p=400, g3=(22, 10), g3b=(16, 16, 8), r3=3000)
    uAA = grid1(-1, t, sizes['p'])
    S3 = grid_AAA(t, *sizes['g3'])
    S3 = np.r_[S3, np.sort(rand_triples(sizes['r3'], (t, t, t)), 1)]
    if withB:
        uAB = grid1(-1, tau, sizes['p'])
        SB = grid_AAB(t, tau, *sizes['g3b'])
        rb = rand_triples(sizes['r3'], (t, tau, tau))
        SB = np.r_[SB, rb]
    cAA, cAB, cAAA, cAAB = cp.Variable(), cp.Variable(), cp.Variable(), cp.Variable()
    res = None
    for r in range(rounds):
        t0 = time.time()
        pA = C.lin(C.point_A())
        cons = [C.lin(C.pair_AA(uAA)) <= cAA, C.lin(C.triple_AAA(*S3.T)) <= cAAA] + ([C.trace() == 1] if NORM == 'trace' else [cAA == -1, C.trace() <= TMAX])
        val = n * pA + comb2(n) * cAA + comb3(n) * cAAA
        if withB:
            pB = C.lin(C.point_B())
            cons += [C.lin(C.pair_AB(uAB)) <= cAB, C.lin(C.triple_AAB(*SB.T)) <= cAAB]
            val = val + pB + n * cAB + comb2(n) * cAAB
        prob = cp.Problem(cp.Minimize(val), cons)
        global LAST
        LAST = dict(cons=cons, uAA=uAA.copy(), uAB=(uAB.copy() if withB else None))
        try:
            prob.solve(solver='CLARABEL', max_iter=500, tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7, static_regularization_constant=1e-6, equilibrate_max_iter=50, tol_ktratio=1e-6, max_threads=int(os.environ.get('CL_THREADS', '1')))
        except Exception as e:
            print('clarabel failed', e, flush=True)
            raise
        # fine check
        uu = grid1(-1, t, 20001)
        vAA = C.lin_value(C.pair_AA(uu)) - cAA.value
        g3 = np.r_[np.sort(rand_triples(40000, (t, t, t)), 1), grid_AAA(t, 60, 25)]
        v3 = np.concatenate([C.lin_value(C.triple_AAA(*g3[i:i + 20000].T)) for i in range(0, len(g3), 20000)]) - cAAA.value
        viol = {'AA': vAA.max(), 'AAA': v3.max()}
        # corrected value: raise each constant by its worst violation
        corr = comb2(n) * max(vAA.max(), 0) + comb3(n) * max(v3.max(), 0)
        if withB:
            uu2 = grid1(-1, tau, 20001)
            vAB = C.lin_value(C.pair_AB(uu2)) - cAB.value
            gb = np.r_[rand_triples(40000, (t, tau, tau)), grid_AAB(t, tau, 40, 40, 16)]
            vb = np.concatenate([C.lin_value(C.triple_AAB(*gb[i:i + 20000].T)) for i in range(0, len(gb), 20000)]) - cAAB.value
            viol.update(AB=vAB.max(), AAB=vb.max())
            corr += n * max(vAB.max(), 0) + comb2(n) * max(vb.max(), 0)
        res = dict(status=prob.status + '/' + SOLINFO.get('status', '?'), val=prob.value, dobj=SOLINFO.get('dobj'), corrected=prob.value + corr, viol=viol,
                   pA=float(pA.value[0]) if hasattr(pA.value, '__len__') else float(pA.value),
                   cAA=float(cAA.value), cAAA=float(cAAA.value), trace=float(C.trace().value))
        if withB:
            res.update(cAB=float(cAB.value), cAAB=float(cAAB.value))
        if verbose:
            print('  %s d=%d kappa=%.4f tau=%.4f n=%.3f round %d: %s  Val=%.3e (dual bound %.3e) corrected=%.3e trace=%.1f viol %s  [%d+%d rows, %.0fs]'
                  % (mode, d, kap, tau, n, r + 1, res['status'], prob.value, res['dobj'], res['corrected'], res['trace'],
                     {k: '%.1e' % v for k, v in viol.items()}, len(S3), len(SB) if withB else 0, time.time() - t0), flush=True)
        # refine
        uAA = np.r_[uAA, uu[np.argsort(vAA)[-100:]]]
        S3 = np.r_[S3, g3[np.argsort(v3)[-800:]]]
        if withB:
            uAB = np.r_[uAB, uu2[np.argsort(vAB)[-100:]]]
            SB = np.r_[SB, gb[np.argsort(vb)[-1200:]]]
    try:
        out = {'f%d' % k: C.f[k].value for k in range(1, d + 1)}
        out.update({'FA%d' % k: C.FA[k].value for k in range(d + 1)})
        if withB:
            out.update({'FB%d' % k: C.FB[k].value for k in range(d + 1)})
        out.update({k: v for k, v in res.items() if isinstance(v, (int, float))})
        np.savez('cert_%s_d%d_k%.4f_t%.4f_n%.2f.npz' % (mode, d, kap, tau, n), **out)
    except Exception as e:
        print('could not save certificate', e)
    return res, C


def comb2(n):
    return n * (n - 1) / 2


def comb3(n):
    return n * (n - 1) * (n - 2) / 6


if __name__ == '__main__':
    mode, d, kap, tau, n = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    rounds = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    res, _ = solve(mode, d, kap, tau, n, rounds)
    print('RESULT', mode, d, kap, tau, n, res, flush=True)
