#!/usr/bin/env python3
"""
combo_gen.py -- combo30.py for a case given in a JSON file (argv[3]): statement (C)
at M centres: a two-point kernel with the distance as a
continuous label (as in radial_case_sdp.py) PLUS a typed three-point kernel on
the directions (typed3pt.py), whose types are distance ranges.

Fifth case of thm:count30 (with prop:C-radial): 30 centres within sqrt6, of which
  >= 22 within 2.05, >= 23 within 2.15, 24 within 2.25, and 6 in [2.4, sqrt6).
Types: A = [2, 2.05], B = (2.05, 2.25], F = [2.4, sqrt6).  Two-point bins:
  A, B1 = (2.05, 2.15], B2 = (2.15, 2.25], F.
Bin-count vectors: (nA, nB1, nB2, nF) in {(22,1,1,6), (22,2,0,6), (23,0,1,6), (23,1,0,6), (24,0,0,6)}.

For every such packing set Y (with U(Y) = sum S - sum Pi the union of the caps),
  U(Y) <= sum_y [f(|y|) + p_{type(y)}] + t/2 + sum_{pairs} [K - Pi + PAIR3_{st}(u)]
          + sum_{triples} TRIPLE3,
  f(d) = S(d) + K(d,d,1)/2 - z.p(d).
If K - Pi + PAIR3_st <= c_st on the admissible pairs of types (s,t), TRIPLE3 <= c_str
on the admissible typed triples, and f <= m_b on bin b, then
  U(Y) <= Bound(n) = sum_b n_b (m_b + p_{type(b)}) + t/2 + sum N_st(n) c_st + sum N_str(n) c_str.
(C) holds in the residual case if max_n Bound(n) < 9 pi^2/8 - 8.
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'multi_cap'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'C30'))
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from math import comb
from truncated_search import pair, S
from radial_count_sdp import pbasis, ubasis, TARGET
import typed3pt as T3

R6 = 6 ** .5
D2, R2 = int(os.environ.get('D2', '12')), int(os.environ.get('R2', '4'))
rng = np.random.default_rng(int(os.environ.get('SEED', '1')))

def amax(d1, d2):
    return (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)

import json
CASE = json.load(open(sys.argv[3]))
def _num(x):
    return R6 if x == 'sqrt6' else float(x)
TYPES = CASE['types']
TRANGE = {k: (_num(a), _num(b)) for k, (a, b) in CASE['trange'].items()}
BINS = [(t, _num(a), _num(b)) for t, a, b in CASE['bins']]


def _counts():
    """all count vectors over BINS with total M meeting the case constraints:
    each constraint [bins, lo, hi] bounds the number of points in those bins."""
    M, nb = CASE['M'], len(BINS)
    out = []
    def rec(prefix, left):
        if len(prefix) == nb - 1:
            v = prefix + [left]
            if all(lo <= sum(v[i] for i in bs) <= hi for bs, lo, hi in CASE['constraints']):
                out.append(tuple(v))
            return
        for k in range(left + 1):
            rec(prefix + [k], left - k)
    rec([], M)
    return out


COUNTS = _counts()
TAG = CASE.get('tag', 'case')


def tcounts(nb):
    out = {'A': 0, 'B': 0, 'F': 0}
    for (ty, lo, hi), n in zip(BINS, nb):
        out[ty] += n
    return out


def Npair(tc, s, t):
    return comb(tc[s], 2) if s == t else tc[s] * tc[t]


def Ntriple(tc, combo):
    out = 1
    for s in set(combo):
        out *= comb(tc[s], combo.count(s))
    return out


def pair_samples(s, t, n_d=9, n_u=40):
    (a0, a1), (b0, b1) = TRANGE[s], TRANGE[t]
    P, Q, W = [], [], []
    for d1 in np.linspace(a0, a1, n_d):
        for d2 in np.linspace(b0, b1, n_d):
            top = amax(d1, d2)
            for u in -1 + (top + 1) * (1 - np.cos(np.linspace(0, np.pi, n_u))) / 2:
                P.append(d1); Q.append(d2); W.append(u)
    for d1, d2 in ((a0, b0), (a1, b1), (a0, b1), (a1, b0)):
        top = amax(d1, d2)
        for u in -1 + (top + 1) * (1 - np.cos(np.linspace(0, np.pi, 300))) / 2:
            P.append(d1); Q.append(d2); W.append(u)
    return np.array(P), np.array(Q), np.array(W)


def rand_pairs(s, t, n):
    (a0, a1), (b0, b1) = TRANGE[s], TRANGE[t]
    p = a0 + (a1 - a0) * rng.random(n); q = b0 + (b1 - b0) * rng.random(n)
    k = n // 6
    p[:k] = a0; q[k:2 * k] = b0; p[2 * k:3 * k] = a1; q[3 * k:4 * k] = b1
    top = amax(p, q)
    u = -1 + (top + 1) * rng.random(n) ** 0.5
    u[4 * k:5 * k] = top[4 * k:5 * k] - 3e-3 * rng.random(k)
    return p, q, u


def build_and_solve(d3, rounds, psamp, tsamp):
    L = T3.Layout(len(TYPES), d3)
    X3 = {name: cp.Variable((s, s), PSD=True) for name, (off, s) in L.blocks.items()}
    def vec(name):
        M = X3[name]; s = M.shape[0]
        iu = np.triu_indices(s, 1)
        ii = np.r_[np.arange(s), iu[0]]; jj = np.r_[np.arange(s), iu[1]]
        Sel = sp.csr_matrix((np.ones(len(ii)), (np.arange(len(ii)), ii + jj * s)), shape=(len(ii), s * s))
        return Sel @ cp.vec(M, order='F')
    x3 = cp.hstack([vec(n) for n in L.blocks])
    A = [cp.Variable((R2 + 1, R2 + 1), symmetric=True) for _ in range(D2 + 1)]
    z, t = cp.Variable(R2 + 1), cp.Variable()
    Z = cp.bmat([[A[0], cp.reshape(z, (R2 + 1, 1), order='C')], [cp.reshape(z, (1, R2 + 1), order='C'), cp.reshape(t, (1, 1), order='C')]])
    cons = [A[k] >> 0 for k in range(1, D2 + 1)] + [Z >> 0]
    ti = {s: i for i, s in enumerate(TYPES)}
    # point terms of the three-point kernel
    B = T3.Builder(L)
    for s in TYPES:
        T3.add_point(B, np.array([ti[s]]), ti[s])
    pvec = B.matrix(len(TYPES)) @ x3
    # pair constraints
    c2 = {}
    for (s, tt) in itertools.combinations_with_replacement(TYPES, 2):
        P, Q, W = psamp[(s, tt)]
        pv = pair(P / 2, Q / 2, W)
        bp, bq, uk = pbasis(P, R2), pbasis(Q, R2), ubasis(W, D2)
        K = sum(cp.multiply(uk[:, k], cp.sum(cp.multiply(bp @ A[k], bq), axis=1)) for k in range(D2 + 1))
        B = T3.Builder(L); T3.add_pair(B, np.arange(len(W)), ti[s], ti[tt], W)
        c2[(s, tt)] = cp.Variable()
        cons.append(K - pv + B.matrix(len(W)) @ x3 <= c2[(s, tt)])
    # triple constraints
    c3 = {}
    for combo in itertools.combinations_with_replacement(TYPES, 3):
        if all(Ntriple(tcounts(nb), list(combo)) == 0 for nb in COUNTS):
            continue
        g = tsamp[combo]
        B = T3.Builder(L); T3.add_triple(B, np.arange(len(g)), tuple(ti[x] for x in combo), g[:, 0], g[:, 1], g[:, 2])
        c3[combo] = cp.Variable()
        cons.append(B.matrix(len(g)) @ x3 <= c3[combo])
    # bin brackets
    m = cp.Variable(len(BINS))
    for b, (ty, lo, hi) in enumerate(BINS):
        dd = np.linspace(lo, hi, 120)
        bd = pbasis(dd, R2)
        Kd = sum(cp.sum(cp.multiply(bd @ A[k], bd), axis=1) for k in range(D2 + 1))
        cons.append(S(dd) + Kd / 2 - bd @ z <= m[b])
    eta = cp.Variable()
    for nb in COUNTS:
        tc = tcounts(nb)
        bound = sum(nb[b] * (m[b] + pvec[ti[BINS[b][0]]]) for b in range(len(BINS))) + t / 2
        bound = bound + sum(Npair(tc, s, tt) * c2[(s, tt)] for (s, tt) in c2)
        bound = bound + sum(Ntriple(tc, list(cb)) * c3[cb] for cb in c3)
        cons.append(bound <= eta)
    prob = cp.Problem(cp.Minimize(eta), cons)
    prob.solve(solver='CLARABEL', max_iter=500)
    return prob, L, X3, A, z, t, m, c2, c3, pvec


def tbound(s, t):
    return amax(TRANGE[s][1], TRANGE[t][1])


def main():
    d3 = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    psamp = {st: pair_samples(*st) for st in itertools.combinations_with_replacement(TYPES, 2)}
    tsamp = {}
    for combo in itertools.combinations_with_replacement(TYPES, 3):
        T12, T13, T23 = tbound(combo[0], combo[1]), tbound(combo[0], combo[2]), tbound(combo[1], combo[2])
        tsamp[combo] = np.r_[T3.triple_grid(T12, T13, T23, 12, 12, 6), T3.random_triples(T12, T13, T23, 1000)]
    if os.environ.get('RESUME'):
        sv = np.load(os.environ['RESUME'])
        for st in psamp:
            psamp[st] = tuple(sv['p_' + ''.join(st)])
        for cb in tsamp:
            if 't_' + ''.join(cb) in sv:
                tsamp[cb] = sv['t_' + ''.join(cb)]
    for rnd in range(rounds):
        t0 = time.time()
        prob, L, X3, A, z, t, m, c2, c3, pvec = build_and_solve(d3, rounds, psamp, tsamp)
        x3v = np.concatenate([np.r_[np.diag(X3[n].value), X3[n].value[np.triu_indices(X3[n].shape[0], 1)]] for n in L.blocks])
        Av = np.array([a.value for a in A])
        ti = {s: i for i, s in enumerate(TYPES)}
        # violations between the samples
        worst_total = 0.0; report = []
        newp, newt = {}, {}
        corr = {nb: 0.0 for nb in COUNTS}
        for st in c2:
            p, q, u = rand_pairs(*st, 300000)
            Kv = np.einsum('nk,na,kab,nb->n', ubasis(u, D2), pbasis(p, R2), Av, pbasis(q, R2))
            B = T3.Builder(L); T3.add_pair(B, np.arange(len(u)), ti[st[0]], ti[st[1]], u)
            v = Kv - pair(p / 2, q / 2, u) + B.matrix(len(u)) @ x3v - c2[st].value
            report.append('%s%s %.1e' % (st[0], st[1], v.max()))
            for nb in COUNTS:
                corr[nb] += Npair(tcounts(nb), *st) * max(v.max(), 0)
            w = np.argsort(v)[-3000:]
            newp[st] = (p[w], q[w], u[w])
        for cb in c3:
            T12, T13, T23 = tbound(cb[0], cb[1]), tbound(cb[0], cb[2]), tbound(cb[1], cb[2])
            g = np.r_[T3.random_triples(T12, T13, T23, 8000), T3.triple_grid(T12, T13, T23, 26, 26, 10)]
            vals = []
            for i in range(0, len(g), 5000):
                gg = g[i:i + 5000]
                B = T3.Builder(L); T3.add_triple(B, np.arange(len(gg)), tuple(ti[x] for x in cb), gg[:, 0], gg[:, 1], gg[:, 2])
                vals.append(B.matrix(len(gg)) @ x3v)
            v = np.concatenate(vals) - c3[cb].value
            report.append('%s %.1e' % (''.join(cb), v.max()))
            for nb in COUNTS:
                corr[nb] += Ntriple(tcounts(nb), list(cb)) * max(v.max(), 0)
            newt[cb] = g[np.argsort(v)[-400:]]
        print('d3=%d round %d: %s  bound %.5f (target %.5f), corrected <= %.5f  [%.0f s]\n   viol: %s' % (
            d3, rnd + 1, prob.status, prob.value, TARGET, prob.value + max(corr.values()), time.time() - t0, ', '.join(report)), flush=True)
        print('   c2: %s' % {''.join(k): round(float(v.value), 6) for k, v in c2.items()}, flush=True)
        print('   c3: %s' % {''.join(k): round(float(v.value), 7) for k, v in c3.items()}, flush=True)
        for st in newp:
            psamp[st] = tuple(np.r_[a, b] for a, b in zip(psamp[st], newp[st]))
        for cb in newt:
            tsamp[cb] = np.r_[tsamp[cb], newt[cb]]
        np.savez('%s_d%d_r%d.npz' % (TAG, d3, rnd + 1), x3=x3v, A=Av, z=z.value, t=t.value, m=m.value,
                 bound=prob.value, c2=np.array([c2[k].value for k in c2]), c3=np.array([c3[k].value for k in c3]),
                 **{'p_' + ''.join(k): np.array(v) for k, v in psamp.items()},
                 **{'t_' + ''.join(k): v for k, v in tsamp.items()})


if __name__ == '__main__':
    main()
