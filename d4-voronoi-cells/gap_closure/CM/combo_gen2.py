#!/usr/bin/env python3
"""
combo_gen2.py -- combo_gen.py with the count vectors grouped by type counts and the
bin part bounded by LP duality, so that the programme stays small.

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
    out = {s: 0 for s in TYPES}
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
    # group the count vectors by their type counts; the bin part max_n sum_b n_b m_b
    # over the (continuous relaxation of the) bin polytope is bounded by LP duality:
    #   max {m.n : G n <= h, E n = e, n >= 0} <= h.y + e.mu  whenever G^T y + E^T mu >= m, y >= 0.
    G, h = [], []
    for bs, lo, hi in CASE['constraints']:
        row = np.zeros(len(BINS)); row[bs] = 1
        G.append(row); h.append(hi); G.append(-row); h.append(-lo)
    G, h = np.array(G), np.array(h, float)
    E = np.array([[1.0 if BINS[b][0] == ty else 0.0 for b in range(len(BINS))] for ty in TYPES])
    seen = set()
    for nb in COUNTS:
        tc = tcounts(nb)
        key = tuple(tc[ty] for ty in TYPES)
        if key in seen:
            continue
        seen.add(key)
        y = cp.Variable(len(h), nonneg=True); mu = cp.Variable(len(TYPES))
        cons.append(G.T @ y + E.T @ mu >= m)
        bound = h @ y + np.array(key, float) @ mu + sum(tc[ty] * pvec[ti[ty]] for ty in TYPES) + t / 2
        bound = bound + sum(Npair(tc, s, tt) * c2[(s, tt)] for (s, tt) in c2)
        bound = bound + sum(Ntriple(tc, list(cb)) * c3[cb] for cb in c3)
        cons.append(bound <= eta)
    print('   %d type-count vectors' % len(seen), flush=True)
    prob = cp.Problem(cp.Minimize(eta), cons)
    prob.solve(solver='CLARABEL', max_iter=500)
    return prob, L, X3, A, z, t, m, c2, c3, pvec


def tbound(s, t):
    return amax(TRANGE[s][1], TRANGE[t][1])


KEEP_P, KEEP_T = int(os.environ.get('KEEP_P', '5000')), int(os.environ.get('KEEP_T', '2500'))
KEEP_BASE = os.environ.get('KEEP_BASE') == '1'   # keep the starting pair grid through every pruning
TCOPLANAR = int(os.environ.get('TCOPLANAR', '0'))  # n > 0: a fixed n-by-n grid of coplanar triples per kind
TSYM = int(os.environ.get('TSYM', '0'))            # n > 0: a fixed n-by-n grid on the symmetric slices per kind
PERM_T = {}                                        # the fixed triples, kept through every pruning


def coplanar_grid(T12, T13, T23, n):
    """triples of unit vectors in one 2-plane, the boundary of the Gram domain: with
    the first at angle 0 and the others at angles a and +-b, the inner products are
    cos a, cos b and cos(a -+ b); a and b run over n equally spaced angles each."""
    a = np.linspace(np.arccos(T12), np.pi, n)
    b = np.linspace(np.arccos(T13), np.pi, n)
    A, B = (x.ravel() for x in np.meshgrid(a, b))
    out = []
    for w in (np.cos(A - B), np.cos(A + B)):
        P = np.stack([np.cos(A), np.cos(B), w], 1)
        out.append(P[w <= T23])
    return np.unique(np.round(np.concatenate(out), 12), axis=0)


def symmetric_grid(cb, T12, T13, T23, n):
    """triples (u12, u13, u23) fixed by the exchange of two points of the same type, where
    the triple inequality of a symmetric kernel has its critical points: u13 = u23 when the
    first two types agree, u12 = u13 when the last two do, and for three equal types also
    the diagonal u12 = u13 = u23.  The free inner products run over n equally spaced values
    each, and only admissible triples (Gram determinant >= 0) are kept."""
    out = []
    if cb[0] == cb[1]:                      # u12 = x, u13 = u23 = y
        X, Y = (v.ravel() for v in np.meshgrid(np.linspace(-1, T12, n), np.linspace(-1, min(T13, T23), n)))
        out.append(np.stack([X, Y, Y], 1))
    if cb[1] == cb[2]:                      # u23 = x, u12 = u13 = y
        X, Y = (v.ravel() for v in np.meshgrid(np.linspace(-1, T23, n), np.linspace(-1, min(T12, T13), n)))
        out.append(np.stack([Y, Y, X], 1))
    if cb[0] == cb[1] == cb[2]:
        d = np.linspace(-0.5, T12, 4 * n)
        out.append(np.stack([d, d, d], 1))
    if not out:
        return np.zeros((0, 3))
    P = np.concatenate(out)
    P = P[1 + 2 * P[:, 0] * P[:, 1] * P[:, 2] - (P ** 2).sum(1) >= 0]
    return np.unique(np.round(P, 12), axis=0)
NEWP, NEWT = int(os.environ.get('NEWP', '3000')), int(os.environ.get('NEWT', '400'))   # worst new samples added per kind and round


def prune(psamp, tsamp, x3v, Av, c2v, c3v, L):
    """keep the KEEP_P pair samples and KEEP_T triple samples of each kind that come
    closest to violating their constraint at the solution (x3v, Av, c2v, c3v)."""
    ti = {s: i for i, s in enumerate(TYPES)}
    for st in psamp:
        p, q, u = psamp[st]
        if len(u) <= KEEP_P:
            continue
        Kv = np.einsum('nk,na,kab,nb->n', ubasis(u, D2), pbasis(p, R2), Av, pbasis(q, R2))
        B = T3.Builder(L); T3.add_pair(B, np.arange(len(u)), ti[st[0]], ti[st[1]], u)
        v = Kv - pair(p / 2, q / 2, u) + B.matrix(len(u)) @ x3v - c2v[st]
        w = np.argsort(v)[-KEEP_P:]
        psamp[st] = (p[w], q[w], u[w])
        if KEEP_BASE:
            # the starting grid stays, so that no region of the pairs is left unsampled
            # after a round in which it was far from binding
            X = np.unique(np.r_[np.stack(psamp[st], 1), np.stack(pair_samples(*st), 1)], axis=0)
            psamp[st] = (X[:, 0], X[:, 1], X[:, 2])
    for cb in tsamp:
        g = tsamp[cb]
        if len(g) <= KEEP_T or cb not in c3v:
            continue
        B = T3.Builder(L); T3.add_triple(B, np.arange(len(g)), tuple(ti[x] for x in cb), g[:, 0], g[:, 1], g[:, 2])
        v = B.matrix(len(g)) @ x3v - c3v[cb]
        tsamp[cb] = g[np.argsort(v)[-KEEP_T:]]
        if cb in PERM_T:
            tsamp[cb] = np.unique(np.r_[tsamp[cb], PERM_T[cb]], axis=0)


REFINE = int(os.environ.get('REFINE', '60'))   # starts of the local ascent per kind and round (0: none)
_E6 = np.vstack([np.eye(3), -np.eye(3)])


def pair_values(st, X, x3v, Av, L, cst):
    """K - Pi + PAIR3_st - c_st at the rows (p, q, u) of X"""
    ti = {s: i for i, s in enumerate(TYPES)}
    p, q, u = X[:, 0], X[:, 1], X[:, 2]
    Kv = np.einsum('nk,na,kab,nb->n', ubasis(u, D2), pbasis(p, R2), Av, pbasis(q, R2))
    B = T3.Builder(L); T3.add_pair(B, np.arange(len(u)), ti[st[0]], ti[st[1]], u)
    return Kv - pair(p / 2, q / 2, u) + B.matrix(len(u)) @ x3v - cst


def triple_values(cb, X, x3v, L, cst):
    """TRIPLE3_cb - c_cb at the rows (u12, u13, u23) of X"""
    ti = {s: i for i, s in enumerate(TYPES)}
    B = T3.Builder(L); T3.add_triple(B, np.arange(len(X)), tuple(ti[x] for x in cb), X[:, 0], X[:, 1], X[:, 2])
    return B.matrix(len(X)) @ x3v - cst


def climb(f, X, proj, feas, h0=0.02, hmin=1e-7, iters=300):
    """compass search from each row of X for a local maximum of f over the admissible
    set (proj maps onto the box constraints, feas tests the rest); it only ever moves
    to admissible points of larger value, so the values it returns are attained."""
    X = proj(np.array(X, float)); K = len(X)
    val = f(X); h = np.full(K, h0)
    for _ in range(iters):
        live = h >= hmin
        if not live.any():
            break
        idx = np.flatnonzero(live)
        Cd = proj((X[idx, None, :] + h[idx, None, None] * _E6[None]).reshape(-1, 3))
        ok = feas(Cd)
        v = np.full(len(Cd), -np.inf)
        if ok.any():
            v[ok] = f(Cd[ok])
        v = v.reshape(len(idx), 6); j = v.argmax(1); best = v[np.arange(len(idx)), j]
        up = best > val[idx]
        X[idx[up]] = Cd.reshape(len(idx), 6, 3)[up, j[up]]; val[idx[up]] = best[up]
        h[idx[up]] *= 1.5; h[idx[~up]] /= 2
    return X, val


def pair_proj(st):
    (a0, a1), (b0, b1) = TRANGE[st[0]], TRANGE[st[1]]

    def proj(X):
        X[:, 0] = np.clip(X[:, 0], a0, a1); X[:, 1] = np.clip(X[:, 1], b0, b1)
        X[:, 2] = np.clip(X[:, 2], -1, amax(X[:, 0], X[:, 1]))
        return X
    return proj


def triple_proj(cb):
    T = np.array([tbound(cb[0], cb[1]), tbound(cb[0], cb[2]), tbound(cb[1], cb[2])])

    def proj(X):
        return np.clip(X, -1, T)
    return proj


def gram_ok(X):
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    return 1 + 2 * a * b * c - a * a - b * b - c * c >= 0


def main():
    d3 = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    psamp = {st: pair_samples(*st) for st in itertools.combinations_with_replacement(TYPES, 2)}
    tsamp = {}
    for combo in itertools.combinations_with_replacement(TYPES, 3):
        T12, T13, T23 = tbound(combo[0], combo[1]), tbound(combo[0], combo[2]), tbound(combo[1], combo[2])
        g1, g2, g3 = (int(x) for x in os.environ.get('TGRID', '12,12,6').split(','))
        tsamp[combo] = np.r_[T3.triple_grid(T12, T13, T23, g1, g2, g3),
                             T3.random_triples(T12, T13, T23, int(os.environ.get('TRAND', '1000')))]
    if os.environ.get('RESUME'):
        sv = np.load(os.environ['RESUME'])
        for st in psamp:
            psamp[st] = tuple(sv['p_' + ''.join(st)])
        for cb in tsamp:
            if 't_' + ''.join(cb) in sv:
                tsamp[cb] = sv['t_' + ''.join(cb)]
        L0 = T3.Layout(len(TYPES), d3)
        c2keys = list(itertools.combinations_with_replacement(TYPES, 2))
        c3keys = [cb for cb in itertools.combinations_with_replacement(TYPES, 3)
                  if not all(Ntriple(tcounts(nb), list(cb)) == 0 for nb in COUNTS)]
        if os.environ.get('RESUME_NOPRUNE') == '1':
            # samples of a certificate of another degree: keep them all, since its
            # thresholds say nothing about the kernel of this degree
            print('resumed from %s, all samples kept' % os.environ['RESUME'], flush=True)
        else:
            prune(psamp, tsamp, sv['x3'], sv['A'], dict(zip(c2keys, sv['c2'])), dict(zip(c3keys, sv['c3'])), L0)
            print('resumed from %s, pruned to %d pair and %d triple samples per kind' % (os.environ['RESUME'], KEEP_P, KEEP_T), flush=True)
    if TCOPLANAR:
        for cb in tsamp:
            PERM_T[cb] = coplanar_grid(tbound(cb[0], cb[1]), tbound(cb[0], cb[2]), tbound(cb[1], cb[2]), TCOPLANAR)
            tsamp[cb] = np.unique(np.r_[tsamp[cb], PERM_T[cb]], axis=0)
        print('fixed coplanar triples per kind: %s' % {''.join(k): len(v) for k, v in PERM_T.items()}, flush=True)
    if TSYM:
        for cb in tsamp:
            sg = symmetric_grid(cb, tbound(cb[0], cb[1]), tbound(cb[0], cb[2]), tbound(cb[1], cb[2]), TSYM)
            PERM_T[cb] = np.unique(np.r_[PERM_T.get(cb, np.zeros((0, 3))), sg], axis=0)
            tsamp[cb] = np.unique(np.r_[tsamp[cb], sg], axis=0)
        print('fixed triples per kind with the symmetric slices: %s' % {''.join(k): len(v) for k, v in PERM_T.items()}, flush=True)
    for rnd in range(rounds):
        t0 = time.time()
        prob, L, X3, A, z, t, m, c2, c3, pvec = build_and_solve(d3, rounds, psamp, tsamp)
        x3v = np.concatenate([np.r_[np.diag(X3[n].value), X3[n].value[np.triu_indices(X3[n].shape[0], 1)]] for n in L.blocks])
        Av = np.array([a.value for a in A])
        ti = {s: i for i, s in enumerate(TYPES)}
        # violations between the samples
        worst_total = 0.0; report = []
        newp, newt = {}, {}
        TCK = sorted({tuple(tcounts(nb)[ty] for ty in TYPES) for nb in COUNTS})
        corr = {k: 0.0 for k in TCK}
        for st in c2:
            p, q, u = rand_pairs(*st, 300000)
            Kv = np.einsum('nk,na,kab,nb->n', ubasis(u, D2), pbasis(p, R2), Av, pbasis(q, R2))
            x3part = []
            for i in range(0, len(u), 20000):
                B = T3.Builder(L); T3.add_pair(B, np.arange(len(u[i:i + 20000])), ti[st[0]], ti[st[1]], u[i:i + 20000])
                x3part.append(B.matrix(len(u[i:i + 20000])) @ x3v)
            v = Kv - pair(p / 2, q / 2, u) + np.concatenate(x3part) - c2[st].value
            w = np.argsort(v)[-NEWP:]
            newp[st] = (p[w], q[w], u[w])
            vmax = v.max()
            if REFINE:
                old = np.stack(psamp[st], 1)
                vo = pair_values(st, old, x3v, Av, L, c2[st].value)
                starts = np.r_[np.stack([p[w[-REFINE:]], q[w[-REFINE:]], u[w[-REFINE:]]], 1), old[np.argsort(vo)[-REFINE:]]]
                X, val = climb(lambda Y: pair_values(st, Y, x3v, Av, L, c2[st].value), starts, pair_proj(st),
                               lambda Y: np.ones(len(Y), bool))
                vmax = max(vmax, val.max())
                newp[st] = tuple(np.r_[a, X[:, i]] for i, a in enumerate(newp[st]))
            report.append('%s%s %.1e' % (st[0], st[1], vmax))
            for k in TCK:
                corr[k] += Npair(dict(zip(TYPES, k)), *st) * max(vmax, 0)
        for cb in c3:
            T12, T13, T23 = tbound(cb[0], cb[1]), tbound(cb[0], cb[2]), tbound(cb[1], cb[2])
            g = np.r_[T3.random_triples(T12, T13, T23, int(os.environ.get('TPROBE', '8000'))), T3.triple_grid(T12, T13, T23, 26, 26, 10)]
            if TCOPLANAR:
                g = np.r_[g, coplanar_grid(T12, T13, T23, 2 * TCOPLANAR + 1)]
            if TSYM:
                g = np.r_[g, symmetric_grid(cb, T12, T13, T23, 2 * TSYM + 1)]
            vals = []
            for i in range(0, len(g), 5000):
                gg = g[i:i + 5000]
                B = T3.Builder(L); T3.add_triple(B, np.arange(len(gg)), tuple(ti[x] for x in cb), gg[:, 0], gg[:, 1], gg[:, 2])
                vals.append(B.matrix(len(gg)) @ x3v)
            v = np.concatenate(vals) - c3[cb].value
            newt[cb] = g[np.argsort(v)[-NEWT:]]
            vmax = v.max()
            if REFINE:
                old = tsamp[cb]
                vo = triple_values(cb, old, x3v, L, c3[cb].value)
                starts = np.r_[g[np.argsort(v)[-REFINE:]], old[np.argsort(vo)[-REFINE:]]]
                X, val = climb(lambda Y: triple_values(cb, Y, x3v, L, c3[cb].value), starts, triple_proj(cb), gram_ok)
                vmax = max(vmax, val.max())
                newt[cb] = np.r_[newt[cb], X]
            report.append('%s %.1e' % (''.join(cb), vmax))
            for k in TCK:
                corr[k] += Ntriple(dict(zip(TYPES, k)), list(cb)) * max(vmax, 0)
        print('d3=%d round %d: %s  bound %.5f (target %.5f), corrected <= %.5f  [%.0f s]\n   viol: %s' % (
            d3, rnd + 1, prob.status, prob.value, TARGET, prob.value + max(corr.values()), time.time() - t0, ', '.join(report)), flush=True)
        print('   c2: %s' % {''.join(k): round(float(v.value), 6) for k, v in c2.items()}, flush=True)
        print('   c3: %s' % {''.join(k): round(float(v.value), 7) for k, v in c3.items()}, flush=True)
        prune(psamp, tsamp, x3v, Av, {k: v.value for k, v in c2.items()}, {k: v.value for k, v in c3.items()}, L)
        for st in newp:
            psamp[st] = tuple(np.r_[a, b] for a, b in zip(psamp[st], newp[st]))
        for cb in newt:                                   # the climbs often end at the same point: keep each once
            tsamp[cb] = np.unique(np.round(np.r_[tsamp[cb], newt[cb]], 12), axis=0)
        np.savez('%s_d%d_r%d.npz' % (TAG, d3, rnd + 1), x3=x3v, A=Av, z=z.value, t=t.value, m=m.value,
                 bound=prob.value, c2=np.array([c2[k].value for k in c2]), c3=np.array([c3[k].value for k in c3]),
                 **{'p_' + ''.join(k): np.array(v) for k, v in psamp.items()},
                 **{'t_' + ''.join(k): v for k, v in tsamp.items()})


if __name__ == '__main__':
    main()
