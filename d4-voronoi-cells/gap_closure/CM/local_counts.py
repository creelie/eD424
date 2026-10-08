#!/usr/bin/env python3
"""
local_counts.py d3 case.json -- one round of combo_direct.py with local count constraints.

For a centre of type s and a band u >= alpha of inner products of directions, the
number of centres of type t whose directions lie in that band is at most
kappa(s, t, alpha): writing their directions as c e + sqrt(1 - c^2) v with e the
direction of the centre and v on the link sphere S^2, two of them with inner product
at most tau = a(top_t, top_t) have <v, v'> <= (tau - c c')/sqrt((1 - c^2)(1 - c'^2)),
and the solved cases of the Tammes problem (N <= 14; the bound of Fejes Toth above)
limit how many such v fit.  Summed over the centres,
    sum over pairs {i, j} with u_ij >= alpha of (lam_{s_i t_j} + lam_{s_j t_i})
        <= sum_s n_s sum_t kappa(s, t, alpha) lam_{s t},
so each pair inequality may be relaxed by the multipliers in its band and the bound
pays for them through the counts.  BANDS=a1,a2,... (empty: the plain programme).

It prints the value, the multipliers in use, and the dual of the pair constraints as
the pair distribution of the relaxed configuration: for the class of count vectors
with the largest weight, how many neighbours of each type a centre has in each band.
Floating point; used in no proof.  Run from gap_closure/CM.
"""
import itertools
import os
import sys
from math import pi, tan, acos

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d3 = int(sys.argv[1])
case = sys.argv[2]
sys.argv = [sys.argv[0], str(d3), '1', os.path.join(HERE, case)]
sys.path.insert(0, HERE)
import combo_direct as CD  # noqa: E402
from combo_direct import (G, T3, TYPES, BINS, COUNTS, CASE, D2, R2, pbasis, ubasis, pair, S,  # noqa: E402
                          tcounts, Npair, Ntriple, sp, clarabel, SQ2, Val, Prob, time, gc, math)




def link_cos(alpha, beta, tau, n=401):
    c = np.linspace(alpha, beta, n)
    cj, ck = np.meshgrid(c, c)
    f = (tau - cj * ck) / np.sqrt((1 - cj ** 2) * (1 - ck ** 2))
    return f.max()


def ft_angle(N):
    """the largest possible least angle of N points on S^2 is at most this (Fejes Toth)"""
    om = N * pi / (6 * (N - 2))
    return acos((1 / tan(om) ** 2 - 1) / 2)


# the largest least angle of N points on S^2 (Tammes problem), in degrees, rounded up:
# N <= 6 classical, 7 to 9 Schuette and van der Waerden, 10 and 11 Danzer, 12 Fejes Toth,
# 13 and 14 Musin and Tarasov
TAMMES = {3: 120.0, 4: 109.4713, 5: 90.0001, 6: 90.0001, 7: 77.8696, 8: 74.8585, 9: 70.5288,
          10: 66.1469, 11: 63.4350, 12: 63.4350, 13: 57.1368, 14: 55.6706}


def tammes_upper(N):
    """an upper bound on the largest least angle (radians) of N points on S^2"""
    if N in TAMMES:
        return TAMMES[N] * pi / 180
    return ft_angle(N)


def kappa(alpha, beta, tau, exact=True):
    s = link_cos(alpha, beta, tau)
    if s >= 1:
        return 10 ** 9
    phi = acos(max(-1.0, s))
    up = tammes_upper if exact else ft_angle
    N = 2
    while up(N + 1) >= phi:
        N += 1
    return N


def amax(d1, d2):
    return (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)



def build_and_solve(d3, rounds, psamp, tsamp):
    t0 = time.time()
    L = T3.Layout(len(TYPES), d3)
    n3 = L.n
    nA = R2 + 1
    off = n3
    aidx = np.zeros((D2 + 1, nA, nA), dtype=np.int64)
    for k in range(D2 + 1):
        for i in range(nA):
            for j in range(i, nA):
                aidx[k, i, j] = aidx[k, j, i] = off; off += 1
    zi = off + np.arange(nA); off += nA
    tix = off; off += 1
    c2keys = list(itertools.combinations_with_replacement(TYPES, 2))
    c2i = {key: off + i for i, key in enumerate(c2keys)}; off += len(c2keys)
    c3keys = [cb for cb in itertools.combinations_with_replacement(TYPES, 3)
              if not all(Ntriple(tcounts(nb), list(cb)) == 0 for nb in COUNTS)]
    c3i = {key: off + i for i, key in enumerate(c3keys)}; off += len(c3keys)
    nb = len(BINS)
    mi = off + np.arange(nb); off += nb
    etai = off; off += 1
    lami = {}
    for s in TYPES:
        for tt in TYPES:
            for al in BANDS:
                lami[(s, tt, al)] = off; off += 1
    Gm, h = [], []
    for bs, lo, hi in CASE['constraints']:
        row = np.zeros(nb); row[bs] = 1
        Gm.append(row); h.append(hi); Gm.append(-row); h.append(-lo)
    Gm, h = np.array(Gm), np.array(h, float)
    E = np.array([[1.0 if BINS[b][0] == ty else 0.0 for b in range(nb)] for ty in TYPES])
    keys = sorted({tuple(tcounts(v)[ty] for ty in TYPES) for v in COUNTS})
    yi, mui = {}, {}
    for key in keys:
        yi[key] = off + np.arange(len(h)); off += len(h)
        mui[key] = off + np.arange(len(TYPES)); off += len(TYPES)
    nvar = off
    ti = {s: i for i, s in enumerate(TYPES)}

    blocks, bvec = [], []          # nonnegative cone: A x <= b

    def kcoef(bp, bq, uk):
        """(N, (D2+1)*nA*nA) dense coefficients of K = sum_k uk_k bp^T A_k bq on the A entries,
        returned with the column indices (entries (i,j) and (j,i) share a column)."""
        N = bp.shape[0]
        outer = bp[:, :, None] * bq[:, None, :]                    # (N, nA, nA)
        coef = uk[:, :, None, None] * outer[:, None, :, :]          # (N, D2+1, nA, nA)
        return coef.reshape(N, -1), aidx.reshape(-1)

    def add(rowsmat, b):
        blocks.append(rowsmat.tocsr()); bvec.append(np.asarray(b, float))

    # pair constraints: K - Pi + PAIR3 - c2 <= 0
    for (s, tt) in c2keys:
        P, Q, W = psamp[(s, tt)]
        N = len(W)
        for i0 in range(0, N, 4000):
            sl = slice(i0, min(N, i0 + 4000)); n = sl.stop - sl.start
            cf, cols = kcoef(pbasis(P[sl], R2), pbasis(Q[sl], R2), ubasis(W[sl], D2))
            Ka = sp.csr_matrix((cf.ravel(), (np.repeat(np.arange(n), cf.shape[1]), np.tile(cols, n))), shape=(n, nvar))
            B = T3.Builder(L); T3.add_pair(B, np.arange(n), ti[s], ti[tt], W[sl])
            Bx = B.matrix(n)
            Bx = sp.csr_matrix((Bx.data, Bx.indices, Bx.indptr), shape=(n, nvar))
            Cc = sp.csr_matrix((-np.ones(n), (np.arange(n), np.full(n, c2i[(s, tt)]))), shape=(n, nvar))
            br, bc, bv = [], [], []
            for al in BANDS:
                rr = np.flatnonzero(W[sl] >= al)
                for key in {(s, tt, al), (tt, s, al)}:
                    br.append(rr); bc.append(np.full(len(rr), lami[key])); bv.append(np.full(len(rr), -2.0 if s == tt else -1.0))
            Lc = sp.csr_matrix((np.concatenate(bv) if bv else [], (np.concatenate(br) if br else [], np.concatenate(bc) if bc else [])), shape=(n, nvar))
            add(Ka + Bx + Cc + Lc, pair(P[sl] / 2, Q[sl] / 2, W[sl]))
    # triple constraints: TRIPLE3 - c3 <= 0
    for cb in c3keys:
        g = tsamp[cb]
        for i0 in range(0, len(g), 4000):
            gg = g[i0:i0 + 4000]; n = len(gg)
            B = T3.Builder(L); T3.add_triple(B, np.arange(n), tuple(ti[x] for x in cb), gg[:, 0], gg[:, 1], gg[:, 2])
            Bx = B.matrix(n)
            Bx = sp.csr_matrix((Bx.data, Bx.indices, Bx.indptr), shape=(n, nvar))
            Cc = sp.csr_matrix((-np.ones(n), (np.arange(n), np.full(n, c3i[cb]))), shape=(n, nvar))
            add(Bx + Cc, np.zeros(n))
    # bin brackets: S + K(d,d,1)/2 - z.p - m_b <= 0
    for b, (ty, lo, hi) in enumerate(BINS):
        dd = np.linspace(lo, hi, 120); n = len(dd)
        bd = pbasis(dd, R2)
        cf, cols = kcoef(bd, bd, np.ones((n, D2 + 1)))
        Ka = sp.csr_matrix((0.5 * cf.ravel(), (np.repeat(np.arange(n), cf.shape[1]), np.tile(cols, n))), shape=(n, nvar))
        Zc = sp.csr_matrix((-bd.ravel(), (np.repeat(np.arange(n), nA), np.tile(zi, n))), shape=(n, nvar))
        Mc = sp.csr_matrix((-np.ones(n), (np.arange(n), np.full(n, mi[b]))), shape=(n, nvar))
        add(Ka + Zc + Mc, -S(dd))
    # point terms p_s as rows over x3
    Bp = T3.Builder(L)
    for s in TYPES:
        T3.add_point(Bp, np.array([ti[s]]), ti[s])
    Pm = Bp.matrix(len(TYPES)).toarray()                               # (ntypes, n3)
    if lami:
        kl = list(lami.values())
        add(sp.csr_matrix((-np.ones(len(kl)), (np.arange(len(kl)), kl)), shape=(len(kl), nvar)), np.zeros(len(kl)))
    # LP duals, one per class of count vectors
    for key in keys:
        tc = dict(zip(TYPES, key))
        r, c, v = [], [], []
        for b in range(nb):                                             # m_b - G^T y - E^T mu <= 0
            r.append(b); c.append(mi[b]); v.append(1.0)
            for q in range(len(h)):
                if Gm[q, b]:
                    r.append(b); c.append(yi[key][q]); v.append(-Gm[q, b])
            for si in range(len(TYPES)):
                if E[si, b]:
                    r.append(b); c.append(mui[key][si]); v.append(-E[si, b])
        for q in range(len(h)):                                         # -y <= 0
            r.append(nb + q); c.append(yi[key][q]); v.append(-1.0)
        row = nb + len(h)                                               # the bound <= eta
        for q in range(len(h)):
            r.append(row); c.append(yi[key][q]); v.append(h[q])
        for si, s in enumerate(TYPES):
            r.append(row); c.append(mui[key][si]); v.append(float(key[si]))
            nz = np.nonzero(Pm[si])[0]
            r += [row] * len(nz); c += list(nz); v += list(tc[s] * Pm[si][nz])
        r.append(row); c.append(tix); v.append(0.5)
        for kk in c2keys:
            r.append(row); c.append(c2i[kk]); v.append(float(Npair(tc, *kk)))
        for cb in c3keys:
            r.append(row); c.append(c3i[cb]); v.append(float(Ntriple(tc, list(cb))))
        for (s, tt, al), ix in lami.items():
            r.append(row); c.append(ix); v.append(float(tc[s]) * KAPPA[(s, tt, al)])
        r.append(row); c.append(etai); v.append(-1.0)
        add(sp.csr_matrix((v, (r, c)), shape=(row + 1, nvar)), np.zeros(row + 1))
    Anon = sp.vstack(blocks, format='csr'); bnon = np.concatenate(bvec)
    blocks.clear(); bvec.clear()                                    # free the pieces at once
    nnon = Anon.shape[0]
    # PSD cones
    cones = [clarabel.NonnegativeConeT(nnon)]
    pr, pc, pv, pb = [], [], [], 0

    def psd(entry):
        """entry(i, j) -> variable index for i <= j of an n x n matrix; appends its svec rows"""
        nonlocal pb
        n = entry.n
        for j in range(n):
            for i in range(j + 1):
                pr.append(pb); pc.append(entry(i, j)); pv.append(-1.0 if i == j else -SQ2); pb += 1
        cones.append(clarabel.PSDTriangleConeT(n))

    class Ent:
        def __init__(self, n, f):
            self.n, self.f = n, f

        def __call__(self, i, j):
            return self.f(i, j)
    for k in range(1, D2 + 1):
        psd(Ent(nA, lambda i, j, k=k: aidx[k, i, j]))
    psd(Ent(nA + 1, lambda i, j: aidx[0, i, j] if j < nA else (zi[i] if i < nA else tix)))
    for name, (o, s) in L.blocks.items():
        I = L.idx(name)
        psd(Ent(s, lambda i, j, I=I: I[i, j]))
    Apsd = sp.csc_matrix((pv, (pr, pc)), shape=(pb, nvar))
    A = sp.vstack([Anon, Apsd], format='csr')
    del Anon, Apsd
    A = A.tocsc()
    b = np.r_[bnon, np.zeros(pb)]
    q = np.zeros(nvar); q[etai] = 1.0
    P = sp.csc_matrix((nvar, nvar))
    st = clarabel.DefaultSettings()
    st.max_iter = int(os.environ.get('MAX_ITER', '500'))
    st.verbose = bool(int(os.environ.get('VERBOSE', '0')))
    st.direct_solve_method = os.environ.get('DSM', st.direct_solve_method)
    st.max_threads = int(os.environ.get('THREADS', str(st.max_threads)))
    print('   direct: %d variables, %d inequalities, %d psd rows in %d cones, %d nonzeros [%.0f s to build]'
          % (nvar, nnon, pb, len(cones) - 1, A.nnz, time.time() - t0), flush=True)
    solver = clarabel.DefaultSolver(P, q, A, b, cones, st)
    del A                       # Clarabel keeps its own copy; drop ours before the factorisations
    gc.collect()
    sol = solver.solve()
    STASH['z'] = np.array(sol.z); STASH['x'] = np.array(sol.x); STASH['lami'] = lami
    del solver
    sname = str(sol.status).split('.')[-1]
    status = {'Solved': 'optimal', 'AlmostSolved': 'optimal_inaccurate'}.get(sname, sname)
    x = np.array(sol.x)
    X3 = {name: Val(x[L.idx(name)]) for name in L.blocks}
    Av = [Val(x[aidx[k]]) for k in range(D2 + 1)]
    z, t = Val(x[zi]), Val(float(x[tix]))
    m = Val(x[mi])
    c2 = {kk: Val(float(x[c2i[kk]])) for kk in c2keys}
    c3 = {cb: Val(float(x[c3i[cb]])) for cb in c3keys}
    pvec = Pm @ x[:n3]
    print('   %d type-count vectors' % len(keys), flush=True)
    return Prob(status, float(x[etai])), L, X3, Av, z, t, m, c2, c3, pvec



BANDS = [float(x) for x in os.environ.get('BANDS', '').split(',') if x]
KAPPA = {}
for s in TYPES:
    for tt in TYPES:
        top = G.tbound(s, tt)
        tau = G.tbound(tt, tt)
        for al in BANDS:
            KAPPA[(s, tt, al)] = float(kappa(al, top, tau)) if al < top else 0.0
print('bands %s; kappa %s' % (BANDS, {'%s%s%.2f' % k: int(v) for k, v in KAPPA.items()}), flush=True)
STASH = {}

psamp = {st: G.pair_samples(*st) for st in itertools.combinations_with_replacement(TYPES, 2)}
for st in psamp:
    extra = G.rand_pairs(*st, 6000)
    psamp[st] = tuple(np.r_[a, b] for a, b in zip(psamp[st], extra))
tsamp = {}
for combo in itertools.combinations_with_replacement(TYPES, 3):
    T12, T13, T23 = G.tbound(combo[0], combo[1]), G.tbound(combo[0], combo[2]), G.tbound(combo[1], combo[2])
    tsamp[combo] = np.r_[G.T3.triple_grid(T12, T13, T23, 12, 12, 6), G.T3.random_triples(T12, T13, T23, 1000)]
prob, L, X3, A, z, t, m, c2, c3, pvec = build_and_solve(d3, 1, psamp, tsamp)
print('status %s value %.5f (target 3.10330)' % (prob.status, prob.value), flush=True)
Z, X, lami = STASH['z'], STASH['x'], STASH['lami']
if lami:
    print('band multipliers: %s' % {'%s%s%.2f' % k: round(float(X[i]), 5) for k, i in lami.items() if X[i] > 1e-6})
c2keys = list(itertools.combinations_with_replacement(TYPES, 2))
off = 0
prow = {}
for st in c2keys:
    n = len(psamp[st][2])
    prow[st] = Z[off:off + n]; off += n
c3keys = [cb for cb in itertools.combinations_with_replacement(TYPES, 3)
          if not all(G.Ntriple(G.tcounts(nb), list(cb)) == 0 for nb in G.COUNTS)]
for cb in c3keys:
    off += len(tsamp[cb])
off += 120 * len(G.BINS) + len(lami)
keys = sorted({tuple(G.tcounts(v)[ty] for ty in TYPES) for v in G.COUNTS})
nb, nh = len(G.BINS), 2 * len(G.CASE['constraints'])
w = {}
for key in keys:
    off += nb + nh
    w[key] = Z[off]; off += 1
print('class weights (sum %.4f):' % sum(w.values()))
top = sorted(w.items(), key=lambda kv: -kv[1])[:6]
for k, v in top:
    print('   %s %.4f' % (dict(zip(TYPES, k)), v))
tc = dict(zip(TYPES, top[0][0]))
edges = [-1.0, -0.5, 0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
for st in c2keys:
    p, q, u = psamp[st]
    zz = prow[st]
    Sm = zz.sum()
    if Sm <= 1e-9 or G.Npair(tc, *st) == 0:
        continue
    cnt = zz / Sm * G.Npair(tc, *st)
    s, tt = st
    tb = G.tbound(*st)
    e = [x for x in edges if x < tb] + [tb + 1e-9]
    h, _ = np.histogram(u, bins=e, weights=cnt)
    out = []
    for i in range(len(h)):
        if h[i] > 1e-3:
            if s == tt:
                out.append('[%.2f,%.2f) %.2f' % (e[i], min(e[i + 1], tb), 2 * h[i] / tc[s]))
            else:
                out.append('[%.2f,%.2f) %.2f/%.2f' % (e[i], min(e[i + 1], tb), h[i] / tc[s], h[i] / tc[tt]))
    print('pair %s%s (top %.4f), neighbours per centre%s: %s' % (s, tt, tb, '' if s == tt else ' (%s of an %s / %s of an %s)' % (tt, s, s, tt), '; '.join(out)), flush=True)
    sel = u > tb - 0.05
    if zz[sel].sum() > 0:
        print('   near the top: mean p %.4f q %.4f u %.4f' % tuple(np.average(np.stack([p[sel], q[sel], u[sel]]), axis=1, weights=zz[sel])))
