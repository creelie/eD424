#!/usr/bin/env python3
"""
combo_direct.py -- combo_gen2.py with the programme handed to Clarabel in its own
standard form instead of through cvxpy.  The programme, the samples, the rounds, the
pruning and the files written are those of combo_gen2.py; only the construction of
the conic problem changes, which keeps the memory to the size of the constraint
matrix itself (cvxpy needed about 11 GB at 29 centres and degree 8).

Variables: the typed three-point blocks (each as diagonal plus strict upper triangle,
the layout of typed3pt.Layout), the upper triangles of A_0..A_D, z, t, the pair and
triple thresholds c2, c3, the bin brackets m, the objective eta, and for each class of
count vectors with equal type counts the multipliers (y, mu) of the LP dual that bounds
the bin part.  Cones: one nonnegative cone for all inequalities, and the PSD cones of
A_1..A_D, [[A_0, z], [z^T, t]] and the three-point blocks (Clarabel's triangle form:
upper triangle by columns, off-diagonal entries times sqrt 2).

Usage: python3 combo_direct.py d3 rounds case.json        (as combo_gen2.py)

With TRACE_REG=<eps> the objective is eta + eps * (the sum of the traces of the
three-point blocks) instead of eta.  The value printed is still eta, the bound.  The
traces grow with the degree (about 9 at degree 8 and 33 at degree 10 at 28 centres), and
large blocks make the triple polynomials swing between the sample points; the penalty
trades a little of the bound for smaller swings.
"""
import gc
import itertools
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import clarabel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combo_gen2 as G  # noqa: E402  (reads the case file from argv[3])
from combo_gen2 import T3, TYPES, BINS, COUNTS, CASE, D2, R2, pbasis, ubasis, pair, S, tcounts, Npair, Ntriple  # noqa: E402

SQ2 = math.sqrt(2.0)


class Val:
    def __init__(self, value):
        self.value = value
        self.shape = np.shape(value)


class Prob:
    def __init__(self, status, value):
        self.status, self.value = status, value


def svec_rows(entries):
    """entries: list over the upper triangle (column-major) of (variable index or None, scale);
    returns (cols, vals) of A = -scale * e_var, so that s = b - A x = svec(M)."""
    cols = np.array([v for v, _ in entries]); vals = np.array([-sc for _, sc in entries])
    return cols, vals


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
            add(Ka + Bx + Cc, pair(P[sl] / 2, Q[sl] / 2, W[sl]))
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
    reg = float(os.environ.get('TRACE_REG', '0'))
    if reg:
        for name, (o, s) in L.blocks.items():
            I = L.idx(name)
            q[[I[i, i] for i in range(s)]] += reg
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


G.build_and_solve = build_and_solve
G.TAG = os.environ.get('TAG', G.TAG + 'x')

if __name__ == '__main__':
    G.main()
