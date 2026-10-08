"""c28_hull.py -- the best convex combination of existing combined certificates at 28 centres.

A convex combination of certificates is a certificate: the positivity conditions are
convex and every inequality is linear in the kernel, with thresholds recomputed for the
combination.  With only the weights as unknowns the programme is a small LP, so it can
take far more samples than the full programme, and the bulges of single certificates
between their samples can cancel.  Float throughout; proves nothing.

usage: python3 c28_hull.py TARGET ITERS cert.npz [cert.npz ...]
TARGET: k15f4, k16f1, k15f4+k16f1, lo16, ...  (see SETS)
env: NP, NT (random samples per kind), GRID (grid points per inner product), OUT (npz of
the best combination, written after every iteration).  A certificate of lower three-point
degree enters with its blocks embedded in the layout of the highest degree (zero blocks
for the extra degrees and zero rows for the extra Chebyshev indices), which leaves every
function of it unchanged.
"""
import sys, os, time, itertools
import numpy as np
from math import comb
from scipy.optimize import linprog
import scipy.sparse as sp


def blocks(x3, m, d):
    """the three-point blocks of typed3pt.Layout(m, d) from the stored vector"""
    names = [('f', k) for k in range(1, d + 1)] + [('F', P, k) for P in range(m) for k in range(d + 1)]
    out, off = {}, 0
    for name in names:
        s = m if name[0] == 'f' else m * (d - name[2] + 1)
        n = s * (s + 1) // 2
        v = x3[off:off + n]; off += n
        M = np.zeros((s, s)); M[np.arange(s), np.arange(s)] = v[:s]
        iu = np.triu_indices(s, 1); M[iu] = v[s:]; M[(iu[1], iu[0])] = v[s:]
        out[name] = M
    assert off == len(x3)
    return out


def to_x3(B, m, D):
    """the stored vector of typed3pt.Layout(m, D) from blocks (missing ones zero)"""
    names = [('f', k) for k in range(1, D + 1)] + [('F', P, k) for P in range(m) for k in range(D + 1)]
    parts = []
    for name in names:
        s = m if name[0] == 'f' else m * (D - name[2] + 1)
        M = B.get(name, np.zeros((s, s)))
        iu = np.triu_indices(s, 1)
        parts += [np.diag(M), M[iu]]
    return np.concatenate(parts)


def embed(x3, m, d, D):
    """x3 of degree d in the layout of degree D >= d"""
    if d == D:
        return np.array(x3, float)
    B = blocks(np.array(x3, float), m, d)
    out = {}
    for name, M in B.items():
        if name[0] == 'f':
            out[name] = M
        else:
            k = name[2]; mk, MK = d - k + 1, D - k + 1
            idx = np.array([Q * MK + i for Q in range(m) for i in range(mk)])
            E = np.zeros((m * MK, m * MK)); E[np.ix_(idx, idx)] = M
            out[name] = E
    return to_x3(out, m, D)


CM = os.path.dirname(os.path.abspath(__file__)) + '/'
target, iters, certs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
sys.argv = ['x', '8', '1', CM + 'case28_lo16.json']
sys.path.insert(0, CM)
import combo_gen2 as G
import typed3pt as T3
t0 = time.time()
TYPES = G.TYPES; ti = {s: i for i, s in enumerate(TYPES)}
BT = [b[0] for b in G.BINS]; nb = len(G.BINS)
DEG = {2052: 6, 4098: 8, 7188: 10, 11538: 12}
Zs = [np.load(c if c.startswith('/') else CM + c) for c in certs]
D = max(DEG[len(Z['x3'])] for Z in Zs)
K = len(Zs)


def clip(M):
    """the nearest positive semidefinite matrix: negative eigenvalues set to 0, as the exact
    check does before it rounds (solutions that stopped on a numerical error can have
    eigenvalues near -1e-6, which move the triple polynomials by up to 1e-5)"""
    M = (np.array(M, float) + np.array(M, float).T) / 2
    w, V = np.linalg.eigh(M)
    return (V * np.maximum(w, 0)) @ V.T


def projected(Z):
    d = DEG[len(Z['x3'])]
    B = {name: clip(M) for name, M in blocks(np.array(Z['x3'], float), 3, d).items()}
    A = np.array(Z['A'], float)
    nA = A.shape[1]
    Bd = np.zeros((nA + 1, nA + 1))
    Bd[:nA, :nA] = A[0]; Bd[:nA, nA] = Bd[nA, :nA] = np.array(Z['z'], float); Bd[nA, nA] = float(Z['t'])
    Bd = clip(Bd)
    A = np.stack([Bd[:nA, :nA]] + [clip(a) for a in A[1:]])
    return embed(to_x3(B, 3, d), 3, d, D), A, Bd[:nA, nA], Bd[nA, nA]


PJ = [projected(Z) for Z in Zs]
X3 = np.stack([p[0] for p in PJ], 1)                                                         # n3 x K
AV = np.stack([p[1] for p in PJ])                                                            # K x 13 x 5 x 5
ZV = np.stack([p[2] for p in PJ])                                                            # K x 5
TV = np.array([p[3] for p in PJ])
L = T3.Layout(3, D)
Bp = T3.Builder(L)
for s in TYPES:
    T3.add_point(Bp, np.array([ti[s]]), ti[s])
PT = Bp.matrix(3) @ X3                                                                       # 3 x K
rng = np.random.default_rng(int(os.environ.get('SEED', '3')))
NP, NT = int(os.environ.get('NP', '60000')), int(os.environ.get('NT', '60000'))
print('%d certificates, three-point degree %d, n3 %d [%.0f s]' % (K, D, L.n, time.time() - t0), flush=True)


def pair_vals(st, X):
    out = []
    for i0 in range(0, len(X), 10000):
        x = X[i0:i0 + 10000]; p, q, u = x[:, 0], x[:, 1], x[:, 2]
        U, Pp, Pq = G.ubasis(u, G.D2), G.pbasis(p, G.R2), G.pbasis(q, G.R2)
        Kv = np.einsum('nk,na,ikab,nb->ni', U, Pp, AV, Pq, optimize=True)
        B = T3.Builder(L); T3.add_pair(B, np.arange(len(u)), ti[st[0]], ti[st[1]], u)
        out.append((Kv + B.matrix(len(u)) @ X3 - G.pair(p / 2, q / 2, u)[:, None]).astype(np.float32))
    return np.concatenate(out)


def trip_vals(cb, X):
    out = []
    for i0 in range(0, len(X), 10000):
        x = X[i0:i0 + 10000]
        B = T3.Builder(L); T3.add_triple(B, np.arange(len(x)), tuple(ti[c] for c in cb), x[:, 0], x[:, 1], x[:, 2])
        out.append((B.matrix(len(x)) @ X3).astype(np.float32))
    return np.concatenate(out)


def bin_vals(d):
    Pd = G.pbasis(d, G.R2); U1 = G.ubasis(np.ones(1), G.D2)[0]
    Kv = np.einsum('k,na,ikab,nb->ni', U1, Pd, AV, Pd)
    return G.S(d)[:, None] + 0.5 * Kv - Pd @ ZV.T


def rand_gram(T, n):
    outs = []
    for dim in (4, 3, 2):
        V = rng.normal(size=(8 * n, 3, dim)); V /= np.linalg.norm(V, axis=2, keepdims=True)
        g = np.stack([(V[:, 0] * V[:, 1]).sum(1), (V[:, 0] * V[:, 2]).sum(1), (V[:, 1] * V[:, 2]).sum(1)], 1)
        outs.append(g[(g[:, 0] <= T[0]) & (g[:, 1] <= T[1]) & (g[:, 2] <= T[2])][:n])
    return np.concatenate(outs)


pkeys = list(itertools.combinations_with_replacement(TYPES, 2))
tkeys = list(itertools.combinations_with_replacement(TYPES, 3))
PS, TS = {}, {}
for st in pkeys:
    P, Q, W = G.pair_samples(*st, n_d=17, n_u=60)
    p, q, u = G.rand_pairs(st[0], st[1], NP)
    X = np.r_[np.stack([P, Q, W], 1), np.stack([p, q, u], 1)]
    PS[st] = [X, pair_vals(st, X)]
for cb in tkeys:
    T = (G.tbound(cb[0], cb[1]), G.tbound(cb[0], cb[2]), G.tbound(cb[1], cb[2]))
    ax = [np.linspace(-1, t, int(os.environ.get('GRID', '48'))) for t in T]
    Gd = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    Gd = Gd[G.gram_ok(Gd)]
    X = np.r_[rand_gram(T, NT), Gd, G.coplanar_grid(*T, 64), G.symmetric_grid(cb, *T, 64)]
    TS[cb] = [X, trip_vals(cb, X)]
BD = [np.linspace(lo, hi, 600) for _, lo, hi in G.BINS]
BV = [bin_vals(d) for d in BD]
print('samples: pairs %d, triples %d [%.0f s]' % (sum(len(v[0]) for v in PS.values()), sum(len(v[0]) for v in TS.values()),
                                                   time.time() - t0), flush=True)


def vectors():
    out, lows = [], [0, 6, 21, 22, 23, 24]
    def rec(prefix, left):
        j = len(prefix)
        if j == 6:
            out.append(tuple(prefix + [left])); return
        for n in range(left + 1):
            if sum(prefix) + n < lows[j]:
                continue
            rec(prefix + [n], left - n)
    rec([], 28)
    return out


SETS = {'k15f4': lambda v: v[0] == 15 and v[6] == 4, 'k16f1': lambda v: v[0] == 16 and v[6] == 1,
        'k16f2': lambda v: v[0] == 16 and v[6] == 2, 'k16f3': lambda v: v[0] == 16 and v[6] == 3,
        'k16f4': lambda v: v[0] == 16 and v[6] == 4, 'k16': lambda v: v[0] == 16, 'lo16': lambda v: v[0] <= 16, 'lo13': lambda v: v[0] <= 13,
        'k17': lambda v: v[0] == 17, 'lo17': lambda v: v[0] <= 17, 'all': lambda v: True,
        'hi17': lambda v: v[0] >= 17, 'rest': lambda v: (v[0] == 15 and v[6] == 4) or (v[0] == 16 and v[6] >= 1) or v[0] >= 17}
ALLV = vectors()
VS = [v for v in ALLV if any(SETS[t](v) for t in target.split('+'))]


def tc(v):
    return {s: sum(n for n, b in zip(v, BT) if b == s) for s in TYPES}


def npair(c, st):
    return comb(c[st[0]], 2) if st[0] == st[1] else c[st[0]] * c[st[1]]


def ntrip(c, cb):
    out = 1
    for s in set(cb):
        out *= comb(c[s], cb.count(s))
    return out


# variable layout: lam (K), c2 (6), c3 (10), m (nb), eta
iL = np.arange(K); i2 = K + np.arange(6); i3 = K + 6 + np.arange(10); iM = K + 16 + np.arange(nb); iE = K + 16 + nb
NV = iE + 1
vrows = []
for v in VS:
    c = tc(v); r = np.zeros(NV)
    r[iL] = TV / 2 + sum(v[b] * PT[ti[BT[b]]] for b in range(nb))
    r[i2] = [npair(c, st) for st in pkeys]; r[i3] = [ntrip(c, cb) for cb in tkeys]
    r[iM] = v; r[iE] = -1
    vrows.append(r)
vrows = np.unique(np.array(vrows), axis=0)
print('%d count vectors in %s (%d distinct rows)' % (len(VS), target, len(vrows)), flush=True)


def lp(active):
    rows, b = [vrows], [np.zeros(len(vrows))]
    for j, st in enumerate(pkeys):
        V = PS[st][1][active['p'][st]]
        R = np.zeros((len(V), NV)); R[:, iL] = V.astype(float); R[:, i2[j]] = -1; rows.append(R); b.append(np.zeros(len(V)))
    for j, cb in enumerate(tkeys):
        V = TS[cb][1][active['t'][cb]]
        R = np.zeros((len(V), NV)); R[:, iL] = V.astype(float); R[:, i3[j]] = -1; rows.append(R); b.append(np.zeros(len(V)))
    for bb in range(nb):
        V = BV[bb]
        R = np.zeros((len(V), NV)); R[:, iL] = V.astype(float); R[:, iM[bb]] = -1; rows.append(R); b.append(np.zeros(len(V)))
    A = sp.vstack([sp.csr_matrix(R) for R in rows], format='csr'); bv = np.concatenate(b); rows.clear()
    Aeq = np.zeros((1, NV)); Aeq[0, iL] = 1
    cobj = np.zeros(NV); cobj[iE] = 1
    bounds = [(0, None)] * K + [(None, None)] * (NV - K)
    r = linprog(cobj, A_ub=A, b_ub=bv, A_eq=Aeq, b_eq=[1], bounds=bounds, method='highs')
    return r


def top_active(lam=None, keep=4000):
    act = {'p': {}, 't': {}}
    for st in pkeys:
        V = PS[st][1]; s = V.max(1) if lam is None else V @ lam.astype(np.float32)
        act['p'][st] = np.argsort(s)[-keep:]
    for cb in tkeys:
        V = TS[cb][1]; s = V.max(1) if lam is None else V @ lam.astype(np.float32)
        act['t'][cb] = np.argsort(s)[-keep:]
    return act


def climb_max(lam, x3, Av):
    """local maxima of the combination from its best samples: compass search"""
    newp, newt, cp, ct = {}, {}, {}, {}
    for st in pkeys:
        X, V = PS[st]; s = (V @ lam.astype(np.float32)).astype(float); st_ = X[np.argsort(s)[-60:]]
        f = lambda Y: G.pair_values(st, Y, x3, Av, L, 0)
        Y, val = G.climb(f, st_, G.pair_proj(st), lambda Y: np.ones(len(Y), bool))
        newp[st] = Y; cp[st] = max(val.max(), s.max())
    for cb in tkeys:
        X, V = TS[cb]; s = (V @ lam.astype(np.float32)).astype(float); st_ = X[np.argsort(s)[-60:]]
        f = lambda Y: G.triple_values(cb, Y, x3, L, 0)
        Y, val = G.climb(f, st_, G.triple_proj(cb), G.gram_ok)
        newt[cb] = Y; ct[cb] = max(val.max(), s.max())
    return newp, newt, cp, ct


def bound_with(lam, c2, c3, m, vs):
    best = (-1, None)
    for v in vs:
        c = tc(v)
        val = lam @ (TV / 2) + sum(v[b] * (m[b] + PT[ti[BT[b]]] @ lam) for b in range(nb)) \
            + sum(npair(c, st) * c2[st] for st in pkeys) + sum(ntrip(c, cb) * c3[cb] for cb in tkeys)
        best = max(best, (val, v))
    return best


MG2 = {st: 1e-6 for st in pkeys}; MG2[('A', 'A')] = 5e-6
MG3 = {cb: 5e-8 for cb in tkeys}; MG3[('F', 'F', 'F')] = 5e-8 + 1e-5
act = top_active()
for it in range(iters):
    r = lp(act)
    lam = np.clip(r.x[iL], 0, None); lam /= lam.sum()
    x3 = X3 @ lam; Av = np.einsum('i,ikab->kab', lam, AV)
    newp, newt, cp, ct = climb_max(lam, x3, Av)
    mtrue = [(BV[b] @ lam).max() for b in range(nb)]
    est = bound_with(lam, {st: cp[st] + MG2[st] for st in pkeys}, {cb: ct[cb] + MG3[cb] for cb in tkeys},
                     [mm + 1e-7 for mm in mtrue], VS)
    gaps = {''.join(k): ct[k] - r.x[i3[j]] for j, k in enumerate(tkeys)}
    gaps.update({''.join(k): cp[k] - r.x[i2[j]] for j, k in enumerate(pkeys)})
    print('iter %d: LP %.6f; with climbed maxima and the margins of the check %.6f at %s; weights %s; largest gaps %s [%.0f s]'
          % (it, r.x[iE], est[0], est[1], {os.path.basename(certs[i]): round(float(lam[i]), 4) for i in range(K) if lam[i] > 1e-4},
             sorted(((round(v, 9), k) for k, v in gaps.items()), reverse=True)[:4], time.time() - t0), flush=True)
    # add the climbed points and every sample now above the LP thresholds
    for st in pkeys:
        PS[st][0] = np.r_[PS[st][0], newp[st]]; PS[st][1] = np.r_[PS[st][1], pair_vals(st, newp[st])]
    for cb in tkeys:
        TS[cb][0] = np.r_[TS[cb][0], newt[cb]]; TS[cb][1] = np.r_[TS[cb][1], trip_vals(cb, newt[cb])]
    a2 = top_active(lam)
    for st in pkeys:
        a2['p'][st] = np.union1d(np.union1d(act['p'][st], a2['p'][st]), np.arange(len(PS[st][0]) - len(newp[st]), len(PS[st][0])))
    for cb in tkeys:
        a2['t'][cb] = np.union1d(np.union1d(act['t'][cb], a2['t'][cb]), np.arange(len(TS[cb][0]) - len(newt[cb]), len(TS[cb][0])))
    act = a2
    if os.environ.get('OUT'):
        mfine = [(bin_vals(np.linspace(lo, hi, 20001)) @ lam).max() + 2e-7 for _, lo, hi in G.BINS]
        np.savez(os.environ['OUT'], lam=lam, certs=np.array(certs), x3=x3, A=Av, z=lam @ ZV, t=lam @ TV,
                 m=np.array(mfine), d3=D, bound=est[0],
                 c2=np.array([cp[st] for st in pkeys]), c3=np.array([ct[cb] for cb in tkeys]))
