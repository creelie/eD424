#!/usr/bin/env python3
"""
typed4pt.py -- a four-point bound for spherical codes on S^3 whose points carry TYPES:
the typed three-point bound of typed3pt.py together with kernels that fix two points of
the code.  Floating point exploration (sampled constraints, refinement rounds); a value
below 0 is evidence that no code with the given counts exists, to be certified separately.

Two-pole kernels.  Fix two distinct points e1, e2 of the code, s = <e1, e2>.  A point x
has u1 = <x, e1>, u2 = <x, e2> and a component w_x in the plane orthogonal to e1 and e2.
With D = 1 - s^2,
  R_x  = D |w_x|^2     = 1 - s^2 - u1^2 - u2^2 + 2 s u1 u2           (a Gram determinant),
  C_xy = D <w_x, w_y>  = D t - (u1x u1y + u2x u2y - s (u1x u2y + u2x u1y)),   t = <x, y>,
and Z_k(x, y) = Re((sqrt(D) w_x conj(sqrt(D) w_y))^k) = (R_x R_y)^(k/2) T_k(C_xy / sqrt(R_x R_y))
is a polynomial and a positive kernel for each k (the plane is identified with C; the value
does not depend on the identification).  For psd matrices G^{PQ}_k,
  K_{e1,e2}(x, y) = sum_{k=0..d4} Z_k(x, y) b_k(x)^T G^{type(e1) type(e2)}_k b_k(y),
  b_k(x) = e(type(x)) (x) (T_i(u1) T_j(u2))_{i+j <= d4-k}, or with WITH_S=1
  (T_a(s) T_i(u1) T_j(u2))_{a+i+j <= d4-k}, s being fixed with the poles,
is positive, so S = sum over ordered pairs e1 != e2 of sum_{x,y in C} K_{e1,e2}(x, y) >= 0.
Grouping the terms by the set {e1, e2, x, y}:
  pairs {p, q}:       x, y in {e1, e2}                                       (PAIR4)
  triples {p, q, z}:  2 K(e1, z) + 2 K(e2, z) + K(z, z), over the six ordered poles  (TRIPLE4)
  quadruples:         2 K(z, z'), over the twelve ordered poles                 (QUAD4)
Added to the three-point identity of typed3pt.py,
  Val = sum_s n_s p_s + sum N_st c_st + sum N_str c_str + sum N_strv c_strv < 0,
with PAIR <= c, TRIPLE <= c and QUAD <= c on the admissible inner products, shows that no
such code exists.
"""
import sys, os, time, itertools
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from math import comb
from numpy.polynomial import chebyshev as CH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import typed3pt as T3

rng = np.random.default_rng(int(os.environ.get('SEED', '5')))
PERMS2 = {n: list(itertools.permutations(range(n), 2)) for n in (2, 3, 4)}


def mon2(n):
    return [(0, i, j) for i in range(n + 1) for j in range(n + 1 - i)]


def mon3(n):
    return [(a, i, j) for a in range(n + 1) for i in range(n + 1 - a) for j in range(n + 1 - a - i)]


class Layout4(T3.Layout):
    def __init__(self, m, d, d4, poles, with_s=False):
        super().__init__(m, d)
        self.d4 = d4
        self.mons = {k: (mon3 if with_s else mon2)(d4 - k) for k in range(d4 + 1)}
        off = self.n
        for (P, Q) in poles:
            for k in range(d4 + 1):
                s = m * len(self.mons[k])
                self.blocks[('G', P, Q, k)] = (off, s); off += s * (s + 1) // 2
        self.n = off
        self.poles = set(poles)


def feat4(L, typ, k, u1, u2, s):
    ms = L.mons[k]; nm = len(ms); deg = L.d4 - k
    C1 = T3.cheb(u1, deg + 1); C2 = T3.cheb(u2, deg + 1); C0 = T3.cheb(s, deg + 1)
    V = np.stack([C0[:, a] * C1[:, i] * C2[:, j] for a, i, j in ms], -1)
    return np.arange(typ * nm, (typ + 1) * nm), V


TK = {}


def Zk(k, Rx, Ry, C):
    if k not in TK:
        c = np.zeros(k + 1); c[k] = 1.0
        TK[k] = CH.cheb2poly(c)
    a = TK[k]; RR = Rx * Ry
    out = np.zeros_like(C)
    for m in range(k % 2, k + 1, 2):
        out = out + a[m] * C ** m * RR ** ((k - m) // 2)
    return out


def Rgram(s, u1, u2):
    return 1 - s * s - u1 * u1 - u2 * u2 + 2 * s * u1 * u2


def add_pair4(B, rows, P0, Q0, s):
    L = B.L; one = np.ones(len(s))
    if L.d4 < 0:
        return
    for (P, Q) in ((P0, Q0), (Q0, P0)):
        if (P, Q) not in L.poles:
            continue
        name = ('G', P, Q, 0)
        i1, v1 = feat4(L, P, 0, one, s, s)
        i2, v2 = feat4(L, Q, 0, s, one, s)
        B.add_bilinear(rows, name, i1, v1, i1, v1, one)
        B.add_bilinear(rows, name, i2, v2, i2, v2, one)
        B.add_bilinear(rows, name, i1, v1, i2, v2, 2 * one)


def add_triple4(B, rows, types, U):
    """U[(i, j)] = inner product of points i and j (i != j), arrays of length N."""
    L = B.L; N = len(rows); one = np.ones(N)
    if L.d4 < 0:
        return
    for (i, j) in PERMS2[3]:
        k = 3 - i - j
        P, Q, Z = types[i], types[j], types[k]
        if (P, Q) not in L.poles:
            continue
        s, u1, u2 = U[(i, j)], U[(i, k)], U[(j, k)]
        i1, v1 = feat4(L, P, 0, one, s, s)
        i2, v2 = feat4(L, Q, 0, s, one, s)
        iz, vz = feat4(L, Z, 0, u1, u2, s)
        B.add_bilinear(rows, ('G', P, Q, 0), i1, v1, iz, vz, 2 * one)
        B.add_bilinear(rows, ('G', P, Q, 0), i2, v2, iz, vz, 2 * one)
        R = np.maximum(Rgram(s, u1, u2), 0)
        for kk in range(L.d4 + 1):
            iz, vz = feat4(L, Z, kk, u1, u2, s)
            B.add_bilinear(rows, ('G', P, Q, kk), iz, vz, iz, vz, R ** kk)


def add_quad4(B, rows, types, U):
    L = B.L
    for (i, j) in PERMS2[4]:
        k, l = [r for r in range(4) if r not in (i, j)]
        P, Q = types[i], types[j]
        if (P, Q) not in L.poles:
            continue
        s = U[(i, j)]
        u1x, u2x, u1y, u2y, t = U[(i, k)], U[(j, k)], U[(i, l)], U[(j, l)], U[(k, l)]
        Rx = np.maximum(Rgram(s, u1x, u2x), 0); Ry = np.maximum(Rgram(s, u1y, u2y), 0)
        C = (1 - s * s) * t - (u1x * u1y + u2x * u2y - s * (u1x * u2y + u2x * u1y))
        for kk in range(L.d4 + 1):
            ix, vx = feat4(L, types[k], kk, u1x, u2x, s)
            iy, vy = feat4(L, types[l], kk, u1y, u2y, s)
            B.add_bilinear(rows, ('G', P, Q, kk), ix, vx, iy, vy, 2 * Zk(kk, Rx, Ry, C))


def gram_dict(G, idx):
    n = len(idx)
    return {(a, b): G[:, idx[a], idx[b]] for a in range(n) for b in range(n) if a != b}


def tri_U(g):
    U = {(0, 1): g[:, 0], (0, 2): g[:, 1], (1, 2): g[:, 2]}
    U.update({(b, a): v for (a, b), v in list(U.items())})
    return U


def quad_U(q):
    """q: (N, 6) inner products in the order 01, 02, 03, 12, 13, 23."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    U = {p: q[:, n] for n, p in enumerate(pairs)}
    U.update({(b, a): v for (a, b), v in list(U.items())})
    return U


QP = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def admissible(q, types, Tmat):
    ok = np.ones(len(q), bool)
    for n, (a, b) in enumerate(QP):
        ok &= q[:, n] <= Tmat[types[a]][types[b]] + 1e-12
    return ok


def vecs_to_q(X):
    return np.stack([np.einsum('ni,ni->n', X[:, a], X[:, b]) for a, b in QP], 1)


def roots_D4():
    out = []
    for i, j in itertools.combinations(range(4), 2):
        for a in (1, -1):
            for b in (1, -1):
                v = np.zeros(4); v[i] = a; v[j] = b; out.append(v / 2 ** .5)
    return np.array(out)


def random_quads(types, Tmat, n):
    outs = []
    for dim in (4, 3, 2):
        need = n if dim == 4 else n // 4
        got = 0
        for _ in range(60):
            X = rng.normal(size=(20 * need, 4, dim)); X /= np.linalg.norm(X, axis=2, keepdims=True)
            q = vecs_to_q(X); q = q[admissible(q, types, Tmat)]
            outs.append(q[:need - got]); got += len(outs[-1])
            if got >= need:
                break
    # near the root system: points of type 0 near roots of D4, others random in the holes
    Rt = roots_D4(); X = []
    for _ in range(40 * n):
        Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        pts = []
        for ty in types:
            if ty == 0:
                pts.append(Rt[rng.integers(24)] @ Q.T)
            else:
                pts.append(rng.normal(size=4))
        P = np.array(pts) + 0.03 * rng.normal(size=(4, 4))
        X.append(P / np.linalg.norm(P, axis=1, keepdims=True))
    X = np.array(X); q = vecs_to_q(X); q = q[admissible(q, types, Tmat)]
    outs.append(q[:n])
    return np.concatenate(outs)


def quad_chunk(L):
    """quadruples per chunk, so that one chunk holds about 2e7 coordinate entries."""
    per = 12 * sum(len(L.mons[k]) ** 2 for k in range(L.d4 + 1))
    return max(50, int(2e7 // per))


def tri_chunk(L):
    n0 = len(L.mons[0]) if L.d4 >= 0 else 0
    per = 6 * (2 * n0 ** 2 + sum(len(L.mons[k]) ** 2 for k in range(L.d4 + 1))) + 3 * (L.d + 1) ** 3
    return max(50, int(2e7 // per))


def tri_matrix(L, c, g, chunk=None):
    """constraint rows of TRIPLE + TRIPLE4 for the triples g (columns u12, u13, u23), in chunks."""
    out = []; chunk = chunk or tri_chunk(L)
    for i in range(0, len(g), chunk):
        gg = g[i:i + chunk]; rows = np.arange(len(gg))
        B = T3.Builder(L); T3.add_triple(B, rows, c, gg[:, 0], gg[:, 1], gg[:, 2]); add_triple4(B, rows, c, tri_U(gg))
        out.append(B.matrix(len(gg)))
    return sp.vstack(out).tocsr()


def quad_matrix(L, types, q, chunk=None):
    """constraint rows of QUAD4 for the samples q, built in chunks so that the coordinate lists
    (about 10^4 entries per quadruple at d4 = 5) never hold more than one chunk."""
    out = []; chunk = chunk or quad_chunk(L)
    for i in range(0, len(q), chunk):
        qq = q[i:i + chunk]
        B = T3.Builder(L); add_quad4(B, np.arange(len(qq)), types, quad_U(qq))
        out.append(B.matrix(len(qq)))
    return sp.vstack(out).tocsr()


def quad_values(L, xv, types, q):
    vals = []; chunk = quad_chunk(L)
    for i in range(0, len(q), chunk):
        qq = q[i:i + chunk]
        B = T3.Builder(L); add_quad4(B, np.arange(len(qq)), types, quad_U(qq))
        vals.append(B.matrix(len(qq)) @ xv)
    return np.concatenate(vals)


def refine_quads(L, xv, types, Tmat, q0, cval, iters=40):
    """hill-climb the worst quadruples (as Gram vectors realised by points) to raise QUAD4."""
    out = []
    for q in q0:
        G = np.ones((4, 4))
        for n, (a, b) in enumerate(QP):
            G[a, b] = G[b, a] = q[n]
        w, V = np.linalg.eigh(G); w = np.maximum(w, 0)
        X = V * np.sqrt(w)[None, :]
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        cur = quad_values(L, xv, types, vecs_to_q(X[None]))[0]
        sig = 0.02
        for it in range(iters):
            Y = X + sig * rng.normal(size=(8, 4, 4))
            Y /= np.linalg.norm(Y, axis=2, keepdims=True)
            qy = vecs_to_q(Y); ok = admissible(qy, types, Tmat)
            if ok.any():
                vy = quad_values(L, xv, types, qy[ok])
                b = vy.argmax()
                if vy[b] > cur:
                    cur = vy[b]; X = Y[ok][b]; continue
            sig *= 0.7
        out.append(vecs_to_q(X[None])[0])
    return np.array(out)


def solve(counts, Tmat, d, d4, rounds=3, nquad=3000, sizes=(14, 14, 7), nrand=1500, verbose=True, with_s=False):
    m = len(counts)
    pair_list = [(s, t) for s in range(m) for t in range(s, m) if (comb(counts[s], 2) if s == t else counts[s] * counts[t]) > 0]
    poles = sorted({(s, t) for (a, b) in pair_list for (s, t) in ((a, b), (b, a))})
    L = Layout4(m, d, d4, poles, with_s)
    triple_list = [c for c in itertools.combinations_with_replacement(range(m), 3) if T3.ncount(counts, list(c)) > 0]
    quad_list = [c for c in itertools.combinations_with_replacement(range(m), 4) if T3.ncount(counts, list(c)) > 0 and d4 >= 0]
    psamp = {p: np.unique(np.r_[T3.grid1(Tmat[p[0]][p[1]], 600), Tmat[p[0]][p[1]]]) for p in pair_list}
    tsamp = {}
    for c in triple_list:
        T12, T13, T23 = Tmat[c[0]][c[1]], Tmat[c[0]][c[2]], Tmat[c[1]][c[2]]
        tsamp[c] = np.r_[T3.triple_grid(T12, T13, T23, *sizes), T3.random_triples(T12, T13, T23, nrand)]
    qsamp = {c: random_quads(c, Tmat, nquad) for c in quad_list}
    if verbose:
        print('   layout: %d variables; quadruple kinds %s with %s samples' % (
            L.n, quad_list, [len(qsamp[c]) for c in quad_list]), flush=True)
    for rnd in range(rounds):
        t0 = time.time()
        X = {name: cp.Variable((s, s), PSD=True) for name, (off, s) in L.blocks.items()}
        def vec(name):
            M = X[name]; s = M.shape[0]
            iu = np.triu_indices(s, 1)
            ii = np.r_[np.arange(s), iu[0]]; jj = np.r_[np.arange(s), iu[1]]
            Sel = sp.csr_matrix((np.ones(len(ii)), (np.arange(len(ii)), ii + jj * s)), shape=(len(ii), s * s))
            return Sel @ cp.vec(M, order='F')
        xall = cp.hstack([vec(name) for name in L.blocks])
        cons = []; cvars = {}
        B = T3.Builder(L)
        for s in range(m):
            T3.add_point(B, np.array([s]), s)
        p = B.matrix(m) @ xall
        val = sum(counts[s] * p[s] for s in range(m))
        for pr in pair_list:
            u = psamp[pr]; rows = np.arange(len(u))
            B = T3.Builder(L); T3.add_pair(B, rows, pr[0], pr[1], u); add_pair4(B, rows, pr[0], pr[1], u)
            cv = cp.Variable(); cvars[pr] = cv
            cons.append(B.matrix(len(u)) @ xall <= cv)
            val = val + (comb(counts[pr[0]], 2) if pr[0] == pr[1] else counts[pr[0]] * counts[pr[1]]) * cv
        for c in triple_list:
            g = tsamp[c]
            cv = cp.Variable(); cvars[c] = cv
            cons.append(tri_matrix(L, c, g) @ xall <= cv)
            val = val + T3.ncount(counts, list(c)) * cv
        for c in quad_list:
            q = qsamp[c]
            cv = cp.Variable(); cvars[c] = cv
            cons.append(quad_matrix(L, c, q) @ xall <= cv)
            val = val + T3.ncount(counts, list(c)) * cv
        cons.append(sum(cp.trace(X[name]) for name in L.blocks) <= 1)
        prob = cp.Problem(cp.Minimize(val), cons)
        try:
            prob.solve(solver='CLARABEL', max_iter=400)
        except cp.error.SolverError:
            print('   CLARABEL failed; SCS', flush=True)
            prob.solve(solver='SCS', eps=1e-7, max_iters=200000)
        xv = np.concatenate([np.r_[np.diag(X[n].value), X[n].value[np.triu_indices(X[n].shape[0], 1)]] for n in L.blocks])
        corr = 0.0; viol = {}
        for pr in pair_list:
            u = np.linspace(-1, Tmat[pr[0]][pr[1]], 20001); rows = np.arange(len(u))
            B = T3.Builder(L); T3.add_pair(B, rows, pr[0], pr[1], u); add_pair4(B, rows, pr[0], pr[1], u)
            vv = B.matrix(len(u)) @ xv - cvars[pr].value
            viol[pr] = vv.max(); corr += (comb(counts[pr[0]], 2) if pr[0] == pr[1] else counts[pr[0]] * counts[pr[1]]) * max(vv.max(), 0)
            psamp[pr] = np.r_[psamp[pr], u[np.argsort(vv)[-60:]]]
        for c in triple_list:
            T12, T13, T23 = Tmat[c[0]][c[1]], Tmat[c[0]][c[2]], Tmat[c[1]][c[2]]
            g = np.r_[T3.random_triples(T12, T13, T23, 12000), T3.triple_grid(T12, T13, T23, 30, 30, 12)]
            vv = []
            for i in range(0, len(g), 5000):
                vv.append(tri_matrix(L, c, g[i:i + 5000]) @ xv)
            vv = np.concatenate(vv) - cvars[c].value
            viol[c] = vv.max(); corr += T3.ncount(counts, list(c)) * max(vv.max(), 0)
            tsamp[c] = np.r_[tsamp[c], g[np.argsort(vv)[-400:]]]
        for c in quad_list:
            q = random_quads(c, Tmat, 4 * nquad)
            vv = quad_values(L, xv, c, q) - cvars[c].value
            top = q[np.argsort(vv)[-60:]]
            ref = refine_quads(L, xv, c, Tmat, top, cvars[c].value)
            vr = quad_values(L, xv, c, ref) - cvars[c].value
            viol[c] = max(vv.max(), vr.max()); corr += T3.ncount(counts, list(c)) * max(viol[c], 0)
            qsamp[c] = np.r_[qsamp[c], q[np.argsort(vv)[-600:]], ref]
        if verbose:
            print('d=%d d4=%d round %d: %s Val=%.4e corrected=%.4e  viol %s  [%.0f s]' % (
                d, d4, rnd + 1, prob.status, prob.value, prob.value + corr,
                {''.join('ABCDEFGH'[i] for i in k): float('%.1e' % v) for k, v in viol.items()}, time.time() - t0), flush=True)
    return prob.value, prob.value + corr, X, cvars


def selftest(d=4, d4=3, seed=0, with_s=True):
    """compare the grouped identity with the direct double sum over ordered poles, on a code
    with two types and random psd blocks."""
    r = np.random.default_rng(seed)
    Rt = roots_D4()
    C = np.r_[Rt[:9], r.normal(size=(4, 4))]
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    types = [0] * 9 + [1] * 4
    n = len(C); m = 2
    L = Layout4(m, d, d4, [(0, 0), (0, 1), (1, 0), (1, 1)], with_s)
    Gm = {}
    for name, (off, s) in L.blocks.items():
        A = r.normal(size=(s, s)); Gm[name] = A @ A.T / s
    xv = np.concatenate([np.r_[np.diag(Gm[nm]), Gm[nm][np.triu_indices(Gm[nm].shape[0], 1)]] for nm in L.blocks])
    # direct sum of the two-pole kernels
    direct = 0.0
    for e1, e2 in itertools.permutations(range(n), 2):
        P, Q = types[e1], types[e2]; s = C[e1] @ C[e2]
        u1 = C @ C[e1]; u2 = C @ C[e2]; t = C @ C.T
        R = np.maximum(Rgram(s, u1, u2), 0)
        Cm = (1 - s * s) * t - (np.outer(u1, u1) + np.outer(u2, u2) - s * (np.outer(u1, u2) + np.outer(u2, u1)))
        for k in range(d4 + 1):
            ms = L.mons[k]; nm = len(ms)
            b = np.zeros((n, m * nm))
            for x in range(n):
                ii, vv = feat4(L, types[x], k, np.array([u1[x]]), np.array([u2[x]]), np.array([s]))
                b[x, ii] = vv[0]
            Z = Zk(k, R[:, None], R[None, :], Cm)
            direct += np.sum(Z * (b @ Gm[('G', P, Q, k)] @ b.T))
    # grouped sum
    grouped = 0.0
    G = C @ C.T
    for a, b_ in itertools.combinations(range(n), 2):
        B = T3.Builder(L); add_pair4(B, np.array([0]), types[a], types[b_], np.array([G[a, b_]]))
        grouped += (B.matrix(1) @ xv)[0]
    for tr in itertools.combinations(range(n), 3):
        B = T3.Builder(L); add_triple4(B, np.array([0]), [types[x] for x in tr], gram_dict(G[None], tr))
        grouped += (B.matrix(1) @ xv)[0]
    for qd in itertools.combinations(range(n), 4):
        B = T3.Builder(L); add_quad4(B, np.array([0]), [types[x] for x in qd], gram_dict(G[None], qd))
        grouped += (B.matrix(1) @ xv)[0]
    print('selftest: direct %.10f grouped %.10f difference %.1e' % (direct, grouped, abs(direct - grouped)))
    return direct, grouped


if __name__ == '__main__':
    mode = sys.argv[1]
    WS = bool(int(os.environ.get('WITH_S', '0')))
    NQ = int(os.environ.get("NQUAD", "3000"))
    if mode == 'selftest':
        selftest()
    elif mode == 'hole':        # 24 points within dA and nF more within dF, typed by distance
        d, d4, rounds = map(int, sys.argv[2:5])
        nF = int(sys.argv[5]); dA = float(sys.argv[6]); dF = float(sys.argv[7])
        a = T3.a_
        T = [[a(dA, dA), a(dA, dF)], [a(dA, dF), a(dF, dF)]]
        print('24 within %.4f and %d within %.4f; T = %s' % (dA, nF, dF, np.round(T, 5)), flush=True)
        solve([24, nF], T, d, d4, rounds, nquad=NQ, with_s=WS)
    elif mode == 'kiss':        # n points with inner products <= t
        d, d4, rounds, n = map(int, sys.argv[2:6]); t = float(sys.argv[6])
        solve([n], [[t]], d, d4, rounds, nquad=NQ, with_s=WS)
