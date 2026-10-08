#!/usr/bin/env python3
"""
level2_sampled.py -- a sampled second-level programme (floating point,
exploration, not proof).

The kernel blocks X_lambda of the certificate of de Laat, Leijenhorst and de
Muinck Keizer are the unknowns (60 blocks, 5298 unknowns, scaled so that the
deposited certificate has unit diagonal); the sums of squares that certify the
three- and four-point constraints are replaced by those constraints at sampled
configurations, and Clarabel solves the result through its native interface.

  bound KAP NQ ROUNDS        minimise K(empty, empty) subject to A_2K({x}) <= -1
                             and A_2K(Q) <= 0 at sampled Q with inner products
                             <= 1/2 + KAP (NQ quadruples, NQ/4 triples, a grid
                             of pairs), adding the most violated of fresh
                             samples each round
  robust KAP NQ ROUNDS WIN   maximise gamma - (K - 24) with A_2K({x,y}) <= -gamma
                             for pairs outside windows of half-width WIN about
                             -1, -1/2, 0, 1/2 (positive: every pair of a
                             24-point code with slack KAP is forced into them)

Usage: python3 level2_sampled.py DATA PSPICKLE MODE KAP NQ ROUNDS [WIN]
"""
import sys, time, itertools, numpy as np, scipy.sparse as sp, clarabel
import level2_numeric
DATA, PSP = sys.argv[1], sys.argv[2]
mode = sys.argv[3]; KAP = float(sys.argv[4]); NQ = int(sys.argv[5]); ROUNDS = int(sys.argv[6])
WIN = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
TOP = 0.5 + KAP
rng = np.random.default_rng(7)
t0 = time.time()
B = level2_numeric.Blocks(DATA, PSP, with_X=True)
lams = B.lams
S2 = np.sqrt(2.0)
# svec order (column-wise upper triangle), with the scaling X = D Y D, D = sqrt(diag X_LLM24)
IDX, DD, OFF = {}, {}, {}
off = 0
for lam in lams:
    m = B.info[lam]['m']
    ii = [(i, j) for j in range(m) for i in range(j + 1)]
    IDX[lam] = (np.array([p[0] for p in ii]), np.array([p[1] for p in ii]))
    d = np.sqrt(np.maximum(np.diag(B.info[lam]['X']), 1e-300)); DD[lam] = np.outer(d, d)
    OFF[lam] = off; off += len(ii)
NV = off
def pack(Y):
    cols = []
    for lam in lams:
        i, j = IDX[lam]; Yl = Y[lam] * DD[lam][None]
        cols.append(np.where(i == j, Yl[:, i, j], S2 * Yl[:, i, j]))   # C_ij Y_ij + C_ji Y_ji = sqrt2 C_ij (sqrt2 Y_ij)
    return np.concatenate(cols, 1)
def svec_of(X):
    out = []
    for lam in lams:
        i, j = IDX[lam]; Z = X[lam] / DD[lam]
        out.append(np.where(i == j, Z[i, j], S2 * Z[i, j]))
    return np.concatenate(out)
xs = svec_of({lam: B.info[lam]['X'] for lam in lams})
R = []
for a, b in itertools.combinations(range(4), 2):
    for sa in (1, -1):
        for sb in (1, -1):
            w = np.zeros(4); w[a] = sa; w[b] = sb; R.append(w / np.sqrt(2))
R = np.array(R)
def codes(k, n):
    out = []
    while len(out) < n // 3:
        X = rng.normal(size=(k, 4)); X /= np.linalg.norm(X, axis=1, keepdims=True); G = X @ X.T
        if (G[np.triu_indices(k, 1)] <= TOP).all(): out.append(G)
    while len(out) < 2 * n // 3:
        X = R[rng.choice(24, k, replace=False)] + (0.01 + 0.25 * rng.random()) * rng.normal(size=(k, 4))
        X /= np.linalg.norm(X, axis=1, keepdims=True); G = X @ X.T
        if (G[np.triu_indices(k, 1)] <= TOP).all(): out.append(G)
    while len(out) < n:
        X = rng.normal(size=(k, 4)); X /= np.linalg.norm(X, axis=1, keepdims=True)
        for _ in range(80):
            G = X @ X.T; iu = np.triu_indices(k, 1); g = G[iu]
            if TOP - 2e-3 <= g.max() <= TOP: break
            a, b = iu[0][g.argmax()], iu[1][g.argmax()]
            s = 0.3 * (g.max() - TOP); X[a] -= s * X[b]; X[b] -= s * X[a]
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        G = X @ X.T
        if (G[np.triu_indices(k, 1)] <= TOP + 1e-12).all(): out.append(G)
    return np.stack(out)
def rows(G, chunk=2000):
    return np.concatenate([pack(B.rows(G[s:s + chunk])) for s in range(0, len(G), chunk)])
nrm = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
r0 = rows(np.zeros((1, 0, 0)))[0]; r1 = rows(np.ones((1, 1, 1)))[0]
up = np.unique(np.clip(np.r_[np.linspace(-1, TOP, 1500), -1, -0.5, 0, 0.5, TOP], -1, TOP))
Rp = rows(np.stack([np.array([[1, x], [x, 1]]) for x in up]))
inside = (up <= -1 + WIN) | (np.abs(up + 0.5) <= WIN) | (np.abs(up) <= WIN) | (up >= 0.5 - WIN)
R3 = nrm(rows(codes(3, NQ // 4))); R4 = nrm(rows(codes(4, NQ)))
print('%s kappa %.4f: %d unknowns, %d pair, %d triple, %d quadruple rows [%.0fs]' % (mode, KAP, NV, len(Rp), len(R3), len(R4), time.time() - t0), flush=True)
print('  LLM24 on these rows: K = %.6f, A2K(x) = %.6f, max pair %.1e, triple %.1e, quad %.1e'
      % (r0 @ xs, r1 @ xs, (Rp @ xs).max(), (R3 @ xs).max(), (R4 @ xs).max()), flush=True)
cones_psd = [clarabel.PSDTriangleConeT(B.info[lam]['m']) for lam in lams]
for rnd in range(ROUNDS):
    # unknowns: y (NV), and gamma in robust mode
    ng = 1 if mode == 'robust' else 0
    n = NV + ng
    if mode == 'bound':
        q = r0.copy()
        bl = np.r_[-1.0, np.zeros(len(Rp) + len(R3) + len(R4))]
        Lin = np.r_[r1[None], nrm(Rp), R3, R4]
        A_lin = sp.csc_matrix(Lin)
    else:
        q = np.r_[r0, -1.0]                               # minimise (K - 24) - gamma
        Pin, Pout = Rp[inside], Rp[~inside]
        Lin = sp.vstack([sp.csc_matrix(np.r_[r1, 0.0][None]),
                         sp.hstack([sp.csc_matrix(Pin), sp.csc_matrix((len(Pin), 1))]),
                         sp.hstack([sp.csc_matrix(Pout), sp.csc_matrix(np.ones((len(Pout), 1)))]),
                         sp.hstack([sp.csc_matrix(np.r_[R3, R4]), sp.csc_matrix((len(R3) + len(R4), 1))]),
                         sp.csc_matrix(np.r_[np.zeros(NV), -1.0][None]),     # gamma >= 0
                         sp.csc_matrix(np.r_[np.zeros(NV), 1.0][None])])     # gamma <= 1
        bl = np.r_[-1.0, np.zeros(len(Pin) + len(Pout) + len(R3) + len(R4)), 0.0, 1.0]
        A_lin = Lin.tocsc()
    A_psd = sp.hstack([-sp.identity(NV), sp.csc_matrix((NV, ng))]) if ng else -sp.identity(NV)
    A = sp.vstack([A_lin, A_psd]).tocsc()
    b = np.r_[bl, np.zeros(NV)]
    cones = [clarabel.NonnegativeConeT(A_lin.shape[0])] + cones_psd
    P = sp.csc_matrix((n, n))
    st = clarabel.DefaultSettings(); st.verbose = False; st.max_iter = 400
    st.tol_gap_abs = 1e-10; st.tol_gap_rel = 1e-10; st.tol_feas = 1e-10
    t1 = time.time()
    sol = clarabel.DefaultSolver(P, q, A, b, cones, st).solve()
    x = np.array(sol.x)
    y = x[:NV]
    N3 = nrm(rows(codes(3, NQ // 4))); N4 = nrm(rows(codes(4, NQ)))
    v3, v4 = N3 @ y, N4 @ y
    print('round %d: %s, K(0,0) = %.6f%s; fresh max triple %.2e, quad %.2e [%.0fs]'
          % (rnd + 1, sol.status, r0 @ y, (', gamma = %.3e, margin = %.3e' % (x[-1], x[-1] - (r0 @ y - 24))) if ng else '',
             v3.max(), v4.max(), time.time() - t1), flush=True)
    R3 = np.r_[R3, N3[np.argsort(v3)[-NQ // 12:]]]; R4 = np.r_[R4, N4[np.argsort(v4)[-NQ // 3:]]]
