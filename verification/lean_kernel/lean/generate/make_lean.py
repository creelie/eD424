"""Write the data files of the Lean certificate (lean/D4Lean/Data/*.lean) from cert/certificate.txt
and data/equality_points.json.

For every vertex J the data are: the minimal vectors; bound certificates showing that a lattice
point outside a box has J[k] > 4; the box; 15 minimal vectors whose equations determine J, with a
certificate A B = c I; an elimination certificate for det J and det Q.
For every edge J + tY the data are: the end point (T^T J_to T with T in Gamma) or the rank-one
form of Y; the polynomials det(J + tY), det Q(J + tY) with six sample points and their
elimination certificates; break points u/wd and the pieces between them, on each of which
det J > 0 or det Q > 256 is shown by a coefficient test.

Usage: python3 make_lean.py certificate.txt equality_points.json outdir   (outdir = lean/)
"""
import sys, os, json, math
from fractions import Fraction as F

N = 5          # size of the Gram matrix J of (a_1, ..., a_4, b)
IB = N - 1      # index of b; Q is the leading IB x IB block
BOUND = 256    # the bound for det Q



def lcm(a, b):
    return a * b // math.gcd(a, b)


def lcm_list(xs):
    r = 1
    for x in xs:
        r = lcm(r, x)
    return r


def qform(J, k):
    return sum(k[i] * J[i][j] * k[j] for i in range(len(k)) for j in range(len(k)))


def symc(k):
    return [(1 if a == b else 2) * k[a] * k[b] for a in range(N) for b in range(a, N)]


def sym_from_vec(v):
    X = [[0] * N for _ in range(N)]
    t = 0
    for a in range(N):
        for b in range(a, N):
            X[a][b] = X[b][a] = v[t]
            t += 1
    return X


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
            for i in range(len(A))]


def transpose(A):
    return [list(r) for r in zip(*A)]


def inverse(A):
    n = len(A)
    M = [[F(x) for x in row] + [F(1 if i == j else 0) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        inv = 1 / M[c][c]
        M[c] = [x * inv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def det(M):
    n = len(M)
    A = [[F(x) for x in row] for row in M]
    d = F(1)
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None:
            return F(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            d = -d
        d *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            if f:
                for k in range(c, n):
                    A[r][k] -= f * A[c][k]
    return d


def rank_rows(rows):
    """indices of a maximal independent subset of rows (greedy, exact)."""
    basis = []   # reduced rows with pivot columns
    chosen = []
    for idx, r in enumerate(rows):
        v = [F(x) for x in r]
        for piv, b in basis:
            if v[piv] != 0:
                f = v[piv] / b[piv]
                v = [x - f * y for x, y in zip(v, b)]
        nz = next((i for i, x in enumerate(v) if x != 0), None)
        if nz is not None:
            basis.append((nz, v))
            chosen.append(idx)
    return chosen


# ---------- elimination certificates ----------

def elim(A):
    """Integer lower-triangular E with E A upper triangular (leading minors of order <= N - 1 nonzero)."""
    n = len(A)
    M = [[F(x) for x in row] for row in A]
    Lr = [[F(1 if i == j else 0) for j in range(n)] for i in range(n)]   # running E (rational)
    for c in range(n - 1):
        assert M[c][c] != 0, "zero pivot"
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
                Lr[r] = [x - f * y for x, y in zip(Lr[r], Lr[c])]
    E = []
    for row in Lr:
        den = lcm_list([x.denominator for x in row])
        num = [int(x * den) for x in row]
        g = 0
        for x in num:
            g = math.gcd(g, x)
        E.append([x // g for x in num])
    EA = matmul(E, A)
    for i in range(n):
        for j in range(n):
            assert (j <= i or E[i][j] == 0) and (i <= j or EA[i][j] == 0)
    dE = math.prod(E[i][i] for i in range(n))
    assert dE != 0
    return E


def check_det(A, E, d, dq):
    EA = matmul(E, A)
    assert math.prod(EA[i][i] for i in range(N)) == d * math.prod(E[i][i] for i in range(N))
    assert math.prod(EA[i][i] for i in range(N - 1)) == dq * math.prod(E[i][i] for i in range(N - 1))


# ---------- bound certificates ----------

def decompose(M):
    """M rational symmetric PSD: M = sum d_r c_r c_r^T with d_r > 0 (exact)."""
    n = len(M)
    M = [[F(x) for x in row] for row in M]
    terms = []
    while True:
        p = next((i for i in range(n) if M[i][i] != 0), None)
        if p is None:
            assert all(x == 0 for row in M for x in row), "not PSD"
            return terms
        assert M[p][p] > 0, "not PSD"
        col = [M[i][p] for i in range(n)]
        d = 1 / M[p][p]
        terms.append((d, col))
        M = [[M[i][j] - d * col[i] * col[j] for j in range(n)] for i in range(n)]


def bound_certs(J, cd, cn):
    Jp = [[F(cd * J[a][c] + (cn if a == IB and c == IB else 0)) for c in range(N)] for a in range(N)]
    Ji = inverse(Jp)
    certs, bs = [], []
    for i in range(IB):
        lam = 1 / Ji[i][i]
        R = [[Jp[a][c] - (lam if a == i and c == i else 0) for c in range(N)] for a in range(N)]
        terms = [(lam, [1 if a == i else 0 for a in range(N)])]
        for d, col in decompose(R):
            den = lcm_list([x.denominator for x in col])
            m = [int(x * den) for x in col]
            g = 0
            for x in m:
                g = math.gcd(g, x)
            m = [x // g for x in m]
            # d col col^T = d (g/den)^2 m m^T
            terms.append((d * F(g, den) ** 2, m))
        # smallest b with lam (b+1)^2 > 4 cd + cn
        b = 0
        while not lam * (b + 1) ** 2 > 4 * cd + cn:
            b += 1
        certs.append(terms)
        bs.append(b)
    C = lcm_list([e.denominator for terms in certs for e, _ in terms])
    out = [[(int(e * C), m) for e, m in terms] for terms in certs]
    for i, terms in enumerate(out):
        for a in range(N):
            for c in range(N):
                lhs = C * (cd * J[a][c] + (cn if a == IB and c == IB else 0))
                assert lhs == sum(e * m[a] * m[c] for e, m in terms)
        assert C * (4 * cd + cn) < terms[0][0] * (bs[i] + 1) ** 2
    return C, bs, out


def box_size(bs):
    return 2 * math.prod(2 * b + 1 for b in bs)


def best_bounds(J):
    Q = [row[:IB] for row in J[:IB]]
    r = [J[i][IB] for i in range(IB)]
    q = [sum(x * y for x, y in zip(row, r)) for row in inverse(Q)]
    sigma = J[IB][IB] - sum(x * y for x, y in zip(r, q))
    cands = []
    base = max(F(0), -sigma)
    for delta in [F(0), F(1, 16), F(1, 8), F(1, 4), F(1, 2), F(1), F(2), F(4), F(8)]:
        rho = base + delta
        if sigma + rho <= 0:
            continue
        cands.append(rho)
    best = None
    for rho in cands:
        cd, cn = rho.denominator, rho.numerator
        C, bs, certs = bound_certs(J, cd, cn)
        sz = box_size(bs)
        if best is None or sz < best[0]:
            best = (sz, cd, cn, C, bs, certs)
    return best


# ---------- polynomials ----------

def poly_from_values(ts, vals):
    """coefficients (lowest first) of the polynomial of degree < len(ts) through the points."""
    n = len(ts)
    A = [[F(t) ** i for i in range(n)] for t in ts]
    Ai = inverse(A)
    return [sum(Ai[i][j] * vals[j] for j in range(n)) for i in range(n)]


def peval(p, t):
    s = F(0)
    for c in reversed(p):
        s = s * t + c
    return s


def pcomp(p, u, h):
    """coefficients in x of p(u + h x)."""
    out = [F(0)]
    for c in reversed(p):
        # out = out * (u + h x) + c
        new = [F(0)] * (len(out) + 1)
        for i, a in enumerate(out):
            new[i] += a * u
            new[i + 1] += a * h
        new[0] += c
        out = new
    return out


def pos_bound(c):
    i = 0
    while i < len(c) and c[i] == 0:
        i += 1
    if i == len(c):
        return False
    return c[i] + sum(min(x, 0) for x in c[i + 1:]) > 0


def pos_inf(c):
    i = 0
    while i < len(c) and c[i] == 0:
        i += 1
    if i == len(c):
        return False
    return c[i] > 0 and all(x >= 0 for x in c[i + 1:])


def trim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def simple_between(a, b):
    """a rational in [a + (b-a)/3, a + 2(b-a)/3] with small denominator."""
    lo, hi = a + (b - a) / 3, a + 2 * (b - a) / 3
    d = 1
    while True:
        n = math.ceil(lo * d)
        if F(n, d) <= hi:
            return F(n, d)
        d *= 2


class SignFail(Exception):
    pass


def sign_cert(pJ, g, req, tend, eqpts):
    """Break points (sorted rationals from 0) and pieces [(modeJ, dirL)].
    tend: the end t* (bounded) or None (unbounded: the last piece is [u, oo))."""
    def point_ok(t):
        a, b = peval(pJ, t), peval(g, t)
        if not (a > 0 or b >= 0):
            return False
        if a == 0 and b == 0 and t not in eqpts:
            return False
        return True

    def piece(a, b):
        for modeJ in (True, False):
            P = pJ if modeJ else g
            if pos_bound(pcomp(P, a, b - a)):
                return (modeJ, True)
            if pos_bound(pcomp(P, b, a - b)):
                return (modeJ, False)
        return None

    def cover(a, b, depth):
        pc = piece(a, b)
        if pc is not None:
            return [b], [pc]
        if depth > 60:
            raise SignFail(f"no certificate on [{a}, {b}]")
        m = simple_between(a, b)
        if not point_ok(m):
            raise SignFail(f"point {m} fails")
        u1, p1 = cover(a, m, depth + 1)
        u2, p2 = cover(m, b, depth + 1)
        return u1 + u2, p1 + p2

    pts = sorted(set(req))
    tail = None
    if tend is None:
        u = max(pts) if pts else F(0)
        u = max(u, F(1))
        while True:
            if pos_inf(pcomp(pJ, u, 1)):
                tail = True
                break
            if pos_inf(pcomp(g, u, 1)):
                tail = False
                break
            u *= 2
        if u not in pts:
            pts.append(u)
        pts.sort()
    for t in pts:
        if not point_ok(t):
            raise SignFail(f"required point {t} fails")
    us, ps = [pts[0]], []
    for a, b in zip(pts, pts[1:]):
        u1, p1 = cover(a, b, 0)
        us += u1
        ps += p1
    return us, ps, tail


# ---------- parsing ----------

def parse_cert(path):
    lines = open(path).read().split("\n")
    pos = 0

    def nxt():
        nonlocal pos
        l = lines[pos]
        pos += 1
        return l

    w = nxt().split()
    assert w[0] == "VERTICES"
    nv = int(w[1])
    verts = []
    for v in range(nv):
        assert nxt().split() == ["VERTEX", str(v)]
        w = nxt().split()
        assert w[0] == "J"
        Jf = [F(x) for x in w[1:]]
        assert all(x.denominator == 1 for x in Jf)
        J = [[int(Jf[N * i + j]) for j in range(N)] for i in range(N)]
        w = nxt().split()
        mins = [[int(x) for x in nxt().split()] for _ in range(int(w[1]))]
        w = nxt().split()
        gens = [[int(x) for x in nxt().split()] for _ in range(int(w[1]))]
        w = nxt().split()
        rays = [[int(x) for x in nxt().split()] for _ in range(int(w[1]))]
        w = nxt().split()
        reps = []
        for _ in range(int(w[1])):
            p = nxt().split()
            if p[0] == "U":
                reps.append(dict(ray=rays[int(p[1])], idx=int(p[1]), orbit=int(p[2]), bounded=False))
            else:
                T = [int(x) for x in p[5:]]
                reps.append(dict(ray=rays[int(p[1])], idx=int(p[1]), orbit=int(p[2]), bounded=True,
                                 tstar=F(p[3]), to=int(p[4]), T=[T[N * i:N * i + N] for i in range(N)]))
        assert nxt() == "END"
        verts.append(dict(J=J, mins=mins, reps=reps, rays=rays,
                          gens=[[g[N * i:N * i + N] for i in range(N)] for g in gens]))
    w = nxt().split()
    assert w[0] == "EQUALITY"
    eqs = []
    for _ in range(int(w[1])):
        p = nxt().split()
        eqs.append((int(p[1]), int(p[2]), F(p[3])))
    return verts, eqs


# ---------- vertex and edge data ----------

def vertex_data(V, is_eq):
    J = V['J']
    mins = V['mins']
    for k in mins:
        assert qform(J, k) == 4 and k[IB] in (0, 1)
    sz, cd, cn, C, bs, certs = best_bounds(J)
    # box check (exact) mirrors ptOK
    idx = rank_rows([symc(k) for k in mins])
    assert len(idx) == N * (N + 1) // 2
    A = [symc(mins[i]) for i in idx]
    Ai = inverse(A)
    c = lcm_list([x.denominator for row in Ai for x in row])
    Bp = [[int(x * c) for x in row] for row in Ai]
    Bt = transpose(Bp)
    AB = matmul(A, Bp)
    assert all(AB[i][j] == (c if i == j else 0) for i in range(len(A)) for j in range(len(A)))
    E = elim(J)
    dJ = det(J)
    dQ = det([row[:IB] for row in J[:IB]])
    assert dJ.denominator == 1 and dQ.denominator == 1
    dJ, dQ = int(dJ), int(dQ)
    check_det(J, E, dJ, dQ)
    assert dJ > 0 or dQ >= BOUND
    assert (dJ == 0 and dQ == BOUND) == is_eq
    return dict(J=J, mins=mins, cd=cd, cn=cn, C=C, bs=bs, certs=certs, idx=idx, A=A, Bt=Bt, Bp=Bp, c=c,
                E=E, dJ=dJ, dQ=dQ, eq=is_eq, box=sz)


def detpolys(J, Y):
    ts = list(range(N + 1))
    vJ = [det([[J[a][b] + t * Y[a][b] for b in range(N)] for a in range(N)]) for t in ts]
    vQ = [det([[J[a][b] + t * Y[a][b] for b in range(IB)] for a in range(IB)]) for t in ts]
    pJ = trim(poly_from_values(ts, vJ))
    pQ = trim(poly_from_values(ts[:N], vQ[:N]))
    assert all(x.denominator == 1 for x in pJ + pQ)
    assert all(peval(pQ, t) == vQ[t] for t in ts)
    return [int(x) for x in pJ], [int(x) for x in pQ]


def rank_one(Y):
    Y5 = [row[:IB] for row in Y[:IB]]
    y = [Y[a][IB] for a in range(IB)]
    s = Y[IB][IB]
    if all(x == 0 for row in Y5 for x in row):
        assert all(x == 0 for x in y) and s >= 0
        return 0, 0, s, 0, [0] * IB
    p = next(a for a in range(IB) if Y5[a][a] != 0)
    lam0 = Y5[p][p]
    # w proportional to row p
    w = Y5[p][:]
    g = 0
    for x in w:
        g = math.gcd(g, x)
    w = [x // g for x in w]
    if w[p] < 0:
        w = [-x for x in w]
    lam = F(Y5[p][p], w[p] * w[p])
    assert lam.denominator == 1
    lam = int(lam)
    for a in range(IB):
        for b in range(IB):
            assert Y5[a][b] == lam * w[a] * w[b]
    mu = F(y[p], w[p])
    assert mu.denominator == 1
    mu = int(mu)
    assert all(y[a] == mu * w[a] for a in range(IB))
    n0 = math.floor(F(-mu, lam))
    assert lam * n0 <= -mu < lam * (n0 + 1)
    assert lam * n0 * n0 + 2 * mu * n0 + s >= 0
    assert lam * (n0 + 1) ** 2 + 2 * mu * (n0 + 1) + s >= 0
    return lam, mu, s, n0, w


def edge_data(src, dst, rep, eqpts):
    J = src['J']
    Y = sym_from_vec(rep['ray'])
    pJ, pQ = detpolys(J, Y)
    g = list(pQ) if pQ else [0]
    g = [g[0] - BOUND] + g[1:]
    d = dict(Y=Y, bounded=rep['bounded'], pJ=pJ, pQ=pQ)
    if rep['bounded']:
        ts = rep['tstar']
        T = rep['T']
        Ti = inverse(T)
        assert all(x.denominator == 1 for row in Ti for x in row)
        Ti = [[int(x) for x in row] for row in Ti]
        Jto = dst['J']
        end = matmul(matmul(transpose(T), Jto), T)
        assert all(J[a][b] + ts * Y[a][b] == end[a][b] for a in range(N) for b in range(N))
        assert all(T[IB][j] == 0 for j in range(IB)) and T[IB][IB] in (1, -1)
        kmax = None
        # minimal vectors of the end point are Tinv k for minimal k of J_to
        for k in dst['mins']:
            kk = [sum(Ti[i][j] * k[j] for j in range(N)) for i in range(N)]
            if kk[IB] == -1:
                kk = [-x for x in kk]
            if qform(Y, kk) < 0:
                kmax = kk
                break
        assert kmax is not None and kmax[IB] in (0, 1)
        assert qform(J, kmax) + ts * qform(Y, kmax) == 4
        d.update(tn=ts.numerator, td=ts.denominator, target=rep['to'], T=T, Tinv=Ti, kmax=kmax)
        tend = ts
        sts = [F(j, N) * ts for j in range(N + 1)]
    else:
        lam, mu, s, n0, w = rank_one(Y)
        d.update(lam=lam, mu=mu, s=s, n0=n0, w=w)
        tend = None
        sts = [F(j) for j in range(N + 1)]
    sd = lcm_list([x.denominator for x in sts])
    sn = [int(x * sd) for x in sts]
    Es = []
    for x in sn:
        A = [[sd * J[a][b] + x * Y[a][b] for b in range(N)] for a in range(N)]
        E = elim(A)
        hJ = sum(c * x ** i * sd ** (N - i) for i, c in enumerate(pJ))
        hQ = sum(c * x ** i * sd ** (IB - i) for i, c in enumerate(pQ))
        check_det(A, E, hJ, hQ)
        Es.append(E)
    eqpts = list(eqpts) + ([F(0)] if src['eq'] else []) + \
        ([tend] if rep['bounded'] and dst['eq'] else [])
    req = [F(0)] + ([tend] if tend is not None else []) + list(eqpts)
    us, ps, tail = sign_cert([F(x) for x in pJ], [F(x) for x in g], req, tend,
                             set(F(x) for x in eqpts))
    wd = lcm_list([x.denominator for x in us])
    d.update(sd=sd, sn=sn, Es=Es, wd=wd, us=[int(x * wd) for x in us], ps=ps,
             tailJ=bool(tail) if tail is not None else False,
             eqs=sorted(set(int(x * wd) for x in eqpts)))
    return d


# ---------- all rays of a vertex ----------

def vec_of_sym(Y):
    return [Y[a][b] for a in range(N) for b in range(a, N)]


def congr(g, Y):
    return matmul(matmul(transpose(g), Y), g)


def expand(V):
    """For every ray of the vertex: (number of its representative, g) with Y = g^T Y_rep g,
    g^T J g = J and g in Gamma, by a search over the generators of the stabiliser."""
    J = V['J']
    ident = [[1 if i == j else 0 for j in range(N)] for i in range(N)]
    for g in V['gens']:
        assert congr(g, J) == J
        assert all(g[IB][j] == 0 for j in range(IB)) and g[IB][IB] in (1, -1)
        assert det(g) in (1, -1)
    pos = {tuple(r): i for i, r in enumerate(V['rays'])}
    assert len(pos) == len(V['rays'])
    assign = [None] * len(V['rays'])
    for ri, rep in enumerate(V['reps']):
        assert assign[rep['idx']] is None
        assign[rep['idx']] = (ri, ident)
        queue, count = [(rep['ray'], ident)], 1
        while queue:
            ray, g = queue.pop()
            for h in V['gens']:
                r2 = vec_of_sym(congr(h, sym_from_vec(ray)))
                j = pos[tuple(r2)]
                if assign[j] is None:
                    g2 = matmul(g, h)
                    assign[j] = (ri, g2)
                    queue.append((r2, g2))
                    count += 1
        assert count == rep['orbit'], (count, rep['orbit'])
    assert all(a is not None for a in assign)
    for j, (ri, g) in enumerate(assign):
        assert vec_of_sym(congr(g, sym_from_vec(V['reps'][ri]['ray']))) == V['rays'][j]
    return assign


# ---------- the cone at a vertex ----------

MASK_BITS = 20      # rows of a vertex, at most 20
FIDX_BASE = 4096    # flat numbers, below 4096
FIDX_PER = 64       # flat numbers packed in one natural number


def kernel_vec(rows):
    """A primitive integral vector spanning the kernel of 14 rows of length 15, or None."""
    n, m = len(rows[0]), len(rows)
    M = [list(r) for r in rows]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        for i in range(m):
            if i != r and M[i][c] != 0:
                f, g = M[i][c], M[r][c]
                M[i] = [g * x - f * y for x, y in zip(M[i], M[r])]
                gg = 0
                for x in M[i]:
                    gg = math.gcd(gg, x)
                if gg > 1:
                    M[i] = [x // gg for x in M[i]]
        piv.append(c)
        r += 1
        if r == m:
            break
    if r < m:
        return None
    free = [c for c in range(n) if c not in piv][0]
    Lc = 1
    for i in range(m):
        Lc = lcm(Lc, abs(M[i][piv[i]]))
    y = [0] * n
    y[free] = Lc
    for i in range(m):
        y[piv[i]] = -M[i][free] * Lc // M[i][piv[i]]
    g = 0
    for x in y:
        g = math.gcd(g, x)
    y = [x // g for x in y]
    assert all(sum(a * b for a, b in zip(row, y)) == 0 for row in rows)
    return y


def dotv(a, b):
    return sum(x * y for x, y in zip(a, b))


def cone_data(V, vdat):
    """Rank certificate, hyperplane flats and, for every set of 14 rows, a flat containing it."""
    import itertools
    rows = [symc(k) for k in V['mins']]
    m = len(rows)
    assert m <= MASK_BITS
    d = len(rows[0])
    rays = V['rays']
    rpos = {tuple(r): i for i, r in enumerate(rays)}
    for r in rays:
        assert all(dotv(a, r) >= 0 for a in rows)
    flats, fpos = [], {}
    subsets = list(itertools.combinations(range(m), d - 1))
    fidx = [None] * len(subsets)
    for si, S in enumerate(subsets):
        y = kernel_vec([rows[i] for i in S])
        if y is None:
            continue
        vals = [dotv(a, y) for a in rows]
        mask = sum(1 << i for i in range(m) if vals[i] == 0)
        if mask not in fpos:
            if any(x > 0 for x in vals) and any(x < 0 for x in vals):
                p = next(i for i in range(m) if vals[i] > 0)
                q = next(i for i in range(m) if vals[i] < 0)
                kind = ('mixed', y, p, q)
            elif tuple(y) in rpos:
                kind = ('ray', rpos[tuple(y)], False)
            else:
                kind = ('ray', rpos[tuple(-x for x in y)], True)
            fpos[mask] = len(flats)
            flats.append((mask, kind))
        fidx[si] = fpos[mask]
    masks = [f[0] for f in flats]
    for si, S in enumerate(subsets):
        if fidx[si] is None:
            sm = sum(1 << i for i in S)
            fidx[si] = next(j for j, F_ in enumerate(masks) if sm & F_ == sm)
    assert len(flats) < FIDX_BASE
    assert sorted(f[1][1] for f in flats if f[1][0] == 'ray') == list(range(len(rays)))
    ymax = max([abs(x) for f in flats if f[1][0] == 'mixed' for x in f[1][1]] + [0])
    w = 1
    while 2 ** (w - 1) <= ymax:
        w += 1
    packed = []
    for mask, kind in flats:
        if kind[0] == 'ray':
            rest = 2 * (int(kind[2]) + 2 * kind[1])
        else:
            _, y, p, q = kind
            assert p < 32 and q < 32
            yv = sum((x + 2 ** (w - 1)) << (w * i) for i, x in enumerate(y))
            rest = 1 + 2 * (p + 32 * q + 1024 * yv)
        packed.append(mask + (rest << MASK_BITS))
    nmixed = sum(1 for f in flats if f[1][0] == 'mixed')
    return dict(idx=vdat['idx'], B=vdat['Bp'], c=vdat['c'], w=w, flats=packed, fidx=fidx, m=m,
                nflats=len(flats), nmixed=nmixed, nsub=len(subsets))


def pack_idx(xs):
    """Flat indices, FIDX_PER to a number in base FIDX_BASE."""
    return [sum(x * FIDX_BASE ** i for i, x in enumerate(xs[c:c + FIDX_PER]))
            for c in range(0, len(xs), FIDX_PER)]


def sub_pieces(k, l, acc, start, out, name):
    """Split the walk of subsOK over the k-element sublists of l (in the order of combs) into
    pieces of at most SUB_CHUNK sets. Appends (k, l, acc, start, count) for every piece to out and
    returns the proof term built from the pieces (named by name) with subsOK_node, and the number
    of sets."""
    n = math.comb(len(l), k)
    if n <= SUB_CHUNK or k == 0:
        out.append((k, l, acc, start, n))
        return name(len(out) - 1), n
    left, n1 = sub_pieces(k - 1, l[1:], acc + [l[0]], start, out, name)
    right, n2 = sub_pieces(k, l[1:], acc, start + n1, out, name)
    return f"(subsOK_node {left} {right})", n1 + n2


def acc_lean(acc):
    """The mask of the rows chosen before, as subsOK builds it."""
    return "(0" + "".join(f" ||| 2 ^ {x}" for x in acc) + ")"


# ---------- Lean output ----------

def L(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x) if x >= 0 else f"({x})"
    if isinstance(x, tuple):
        return "(" + ", ".join(L(y) for y in x) + ")"
    if isinstance(x, list):
        return "[" + ", ".join(L(y) for y in x) + "]"
    raise TypeError(x)


def lean_vertex(name, v):
    certs = [[(e, m) for e, m in terms] for terms in v['certs']]
    return f"""def {name} : Vtx where
  J := {L(v['J'])}
  mins := {L(v['mins'])}
  cd := {L(v['cd'])}
  cn := {L(v['cn'])}
  C := {L(v['C'])}
  bs := {L(v['bs'])}
  certs := {L(certs)}
  E := {L(v['E'])}
  dJ := {L(v['dJ'])}
  dQ := {L(v['dQ'])}
  eq := {L(v['eq'])}
"""


def lean_edge(name, e):
    z6 = [[0] * N for _ in range(N)]
    f = dict(Y=e['Y'], bounded=e['bounded'], tn=e.get('tn', 0), td=e.get('td', 1),
             target=e.get('target', 0), T=e.get('T', z6), Tinv=e.get('Tinv', z6),
             kmax=e.get('kmax', []), lam=e.get('lam', 0), mu=e.get('mu', 0), s=e.get('s', 0),
             n0=e.get('n0', 0), w=e.get('w', []), pJ=e['pJ'], pQ=e['pQ'], sd=e['sd'], sn=e['sn'],
             Es=e['Es'], wd=e['wd'], us=e['us'], ps=[tuple(p) for p in e['ps']], tailJ=e['tailJ'],
             eqs=e['eqs'])
    body = "\n".join(f"  {k} := {L(v)}" for k, v in f.items())
    return f"def {name} : Edge where\n{body}\n"


HEADER = """/- Generated by lean/generate/make_lean.py from cert/certificate.txt; do not edit. -/
"""


EDGES_PER_FILE = 40   # edges in one file of D4Lean/Cert, so that the files are checked in parallel
LIST_CHUNK = 300      # elements of one list literal
FLAT_CHUNK = 100      # flats checked in one theorem
SUB_CHUNK = 2000      # sets of 14 rows checked in one theorem


def lean_list(name, typ, xs):
    """A long list as the concatenation of literals of at most LIST_CHUNK elements."""
    if len(xs) <= LIST_CHUNK:
        return f"def {name} : List {typ} := {L(xs)}\n"
    parts = [xs[c:c + LIST_CHUNK] for c in range(0, len(xs), LIST_CHUNK)]
    out = "".join(f"def {name}_{j} : List {typ} := {L(p)}\n" for j, p in enumerate(parts))
    return out + f"def {name} : List {typ} :=\n  " + " ++ ".join(
        f"{name}_{j}" for j in range(len(parts))) + "\n"


def write_lean(vd, edges, cones, outdir):
    ddir = os.path.join(outdir, "D4Lean", "Data")
    cdir = os.path.join(outdir, "D4Lean", "Cert")
    os.makedirs(ddir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)
    for f in os.listdir(cdir):
        if f.endswith(".lean"):
            os.remove(os.path.join(cdir, f))
    n = len(vd)
    with open(os.path.join(ddir, "Vertices.lean"), "w") as fh:
        fh.write(HEADER + "import D4Lean.Check\n\nnamespace D4\n\n")
        for i, v in enumerate(vd):
            fh.write(lean_vertex(f"v{i}", v) + "\n")
        fh.write(f"/-- The {n} vertices. -/\ndef vertices : List Vtx :=\n  [" +
                 ", ".join(f"v{i}" for i in range(n)) + "]\n\n")
        fh.write("/-- Vertex `i` of the list. -/\ndef vtx (i : Nat) : Vtx := vertices.getD i default\n\n")
        fh.write("/-- The check of an edge at vertex `src`: the end point of a bounded edge is one of the\n"
                 f"vertices, number `target`. -/\ndef edgeOK (src : Nat) (e : Edge) : Bool :=\n"
                 f"  (!e.bounded || decide (e.target < {n})) &&\n"
                 "    Edge.ok (vtx src).J (vtx (if e.bounded then e.target else src)).J e\n\n")
        fh.write("end D4\n")
    for i in range(n):
        es = [e for s_, e in edges if s_ == i]
        chunks = [list(range(c, min(c + EDGES_PER_FILE, len(es))))
                  for c in range(0, len(es), EDGES_PER_FILE)]
        for ci, ch in enumerate(chunks):
            with open(os.path.join(cdir, f"V{i}E{ci}.lean"), "w") as fh:
                fh.write(HEADER + "import D4Lean.Data.Vertices\n\nnamespace D4\n\n")
                for j in ch:
                    fh.write(lean_edge(f"e{i}_{j}", es[j]) + "\n")
                    fh.write(f"theorem e{i}_{j}_ok : edgeOK {i} e{i}_{j} = true := by decide +kernel\n\n")
                fh.write(f"/-- Edges {ch[0]} to {ch[-1]} at vertex {i}. -/\n")
                fh.write(f"def edges{i}c{ci} : List Edge :=\n  [" + ", ".join(f"e{i}_{j}" for j in ch) + "]\n\n")
                fh.write(f"theorem edges{i}c{ci}_ok : edges{i}c{ci}.all (edgeOK {i}) = true := by\n")
                fh.write(f"  simp only [edges{i}c{ci}, List.all_cons, List.all_nil, " +
                         "".join(f"e{i}_{j}_ok, " for j in ch) + "Bool.true_and]\n\n")
                fh.write("end D4\n")
        with open(os.path.join(cdir, f"V{i}.lean"), "w") as fh:
            fh.write(HEADER + "".join(f"import D4Lean.Cert.V{i}E{ci}\n" for ci in range(len(chunks))) +
                     "\nnamespace D4\n\n")
            fh.write(f"theorem v{i}_ok : v{i}.ok = true := by decide +kernel\n\n")
            fh.write(f"/-- The edges at vertex {i}, one for each ray of its tangent cone, in the order of the\n"
                     "certificate. -/\n")
            fh.write(f"def edges{i} : List Edge :=\n  " +
                     " ++ ".join(f"edges{i}c{ci}" for ci in range(len(chunks))) + "\n\n")
            if len(chunks) == 1:
                fh.write(f"theorem edges{i}_ok : edges{i}.all (edgeOK {i}) = true := edges{i}c0_ok\n\n")
            else:
                fh.write(f"theorem edges{i}_ok : edges{i}.all (edgeOK {i}) = true := by\n")
                fh.write(f"  simp only [edges{i}, List.all_append, " +
                         "".join(f"edges{i}c{ci}_ok, " for ci in range(len(chunks))) + "Bool.and_self]\n\n")
            fh.write("end D4\n")
        cd = cones[i]
        rows = f"(v{i}.mins.map symcL)"
        rays = f"(edges{i}.map fun e => psiL e.Y)"
        fl = [cd['flats'][c:c + FLAT_CHUNK] for c in range(0, len(cd['flats']), FLAT_CHUNK)]
        pieces = []
        term, nsub = sub_pieces(14, list(range(cd['m'])), [], 0, pieces, lambda j_: f"cone{i}_sub{j_}")
        assert nsub == cd['nsub'] == len(cd['fidx'])
        with open(os.path.join(cdir, f"C{i}.lean"), "w") as fh:
            fh.write(HEADER + f"import D4Lean.Cert.V{i}\n\nnamespace D4\n\n")
            for j_, part in enumerate(fl):
                fh.write(f"def cone{i}Flats{j_} : List Nat := {L(part)}\n\n")
            fh.write(f"/-- The cone at vertex {i}: {cd['nflats']} flats ({cd['nmixed']} of them not faces). -/\n")
            fh.write(f"def cone{i} : ConeData where\n  idx := {L(cd['idx'])}\n  B := {L(cd['B'])}\n"
                     f"  c := {L(cd['c'])}\n  w := {L(cd['w'])}\n  flats := " +
                     " ++ ".join(f"cone{i}Flats{j_}" for j_ in range(len(fl))) + "\n\n")
            fh.write(f"theorem cone{i}_rank : rankOK {rows} cone{i}.idx cone{i}.B cone{i}.c = true := by\n"
                     "  decide +kernel\n\n")
            nch = len([e for s_, e in edges if s_ == i])
            nch = (nch + EDGES_PER_FILE - 1) // EDGES_PER_FILE
            for ci in range(nch):
                fh.write(f"theorem cone{i}_rays{ci} : raysOK {rows} (edges{i}c{ci}.map fun e => psiL e.Y) = true := by\n"
                         "  decide +kernel\n\n")
            if nch == 1:
                fh.write(f"theorem cone{i}_rays : raysOK {rows} {rays} = true := cone{i}_rays0\n\n")
            else:
                fh.write(f"theorem cone{i}_rays : raysOK {rows} {rays} = true := by\n"
                         f"  simp only [edges{i}, List.map_append, raysOK_append, " +
                         "".join(f"cone{i}_rays{ci}, " for ci in range(nch)) + "Bool.and_self]\n\n")
            for j_ in range(len(fl)):
                fh.write(f"theorem cone{i}_flats{j_} : flatsOK {rows} {rays} {cd['w']} cone{i}Flats{j_} = true := by\n"
                         "  decide +kernel\n\n")
            ft = "cone%d_flats0" % i
            for j_ in range(1, len(fl)):
                ft = f"(flatsOK_append {ft} cone{i}_flats{j_})"
            fh.write(f"theorem cone{i}_flats : flatsOK {rows} {rays} cone{i}.w cone{i}.flats = true :=\n  {ft}\n\n")
            for pj, (k, l, acc, start, cnt) in enumerate(pieces):
                fh.write(f"theorem cone{i}_sub{pj} : ∃ js, subsOK cone{i}.tree {k} {L(l)} {acc_lean(acc)} js = some [] :=\n"
                         f"  ⟨idxs {L(pack_idx(cd['fidx'][start:start + cnt]))} {cnt}, by decide +kernel⟩\n\n")
            sterm = term
            fh.write(f"theorem cone{i}_subs : ∃ js, subsOK cone{i}.tree 14 (List.range {rows}.length) 0 js = some [] := by\n"
                     f"  rw [show (List.range {rows}.length) = {L(list(range(cd['m'])))} from rfl]\n"
                     f"  exact {sterm}\n\n")
            fh.write(f"theorem cone{i}_ok : ConeOK v{i}.mins {rays} cone{i} :=\n"
                     f"  ⟨cone{i}_rank, cone{i}_rays, cone{i}_flats, cone{i}_subs⟩\n\n")
            fh.write("end D4\n")
    with open(os.path.join(cdir, "All.lean"), "w") as fh:
        fh.write(HEADER + "".join(f"import D4Lean.Cert.C{i}\n" for i in range(n)) + "\nnamespace D4\n\n")
        fh.write(f"/-- The edges at each vertex. -/\ndef edgeLists : List (List Edge) :=\n  [" +
                 ", ".join(f"edges{i}" for i in range(n)) + "]\n\n")
        fh.write("/-- The edges at vertex `i`. -/\ndef edgesOf (i : Nat) : List Edge := edgeLists.getD i []\n\n")
        fh.write(f"/-- The cone data at each vertex. -/\ndef coneList : List ConeData :=\n  [" +
                 ", ".join(f"cone{i}" for i in range(n)) + "]\n\n")
        fh.write("/-- The cone data at vertex `i`. -/\ndef coneOf (i : Nat) : ConeData := coneList.getD i default\n\n")
        fh.write(f"theorem vertices_length : vertices.length = {n} := rfl\n\n")
        fh.write("theorem vertices_ok : vertices.all Vtx.ok = true := by\n")
        fh.write("  simp only [vertices, List.all_cons, List.all_nil, " +
                 "".join(f"v{i}_ok, " for i in range(n)) + "Bool.and_self]\n\n")
        fh.write(f"theorem edges_ok : ∀ i < {n}, (edgesOf i).all (edgeOK i) = true\n")
        for i in range(n):
            fh.write(f"  | {i}, _ => edges{i}_ok\n")
        fh.write(f"  | _ + {n}, h => absurd h (by omega)\n\n")
        fh.write(f"theorem cones_ok : ∀ i < {n},\n"
                 "    ConeOK (vtx i).mins ((edgesOf i).map fun e => psiL e.Y) (coneOf i)\n")
        for i in range(n):
            fh.write(f"  | {i}, _ => cone{i}_ok\n")
        fh.write(f"  | _ + {n}, h => absurd h (by omega)\n")
        fh.write("\nend D4\n")


def main(cpath, epath, outdir):
    verts, eqs = parse_cert(cpath)
    os.makedirs(outdir, exist_ok=True)
    vd = []
    for i, V in enumerate(verts):
        is_eq = any(f == i and r == -1 for f, r, _ in eqs)
        vd.append(vertex_data(V, is_eq))
        print(f"vertex {i}: box {vd[-1]['box']} points", file=sys.stderr)
    total_box = sum(v['box'] for v in vd)
    print(f"box points in all: {total_box}", file=sys.stderr)
    edges, cones = [], []
    for i, V in enumerate(verts):
        assign = expand(V)
        repdata = []
        for rep in V['reps']:
            eqp = [t for f, r, t in eqs if f == i and r == rep['idx']]
            dst = vd[rep['to']] if rep['bounded'] else None
            repdata.append((eqp, edge_data(vd[i], dst, rep, eqp)))
        for j, (ri, g) in enumerate(assign):
            rep = V['reps'][ri]
            eqp, ed = repdata[ri]
            if j == rep['idx']:
                edges.append((i, ed))
                continue
            r2 = dict(rep, ray=V['rays'][j], idx=j)
            if rep['bounded']:
                r2['T'] = matmul(rep['T'], g)
            dst = vd[rep['to']] if rep['bounded'] else None
            e = edge_data(vd[i], dst, r2, eqp)
            for k in ('pJ', 'pQ', 'wd', 'us', 'ps', 'eqs', 'tailJ', 'sd', 'sn'):
                assert e[k] == ed[k]
            edges.append((i, e))
        cones.append(cone_data(V, vd[i]))
        c = cones[-1]
        print(f"vertex {i}: {len(assign)} edges, {c['nflats']} flats ({c['nmixed']} mixed), "
              f"y width {c['w']}", file=sys.stderr)
    print(f"edges: {len(edges)}, pieces: {sum(len(e['ps']) for _, e in edges)}", file=sys.stderr)
    write_lean(vd, edges, cones, outdir)
    return vd, edges


if __name__ == '__main__':
    main(*sys.argv[1:4])
