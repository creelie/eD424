"""
zonal.py -- from the integrals P(S) to the zonal matrices Z_lambda.

Two things happen here.  First the polynomial P(S) computed by psker, which
lives in the seven free entries of the 4 x 2 matrix S, is reduced modulo the
relations S^T S = I to a polynomial in the four entries of its top-left 2 x 2
block, by the procedure of Section 3.4 of

    D. de Laat, N. M. Leijenhorst, W. H. H. de Muinck Keizer,
    Optimality and uniqueness of the D4 root system, arXiv:2404.18794.

Second, an entry of Z_lambda is read off from that polynomial at a given tuple
of inner products: the entries of the top-left block of S are, for two frames
built from pairs of points, quotients of a linear form in the inner products by
square roots of 2(1 +- u), and the powers of those square roots that are left
over are supplied monomial by monomial from the signature and the indices.
"""
from fractions import Fraction as Q

from gl2 import calculatefactors

NS = 7
# s0..s6 are S11 S21 S31 | S12 S22 S32 S42
S11, S21, S31, S12, S22, S32, S42 = range(7)
KEEP = (S11, S21, S12, S22)         # the top-left block, column major


def _mul(p, q):
    r = {}
    for ep, cp in p.items():
        for eq, cq in q.items():
            e = tuple(a + b for a, b in zip(ep, eq))
            r[e] = r.get(e, 0) + cp * cq
    return {e: c for e, c in r.items() if c}


def _pow(p, n):
    r = {(0,) * NS: Q(1)}
    for _ in range(n):
        r = _mul(r, p)
    return r


def _mon(**kw):
    e = [0] * NS
    for k, v in kw.items():
        e[globals()[k]] = v
    return tuple(e)


def replace_mon_by_pol(p, v, q, throwaway):
    """replace the monomial with exponent v by q, dropping any term that still
    has the throwaway variable once the monomial has been divided out"""
    out = {}
    buckets = {}
    for ev, c in p.items():
        k = 0
        while all(ev[i] >= (k + 1) * v[i] for i in range(NS)):
            k += 1
        if all(ev[i] >= k * v[i] + throwaway[i] for i in range(NS)):
            continue
        rest = tuple(ev[i] - k * v[i] for i in range(NS))
        buckets.setdefault(k, {})[rest] = buckets.setdefault(k, {}).get(rest, 0) + c
    for k, part in buckets.items():
        term = part if k == 0 else _mul(part, _pow(q, k))
        for e, c in term.items():
            out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c}


def reducepol(p):
    """p in seven variables -> a polynomial in the four of the top-left block"""
    # S42^2 -> 1 - S12^2 - S22^2 - S32^2
    q = {(0,) * NS: Q(1), _mon(S12=2): Q(-1), _mon(S22=2): Q(-1),
         _mon(S32=2): Q(-1)}
    p = replace_mon_by_pol(p, _mon(S42=2), q, _mon(S42=1))
    # S32*S31 -> -(S12*S11 + S22*S21)
    q = {_mon(S12=1, S11=1): Q(-1), _mon(S22=1, S21=1): Q(-1)}
    p = replace_mon_by_pol(p, _mon(S32=1, S31=1), q, _mon(S32=1))
    # S31^2 -> 1 - S11^2 - S21^2
    q = {(0,) * NS: Q(1), _mon(S11=2): Q(-1), _mon(S21=2): Q(-1)}
    p = replace_mon_by_pol(p, _mon(S31=2), q, _mon(S31=1))
    out = {}
    for ev, c in p.items():
        if ev[S31] or ev[S32] or ev[S42]:
            continue
        out[tuple(ev[i] for i in KEEP)] = c
    return out


def load_ps(path):
    """read psker's output, apply the scalar of Section 3.1, and reduce"""
    raw, cur, den = {}, None, None
    for line in open(path):
        line = line.strip()
        if line.startswith('E '):
            f = line.split()
            cur = ((int(f[1]), int(f[2])), int(f[3]), int(f[4]))
            raw[cur] = {}
        elif line.startswith('D '):
            den = int(line.split()[1])
        elif line == '.':
            cur = None
        elif line and cur is not None:
            f = line.split()
            raw[cur][tuple(int(x) for x in f[:NS])] = Q(int(f[NS]), den)
    out = {}
    for (lam, k1, k2), p in raw.items():
        if sum(lam) == 0:
            fac = Q(1)
        else:
            fac = 2 * Q(str(calculatefactors(lam)[k2]))
        out[(lam, k1, k2)] = reducepol({e: c * fac for e, c in p.items()})
    return out


def countels(lam, j, k):
    return lam[0] - k if j == 1 else lam[1] + k


def scaling_factors(u):
    """the numerators of the top-left block of S, and the squared denominators

    u is (u1, ..., u6): the two inner products inside the two point sets and
    the four between them."""
    a, b, c, d = u[2], u[3], u[4], u[5]
    Sscaled = [a + b + c + d,      # S11
               a + b - c - d,      # S21
               a - b + c - d,      # S12
               a - b - c + d]      # S22
    pols = [2 * (1 + u[0]), 2 * (1 - u[0]), 2 * (1 + u[1]), 2 * (1 - u[1])]
    return Sscaled, pols


def evaluate_zonal_matrix(ps, lam, w1, w2, u, one, zero, coerce=lambda x: x):
    """the (w1, w2) entry of Z_lambda at the inner products u

    `one` and `zero` are the constants of whatever ring the u live in, and
    `coerce` turns a rational coefficient into an element of that ring."""
    i, j = max(w1, w2), min(w1, w2)
    p = ps[(lam, i, j)]
    if i != w1:
        u = [u[1], u[0], u[2], u[4], u[3], u[5]]
    Sscaled, pols = scaling_factors(u)
    row_sums = [countels(lam, 1, i), countels(lam, 2, i)]
    col_sums = [countels(lam, 1, j), countels(lam, 2, j)]
    res = zero
    for ev, c in p.items():
        cur_rows = [ev[0] + ev[2], ev[1] + ev[3]]     # column major 2 x 2
        cur_cols = [ev[0] + ev[1], ev[2] + ev[3]]
        fa_exp = [row_sums[0] - cur_rows[0], row_sums[1] - cur_rows[1],
                  col_sums[0] - cur_cols[0], col_sums[1] - cur_cols[1]]
        if any(x % 2 for x in fa_exp):
            raise RuntimeError('odd half power in the scaling factors: '
                               'lambda=%s w=(%d,%d) ev=%s' % (lam, w1, w2, ev))
        term = one
        for k in range(4):
            if ev[k]:
                term = term * Sscaled[k] ** ev[k]
        for k in range(4):
            if fa_exp[k] > 1:
                term = term * pols[k] ** (fa_exp[k] // 2)
            elif fa_exp[k] < 0:
                raise RuntimeError('negative power in the scaling factors')
        res = res + term * coerce(c)
    return res
