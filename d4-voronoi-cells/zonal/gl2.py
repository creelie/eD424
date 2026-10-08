"""
gl2.py -- the GL(2) matrix coefficients and the scalars of Section 3.1.

rho(lambda, i, j) is the (i,j) entry of the irreducible representation of GL(2)
with signature lambda, as a polynomial in the four entries of the argument:

    rho(lambda,i,j)(A) = det(A)^lambda2 [e1^(m-i) e2^i](A11 e1 + A21 e2)^(m-j)
                                                       (A12 e1 + A22 e2)^j

with m = lambda1 - lambda2.

calculatefactors(lambda) returns the scalars c_k of Section 3.1 of

    D. de Laat, N. M. Leijenhorst, W. H. H. de Muinck Keizer,
    Optimality and uniqueness of the D4 root system, arXiv:2404.18794,

which say how the integral P(S) is expressed through the single tuple J_{0,0,e}
once the symmetry of the integrand is used.  Without them the entries of the
zonal matrix would be off by a scalar that depends on the index, and the
polynomial identities of step 5 of the verification would not hold.

Variables of a 2x2 argument are numbered A11, A21, A12, A22 throughout, matching
the column-major convention of the original.
"""
from functools import lru_cache
from math import comb
from itertools import product as iproduct

from flint import fmpz_mat, fmpq_mat, fmpq

# ---------------------------------------------------------------- polynomials
# a polynomial in A11, A21, A12, A22 is a dict {(e11, e21, e12, e22): coeff}

V11, V21, V12, V22 = 0, 1, 2, 3


def pmul(p, q):
    r = {}
    for ep, cp in p.items():
        for eq, cq in q.items():
            e = (ep[0] + eq[0], ep[1] + eq[1], ep[2] + eq[2], ep[3] + eq[3])
            r[e] = r.get(e, 0) + cp * cq
    return {e: c for e, c in r.items() if c}


def pone():
    return {(0, 0, 0, 0): 1}


def pvar(k):
    e = [0, 0, 0, 0]
    e[k] = 1
    return {tuple(e): 1}


def ppow(p, n):
    r = pone()
    for _ in range(n):
        r = pmul(r, p)
    return r


def pdet():
    return {(1, 0, 0, 1): 1, (0, 1, 1, 0): -1}


@lru_cache(maxsize=None)
def rho(lam, i, j):
    """(i,j) entry of the GL(2) irrep of signature lam, as a dict"""
    l1, l2 = lam
    m = l1 - l2
    # coefficient of e1^(m-i) e2^i in (A11 e1 + A21 e2)^(m-j) (A12 e1 + A22 e2)^j
    out = {}
    for a in range(m - j + 1):          # e2 taken from the first factor
        b = i - a                       # e2 taken from the second
        if b < 0 or b > j:
            continue
        e = (m - j - a, a, j - b, b)
        out[e] = out.get(e, 0) + comb(m - j, a) * comb(j, b)
    if l2:
        out = pmul(out, ppow(pdet(), l2))
    return out


# ------------------------------------------------------------ Section 3.1
def tuplator(l2):
    """representatives of the orbits of tuples of length l2 over {1,2}^2"""
    pairs = [(1, 1), (1, 2), (2, 1), (2, 2)]
    if l2 == 0:
        return [()]
    tup = [(1,), (2,), (3,), (4,)]
    n = l2
    while n > 1:
        nxt = []
        for t in tup:
            for j in range(1, t[0] + 1):
                nxt.append((j,) + t)
        tup = nxt
        n -= 1
    return [tuple(pairs[x - 1] for x in t) for t in tup]


def detpoly(a):
    """prod_i A_{a_i[0],1} A_{a_i[1],2}"""
    p = pone()
    for ai in a:
        p = pmul(p, pvar(V11 if ai[0] == 1 else V21))
        p = pmul(p, pvar(V12 if ai[1] == 1 else V22))
    return p


def countels(lam, j, k):
    return lam[0] - k if j == 1 else lam[1] + k


def Dcount(j, a):
    return sum(1 for ai in a for x in ai if x == j)


def allmonomials(deg):
    out = []
    for i1 in range(deg + 1):
        for i2 in range(deg - i1 + 1):
            for i3 in range(deg - i1 - i2 + 1):
                out.append((i1, i2, i3, deg - i1 - i2 - i3))
    return out


def indices_fun(lam):
    m = lam[0] - lam[1]
    return [(l1, l2, a) for l1 in range(m + 1) for l2 in range(m + 1)
            for a in tuplator(lam[1])]


def indices_fun_small(lam):
    m = lam[0] - lam[1]
    mm = (m, 0)
    out = []
    for (l1, l2, a) in indices_fun(lam):
        c1 = (lam[1] + countels(mm, 2, l1) + countels(mm, 2, l2)
              + Dcount(2, a)) % 2 == 0
        c2 = lam[1] + countels(mm, 1, l1) - countels(mm, 1, l2) - Dcount(1, a) == 0
        c3 = lam[1] + countels(mm, 2, l1) - countels(mm, 2, l2) - Dcount(2, a) == 0
        if c1 and c2 and c3:
            out.append((l1, l2, a))
    return out


def cpoly(k2, l2, a, m):
    return pmul(rho((m, 0), k2, l2), detpoly(a))


def term_constraint_system(k2, lam):
    """rows: one per tuple of IndSmall; columns: [-I | the K_{l1,mu} columns]"""
    m = lam[0] - lam[1]
    M = allmonomials(sum(lam))
    pos = {mo: i for i, mo in enumerate(M)}
    cold = len(M)
    ind = indices_fun_small(lam)
    rows = []
    for (l1, l2, a) in ind:
        v = [0] * (cold * (m + 1))
        for e, c in cpoly(l2, k2, a, m).items():
            v[pos[e] + l1 * cold] = c
        rows.append(v)
    ncol = cold * (m + 1)
    keep = [i for i in range(ncol) if any(r[i] for r in rows)]
    nr = len(rows)
    out = []
    for i, r in enumerate(rows):
        lhs = [0] * nr
        lhs[i] = -1
        out.append(lhs + [r[j] for j in keep])
    return out, nr


def Jsystem(lam):
    """the constraints dPhi(X) J = 0 for X = [0 1; -1 0]"""
    m = lam[0] - lam[1]
    X = ((0, 1), (-1, 0))
    ind = indices_fun(lam)
    small = indices_fun_small(lam)
    where = {t: i for i, t in enumerate(small)}
    nc = len(small)
    rows = []
    for (l1, l2, a) in ind:
        row = [0] * nc
        l3 = l1 + 1
        if l3 <= m and (l3, l2, a) in where:
            row[where[(l3, l2, a)]] += l3
        l3 = l1 - 1
        if l3 >= 0 and (l3, l2, a) in where:
            row[where[(l3, l2, a)]] += -(m - l3)
        l4 = l2 + 1
        if l4 <= m and (l1, l4, a) in where:
            row[where[(l1, l4, a)]] += l4
        l4 = l2 - 1
        if l4 >= 0 and (l1, l4, a) in where:
            row[where[(l1, l4, a)]] += -(m - l4)
        for b, (ai, bi) in perturb_tuple(a):
            bs = tuple(sorted(b))
            if (l1, l2, bs) in where:
                k = where[(l1, l2, bs)]
                for j in range(2):
                    row[k] += X[ai[j] - 1][bi[j] - 1]
        if any(row):
            rows.append(row)
    return rows, nc


def distance_one_tuple(v):
    return [(1, 2), (2, 1)] if v[0] == v[1] else [(1, 1), (2, 2)]


def perturb_tuple(a):
    out = []
    for i in range(len(a)):
        for nb in distance_one_tuple(a[i]):
            b = list(a)
            b[i] = nb
            out.append((tuple(b), (a[i], nb)))
    return out


def is_relevant(t):
    return t[0] == t[1] and all(ai[0] != ai[1] for ai in t[2])


def bigJsystem(lam, k2):
    tcs, nsmall = term_constraint_system(k2, lam)
    ms = len(tcs[0]) - nsmall
    js, nc = Jsystem(lam)
    big = [r + [0] * ms for r in js]
    return big + tcs


def calculatefactor(lam, k2):
    """the scalar c with P(S) = c J_{0,0,e} for this lambda and k2"""
    S = bigJsystem(lam, k2)
    ind = indices_fun_small(lam)
    nrows = len(S)
    m1, m2, sumcoeff = [], [], []
    for tidx in range(len(S[0])):
        col = [S[r][tidx] for r in range(nrows)]
        if tidx < len(ind) and is_relevant(ind[tidx]):
            t = ind[tidx]
            m2.insert(0, col)
            nsg = sum(1 for ai in t[2] if ai == (2, 1))
            sumcoeff.insert(0, (-1) ** nsg * comb(lam[1], nsg))
        else:
            m1.append(col)
    cols = m1 + m2
    mat = fmpz_mat([[cols[c][r] for c in range(len(cols))] for r in range(nrows)])
    H = mat.hnf()
    c1, c2 = len(m1), len(m2)
    final = []
    for q in range(H.nrows()):
        row = [int(H[q, c]) for c in range(H.ncols())]
        if not any(row[:c1]) and any(row[c1:]):
            final.append(row[c1:])
    nullity = c2 - len(final)
    if nullity != 1:
        raise RuntimeError('projected solution space not one dimensional '
                           '(lambda=%s, k2=%d, nullity=%d)' % (lam, k2, nullity))
    # add the row saying that the new variable equals P(S) and row reduce; the
    # factor is minus the last entry of the first row
    sys2 = [[-1] + sumcoeff] + [[0] + r for r in final]
    A = fmpq_mat([[fmpq(x) for x in r] for r in sys2])
    R = A.rref()[0]
    return -fmpq(R[0, R.ncols() - 1])


@lru_cache(maxsize=None)
def calculatefactors(lam):
    m = lam[0] - lam[1]
    return tuple(calculatefactor(lam, k) for k in range(m + 1))


if __name__ == '__main__':
    import sys
    for lam in [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 1), (4, 2), (5, 3)]:
        print(lam, [str(x) for x in calculatefactors(lam)])
