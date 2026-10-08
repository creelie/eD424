"""
o4.py -- exact integration of monomials over the orthogonal group O(n).

    int_{O(n)} prod_{i,j} gamma_{ij}^{M_{ij}} dgamma

by the recursion of

    T. Gorin, Integrals of monomials over the orthogonal group,
    J. Math. Phys. 43, 3342 (2002);
    T. Gorin, G. V. Lopez, Monomial integrals on the classical groups,
    J. Math. Phys. 49, 013503 (2008),

which peels off one column of the exponent matrix at a time.  This is a fresh
implementation in exact rational arithmetic; the test at the bottom is the one
distributed with LasserreSphericalCodes (src/integrate_orthogonal.jl), which
gives seven values that can be checked by hand.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial


def poch(z, n):
    """the rising factorial z (z+1) ... (z+n-1)"""
    r = Q(1)
    for k in range(n):
        r *= (z + k)
    return r


def _bfac(a, b, z1, z2):
    return Q(-1) ** (a - b) * poch(z1, b) * poch(z1, a - b) / poch(z1 - z2, a)


@lru_cache(maxsize=None)
def int_row(d, m):
    """int over O(n) of gamma_{1,1}^m1 ... gamma_{1,n}^mn, one row only"""
    if len(m) == 0:
        return Q(1)
    if any(v % 2 for v in m):
        return Q(0)
    num = Q(1)
    for v in m:
        num *= poch(Q(1, 2), v // 2)
    return num / poch(Q(d, 2), sum(m) // 2)


def _multiexponents(length, total):
    """all nonnegative integer vectors of the given length summing to total"""
    if length == 0:
        return [()] if total == 0 else []
    if length == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for rest in _multiexponents(length - 1, total - first):
            out.append((first,) + rest)
    return out


MEXP = {}


def multiexponents(length, total):
    key = (length, total)
    if key not in MEXP:
        MEXP[key] = _multiexponents(length, total)
    return MEXP[key]


def _even_vectors(m):
    """all vectors k with k_i even and 0 <= k_i <= m_i"""
    out = [()]
    for b in m:
        out = [t + (v,) for t in out for v in range(0, b + 1, 2)]
    return out


@lru_cache(maxsize=None)
def _int_canon(d, M):
    """M is a tuple of rows, already in canonical form and with even margins"""
    nrows, ncols = len(M), len(M[0])
    last = tuple(row[ncols - 1] for row in M)
    if ncols == 1:
        return int_row(d, last)
    if sum(last) % 2:
        return Q(0)
    head = tuple(row[:ncols - 1] for row in M)
    s = Q(0)
    for k in _even_vectors(last):
        si = Q(0)
        rests = [multiexponents(ncols - 1, last[i] - k[i]) for i in range(nrows)]
        for Kv in _product(rests):
            colsums = tuple(sum(Kv[i][j] for i in range(nrows))
                            for j in range(ncols - 1))
            temp = Q(1)
            for i in range(nrows):
                temp *= Q(factorial(last[i] - k[i]),
                          _prod_fact(Kv[i]))
            temp *= int_row(d, colsums)
            temp *= int_mat(d, tuple(tuple(head[i][j] + Kv[i][j]
                                           for j in range(ncols - 1))
                                     for i in range(nrows)))
            si += temp
        if si:
            binom = 1
            for a, b in zip(last, k):
                binom *= comb(a, b)
            s += (binom * int_row(d, k)
                  * _bfac(sum(last) // 2, sum(k) // 2, Q(d, 2), Q(ncols - 1, 2))
                  * si)
    return s


def _prod_fact(v):
    r = 1
    for x in v:
        r *= factorial(x)
    return r


def _product(lists):
    if not lists:
        yield ()
        return
    head, tail = lists[0], lists[1:]
    for h in head:
        for t in _product(tail):
            yield (h,) + t


def _less_key(v):
    return (sum(v), max(v) if v else 0, v)


def canon(M):
    """drop zero rows and columns, make it at least as tall as wide, and put
    the columns in decreasing and the rows in increasing order; the integral is
    invariant under all of this, and it makes the memo table far smaller"""
    rows = [r for r in M if any(r)]
    if not rows:
        return ((0,),)
    ncols = len(rows[0])
    keep = [j for j in range(ncols) if any(r[j] for r in rows)]
    rows = [tuple(r[j] for j in keep) for r in rows]
    if len(rows) < len(rows[0]):
        rows = [tuple(rows[i][j] for i in range(len(rows)))
                for j in range(len(rows[0]))]
    cols = list(zip(*rows))
    cols.sort(key=_less_key, reverse=True)
    rows = [tuple(c[i] for c in cols) for i in range(len(rows))]
    rows.sort(key=_less_key)
    return tuple(rows)


def int_mat(d, M):
    """int over O(d) of prod gamma_{ij}^{M_ij}; M a tuple of row tuples"""
    for r in M:
        if sum(r) % 2:
            return Q(0)
    for j in range(len(M[0])):
        if sum(r[j] for r in M) % 2:
            return Q(0)
    return _int_canon(d, canon(M))


def selftest():
    d = 17
    assert int_mat(d, ((0,),)) == 1
    assert int_row(d, (1,)) == 0
    assert int_row(3, (2,)) == Q(1, 3)
    assert int_row(d, (2, 2, 2)) == Q(1, d * (d + 2) * (d + 4))
    assert int_mat(d, ((1, 1), (1, 1))) == Q(-1, (d - 1) * d * (d + 2))
    assert int_mat(d, ((2, 0), (2, 0), (0, 2))) == \
        Q(d + 3, (d - 1) * d * (d + 2) * (d + 4))
    assert int_mat(d, ((2, 0, 0), (0, 2, 0), (0, 0, 2))) == \
        Q(d * d + 3 * d - 2, (d - 2) * (d - 1) * d * (d + 2) * (d + 4))
    # a few values that can be checked independently: moments of a single
    # column, which is uniform on the sphere
    for n in (3, 4, 5):
        assert int_mat(n, ((2,), (0,), (0,))[:n]) == Q(1, n)
        assert int_mat(n, ((4,), (0,), (0,))[:n]) == Q(3, n * (n + 2))
        assert int_mat(n, ((2,), (2,), (0,))[:n]) == Q(1, n * (n + 2))
    # orthogonality of two columns: E[(v1.v2)^2] = 0 since they are orthogonal
    assert int_mat(4, ((1, 1), (1, 1), (0, 0), (0, 0))) + \
        2 * int_mat(4, ((1, 1), (0, 0), (1, 1), (0, 0))) * 0 == \
        int_mat(4, ((1, 1), (1, 1), (0, 0), (0, 0)))
    print('o4 selftest passed')


if __name__ == '__main__':
    selftest()
