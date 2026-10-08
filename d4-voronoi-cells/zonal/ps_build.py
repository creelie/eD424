"""
ps_build.py -- prepare the three factors of the step 3 integrand for psker.

For every signature lambda of O(4) with |lambda| <= 14 and every admissible pair
k1 >= k2 of indices, the integrand of

    P(S)_{k1,k2} = int_{O(4)} rho_{0,k1}(omega gamma eps)
                              rho_{0,k2}(omega gamma S) dgamma

splits into three polynomials that share no variables of gamma between the
second and the third:

    rho_A = det(A)^l2 A11^(m-k1) A12^k1        (columns 0 and 1 of gamma)
    U     = B11^(l2+m-k2) B12^k2               (rows 0 and 2)
    V     = B22^l2                             (rows 1 and 3)

where A = omega gamma eps and B = omega gamma S, with
omega gamma = [gamma_row0 + i gamma_row2 ; gamma_row1 + i gamma_row3].  Every
occurrence of a variable from row 2 or row 3 carries one factor i, so the real
part of a factor is the part of even degree in those rows, with the sign
(-1)^(half that degree); those signs are folded into the coefficients here so
that the kernel sees plain integers.

The three lists go into a binary file that psker reads.
"""
import os
import struct
import sys

from flint import fmpz_mpoly, fmpz_mpoly_ctx, Ordering

NG, NS = 16, 7
NV = NG + NS
NAMES = ['g%d%d' % (i, j) for i in range(4) for j in range(4)] + \
        ['s%d' % i for i in range(NS)]
CTX = fmpz_mpoly_ctx.get(NAMES, Ordering.lex)
GEN = [CTX.gen(i) for i in range(NV)]
ZERO = CTX.from_dict({})


def g(i, j):
    return GEN[4 * i + j]


def s(k):
    return GEN[NG + k]


# S is 4 x 2, lower half upper triangular: S[.,0] = s0,s1,s2,0; S[.,1] = s3..s6
SCOL = [[s(0), s(1), s(2), ZERO], [s(3), s(4), s(5), s(6)]]


def og(a, c):
    """entry (a, c) of omega gamma, with the factor i left implicit"""
    return g(a, c) + g(a + 2, c)


A11, A12 = og(0, 0), og(0, 1)
A21, A22 = og(1, 0), og(1, 1)
DETA = A11 * A22 - A12 * A21
B11 = sum((og(0, c) * SCOL[0][c] for c in range(4)), ZERO)
B12 = sum((og(0, c) * SCOL[1][c] for c in range(4)), ZERO)       # row 0
B22 = sum((og(1, c) * SCOL[1][c] for c in range(4)), ZERO)       # row 1


def irreps(d1=14):
    out = []
    for l1 in range(d1 + 1):
        for l2 in range(min(l1, d1 - l1) + 1):
            if l1 != l2 or l1 % 2 == 0:
                out.append((l1, l2))
    return out


def weven(lam):
    l1, l2 = lam
    return [k for k in range(l1 - l2 + 1) if (l2 + k) % 2 == 0]


def rowdeg(ev, i):
    return ev[4 * i] + ev[4 * i + 1] + ev[4 * i + 2] + ev[4 * i + 3]


def a_terms(lam, k1):
    """rho_A, real part, signs folded in"""
    l1, l2 = lam
    m = l1 - l2
    p = DETA ** l2 * A11 ** (m - k1) * A12 ** k1
    out = []
    for ev, c in p.terms():
        ev = tuple(int(x) for x in ev)
        d = rowdeg(ev, 2) + rowdeg(ev, 3)
        if d % 2:
            continue
        sg = -1 if (d // 2) % 2 else 1
        if rowdeg(ev, 2) % 2:
            sg = -sg
        out.append((ev[:NG], sg * int(c)))
    return out


def u_terms(lam, k2):
    l1, l2 = lam
    m = l1 - l2
    p = B11 ** (l2 + m - k2) * B12 ** k2
    out = []
    for ev, c in p.terms():
        ev = tuple(int(x) for x in ev)
        su = rowdeg(ev, 2)
        sg = -1 if (su // 2) % 2 else 1
        out.append((ev[:NG], ev[NG:], sg * int(c)))
    return out


def v_terms(lam):
    l1, l2 = lam
    p = B22 ** l2
    out = []
    for ev, c in p.terms():
        ev = tuple(int(x) for x in ev)
        dv = rowdeg(ev, 3)
        sg = -1 if (dv // 2) % 2 else 1
        out.append((ev[:NG], ev[NG:], sg * int(c)))
    return out


def main(path, limit=None):
    jobs = []
    for lam in irreps():
        ws = weven(lam)
        if not ws:
            continue
        if limit is not None and sum(lam) > limit:
            continue
        for i1, k1 in enumerate(ws):
            for k2 in ws[:i1 + 1]:
                jobs.append((lam, k1, k2))
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', len(jobs)))
        maxc = 0
        for lam, k1, k2 in jobs:
            A = a_terms(lam, k1)
            U = u_terms(lam, k2)
            V = v_terms(lam)
            maxc = max([maxc] + [abs(c) for _, c in A]
                       + [abs(c) for _, _, c in U] + [abs(c) for _, _, c in V])
            f.write(struct.pack('<7i', lam[0], lam[1], k1, k2,
                                len(A), len(U), len(V)))
            for ev, c in A:
                f.write(bytes(ev) + struct.pack('<q', c))
            for ev, sv, c in U:
                f.write(bytes(ev) + bytes(sv) + struct.pack('<q', c))
            for ev, sv, c in V:
                f.write(bytes(ev) + bytes(sv) + struct.pack('<q', c))
    print('%d entries, %.1f MB, largest coefficient %d'
          % (len(jobs), os.path.getsize(path) / 1e6, maxc))


if __name__ == '__main__':
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(sys.argv[1], lim)
