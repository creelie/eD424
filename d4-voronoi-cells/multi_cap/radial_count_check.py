#!/usr/bin/env python3
"""
radial_count_check.py -- proves the certificate written by radial_count_sdp.py:
for every packing with at least M centres y, 2 <= |y| < sqrt6, the union of
the caps of prop:truncated has volume below 9 pi^2/8 - 8, so T(Y) > 8
(statement (C) of prop:reduction-G for every count from M on; with M = 31 this
is thm:count31).

Exact rational arithmetic, Bernstein bounds and Arb ball arithmetic; no step
rests on floating point alone.

1. The A_k (k >= 1) are positive semidefinite, by an exact LDL^T; A_0 is
   positive definite, t is set to z^T A_0^{-1} z rounded up, so that
   [[A_0, z], [z^T, t]] is positive semidefinite, again by exact LDL^T.
2. The pair inequality K(d, d', u) <= Pi(d/2, d'/2, u) for 2 <= d <= d' <= dmax
   (dmax = 2.4494898 > sqrt6) and -1 <= u <= a(d, d').  K is a polynomial; its
   tensor Bernstein coefficients on the box [2, dmax]^2 x [-1, a(dmax, dmax)]
   are computed exactly, rounded once, and subdivided at midpoints (de
   Casteljau) in floating point, with the rounding error bounded a priori
   (every midpoint step is an average, so the error grows by at most one unit
   in the last place of the largest coefficient per step).  On a box the
   largest coefficient bounds K.  Pi is decreasing in d and d' and increasing
   in u (lem:pair-closed), so on a box Pi >= Pi(d_hi/2, d'_hi/2, u_lo),
   evaluated from its closed form in Arb.  A box is closed when the bound
   for K is at most 0 or at most that value of Pi, dropped when it lies in
   d > d' or wholly beyond u = a(d, d') (exact rational tests), and halved
   otherwise.
3. The bracket S(d) + K(d, d, 1)/2 - z . p(d) is at most m on [2, dmax]: the
   polynomial part by Bernstein coefficients, S by its value at the left end
   (S decreases), in Arb.
4. m < 0 and M m + t/2 < 9 pi^2/8 - 8, in Arb.

Usage: python3 radial_count_check.py radial_certificates/radial_M.json
"""
import json
import math
import sys
import time
from fractions import Fraction as Fr

import numpy as np
from flint import arb, fmpq, ctx

ctx.prec = 128
Rsq = Fr(3, 2)


def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name, (': ' + detail) if detail else ''), flush=True)
    if not ok:
        sys.exit(1)


def A_(x):
    x = Fr(x)
    return arb(fmpq(x.numerator, x.denominator))


# ---------- exact linear algebra ----------

def ldl_psd(Mx):
    """exact LDL^T; True if Mx is positive semidefinite (zero pivots need zero rows)."""
    n = len(Mx)
    A = [row[:] for row in Mx]
    piv = []
    for k in range(n):
        p = A[k][k]
        if p < 0:
            return False, piv
        if p == 0:
            if any(A[i][k] != 0 for i in range(k + 1, n)):
                return False, piv
            piv.append(p)
            continue
        piv.append(p)
        for i in range(k + 1, n):
            f = A[i][k] / p
            if f:
                for j in range(k + 1, n):
                    A[i][j] -= f * A[k][j]
    return True, piv


def solve_exact(Mx, b):
    n = len(Mx)
    A = [row[:] + [b[i]] for i, row in enumerate(Mx)]
    for k in range(n):
        p = next(i for i in range(k, n) if A[i][k] != 0)
        A[k], A[p] = A[p], A[k]
        for i in range(n):
            if i != k and A[i][k] != 0:
                f = A[i][k] / A[k][k]
                A[i] = [a - f * c for a, c in zip(A[i], A[k])]
    return [A[i][n] / A[i][i] for i in range(n)]


# ---------- exact polynomials in s on [0, 1] ----------

def pmul(p, q):
    out = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return out


def padd(p, q, cp=1, cq=1):
    n = max(len(p), len(q))
    p = p + [Fr(0)] * (n - len(p)); q = q + [Fr(0)] * (n - len(q))
    return [cp * a + cq * b for a, b in zip(p, q)]


def cheb_T(alpha, beta, n):
    """T_0..T_n of x = alpha + beta s, as power series in s."""
    x = [alpha, beta]
    T = [[Fr(1)], x]
    for _ in range(2, n + 1):
        T.append(padd(pmul([2 * alpha, 2 * beta], T[-1]), T[-2], 1, -1))
    return T[:n + 1]


def cheb_U(alpha, beta, n):
    """U_0..U_n of u = alpha + beta s, as power series in s."""
    U = [[Fr(1)], [2 * alpha, 2 * beta]]
    for _ in range(2, n + 1):
        U.append(padd(pmul([2 * alpha, 2 * beta], U[-1]), U[-2], 1, -1))
    return U[:n + 1]


def bernstein(p, n):
    c = p + [Fr(0)] * (n + 1 - len(p))
    return [sum(Fr(math.comb(i, j), math.comb(n, j)) * c[j] for j in range(i + 1)) for i in range(n + 1)]


# ---------- de Casteljau at the midpoint, floating point ----------

def split(B, axis):
    n = B.shape[axis] - 1
    B = np.moveaxis(B, axis, 0)
    left, right = [B[0]], [B[n]]
    cur = B
    for _ in range(n):
        cur = (cur[:-1] + cur[1:]) * 0.5
        left.append(cur[0]); right.append(cur[-1])
    return np.moveaxis(np.array(left), 0, axis), np.moveaxis(np.array(right[::-1]), 0, axis)


# ---------- the cap S and the pair term Pi in Arb ----------

R = A_(Rsq).sqrt()
PI = arb.pi()


def S_arb(d):
    h = A_(d) / 2
    if Fr(d) ** 2 >= 6:
        return arb(0)
    anti = h / 8 * (5 * A_(Rsq) - 2 * h * h) * (A_(Rsq) - h * h).sqrt() + 3 * A_(Rsq) ** 2 / 8 * (h / R).asin()
    return 4 * PI / 3 * (3 * A_(Rsq) ** 2 / 8 * PI / 2 - anti)


def G(h, x):
    tx = x.tan()
    return A_(Rsq) ** 2 * x - 2 * A_(Rsq) * h * h * tx + h ** 4 * (tx + tx ** 3 / 3)


def Pi_lower(d1, d2, u):
    """a rigorous lower bound (float) for Pi(d1/2, d2/2, u), d1, d2, u rational."""
    if Fr(d1) ** 2 >= 6 or Fr(d2) ** 2 >= 6 or u <= -1:
        return 0.0
    h1, h2, ua = A_(d1) / 2, A_(d2) / 2, A_(u)
    g = ua.acos()
    a1, a2 = (h1 / R).acos(), (h2 / R).acos()
    L = (-a1).max(g - a2)
    U = a1.min(g + a2)
    if not (L < U):
        return 0.0
    pc = arb.atan2(h2 - h1 * g.cos(), h1 * g.sin())
    pc = pc.max(L).min(U)
    val = PI / 4 * (G(h2, pc - g) - G(h2, L - g) + G(h1, U) - G(h1, pc))
    lo = float(val.lower())
    if lo > 0 and arb(lo) > val.lower():   # guard the conversion
        lo = float(np.nextafter(lo, -np.inf))
    return max(lo, 0.0)


def main():
    t0 = time.time()
    c = json.load(open(sys.argv[1]))
    M, D, r = c['M'], c['D'], c['r']
    c1, c2, dmax = Fr(c['c1']), Fr(c['c2']), Fr(c['dmax'])
    A = [[[Fr(x) for x in row] for row in a] for a in c['A']]
    z = [Fr(x) for x in c['z']]
    m = Fr(c['m'])
    print('certificate: M = %d, degree %d in the angle, %d in the distance; dmax = %s' % (M, D, r, dmax))
    check('dmax exceeds sqrt 6', dmax ** 2 > 6, '%s^2 = %.10f' % (dmax, float(dmax ** 2)))

    # 1. positivity
    check('A_0, ..., A_%d symmetric' % D, all(a[i][j] == a[j][i] for a in A for i in range(r + 1) for j in range(r + 1)))
    ok = True
    for k in range(1, D + 1):
        good, piv = ldl_psd(A[k])
        ok &= good
    check('A_1, ..., A_%d positive semidefinite (exact LDL^T)' % D, ok)
    good, piv = ldl_psd(A[0])
    check('A_0 positive definite (exact LDL^T)', good and all(p > 0 for p in piv), 'least pivot %.3e' % float(min(piv)))
    w = solve_exact(A[0], z)
    tq = sum(a * b for a, b in zip(z, w))
    t = Fr(math.ceil(tq * 2 ** 48), 2 ** 48)
    Zm = [row[:] + [z[i]] for i, row in enumerate(A[0])] + [z + [t]]
    good, piv = ldl_psd(Zm)
    check('[[A_0, z], [z^T, t]] positive semidefinite (exact LDL^T)', good, 't = %.12f, last pivot %.3e' % (float(t), float(piv[-1])))

    # 2. exact Bernstein coefficients of K on the root box
    alpha, beta = (4 - c1) / c2, 2 * (dmax - 2) / c2          # x = alpha + beta s, d = 2 + (dmax - 2) s
    umax = (2 * dmax ** 2 - 4) / (2 * dmax ** 2)
    Ts = [bernstein(p, r) for p in cheb_T(alpha, beta, r)]
    Us = [bernstein([x / (k + 1) for x in p], D) for k, p in enumerate(cheb_U(Fr(-1), umax + 1, D))]
    B0 = [[[Fr(0)] * (D + 1) for _ in range(r + 1)] for _ in range(r + 1)]
    for k in range(D + 1):
        for a in range(r + 1):
            for b in range(r + 1):
                if A[k][a][b] == 0:
                    continue
                for i in range(r + 1):
                    ci = A[k][a][b] * Ts[a][i]
                    if ci == 0:
                        continue
                    for j in range(r + 1):
                        cij = ci * Ts[b][j]
                        if cij == 0:
                            continue
                        row = B0[i][j]
                        for l in range(D + 1):
                            row[l] += cij * Us[k][l]
    Bf = np.array([[[float(x) for x in row] for row in plane] for plane in B0])
    Cmax = max(abs(x) for plane in B0 for row in plane for x in row)
    u53 = 2.0 ** -53
    depth_max = 80
    err = float(Cmax) * u53 * (1 + depth_max * max(r, D)) * 1.01
    print('  Bernstein coefficients of K on the root box: largest |coefficient| %.4f, error allowance %.2e' % (float(Cmax), err))

    # branch and bound; boxes as integer cells (i, level) per axis in s-coordinates
    def dval(i, n):
        return 2 + (dmax - 2) * Fr(i, 2 ** n)

    def uval(i, n):
        return -1 + (umax + 1) * Fr(i, 2 ** n)

    def amaxq(d1, d2):
        return (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)

    from truncated_search import pair as pair_float
    stack = [((0, 0), (0, 0), (0, 0), Bf, 0)]
    closed = dropped = arbcalls = 0
    memo = {}
    worst = -1.0
    while stack:
        (i1, n1), (i2, n2), (i3, n3), B, depth = stack.pop()
        dlo, dhi = dval(i1, n1), dval(i1 + 1, n1)
        elo, ehi = dval(i2, n2), dval(i2 + 1, n2)
        ulo = uval(i3, n3)
        if dlo > ehi or ulo > amaxq(dhi, ehi):
            dropped += 1
            continue
        kub = float(np.nextafter(float(B.max()) + err, np.inf))
        if kub <= 0:
            closed += 1
            continue
        pf = float(pair_float(float(dhi) / 2, float(ehi) / 2, float(ulo)))
        if kub <= pf - 1e-9:
            key = (dhi, ehi, ulo)
            if key not in memo:
                memo[key] = Pi_lower(dhi, ehi, ulo); arbcalls += 1
            if kub <= memo[key]:
                closed += 1
                worst = max(worst, kub - memo[key])
                continue
        if depth >= depth_max:
            check('pair inequality', False, 'depth limit at d in [%s, %s], d\' in [%s, %s], u from %s' % (dlo, dhi, elo, ehi, ulo))
        w = [Fr(1, 2 ** n1), Fr(1, 2 ** n2), Fr(1, 2 ** n3) * (umax + 1) / (dmax - 2) / 3]
        ax = max(range(3), key=lambda j: w[j])
        L, Rt = split(B, ax)
        if ax == 0:
            stack += [((2 * i1, n1 + 1), (i2, n2), (i3, n3), L, depth + 1), ((2 * i1 + 1, n1 + 1), (i2, n2), (i3, n3), Rt, depth + 1)]
        elif ax == 1:
            stack += [((i1, n1), (2 * i2, n2 + 1), (i3, n3), L, depth + 1), ((i1, n1), (2 * i2 + 1, n2 + 1), (i3, n3), Rt, depth + 1)]
        else:
            stack += [((i1, n1), (i2, n2), (2 * i3, n3 + 1), L, depth + 1), ((i1, n1), (i2, n2), (2 * i3 + 1, n3 + 1), Rt, depth + 1)]
    check('pair inequality K <= Pi on the admissible domain', True,
          '%d boxes closed, %d dropped (d > d\' or beyond u = a(d, d\')), %d values of Pi in Arb [%.0f s]'
          % (closed, dropped, arbcalls, time.time() - t0))

    # 3. the bracket S(d) + K(d, d, 1)/2 - z . p(d) <= m
    Tq = cheb_T(alpha, beta, r)
    Asum = [[sum(A[k][a][b] for k in range(D + 1)) for b in range(r + 1)] for a in range(r + 1)]
    q = [Fr(0)]
    for a in range(r + 1):
        for b in range(r + 1):
            if Asum[a][b]:
                q = padd(q, pmul(Tq[a], Tq[b]), 1, Asum[a][b] / 2)
        q = padd(q, Tq[a], 1, -z[a])
    qb = np.array([float(x) for x in bernstein(q, 2 * r)])
    qmax = max(abs(x) for x in bernstein(q, 2 * r))
    err1 = float(qmax) * u53 * (1 + 60 * 2 * r) * 1.01
    stack = [(0, 0, qb, 0)]
    nb = 0
    gmax = -1e9
    while stack:
        i, n, B, depth = stack.pop()
        dlo = dval(i, n)
        bound = S_arb(dlo) + arb(float(np.nextafter(float(B.max()) + err1, np.inf)))
        if bound < A_(m):
            nb += 1
            gmax = max(gmax, float(bound.upper()))
            continue
        if depth >= 60:
            check('bracket at most m', False, 'at d = %s' % dlo)
        L, Rt = split(B, 0)
        stack += [(2 * i, n + 1, L, depth + 1), (2 * i + 1, n + 1, Rt, depth + 1)]
    check('S(d) + K(d, d, 1)/2 - z . p(d) <= m on [2, dmax]', True, 'm = %.6f, largest bound %.6f, %d intervals' % (float(m), gmax, nb))

    # 4. the conclusion
    check('m < 0, so the bound falls with the count', m < 0)
    total = M * A_(m) + A_(t) / 2
    target = 9 * PI ** 2 / 8 - 8
    check('M m + t/2 < 9 pi^2/8 - 8', total < target, '%s < %s' % (total.str(8), target.str(8)))
    print('PASS: every packing with at least %d centres within sqrt 6 of a centre has T > 8 there [%.0f s]' % (M, time.time() - t0))


if __name__ == '__main__':
    main()
