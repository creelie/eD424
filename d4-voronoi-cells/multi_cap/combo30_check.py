#!/usr/bin/env python3
"""
combo30_check.py -- rigorous verification of a certificate written by
gap_closure/C30/combo30p.py for the fifth case of thm:count30: thirty
centres within sqrt6 of c, twenty-four of them within 2.25 (22 within 2.05,
23 within 2.15) and six in [2.4, sqrt6).  If it passes, every such packing set
has union of caps U(Y) < 9 pi^2/8 - 8, so T(Y) > 8 there.  It passes on
radial_certificates/combo30_d8.npz at d3 = 8 with the margins 2e-5, 5e-6 and
2e-6: the largest bound is 3.087203, at the bin counts (22, 1, 1, 0, 6)
(runs/combo30_check.log, 6891 s).

The certificate combines the distance-labelled two-point kernel of thm:count31,
    K(y, y') = sum_k U_k(<y^,y'^>)/(k+1) p(|y|)^T A_k p(|y'|),
with a three-point kernel on the directions whose points carry the type of
their distance, A = [2, 2.05], B = (2.05, 2.25], F = [2.4, sqrt6) (typed3pt.py):
for every finite set of typed points its sum over all ordered triples is >= 0,
and split by coincidence pattern that sum is
    sum_y p_type(y) + sum_{pairs} PAIR3_st(u) + sum_{triples} TRIPLE3_str(u12, u13, u23).
With U(Y) = sum S(|y|) - sum Pi, the chain of thm:count31 gives
    U(Y) <= sum_y [f(|y|) + p_type(y)] + t/2 + sum_pairs [K - Pi + PAIR3] + sum_triples TRIPLE3,
    f(d) = S(d) + K(d, d, 1)/2 - z . p(d).
So if (all exact)
  (a) A_1..A_D psd, [[A_0, z], [z^T, t]] psd, and the three-point blocks psd;
  (b) K + PAIR3_st - c2_st <= Pi on the admissible pairs of types (s, t)
      (distances in the type ranges, u <= a(d, d'));
  (c) f <= m_b on each bin (A, B1 = (2.05, 2.15], B2 = (2.15, 2.25], F);
  (d) TRIPLE3_str <= c3_str on the admissible typed triples (each inner product
      at most a(.,.) at the upper ends of the two types, Gram determinant >= 0);
then U(Y) <= sum_b n_b (m_b + p_type(b)) + t/2 + sum N_st c2_st + sum N_str c3_str
for the bin counts n of Y, and the check is that the largest of these over the
five count vectors of the case is below 9 pi^2/8 - 8, in Arb.

The floats of the file are made exact once (dyadic rationals; psd blocks
clipped at their least eigenvalues and shifted by 2^-40), and the thresholds
c2, c3, m are the floating-point maxima on dense samples plus a margin, then
proved by branch and bound: (b) with tensor Bernstein coefficients and Pi in
ball arithmetic at the box corner (lem:pair-closed), as radial_count_check.py;
(c) as radial_case_check.py; (d) by second-order Taylor forms in interval
arithmetic, as typed_cardinality_check.py, on the domain reduced by the
symmetry of the repeated types.

Usage: python3 combo30_check.py cert.npz d3 [margin2 margin3 marginm]
"""
import itertools
import math
import os
import sys
import time
from fractions import Fraction as Fr
from math import comb

import numpy as np
from flint import arb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radial_count_check import A_, ldl_psd, solve_exact, cheb_T, cheb_U, bernstein, S_arb, Pi_lower  # noqa: E402
from radial_count_check import pmul as lmul, padd as ladd  # noqa: E402
from radial_count_sdp import C1, C2  # noqa: E402
from radial_case_check import bracket_check, arb_up  # noqa: E402
from certificate_check import (padd, pscale, pmul, univariate, gegenbauer_S3, legendre_coeffs,  # noqa: E402
                               chebyshev, phi_poly, ldl_positive, Poly, derivative, up)
from truncated_search import pair as pair_float, S as S_float  # noqa: E402

D2, R2 = 12, 4
TAYLOR_MONO = os.environ.get('TRIPLE_TAYLOR', '') == 'mono'   # see triple_box_check
DMAX = Fr(24494898, 10000000)                      # above sqrt 6
TYPES = ['A', 'B', 'F']
TRANGE = {'A': (Fr(2), Fr(205, 100)), 'B': (Fr(205, 100), Fr(225, 100)), 'F': (Fr(24, 10), DMAX)}
EDGES = [Fr(2), Fr(205, 100), Fr(215, 100), Fr(225, 100), Fr(24, 10), DMAX]
BINTYPE = ['A', 'B', 'B', None, 'F']               # the bin (2.25, 2.4) holds no centre
COUNTS = [(22, 1, 1, 0, 6), (22, 2, 0, 0, 6), (23, 0, 1, 0, 6), (23, 1, 0, 0, 6), (24, 0, 0, 0, 6)]


def amaxq(d1, d2):
    return (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)


def dyad(x, bits=48):
    return Fr(round(float(x) * 2 ** bits), 2 ** bits)


def above(x):
    """the least double >= x, as a rational"""
    y = float(x)
    if Fr(y) < Fr(x):
        y = float(np.nextafter(y, np.inf))
    return Fr(y)


def tcounts(nb):
    out = {'A': 0, 'B': 0, 'F': 0}
    for ty, n in zip(BINTYPE, nb):
        if ty:
            out[ty] += n
    return out


def Npair(tc, s, t):
    return comb(tc[s], 2) if s == t else tc[s] * tc[t]


def Ntriple(tc, combo):
    out = 1
    for s in set(combo):
        out *= comb(tc[s], combo.count(s))
    return out


# ---------------------------------------------------------------- the certificate, made exact

def psd_exact(M, shift=Fr(1, 2 ** 40)):
    """a symmetric float matrix -> exact dyadic matrix, eigenvalues clipped at 0 and shifted."""
    M = np.array(M, float); M = (M + M.T) / 2
    lam, V = np.linalg.eigh(M)
    M = (V * np.maximum(lam, 0.0)) @ V.T; M = (M + M.T) / 2
    n = len(M)
    out = [[dyad(M[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            out[j][i] = out[i][j]
        out[i][i] += shift
    return out


def blocks_from_x3(x3, d3):
    """the three-point blocks of typed3pt.Layout(3, d3), in its order, from the stored vector
    (each block stored as its diagonal, then its strict upper triangle row by row)."""
    m = len(TYPES)
    names = [('f', k) for k in range(1, d3 + 1)] + [('F', P, k) for P in range(m) for k in range(d3 + 1)]
    out, off = {}, 0
    for name in names:
        s = m if name[0] == 'f' else m * (d3 - name[2] + 1)
        n = s * (s + 1) // 2
        v = x3[off:off + n]; off += n
        M = np.zeros((s, s)); M[np.arange(s), np.arange(s)] = v[:s]
        iu = np.triu_indices(s, 1); M[iu] = v[s:]; M[(iu[1], iu[0])] = v[s:]
        out[name] = M
    assert off == len(x3), (off, len(x3))
    return out


# ---------------------------------------------------------------- the three-point polynomials

def point_term(Bk, d3, s):
    """p_s = sum_k f_k[s][s] + 1^T F^s_0[s-block] 1 (all three points equal)."""
    v = sum(Bk[('f', k)][s][s] for k in range(1, d3 + 1))
    mk = d3 + 1
    blk = Bk[('F', s, 0)]
    v += sum(blk[s * mk + i][s * mk + j] for i in range(mk) for j in range(mk))
    return v


def pair3_poly(Bk, d3, s, t):
    """PAIR3_st(u) as an exact univariate coefficient list (power basis in u)."""
    G = gegenbauer_S3(d3); TC = chebyshev(d3)
    p = [Fr(0)]
    for k in range(1, d3 + 1):
        p = ladd(p, G[k], 1, 2 * Bk[('f', k)][s][t])
    for (P, Q) in ((s, t), (t, s)):
        mk = d3 + 1
        blk = Bk[('F', P, 0)]
        for i in range(mk):                                  # pattern 2: x = pole, y = the other
            for j in range(mk):
                c = blk[P * mk + i][Q * mk + j]
                if c:
                    p = ladd(p, TC[j], 1, 2 * c)
        for k in range(d3 + 1):                              # pattern 3: x = y = the other
            mk = d3 - k + 1
            blk = Bk[('F', P, k)]
            w = [Fr(1)]
            for _ in range(k):
                w = lmul(w, [Fr(1), Fr(0), Fr(-1)])
            q = [Fr(0)]
            for i in range(mk):
                for j in range(mk):
                    c = blk[Q * mk + i][Q * mk + j]
                    if c:
                        q = ladd(q, lmul(TC[i], TC[j]), 1, c)
            p = ladd(p, lmul(w, q))
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def triple3_poly(Bk, d3, types):
    """TRIPLE3(x, y, z) with x = u12, y = u13, z = u23, as an exact dict in 3 variables."""
    LC = legendre_coeffs(d3); TC = chebyshev(d3)
    s1, s2, s3 = (TYPES.index(x) for x in types)
    out = {}
    # (pole, (type a, variable of <pole, a>), (type b, variable of <pole, b>), variable of <a, b>)
    for (P, (Qa, va), (Qb, vb), vt) in ((s1, (s2, 0), (s3, 1), 2),
                                         (s2, (s1, 0), (s3, 2), 1),
                                         (s3, (s1, 1), (s2, 2), 0)):
        for k in range(d3 + 1):
            mk = d3 - k + 1
            blk = Bk[('F', P, k)]
            core = {}
            for i in range(mk):
                for j in range(mk):
                    c = blk[Qa * mk + i][Qb * mk + j]
                    if c:
                        core = padd(core, pscale(pmul(univariate(TC[i], 0), univariate(TC[j], 1)), c))
            if not core:
                continue
            y = pmul(core, phi_poly(k, LC))                 # in (u, v, t) = (<pole,a>, <pole,b>, <a,b>)
            perm = (va, vb, vt)
            ren = {}
            for e, c in y.items():
                ee = [0, 0, 0]
                for src in range(3):
                    ee[perm[src]] += e[src]
                ren[tuple(ee)] = ren.get(tuple(ee), 0) + c
            out = padd(out, pscale(ren, 2))
    return out


def peval(p, x, y, z):
    """float evaluation of a 3-variable dict polynomial on arrays."""
    out = np.zeros_like(x)
    for (a, b, c), co in p.items():
        out = out + float(co) * x ** a * y ** b * z ** c
    return out


# ---------------------------------------------------------------- (b) the pairs

def pair_box_check(A, Pu, c2, box, symmetric, depth_max=70, batch=20000):
    """K(d, d', u) + Pu(u) - c2 <= Pi(d/2, d'/2, u) for d in box[0], d' in box[1],
    -1 <= u <= a(d, d') (and d <= d' if symmetric), by tensor Bernstein bounds."""
    (alo, ahi), (blo, bhi) = box
    umax = amaxq(ahi, bhi)
    Du = max(D2, len(Pu) - 1)
    Ta = [bernstein(p, R2) for p in cheb_T((2 * alo - C1) / C2, 2 * (ahi - alo) / C2, R2)]
    Tb = [bernstein(p, R2) for p in cheb_T((2 * blo - C1) / C2, 2 * (bhi - blo) / C2, R2)]
    Us = [bernstein([x / (k + 1) for x in p], Du) for k, p in enumerate(cheb_U(Fr(-1), umax + 1, D2))]
    # Pu(u) - c2 in s, u = -1 + (umax + 1) s
    q = [Fr(0)]; pw = [Fr(1)]
    for c in Pu:
        q = ladd(q, pw, 1, c); pw = lmul(pw, [Fr(-1), umax + 1])
    q[0] -= c2
    Bq = bernstein(q, Du)
    B0 = [[[Bq[l] for l in range(Du + 1)] for _ in range(R2 + 1)] for _ in range(R2 + 1)]
    for k in range(D2 + 1):
        for a in range(R2 + 1):
            for b in range(R2 + 1):
                if A[k][a][b] == 0:
                    continue
                for i in range(R2 + 1):
                    ci = A[k][a][b] * Ta[a][i]
                    if ci == 0:
                        continue
                    for j in range(R2 + 1):
                        cij = ci * Tb[b][j]
                        if cij == 0:
                            continue
                        row = B0[i][j]
                        for l in range(Du + 1):
                            row[l] += cij * Us[k][l]
    Cmax = max(abs(x) for plane in B0 for row in plane for x in row)
    Bf = np.array([[[float(x) for x in row] for row in plane] for plane in B0])
    err = float(Cmax) * 2.0 ** -53 * (1 + depth_max * max(R2, Du)) * 1.01
    fa, wa, fb, wb, fu = float(alo), float(ahi - alo), float(blo), float(bhi - blo), float(umax + 1)
    I = np.zeros((1, 3), dtype=np.int64); L = np.zeros((1, 3), dtype=np.int64)
    pending = [(I, L, Bf[None])]
    closed = dropped = arbcalls = 0
    memo = {}

    def corner(k3):
        dh = alo + (ahi - alo) * Fr(k3[0] + 1, 2 ** k3[1])
        eh = blo + (bhi - blo) * Fr(k3[2] + 1, 2 ** k3[3])
        ul = -1 + (umax + 1) * Fr(k3[4], 2 ** k3[5])
        return dh, eh, ul

    while pending:
        I, L, Bs = pending.pop()
        if len(I) == 0:
            continue
        if len(I) > batch:
            pending.append((I[batch:], L[batch:], Bs[batch:]))
            I, L, Bs = I[:batch], L[:batch], Bs[:batch]
        if L.max() > depth_max:
            return False, 'depth limit reached'
        sc = 2.0 ** (-L)
        dlo = fa + wa * I[:, 0] * sc[:, 0]; dhi = fa + wa * (I[:, 0] + 1) * sc[:, 0]
        elo = fb + wb * I[:, 1] * sc[:, 1]; ehi = fb + wb * (I[:, 1] + 1) * sc[:, 1]
        ulo = -1.0 + fu * I[:, 2] * sc[:, 2]
        drop = np.zeros(len(I), dtype=bool)
        if symmetric:                                   # the same range on both axes: drop d > d'
            m = np.maximum(L[:, 0], L[:, 1])
            drop = np.array([int(a) * 2 ** (int(mm) - int(la)) > (int(b) + 1) * 2 ** (int(mm) - int(lb))
                             for a, b, la, lb, mm in zip(I[:, 0], I[:, 1], L[:, 0], L[:, 1], m)], dtype=bool)
        am = (dhi * dhi + ehi * ehi - 4) / (2 * dhi * ehi)
        beyond = ulo > am + 1e-12
        unsure = (~beyond) & (ulo > am - 1e-12) & ~drop
        for k in np.nonzero(unsure)[0]:
            dh, eh, ul = corner((int(I[k, 0]), int(L[k, 0]), int(I[k, 1]), int(L[k, 1]), int(I[k, 2]), int(L[k, 2])))
            if ul > amaxq(dh, eh):
                beyond[k] = True
        drop |= beyond
        dropped += int(drop.sum())
        keep = ~drop
        I, L, Bs, dhi, ehi, ulo = I[keep], L[keep], Bs[keep], dhi[keep], ehi[keep], ulo[keep]
        if len(I) == 0:
            continue
        kub = np.nextafter(Bs.reshape(len(Bs), -1).max(1) + err, np.inf)
        ok = kub <= 0
        cand = (~ok) & (kub <= pair_float(dhi / 2, ehi / 2, ulo) - 1e-9)
        for k in np.nonzero(cand)[0]:
            key = (int(I[k, 0]), int(L[k, 0]), int(I[k, 1]), int(L[k, 1]), int(I[k, 2]), int(L[k, 2]))
            if key not in memo:
                memo[key] = Pi_lower(*corner(key))
                arbcalls += 1
            if kub[k] <= memo[key]:
                ok[k] = True
        closed += int(ok.sum())
        I, L, Bs = I[~ok], L[~ok], Bs[~ok]
        if len(I) == 0:
            continue
        w = np.stack([wa * 2.0 ** (-L[:, 0]), wb * 2.0 ** (-L[:, 1]), fu * 2.0 ** (-L[:, 2]) / 3], 1)
        ax = np.argmax(w, 1)
        for a in range(3):
            sel = ax == a
            if not sel.any():
                continue
            Ia, La, Ba = I[sel], L[sel], Bs[sel]
            X = np.moveaxis(Ba, a + 1, 0)
            n = X.shape[0] - 1
            left, right = [X[0]], [X[n]]
            cur = X
            for _ in range(n):
                cur = (cur[:-1] + cur[1:]) * 0.5
                left.append(cur[0]); right.append(cur[-1])
            Bl = np.moveaxis(np.array(left), 0, a + 1); Br = np.moveaxis(np.array(right[::-1]), 0, a + 1)
            Il, Ir = Ia.copy(), Ia.copy()
            Il[:, a] = 2 * Ia[:, a]; Ir[:, a] = 2 * Ia[:, a] + 1
            Ln = La.copy(); Ln[:, a] += 1
            pending.append((Il, Ln, Bl)); pending.append((Ir, Ln.copy(), Br))
    return True, '%d boxes closed, %d set aside, Pi in Arb at %d corners' % (closed, dropped, arbcalls)


# ---------------------------------------------------------------- (d) the triples

def triple_box_check(P, tops, sym, cmax, label='', wmin=1e-6, batch=150000):
    """P <= cmax on {-1 <= x_i <= tops[i], Gram det >= 0} with x0 <= x1 if sym >= 1 and
    x1 <= x2 if sym == 2, by second-order Taylor forms (typed_cardinality_check.verify_3d).
    With TRIPLE_TAYLOR=mono in the environment two sharper forms are used: on a box where
    dP/dx_v has one sign the form is taken on the face where P is larger (it bounds P on the
    whole box), and a diagonal second derivative enters through its upper end cut at 0, since
    (x_v - c_v)^2 >= 0.  Both stay valid bounds; they matter where P comes close to cmax."""
    Pp = Poly(P); Dv = [Poly(derivative(P, v)) for v in range(3)]
    H = {(v, w): Poly(derivative(derivative(P, v), w)) for v in range(3) for w in range(v, 3)}
    det = Poly({(0, 0, 0): Fr(1), (1, 1, 1): Fr(2), (2, 0, 0): Fr(-1), (0, 2, 0): Fr(-1), (0, 0, 2): Fr(-1)})
    lo = np.array([[-1.0], [-1.0], [-1.0]]); hi = np.array([[tops[0]], [tops[1]], [tops[2]]])
    n_done = n_out = 0; worst = -np.inf; level = 0; t0 = time.time()
    while lo.shape[1] > 0:
        n = lo.shape[1]; keep_lo = []; keep_hi = []
        for s in range(0, n, batch):
            L, Hh = lo[:, s:s + batch], hi[:, s:s + batch]
            alive = np.ones(L.shape[1], dtype=bool)
            if sym >= 1:
                alive &= L[0] <= Hh[1]
            if sym == 2:
                alive &= L[1] <= Hh[2]
            _, dhi = det.eval((L, Hh))
            alive &= dhi >= 0
            n_out += int(np.sum(~alive))
            Lt, Ht = L, Hh
            if TAYLOR_MONO:
                # where dP/dx_v > 0 (< 0) on the whole box, P is largest on the face x_v = hi (lo):
                # the bound below is taken on that face, which contains the largest value on the box
                Lt, Ht = L.copy(), Hh.copy()
                for v in range(3):
                    gblo, gbhi = Dv[v].eval((L, Hh))
                    Lt[v] = np.where(gblo > 0, Hh[v], Lt[v]); Ht[v] = np.where(gbhi < 0, L[v], Ht[v])
            c = (Lt + Ht) / 2; r = np.maximum(up(Ht - c), up(c - Lt))      # covers [Lt, Ht] although c is rounded
            _, p_c = Pp.eval((c, c))
            dclo, _ = det.eval((c, c))
            inside = alive & (dclo >= 0)
            if sym >= 1:
                inside &= c[0] <= c[1]
            if sym == 2:
                inside &= c[1] <= c[2]
            worst = max(worst, float(np.max(np.where(inside, p_c, -np.inf), initial=-np.inf)))
            if worst > cmax:
                return False, 'value %.6e above %.6e' % (worst, cmax), n_done
            first = np.zeros(c.shape[1]); contrib = np.zeros_like(c)
            for v in range(3):
                glo, ghi = Dv[v].eval((c, c))
                g = np.maximum(np.abs(glo), np.abs(ghi))
                term = up(g * r[v]); first = up(first + term); contrib[v] = term
            second = np.zeros(c.shape[1])
            for (v, w), Hp in H.items():
                hlo, hhi = Hp.eval((Lt, Ht))
                habs = np.maximum(np.abs(hlo), np.abs(hhi))
                if v == w:
                    # (x_v - c_v)^2 lies in [0, r_v^2], so with TAYLOR_MONO the upper end of d2P/dx_v^2,
                    # cut at 0, bounds its term
                    hdiag = np.maximum(hhi, 0.0) if TAYLOR_MONO else habs
                    term = up(hdiag * up(r[v] * r[v])); second = up(second + term); contrib[v] = up(contrib[v] + term)
                else:
                    term = up(up(2 * habs) * up(r[v] * r[w])); second = up(second + term)
                    contrib[v] = up(contrib[v] + term); contrib[w] = up(contrib[w] + term)
            upper = up(up(p_c + first) + up(second / 2))
            ok = upper <= cmax
            n_done += int(np.sum(alive & ok))
            und = alive & ~ok
            if np.any(und):
                w_ = Hh - L
                if np.max(w_[:, und]) < wmin:
                    return False, 'undecided box below width %g' % wmin, n_done
                Lu, Hu = L[:, und], Hh[:, und]
                axis = np.argmax(contrib[:, und], axis=0)
                if TAYLOR_MONO:
                    # a centre above cmax is outside the domain (else the check has failed): cut the
                    # widest side, so that the Gram condition can discard the parts outside
                    axis = np.where(p_c[und] > cmax, np.argmax(w_[:, und], axis=0), axis)
                mid = (Lu + Hu) / 2
                A_hi = Hu.copy(); B_lo = Lu.copy()
                for v in range(3):
                    sel = axis == v
                    A_hi[v, sel] = mid[v, sel]; B_lo[v, sel] = mid[v, sel]
                keep_lo += [Lu, B_lo]; keep_hi += [A_hi, Hu]
        level += 1
        lo = np.concatenate(keep_lo, axis=1) if keep_lo else np.zeros((3, 0))
        hi = np.concatenate(keep_hi, axis=1) if keep_hi else np.zeros((3, 0))
        if level % 10 == 0:
            print('      %s level %d: %d boxes verified, %d to bisect [%.0f s]' % (label, level, n_done, lo.shape[1], time.time() - t0), flush=True)
    return True, '%d boxes, largest centre value %.4e' % (n_done, worst), n_done


def triple_layout(types):
    """variables (x0, x1, x2) for the triple and the symmetry used: the permutation from
    (u12, u13, u23), and 2 (all equal), 1 (two equal) or 0."""
    s1, s2, s3 = types
    if s1 == s2 == s3:
        return (0, 1, 2), 2
    if s1 == s2:                          # swapping points 1, 2 exchanges u13 and u23
        return (1, 2, 0), 1
    if s2 == s3:                          # swapping points 2, 3 exchanges u12 and u13
        return (0, 1, 2), 1
    return (0, 1, 2), 0


def tmax(s, t):
    return amaxq(TRANGE[s][1], TRANGE[t][1])


def random_gram(T12, T13, T23, n, rng):
    outs = []
    for dim in (4, 3, 2):
        X = rng.normal(size=(30 * n, 3, dim)); X /= np.linalg.norm(X, axis=2, keepdims=True)
        g = np.stack([np.einsum('ni,ni->n', X[:, 0], X[:, 1]), np.einsum('ni,ni->n', X[:, 0], X[:, 2]),
                      np.einsum('ni,ni->n', X[:, 1], X[:, 2])], 1)
        ok = (g[:, 0] <= T12) & (g[:, 1] <= T13) & (g[:, 2] <= T23)
        outs.append(g[ok][:n])
    return np.concatenate(outs)


# ---------------------------------------------------------------- local maxima, to set the thresholds

def kfloat(Af, d, e, u):
    x = (2 * d - float(C1)) / float(C2); y = (2 * e - float(C1)) / float(C2)
    Tx = [1.0, x]; Ty = [1.0, y]
    for _ in range(2, R2 + 1):
        Tx.append(2 * x * Tx[-1] - Tx[-2]); Ty.append(2 * y * Ty[-1] - Ty[-2])
    Uu = [1.0, 2 * u]
    for _ in range(2, D2 + 1):
        Uu.append(2 * u * Uu[-1] - Uu[-2])
    return sum(Uu[k] / (k + 1) * np.array(Tx) @ Af[k] @ np.array(Ty) for k in range(D2 + 1))


def refine_pair(A, Pu, starts, box):
    """the largest value of K + PAIR3 - Pi found by local ascent from the given starts."""
    from scipy.optimize import minimize
    Af = [np.array([[float(v) for v in row] for row in a]) for a in A]
    pc = [float(c) for c in Pu[::-1]]
    (a0, a1), (b0, b1) = [(float(x), float(y)) for x, y in box]

    def f(x):
        d, e, u = x
        return -(kfloat(Af, d, e, u) + np.polyval(pc, u) - float(pair_float(d / 2, e / 2, u)))
    best = -np.inf
    for x0 in starts:
        cons = [{'type': 'ineq', 'fun': lambda x: (x[0] ** 2 + x[1] ** 2 - 4) / (2 * x[0] * x[1]) - x[2]}]
        try:
            r = minimize(f, x0, method='SLSQP', bounds=[(a0, a1), (b0, b1), (-1, 1)], constraints=cons,
                         options={'maxiter': 200, 'ftol': 1e-14})
            x = np.clip(r.x, [a0, b0, -1], [a1, b1, 1])
            x[2] = min(x[2], (x[0] ** 2 + x[1] ** 2 - 4) / (2 * x[0] * x[1]))
            best = max(best, -f(x))
        except Exception:
            pass
    return best


def refine_triple(P, starts, tops):
    """the largest value of P found by local ascent on the admissible triples."""
    from scipy.optimize import minimize
    items = [(np.array(e), float(c)) for e, c in P.items()]
    E = np.array([e for e, _ in items]); Cf = np.array([c for _, c in items])

    def f(x):
        return -float(np.sum(Cf * np.prod(np.asarray(x)[None, :] ** E, axis=1)))
    det = {'type': 'ineq', 'fun': lambda x: 1 + 2 * x[0] * x[1] * x[2] - x[0] ** 2 - x[1] ** 2 - x[2] ** 2}
    best = -np.inf
    for x0 in starts:
        try:
            r = minimize(f, x0, method='SLSQP', bounds=[(-1, tops[0]), (-1, tops[1]), (-1, tops[2])], constraints=[det],
                         options={'maxiter': 200, 'ftol': 1e-15})
            x = np.clip(r.x, -1, tops)
            if 1 + 2 * x[0] * x[1] * x[2] - x[0] ** 2 - x[1] ** 2 - x[2] ** 2 >= -1e-12:
                best = max(best, -f(x))
        except Exception:
            pass
    return best


# ---------------------------------------------------------------- main

def check(name, ok, detail=''):
    print('  [%s] %s%s' % ('ok' if ok else 'FAIL', name, ('  (' + detail + ')') if detail else ''), flush=True)
    if not ok:
        print('FAILED'); sys.exit(1)


def main():
    t0 = time.time()
    path, d3 = sys.argv[1], int(sys.argv[2])
    mg2 = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
    mg3 = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-6
    mgm = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-6
    Z = np.load(path)
    rng = np.random.default_rng(7)
    target = 9 * arb.pi() ** 2 / 8 - 8
    print('residual case of thm:count30: combined certificate %s, two-point degree %d/%d, three-point degree %d'
          % (os.path.basename(path), D2, R2, d3), flush=True)
    check('dmax exceeds sqrt 6', DMAX ** 2 > 6)
    # (a) positivity
    Af = np.array(Z['A'], float)
    A = [psd_exact(Af[k], Fr(1, 2 ** 30)) for k in range(1, D2 + 1)]
    A0 = psd_exact(Af[0], Fr(1, 2 ** 30))
    A = [A0] + A
    z = [dyad(v) for v in np.array(Z['z'], float)]
    good = all(ldl_psd(A[k])[0] for k in range(1, D2 + 1))
    g0, piv = ldl_psd(A0)
    wv = solve_exact(A0, z)
    tq = sum(a * b for a, b in zip(z, wv))
    t = Fr(-(-tq.numerator * 2 ** 48 // tq.denominator), 2 ** 48)
    Zm = [row[:] + [z[i]] for i, row in enumerate(A0)] + [z + [t]]
    gz, _ = ldl_psd(Zm)
    check('A_1..A_D and [[A_0, z], [z^T, t]] positive semidefinite (exact LDL^T)', good and g0 and all(p > 0 for p in piv) and gz,
          't = %.9f, float t %.9f' % (float(t), float(Z['t'])))
    Bf = blocks_from_x3(np.array(Z['x3'], float), d3)
    Bk = {name: psd_exact(M) for name, M in Bf.items()}
    check('three-point blocks positive definite (exact LDL^T)', all(ldl_positive(M) for M in Bk.values()),
          '%d blocks' % len(Bk))
    pt = {s: point_term(Bk, d3, i) for i, s in enumerate(TYPES)}
    print('  point terms: %s' % {s: '%.6f' % float(v) for s, v in pt.items()}, flush=True)
    # (b) pairs
    c2 = {}
    for (s, tt) in itertools.combinations_with_replacement(TYPES, 2):
        i, j = TYPES.index(s), TYPES.index(tt)
        Pu = pair3_poly(Bk, d3, i, j)
        (a0, a1), (b0, b1) = TRANGE[s], TRANGE[tt]
        n = 400000
        p = float(a0) + float(a1 - a0) * rng.random(n); q = float(b0) + float(b1 - b0) * rng.random(n)
        k6 = n // 6
        p[:k6] = float(a0); q[k6:2 * k6] = float(b0); p[2 * k6:3 * k6] = float(a1); q[3 * k6:4 * k6] = float(b1)
        top = (p * p + q * q - 4) / (2 * p * q)
        u = -1 + (top + 1) * rng.random(n) ** 0.5
        u[4 * k6:5 * k6] = top[4 * k6:5 * k6] - 3e-3 * rng.random(k6)
        gd, ge = np.meshgrid(np.linspace(float(a0), float(a1), 25), np.linspace(float(b0), float(b1), 25), indexing='ij')
        gd, ge = gd.ravel(), ge.ravel()
        gt = (gd * gd + ge * ge - 4) / (2 * gd * ge)
        sv = np.r_[0.0, np.linspace(0, 1, 80) ** 2, 1.0]
        gp = np.repeat(gd, len(sv)); gq = np.repeat(ge, len(sv))
        gu = -1 + (np.repeat(gt, len(sv)) + 1) * np.tile(1 - sv[::-1], len(gd))
        p, q, u = np.r_[p, gp], np.r_[q, gq], np.r_[u, gu]
        n = len(u)
        x = (2 * p - float(C1)) / float(C2); y = (2 * q - float(C1)) / float(C2)
        Tx = [np.ones_like(x), x]; Ty = [np.ones_like(y), y]
        for _ in range(2, R2 + 1):
            Tx.append(2 * x * Tx[-1] - Tx[-2]); Ty.append(2 * y * Ty[-1] - Ty[-2])
        Uu = [np.ones_like(u), 2 * u]
        for _ in range(2, D2 + 1):
            Uu.append(2 * u * Uu[-1] - Uu[-2])
        Kv = np.zeros(n)
        for k in range(D2 + 1):
            Ak = np.array([[float(v) for v in row] for row in A[k]])
            Kv += Uu[k] / (k + 1) * np.einsum('na,ab,nb->n', np.stack(Tx, 1), Ak, np.stack(Ty, 1))
        P3 = np.polyval([float(c) for c in Pu[::-1]], u)
        v = Kv + P3 - pair_float(p / 2, q / 2, u)
        w = np.argsort(v)[-40:]
        best = refine_pair(A, Pu, np.stack([p[w], q[w], u[w]], 1), (TRANGE[s], TRANGE[tt]))
        c2[(s, tt)] = above(max(float(v.max()), best) + mg2)
        print('  pair %s%s: float largest K + PAIR3 - Pi %.6e (file c2 %.6e); threshold %.6e'
              % (s, tt, v.max(), float(Z['c2'][len(c2) - 1]), float(c2[(s, tt)])), flush=True)
    jobs = [(A, pair3_poly(Bk, d3, TYPES.index(s), TYPES.index(tt)), c2[(s, tt)], (TRANGE[s], TRANGE[tt]), s == tt)
            for (s, tt) in c2]
    from multiprocessing import Pool
    with Pool(min(4, len(jobs))) as pool:
        res = pool.starmap(pair_box_check, jobs)
    for (s, tt), (ok, msg) in zip(c2, res):
        check('K + PAIR3_%s%s <= Pi + c2 on the admissible pairs' % (s, tt), ok, msg)
    # (c) bins
    mf = np.array(Z['m'], float)
    m = [above(mf[0] + mgm), above(mf[1] + mgm), above(mf[2] + mgm), Fr(10 ** 6), above(mf[3] + mgm)]
    okb, msg = bracket_check(A, z, D2, R2, C1, C2, DMAX, EDGES, m)
    check('f <= m_b on every bin', okb, msg)
    # (d) triples
    c3 = {}
    tjobs = []
    for combo in itertools.combinations_with_replacement(TYPES, 3):
        if all(Ntriple(tcounts(nb), list(combo)) == 0 for nb in COUNTS):
            continue
        P = triple3_poly(Bk, d3, combo)
        T12, T13, T23 = tmax(combo[0], combo[1]), tmax(combo[0], combo[2]), tmax(combo[1], combo[2])
        g = random_gram(float(T12), float(T13), float(T23), 200000, rng)
        ax = [np.r_[-1.0, np.linspace(-1, float(T), 40), float(T)] for T in (T12, T13, T23)]
        G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
        G = G[1 + 2 * G[:, 0] * G[:, 1] * G[:, 2] - (G ** 2).sum(1) >= 0]
        g = np.r_[g, G]
        vals = peval(P, g[:, 0], g[:, 1], g[:, 2])
        best = refine_triple(P, g[np.argsort(vals)[-40:]], (float(T12), float(T13), float(T23)))
        c3[combo] = above(max(float(vals.max()), best) + mg3)
        perm, sym = triple_layout(combo)
        tops = [None] * 3
        for src, val in enumerate((T12, T13, T23)):
            tops[perm.index(src)] = float(above(val))
        Pr = {}
        for e, co in P.items():
            ee = [0, 0, 0]
            for src in range(3):
                ee[perm.index(src)] += e[src]
            Pr[tuple(ee)] = Pr.get(tuple(ee), 0) + co
        print('  triple %s: float largest %.6e on %d samples; threshold %.6e; %d monomials, symmetry %d'
              % (''.join(combo), vals.max(), len(g), float(c3[combo]), len(P), sym), flush=True)
        tjobs.append((combo, (Pr, tops, sym, float(c3[combo]), ''.join(combo))))
    from multiprocessing import Pool
    with Pool(int(os.environ.get('TRIPLE_PROCS', '3'))) as pool:
        res = pool.starmap(triple_box_check, [a for _, a in tjobs])
    for (combo, _), (ok, msg, _) in zip(tjobs, res):
        check('TRIPLE3_%s <= c3 on the admissible triples' % ''.join(combo), ok, msg)
    # the bound
    worst = None
    for nb in COUNTS:
        tc = tcounts(nb)
        val = sum(nb[b] * (m[b] + pt[BINTYPE[b]]) for b in range(len(nb)) if nb[b]) + t / 2
        val += sum(Npair(tc, s, tt) * c2[(s, tt)] for (s, tt) in c2)
        val += sum(Ntriple(tc, list(cb)) * c3[cb] for cb in c3)
        print('  counts %s: bound %.6f' % (nb, float(val)), flush=True)
        worst = val if worst is None else max(worst, val)
    check('largest bound over the five count vectors below 9 pi^2/8 - 8', A_(worst) < target,
          '%.6f < %s' % (float(worst), target.str(8)))
    print('PASS: in the residual case of thm:count30 every packing set has U(Y) <= %.6f < 9 pi^2/8 - 8, so T(Y) > 8 [%.0f s]'
          % (float(worst), time.time() - t0))


if __name__ == '__main__':
    main()
