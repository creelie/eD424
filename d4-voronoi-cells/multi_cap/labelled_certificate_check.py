#!/usr/bin/env python3
"""
labelled_certificate_check.py -- the three-point certificate of thm:certificate
carried from contacts to centres at any distances: a centre of a unit-ball
packing of R^4 with at most 23 other centres within sqrt 6 has a Voronoi cell
of volume greater than 8 (thm:m23 of the paper).

Setting.  c = 0; y_1, ..., y_23 are the centres with |y_i| < sqrt 6, at
distances d_i = |y_i| >= 2, pairwise at least 2 apart; w_i = y_i / d_i,
h_i = d_i / 2, u_ij = <w_i, w_j>.  R^2 = 3/2.  By lem:no-triple (no triple
overlaps in B(R)),

    vol(V_c) >= T = 9 pi^2/8 - sum_i S(d_i) + sum_{i<j} Pair(h_i, h_j, u_ij)
              = A_* + sum_i s(d_i) + sum_{i<j} Pair(h_i, h_j, u_ij),

S(d) the cap of B(R) beyond distance d/2, s(d) = S(2) - S(d) >= 0,
Pair(h_i, h_j, u) the volume of the part of B(R) beyond both hyperplanes,
A_* = 9 pi^2/8 - 23 S(2).  With all d_i = 2 this is thm:certificate's
T = A_* + sum omega(u_ij), omega(u) = Pair(1, 1, u).

The labelled inequality.  Let P be the polynomial of the certificate of
thm:certificate and, for three centres, put (scale 1000, as the solver's)

    Q = 1000 [Pair_12 + Pair_13 + Pair_23 + (s(d_1) + s(d_2) + s(d_3)) / 11]
        - P(u_23, u_13, u_12).

Each pair lies in 21 = N - 2 triples and each centre in 231 = 21 * 11, so
if Q >= 0 for every triple, summing over the triples and using the
certificate lemma (lem:certificate, which holds for any 23 unit vectors) gives
sum Pair + sum s >= B / 1000 > 8 - A_*, hence vol(V_c) > 8.  A centre at
distance above D = 2.1648 settles the case by itself (s(D) > 8 - A_*, step
2), so d_i in [2, D].  Q is symmetric under relabelling the three centres,
so label them so that t = u_12 >= v = u_13 >= u = u_23.  The packing gives
u_ij <= amax(d_i, d_j) = (d_i^2 + d_j^2 - 4) / (2 d_i d_j) <= a_D = amax(D, D).

The reduction to three variables.  Let A(tau) = (4 pi / 3)(R^2 - tau^2)^(3/2)
be the volume of the section of B(R) at height tau (so s(2h) is the
integral of A from 1 to h), and let fr(tau, x) be the share of that section
(a 3-ball of radius rho = (R^2 - tau^2)^(1/2)) on the far side of the
hyperplane of a second centre at height 1 and inner product x:

    fr(tau, x) = (1 - q)^2 (2 + q) / 4,   q = (1 - tau x) / ((1 - x^2)^(1/2) rho),
    fr = 0 for q >= 1.

Moving the hyperplane of centre i from height 1 to h_i changes Pair by minus
the integral of A times the share cut by the other centre; that share
decreases in the other centre's height and in the own height, and increases
in x (step 2 checks the two numerical facts this uses).  Hence

    Q >= Q0(u, v, t) + 1000 [Gamma_1(h_1) + Gamma_2(h_2) + Gamma_3(h_3)],
    Q0 = 1000 (omega(u) + omega(v) + omega(t)) - P(u, v, t),
    Gamma_i(h) = int_1^h A(tau) [1/11 - fr(tau, a_i) - fr(tau, b_i)] dtau,

(a_1, b_1) = (t, v), (a_2, b_2) = (t, u), (a_3, b_3) = (v, u).  Q0 >= 0 on
u, v, t <= 1/2 is the inequality (cert) that certificate_check.py verifies.

  Region I, t <= 1/2: fr(tau, x) <= fr(1, 1/2) < 1/22, so every Gamma_i >= 0
      and Q >= Q0 >= 0 by (cert).  Step 3 reruns certificate_check's
      verification of (cert) (skip with --skip-region-1; its log is
      multi_cap/runs/certificate_check.log of the supplement).
  Region II_s, 1/2 < t <= 1/2 + tau_1 (tau_1 = 1/100): the integrands are at
      least r = 1/11 - 2 fr(1, 1/2 + tau_1) > 0, so Gamma_1 + Gamma_2 >=
      r (s(d_1) + s(d_2)).  amax(d_1, d_2) <= 1/2 + (d_1 + d_2 - 4)/4
      (the difference is (d_1-2)(d_2-2)(d_1+d_2+2)/(4 d_1 d_2)), so the packing
      forces (d_1 - 2) + (d_2 - 2) >= 4 (t - 1/2); s(2 + x) is concave with
      s(2) = 0, so s(d_1) + s(d_2) >= s(2 + 4(t - 1/2)) >= kappa 4 (t - 1/2),
      kappa = s(2 + 4 tau_1) / (4 tau_1).  Step 4 verifies
          Q0(u, v, t) + c (t - 1/2) >= 0,   c = 4000 r kappa,
      on this slab, by the branch and bound of certificate_check.py (second
      order Taylor forms, omega tabulated in interval arithmetic) with the
      linear term added.
  Region II_f, 1/2 + tau_1 <= t <= a_D: step 5 bounds, on each box of
      (u, v, t), Q0 by the same Taylor form and 1000 sum Gamma_i from below
      by tabulating Gamma_i on 256 cells of h in [1, D/2] in interval
      arithmetic (fr at the lower end of the cell, the right-hand minimum
      envelope, so that the bound is nondecreasing in h) and minimising
      over the cells that the packing condition allows for the lower ends
      of the box.

Tables are computed with python-flint (arb, 128 bits) and stored as floats
rounded outward; box computations are in floating point with outward
rounding (numpy nextafter), as in certificate_check.py, and the summations
of at most 300 terms of size below 1 are charged 1e-12 each.

Usage: python3 labelled_certificate_check.py [--skip-region-1] [min box width]
Exit status 0 when every step passes.
"""
import math
import sys
import time
from fractions import Fraction as Fr

import numpy as np
from flint import arb, ctx
from mpmath import iv

import certificate_check as CC

ctx.prec = 128
N, SCALE = 23, 1000
D = Fr(21648, 10000)                  # distances in [2, D]; beyond D a centre settles the case alone
H = D / 2
TAU1 = Fr(1, 100)                     # width of the slab II_s
K = 256                               # cells of h in [1, D/2]
MX = 1600                             # grid of inner products for the table of fr
down, up = CC.down, CC.up
FAILS = []


def check(name, ok, detail=''):
    print(('[PASS] ' if ok else '[FAIL] ') + name)
    if detail:
        print('       ' + detail.replace('\n', '\n       '))
    if not ok:
        FAILS.append(name)


def flo(x):
    """a float <= the arb (or Fraction) x"""
    if isinstance(x, Fr):
        f = float(x)
        return f if Fr(f) <= x else float(np.nextafter(f, -np.inf))
    return float(np.nextafter(float(x.lower()), -np.inf))


def fhi(x):
    """a float >= the arb (or Fraction) x"""
    if isinstance(x, Fr):
        f = float(x)
        return f if Fr(f) >= x else float(np.nextafter(f, np.inf))
    return float(np.nextafter(float(x.upper()), np.inf))


def A(x):
    return arb(x.numerator) / x.denominator if isinstance(x, Fr) else arb(x)


PI = arb.pi()
R2 = arb(3) / 2
Rr = R2.sqrt()


def S(d):
    """volume of the cap {x in B(sqrt(3/2)) : <x, w> > d/2}, 2 <= d <= sqrt 6"""
    h = A(d) / 2
    s = R2 - h * h
    return 4 * PI / 3 * (3 * R2 * R2 / 8 * PI / 2 - (h / 8 * (5 * R2 - 2 * h * h) * s.sqrt() + 3 * R2 * R2 / 8 * (h / Rr).asin()))


S2 = S(2)


def s_(d):
    return S2 - S(d)


def fr_hi(tau, x):
    """an upper bound of fr(tau, x): the share of the section of B(R) at height tau
    beyond the hyperplane of a centre at height 1 with inner product x"""
    tau, x = A(tau), A(x)
    q = (1 - tau * x) / ((1 - x * x).sqrt() * (R2 - tau * tau).sqrt())
    ql = q.lower()
    if ql >= 1:
        return 0.0
    return fhi((1 - ql) ** 2 * (2 + ql) / 4)


def amax(d1, d2):
    return (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)


# ---------------------------------------------------------------- the omega tables, up to a_D
def omega_tables(om0, om1, om2, r_star, umax, n=18000, n2=4000):
    """As certificate_check.omega_tables, from 2 r_* down to an angle whose cosine
    exceeds umax (instead of 60 degrees)."""
    g_hi = float(np.nextafter(float((2 * r_star).b), np.inf))
    g_lo = math.acos(umax) - 1e-6                    # its cosine is checked to exceed umax below
    gams = np.linspace(g_hi, g_lo, n + 1)
    us = np.zeros(n + 1); olow = np.zeros(n + 1); ohigh = np.zeros(n + 1); d1lo = np.zeros(n + 1); d1hi = np.zeros(n + 1)
    for i, g in enumerate(gams):
        gi = iv.mpf(g)
        us[i] = np.nextafter(float(iv.cos(gi).b), np.inf)
        w = om0(gi); olow[i] = max(np.nextafter(float(w.a), -np.inf), 0.0); ohigh[i] = max(np.nextafter(float(w.b), np.inf), 0.0)
        w1 = om1(gi); d1lo[i] = np.nextafter(float(w1.a), -np.inf); d1hi[i] = np.nextafter(float(w1.b), np.inf)
    assert np.all(np.diff(us) > 0) and np.all(np.diff(olow) >= 0)
    assert float(iv.cos(iv.mpf(gams[-1])).a) >= umax
    du = np.nextafter(float(np.max(np.diff(us))) + 1e-12, np.inf)
    d1lo[0] = min(d1lo[0], -1e-300)
    M2 = 0.0
    edges = np.linspace(g_hi, g_lo, n2 + 1)
    for a, b in zip(edges[1:], edges[:-1]):
        w2 = om2(iv.mpf([a, b]))
        M2 = max(M2, abs(float(w2.a)), abs(float(w2.b)))
    return us, olow, ohigh, d1lo, d1hi, du, np.nextafter(M2, np.inf)


# ---------------------------------------------------------------- the tables of Gamma
class GammaTables:
    """Cells [tau_k, tau_{k+1}] of h in [1, D/2]; ds_k the integral of A over the cell;
    FR[k, m] >= fr(tau_k, x_m); AMX[k, l] >= amax(2 tau_{k+1}, 2 tau_{l+1})."""

    def __init__(self):
        taus = [1 + (H - 1) * Fr(k, K) for k in range(K + 1)]
        Ss = [S(2 * t) for t in taus]
        ds = [Ss[k] - Ss[k + 1] for k in range(K)]
        self.ds_lo = np.array([flo(x) for x in ds]); self.ds_hi = np.array([fhi(x) for x in ds])
        assert np.all(self.ds_lo > 0)
        self.aD = amax(D, D)
        xtop = Fr(fhi(self.aD))                    # the float bound of a_D that the boxes use
        self.xs = [Fr(1, 3) + (xtop - Fr(1, 3)) * Fr(m, MX) for m in range(MX + 1)]
        self.xs_lo = np.array([flo(x) for x in self.xs])
        self.third_lo = flo(Fr(1, 3))
        FR = np.zeros((K, MX + 2))                   # column 0: x <= 1/3, where fr = 0
        for k in range(K):
            for m, x in enumerate(self.xs):
                FR[k, m + 1] = fr_hi(taus[k], x)
        assert np.all(np.diff(FR[:, 1:], axis=1) >= 0) and np.all(np.diff(FR, axis=0) <= 0)
        self.FR = FR
        AMX = np.zeros((K, K))
        for k in range(K):
            for l in range(K):
                AMX[k, l] = fhi(amax(2 * taus[k + 1], 2 * taus[l + 1]))
        assert np.all(np.diff(AMX, axis=1) > 0)
        self.AMX = AMX

    def col(self, xhi):
        """column of FR bounding fr(., x) for every x <= xhi"""
        m = np.searchsorted(self.xs_lo, xhi, side='left') + 1
        m = np.where(xhi <= self.third_lo, 0, m)
        if np.any(m > MX + 1):
            raise ValueError('inner product beyond a_D')
        return m

    def gamma_env(self, ahi, bhi):
        """for n boxes: lower bounds (n, K) of min_{h' >= h} Gamma(h') for h in cell k"""
        fa = self.FR[:, self.col(ahi)].T; fb = self.FR[:, self.col(bhi)].T          # (n, K)
        cut = up(self.ds_hi[None, :] * up(fa + fb))
        c_lo = down(down(self.ds_lo / 11)[None, :] - cut)
        before = np.concatenate([np.zeros((c_lo.shape[0], 1)), np.cumsum(c_lo, axis=1)[:, :-1]], axis=1)
        g = before - cut - 1e-12                         # the part of cell k up to h costs at most cut_k
        return np.minimum.accumulate(g[:, ::-1], axis=1)[:, ::-1]

    def kappa(self, ylo):
        """kap[n, k]: the least cell l with AMX[k, l] >= ylo (K if none)"""
        return np.stack([np.searchsorted(self.AMX[k], ylo, side='left') for k in range(K)], axis=1)

    def gain(self, L, Hh):
        """lower bounds, in units of 1/1000 of Q, of Gamma_1(h_1) + Gamma_2(h_2) + Gamma_3(h_3)
        over the heights allowed by the packing for the boxes [L, Hh] (rows u, v, t)"""
        u_lo, v_lo, t_lo = L
        u_hi, v_hi, t_hi = Hh
        n = L.shape[1]
        G1 = self.gamma_env(t_hi, v_hi); G2 = self.gamma_env(t_hi, u_hi); G3 = self.gamma_env(v_hi, u_hi)
        INF = np.full((n, 1), np.inf)
        G2x = np.concatenate([G2, INF], axis=1); G3x = np.concatenate([G3, INF], axis=1)
        kt = self.kappa(t_lo); kv = self.kappa(v_lo)
        rows = np.arange(n)[:, None]
        out = np.full(n, np.inf)
        easy = u_lo <= 0.5                         # the pair (2, 3) is unconstrained
        if np.any(easy):
            e = np.where(easy)[0]
            val = G1[e] + G2x[rows[e], kt[e]] + G3x[rows[e], kv[e]]
            out[e] = np.min(val, axis=1)
        hard = np.where(~easy)[0]
        for s in range(0, len(hard), 64):
            e = hard[s:s + 64]
            ku = self.kappa(u_lo[e])                              # (m, K)
            k3 = np.maximum(kv[e][:, :, None], ku[:, None, :])    # (m, K, K): k1, k2
            val = G1[e][:, :, None] + G2[e][:, None, :] + np.take_along_axis(
                G3x[e][:, None, :].repeat(K, axis=1), k3, axis=2)
            ok = np.arange(K)[None, None, :] >= kt[e][:, :, None]
            val = np.where(ok, val, np.inf)
            out[e] = np.min(val, axis=(1, 2))
        return down(SCALE * out - 1e-9)


# ---------------------------------------------------------------- branch and bound
def verify(P, tables, lo0, hi0, wmin, extra, label, batch=100000):
    """certificate_check.verify_domain on the ordered box lo0..hi0 (u <= v <= t), for
    Q0 + extra: extra = ('lin', c) adds c (t - 1/2) to the Taylor form; extra =
    ('gain', G) adds the lower bound G.gain(box) to the bound of each box."""
    us, olow, ohigh, d1lo, d1hi, du, M2 = tables
    olow = down(SCALE * olow); ohigh = up(SCALE * ohigh); d1lo = down(SCALE * d1lo); d1hi = up(SCALE * d1hi)
    M2 = up(SCALE * M2); dM = up(M2 * du)
    Pp = CC.Poly(P); Dp = [CC.Poly(CC.derivative(P, v)) for v in range(3)]
    Hs = {(v, w): CC.Poly(CC.derivative(CC.derivative(P, v), w)) for v in range(3) for w in range(v, 3)}
    det = CC.Poly({(0, 0, 0): Fr(1), (1, 1, 1): Fr(2), (2, 0, 0): Fr(-1), (0, 2, 0): Fr(-1), (0, 0, 2): Fr(-1)})
    lo = np.array(lo0, dtype=float).reshape(3, 1); hi = np.array(hi0, dtype=float).reshape(3, 1)
    t0 = time.time(); level = 0; n_done = 0; n_out = 0; worst = np.inf
    while lo.shape[1] > 0:
        n = lo.shape[1]; keep_lo = []; keep_hi = []
        for s in range(0, n, batch):
            L, Hh = lo[:, s:s + batch], hi[:, s:s + batch]
            alive = (L[0] <= Hh[1]) & (L[1] <= Hh[2])
            dlo, dhi = det.eval((L, Hh))
            alive &= dhi >= 0
            n_out += int(np.sum(~alive))
            L, Hh = L[:, alive], Hh[:, alive]
            if L.shape[1] == 0:
                continue
            c = (L + Hh) / 2; r = np.maximum(up(Hh - c), up(c - L))      # covers [L, Hh] although c is rounded
            q0 = np.zeros(c.shape[1]); grad_lo = np.zeros_like(c); grad_hi = np.zeros_like(c); m2 = np.zeros_like(c)
            for v in range(3):
                inside = L[v] >= CC.THIRD
                idx = np.clip(np.searchsorted(us, c[v], side='right') - 1, 0, len(us) - 1)
                q0 = down(q0 + np.where(inside, olow[idx], 0.0))
                grad_lo[v] = np.where(inside, down(d1lo[idx] - dM), 0.0)
                grad_hi[v] = np.where(inside, up(d1hi[idx] + dM), 0.0)
                m2[v] = np.where(inside, M2, 0.0)
            plo, phi = Pp.eval((c, c))
            q0 = down(q0 - phi)
            if extra[0] == 'lin':
                cc = extra[1]
                q0 = down(q0 + down(cc * (c[2] - 0.5)))
                grad_lo[2] = down(grad_lo[2] + cc); grad_hi[2] = up(grad_hi[2] + cc)
            first = np.zeros(c.shape[1]); contrib = np.zeros_like(c)
            for v in range(3):
                glo, ghi = Dp[v].eval((c, c))
                dq_lo = down(grad_lo[v] - ghi); dq_hi = up(grad_hi[v] - glo)
                g_abs = np.maximum(np.abs(dq_lo), np.abs(dq_hi))
                first = up(first + up(g_abs * r[v])); contrib[v] = up(g_abs * r[v])
            second = np.zeros(c.shape[1])
            for (v, w), Hp in Hs.items():
                hlo, hhi = Hp.eval((L, Hh))
                habs = np.maximum(np.abs(hlo), np.abs(hhi))
                if v == w:
                    term = up(up(habs + m2[v]) * up(r[v] * r[v]))
                    second = up(second + term); contrib[v] = up(contrib[v] + term)
                else:
                    term = up(up(2 * habs) * up(r[v] * r[w]))
                    second = up(second + term); contrib[v] = up(contrib[v] + term); contrib[w] = up(contrib[w] + term)
            qlo = down(down(q0 - first) - up(second / 2))
            if extra[0] == 'gain':
                need = qlo < 0
                if np.any(need):
                    gl = extra[1].gain(L[:, need], Hh[:, need])
                    qlo[need] = down(qlo[need] + gl)
                for v in range(3):                      # the gain moves with the lower ends above 1/2
                    contrib[v] = contrib[v] + np.where(Hh[v] > 0.5, 130.0 * (Hh[v] - L[v]), 0.0)
            ok = qlo >= 0
            worst = min(worst, float(np.min(qlo)))
            n_done += int(np.sum(ok))
            und = ~ok
            if np.any(und):
                w_ = Hh - L
                if np.max(w_[:, und]) < wmin:
                    i = int(np.argmax(und))
                    print(f'  undecided box of width below {wmin}: u, v, t in [{L[0][i]:.7f},{Hh[0][i]:.7f}] x '
                          f'[{L[1][i]:.7f},{Hh[1][i]:.7f}] x [{L[2][i]:.7f},{Hh[2][i]:.7f}], bound {qlo[i]:.3e}')
                    return False, n_done, n_out
                Lu, Hu = L[:, und], Hh[:, und]
                axis = np.argmax(contrib[:, und], axis=0)
                mid = (Lu + Hu) / 2
                A_hi = Hu.copy(); B_lo = Lu.copy()
                for v in range(3):
                    sel = axis == v
                    A_hi[v, sel] = mid[v, sel]; B_lo[v, sel] = mid[v, sel]
                keep_lo += [Lu, B_lo]; keep_hi += [A_hi, Hu]
        level += 1
        lo = np.concatenate(keep_lo, axis=1) if keep_lo else np.zeros((3, 0))
        hi = np.concatenate(keep_hi, axis=1) if keep_hi else np.zeros((3, 0))
        print(f'  {label} level {level:2d}: {n_done} boxes verified, {n_out} outside the domain, '
              f'{lo.shape[1]} to bisect  [{time.time() - t0:.0f}s]', flush=True)
    return True, n_done, n_out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    wmin = float(args[0]) if args else 1e-7
    t_start = time.time()
    # ------------------------------------------------------------ 1. the certificate
    d = 8
    Z = np.load('continuation_out/certificate_d%d.npz' % d)
    f = [Fr(float(x)) for x in Z['f']]
    F = [[[Fr(float(x)) for x in row] for row in Z['F%d' % k]] for k in range(d + 1)]
    f = [f[0]] + [max(x, Fr(0)) for x in f[1:]]
    check('step 1: f_k >= 0 for k >= 1 and every F_k positive definite (exact LDL^T)',
          all(CC.ldl_positive(F[k]) for k in range(d + 1)))
    P, F111 = CC.build_P(d, f, F)
    B = N * (N * f[0] - sum(f)) / 2 - N * F111 / (6 * (N - 2))
    sym = all(P.get((e[p[0]], e[p[1]], e[p[2]]), 0) == c for e, c in P.items() for p in CC.PERMS)
    om0, om1, om2, A_star, r_star = CC.closed_forms()
    Ast = A_star()
    gap_hi = Fr(float(np.nextafter(float(8 - Ast.a), np.inf)))          # a rational above 8 - A_*
    check('step 1: B / 1000 = %.10f exceeds 8 - A_* in [%.12f, %.12f]; P is symmetric (%d monomials)'
          % (float(B / SCALE), float(8 - Ast.b), float(8 - Ast.a), len(P)), B / SCALE > gap_hi and sym)
    # ------------------------------------------------------------ 2. constants
    gap_arb = A(gap_hi)
    sD = s_(D)
    check('step 2: s(D) > 8 - A_* for D = %s: a centre at distance above D gives T > 8 by itself' % float(D),
          sD.lower() > gap_arb, 's(D) = %s' % sD.str(12, radius=False))
    aD = amax(D, D)
    aD_f = fhi(aD)
    mono = A(aD) / (1 - A(H) * A(aD))
    check('step 2: the share fr decreases in both heights: a_D / (1 - (D/2) a_D) < 2 <= tau / (R^2 - tau^2) for tau >= 1',
          mono.upper() < 2, 'a_D = amax(D, D) = %.10f; a_D / (1 - (D/2) a_D) = %s; 1 / (R^2 - 1) = 2'
          % (float(aD), mono.str(8, radius=False)))
    f12 = fr_hi(Fr(1), Fr(1, 2))
    check('step 2: fr(1, 1/2) < 1/22, so Gamma_i >= 0 when every inner product is at most 1/2',
          Fr(f12) < Fr(1, 22), 'fr(1, 1/2) <= %.8f < 1/22 = %.8f' % (f12, 1 / 22))
    # the slab constant
    fs = fr_hi(Fr(1), Fr(1, 2) + TAU1)
    rr = Fr(1, 11) - 2 * Fr(fs)
    kap = s_(2 + 4 * TAU1) / A(4 * TAU1)
    c_slab = flo(4000 * A(rr) * kap)
    check('step 2: slab constant c = 4000 r kappa > 0 for tau_1 = %s' % float(TAU1), rr > 0 and c_slab > 0,
          'fr(1, 1/2 + tau_1) <= %.8f, r = 1/11 - 2 fr >= %.8f, kappa = s(2 + 4 tau_1)/(4 tau_1) >= %.8f, c >= %.6f'
          % (fs, float(rr), flo(kap), c_slab))
    tables = omega_tables(om0, om1, om2, r_star, aD_f)
    us, olow, ohigh, d1lo, d1hi, du, M2 = tables
    print('  omega tabulated at %d angles up to u = %.6f (cells of width at most %.2e in u); |omega\'\'| <= %.6f on [1/3, a_D]'
          % (len(us), us[-1], du, M2))
    # ------------------------------------------------------------ 3. region I
    if '--skip-region-1' in sys.argv:
        print('[SKIP] step 3: region I is the inequality (cert) of certificate_check.py (see its log)')
    else:
        tab1 = CC.omega_tables(om0, om1, om2, r_star)
        ok = CC.verify_domain(P, tab1, 1e-5)
        check('step 3: region I, u <= v <= t <= 1/2: the inequality (cert) of thm:certificate', ok)
    # ------------------------------------------------------------ 4. the slab
    t1 = float(Fr(1, 2) + TAU1)
    ok, nd, no = verify(P, tables, [-1.0, -1.0, 0.5], [t1, t1, t1], wmin, ('lin', c_slab), 'II_s')
    check('step 4: region II_s, 1/2 <= t <= %.2f: Q0 + c (t - 1/2) >= 0' % t1, ok,
          '%d boxes verified, %d outside the domain' % (nd, no))
    # ------------------------------------------------------------ 5. the rest
    t0 = time.time()
    G = GammaTables()
    print('  Gamma tables: %d cells of h in [1, %.4f], fr on %d inner products in [1/3, a_D]  [%.0fs]'
          % (K, float(H), MX + 1, time.time() - t0))
    ok, nd, no = verify(P, tables, [-1.0, -1.0, t1], [aD_f, aD_f, aD_f], wmin, ('gain', G), 'II_f')
    check('step 5: region II_f, %.2f <= t <= a_D: Q0 + 1000 sum Gamma_i >= 0' % t1, ok,
          '%d boxes verified, %d outside the domain' % (nd, no))
    print()
    if FAILS:
        print('FAILED: ' + '; '.join(FAILS))
        sys.exit(1)
    print('PASS: Q >= 0 for every three centres at distances in [2, %.4f] from c, pairwise at least 2 apart; '
          'so every centre with at most 23 other centres within sqrt 6 has vol(V_c) >= A_* + B / 1000 > 8.  [%.0fs]'
          % (float(D), time.time() - t_start))


if __name__ == '__main__':
    main()
