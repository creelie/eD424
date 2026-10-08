#!/usr/bin/env python3
"""
certificate_check.py -- verification of a three-point certificate for the
pair inequality of thm:strict-reduction, in exact rational and interval arithmetic
(the proof of thm:certificate).

Input: continuation_out/certificate_d<d>.npz, written by three_point_sdp.py
in certificate mode: floating-point numbers f_0, ..., f_d and symmetric
matrices F_0, ..., F_d, with F_k of size d - k + 1.  The numbers are taken
as the exact rationals they denote (no rounding is applied to them), and
the following is established.

  1.  f_k >= 0 for k >= 1 (any negative f_k is replaced by 0), and every
      F_k is positive definite: exact LDL^T decomposition over the
      rationals, every pivot positive.

  2.  The bound B = N (N f_0 - f(1)) / 2 - N F(1,1,1) / (6 (N - 2)), with
      N = 23, computed exactly, divided by the scale 1000 of the solver,
      exceeds 8 - A_*, where A_* is evaluated in interval arithmetic from
      its closed form (the integrals in lem:pair-closed are elementary):
      A_* = 9 pi^2 / 8 - 207 pi r_* / 8 + 253 pi / (12 sqrt 2), r_* = arctan(1/sqrt 2).

  3.  The polynomial P(u, v, t) = f(u) + f(v) + f(t) + F(u, v, t) +
      (the matrices Y_k written in the basis T_i(u) T_j(v) of Chebyshev
      polynomials, as three_point_sdp.py does) +
      (F(1,u,u) + F(1,v,v) + F(1,t,t)) / (N - 2) is expanded exactly, as a
      polynomial with rational coefficients in u, v, t, and the inequality
          1000 (omega(u) + omega(v) + omega(t)) >= P(u, v, t)          (cert)
      is verified on the whole admissible domain -1 <= u <= v <= t <= 1/2,
      1 + 2uvt - u^2 - v^2 - t^2 >= 0 (P is symmetric, so this suffices)
      by interval branch and bound.  omega, its first and its second
      derivative in u are elementary (the lens integral has a closed form,
      differentiated symbolically) and are tabulated in interval arithmetic
      at 12001 angles; omega is increasing in u, and |omega''| is bounded
      on [1/3, 1/2].  On each box the function Q = 1000 (omega(u) + omega(v)
      + omega(t)) - P is bounded below by its second-order Taylor form
      about the centre: the value and gradient at the centre from the
      tables and the exact polynomial, the second derivatives from the
      bound on omega'' and an interval evaluation of the second derivatives
      of P over the box.  A variable whose range is not inside (1/3, 1/2]
      has its omega term replaced by 0.  Boxes on which (cert) is not decided
      are bisected along the direction that contributes most to the
      remainder; boxes outside the domain are discarded; an admissible
      centre at which (cert) is negative stops the run with a counterexample.

If all three steps pass, then for every contact configuration of 23
directions sum_{i<j} omega(gamma_ij) >= B / 1000 > 8 - A_*, which is the
hypothesis of thm:strict-reduction.  The script prints what it verifies and
stops with a message at the first failure.

Usage: python3 certificate_check.py [degree] [minimum box width]
  (a third argument 'continue' lets step 3 run even if step 2 fails, for testing)
"""
import sys, time
from fractions import Fraction as Fr
import numpy as np
from mpmath import iv, mp

N = 23
SCALE = 1000
THIRD = float(np.nextafter(1.0 / 3.0, 1.0))      # a float above 1/3; omega vanishes below 1/3

# ---------------------------------------------------------------- exact polynomials
def padd(p, q):
    r = dict(p)
    for k, c in q.items():
        r[k] = r.get(k, 0) + c
    return {k: c for k, c in r.items() if c != 0}

def pscale(p, s):
    return {k: c * s for k, c in p.items()} if s != 0 else {}

def pmul(p, q):
    r = {}
    for k1, c1 in p.items():
        for k2, c2 in q.items():
            k = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2])
            r[k] = r.get(k, 0) + c1 * c2
    return {k: c for k, c in r.items() if c != 0}

def ppow(p, n):
    r = {(0, 0, 0): Fr(1)}
    for _ in range(n): r = pmul(r, p)
    return r

U, V, T = {(1, 0, 0): Fr(1)}, {(0, 1, 0): Fr(1)}, {(0, 0, 1): Fr(1)}
ONE = {(0, 0, 0): Fr(1)}

def legendre_coeffs(d):
    """Monomial coefficients of P_0, ..., P_d, exact."""
    P = [[Fr(1)], [Fr(0), Fr(1)]]
    for k in range(1, d):
        a = [Fr(0)] + [(2 * k + 1) * c for c in P[k]]          # (2k+1) x P_k
        b = P[k - 1] + [Fr(0)] * (len(a) - len(P[k - 1]))
        P.append([(a[i] - k * b[i]) / (k + 1) for i in range(len(a))])
    return P[:d + 1]

def gegenbauer_S3(d):
    """G_k(u) = U_k(u)/(k+1) as exact univariate coefficient lists."""
    Uc = [[Fr(1)], [Fr(0), Fr(2)]]
    for k in range(1, d):
        a = [Fr(0)] + [2 * c for c in Uc[k]]
        b = Uc[k - 1] + [Fr(0)] * (len(a) - len(Uc[k - 1]))
        Uc.append([a[i] - b[i] for i in range(len(a))])
    return [[c / (k + 1) for c in Uc[k]] for k in range(d + 1)]

def univariate(coeffs, var):
    p = {}
    for i, c in enumerate(coeffs):
        if c != 0:
            k = [0, 0, 0]; k[var] = i; p[tuple(k)] = c
    return p

def phi_poly(k, LC):
    x = padd(T, pscale(pmul(U, V), -1))                      # t - uv
    s2 = pmul(padd(ONE, pscale(pmul(U, U), -1)), padd(ONE, pscale(pmul(V, V), -1)))
    out = {}
    for m in range(k % 2, k + 1, 2):
        if LC[k][m] != 0:
            out = padd(out, pscale(pmul(ppow(x, m), ppow(s2, (k - m) // 2)), LC[k][m]))
    return out

def permute(p, perm):
    """Rename the variables: monomial exponent e -> e[perm]."""
    return {(e[perm[0]], e[perm[1]], e[perm[2]]): c for e, c in p.items()}

PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

def chebyshev(n):
    """T_0, ..., T_n as exact coefficient lists."""
    Tc = [[Fr(1)], [Fr(0), Fr(1)]]
    for k in range(1, n):
        a = [Fr(0)] + [2 * c for c in Tc[k]]
        b = Tc[k - 1] + [Fr(0)] * (len(a) - len(Tc[k - 1]))
        Tc.append([a[i] - b[i] for i in range(len(a))])
    return Tc[:n + 1]

def S_entry(k, i, j, phik, TC):
    y = pmul(pmul(univariate(TC[i], 0), univariate(TC[j], 1)), phik)
    s = {}
    for perm in PERMS: s = padd(s, permute(y, perm))
    return pscale(s, Fr(1, 6))

def at_1uu(p):
    """p(1, u, u) as a polynomial in u."""
    r = {}
    for (a, b, c), co in p.items():
        k = (b + c, 0, 0); r[k] = r.get(k, 0) + co
    return {k: c for k, c in r.items() if c != 0}

def at_111(p):
    return sum(p.values(), Fr(0))

def build_P(d, f, F):
    LC = legendre_coeffs(d); G = gegenbauer_S3(d); TC = chebyshev(d)
    P = {}
    for k in range(d + 1):
        if f[k] != 0:
            for var in range(3):
                P = padd(P, pscale(univariate(G[k], var), f[k]))
    F111 = Fr(0)
    for k in range(d + 1):
        phik = phi_poly(k, LC); n = d - k + 1
        for i in range(n):
            for j in range(n):
                if F[k][i][j] == 0: continue
                s = S_entry(k, i, j, phik, TC)
                P = padd(P, pscale(s, F[k][i][j]))
                s1 = at_1uu(s)
                for perm in ((0, 1, 2), (1, 0, 2), (2, 1, 0)):      # u -> u, v, t
                    P = padd(P, pscale(permute(s1, perm), F[k][i][j] / (N - 2)))
                F111 += F[k][i][j] * at_111(s)
    return P, F111

# ---------------------------------------------------------------- exact linear algebra
def ldl_positive(M):
    """Exact LDL^T over the rationals; True iff every pivot is positive."""
    n = len(M); A = [[Fr(x) for x in row] for row in M]
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]: return False
    for k in range(n):
        piv = A[k][k]
        if piv <= 0: return False
        for i in range(k + 1, n):
            m = A[i][k] / piv
            for j in range(k + 1, n):
                A[i][j] -= m * A[k][j]
    return True

# ---------------------------------------------------------------- closed forms in interval arithmetic
def closed_forms():
    """omega(gamma), d omega/du and d^2 omega/du^2 (u = cos gamma) from the
    closed form of the lens integral, differentiated symbolically, as
    functions of interval arguments; and A_*."""
    import sympy as sp
    mp.dps = 40
    g, rs = sp.symbols('g rs', positive=True)
    I_lin = lambda r: 1 / sp.cos(r) ** 4 / 4
    I_r = lambda r: r / sp.cos(r) ** 4 / 4 - (sp.tan(r) + sp.tan(r) ** 3 / 3) / 4
    I_sin2r = lambda r: 2 * sp.tan(r) ** 3 / 3
    I_sin2 = lambda r: sp.tan(r) ** 4 / 4
    K = -g / 4 + sp.sin(g) / 4 + sp.tan(g / 2) / 2 * sp.sin(g / 2) ** 2
    prim = lambda r: I_r(r) / 2 + K * I_lin(r) - I_sin2r(r) / 4 - sp.tan(g / 2) / 2 * I_sin2(r)
    om = 4 * sp.pi * (prim(rs) - prim(g / 2))
    om_g = sp.diff(om, g); om_gg = sp.diff(om_g, g)
    om_u = -om_g / sp.sin(g)                                   # d/du = -(1/sin g) d/dg
    om_uu = om_gg / sp.sin(g) ** 2 - om_g * sp.cos(g) / sp.sin(g) ** 3
    ns = [{'sin': iv.sin, 'cos': iv.cos, 'tan': iv.tan, 'pi': iv.pi, 'sqrt': iv.sqrt}]
    f0 = sp.lambdify((g, rs), om, modules=ns)
    f1 = sp.lambdify((g, rs), om_u, modules=ns)
    f2 = sp.lambdify((g, rs), om_uu, modules=ns)
    r_star = iv.atan2(iv.mpf(1), iv.sqrt(2))       # tan r_* = 1/sqrt2, that is sin r_* = 1/sqrt3
    pi = iv.pi
    def A_star():
        def I_lin(r): return 1 / iv.cos(r) ** 4 / 4
        def I_r(r): return r / iv.cos(r) ** 4 / 4 - (iv.tan(r) + iv.tan(r) ** 3 / 3) / 4
        def I_sin2r(r): return 2 * iv.tan(r) ** 3 / 3
        def prim(r): return 4 * (2 * pi ** 2 * I_lin(r) - 23 * pi * (2 * I_r(r) - I_sin2r(r)))
        return (2 * pi ** 2 + prim(r_star) - prim(iv.mpf(0))) / 4
    return (lambda gg: f0(gg, r_star)), (lambda gg: f1(gg, r_star)), (lambda gg: f2(gg, r_star)), A_star, r_star

def omega_tables(om0, om1, om2, r_star, n=12000, n2=3000):
    """Angles gamma_i from 2 r_* down to 60 degrees.  us[i] is an upper bound
    of cos(gamma_i), olow[i] a lower bound of omega(gamma_i), d1lo/d1hi[i]
    enclose d omega/du at gamma_i, du is an upper bound of every cell width
    in u, and M2 bounds |d^2 omega/du^2| on [1/3, 1/2] (interval evaluation
    on n2 subintervals).  For u >= us[i] the angle arccos u is at most
    gamma_i, so omega(u) >= omega(gamma_i) >= olow[i] since omega increases
    with u; and |omega'(u) - omega'(u_i)| <= M2 (u - u_i)."""
    g_hi = float(np.nextafter(float((2 * r_star).b), np.inf))
    g_lo = float(np.pi / 3)
    gams = np.linspace(g_hi, g_lo, n + 1)
    us = np.zeros(n + 1); olow = np.zeros(n + 1); ohigh = np.zeros(n + 1); d1lo = np.zeros(n + 1); d1hi = np.zeros(n + 1)
    for i, g in enumerate(gams):
        gi = iv.mpf(g)
        us[i] = np.nextafter(float(iv.cos(gi).b), np.inf)
        w = om0(gi); olow[i] = max(np.nextafter(float(w.a), -np.inf), 0.0); ohigh[i] = max(np.nextafter(float(w.b), np.inf), 0.0)
        w1 = om1(gi); d1lo[i] = np.nextafter(float(w1.a), -np.inf); d1hi[i] = np.nextafter(float(w1.b), np.inf)
    assert np.all(np.diff(us) > 0) and np.all(np.diff(olow) >= 0)
    du = np.nextafter(float(np.max(np.diff(us))) + 1e-12, np.inf)
    d1lo[0] = min(d1lo[0], -1e-300)                 # the first angle may lie beyond 2 r_*, where omega' = 0
    M2 = 0.0
    edges = np.linspace(g_hi, g_lo, n2 + 1)
    for a, b in zip(edges[1:], edges[:-1]):
        w2 = om2(iv.mpf([a, b]))
        M2 = max(M2, abs(float(w2.a)), abs(float(w2.b)))
    M2 = np.nextafter(M2, np.inf)
    return us, olow, ohigh, d1lo, d1hi, du, M2

# ---------------------------------------------------------------- interval arithmetic on arrays
def down(x): return np.nextafter(x, -np.inf)
def up(x): return np.nextafter(x, np.inf)

def iadd(a, b): return down(a[0] + b[0]), up(a[1] + b[1])
def isub(a, b): return down(a[0] - b[1]), up(a[1] - b[0])
def imul(a, b):
    p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return down(np.minimum.reduce(p)), up(np.maximum.reduce(p))
def ipow_list(a, n):
    out = [(np.ones_like(a[0]), np.ones_like(a[0]))]
    for _ in range(n): out.append(imul(out[-1], a))
    return out

class Poly:
    """A polynomial with interval coefficients, for evaluation on boxes."""
    def __init__(self, p):
        items = sorted(p.items())
        self.exps = np.array([e for e, _ in items], dtype=int)
        cf = np.array([float(c) for _, c in items])
        self.clo, self.chi = down(cf), up(cf)
        self.deg = self.exps.max(axis=0)
    def eval(self, box):
        """box = (lo, hi) arrays of shape (3, n); returns (lo, hi) of the polynomial."""
        pw = [ipow_list((box[0][v], box[1][v]), int(self.deg[v])) for v in range(3)]
        lo = np.zeros(box[0].shape[1]); hi = np.zeros(box[0].shape[1])
        for (a, b, c), cl, ch in zip(self.exps, self.clo, self.chi):
            m = imul(imul(pw[0][a], pw[1][b]), pw[2][c])
            m = imul(m, (np.full_like(lo, cl), np.full_like(hi, ch)))
            lo, hi = iadd((lo, hi), m)
        return lo, hi

def derivative(p, var):
    r = {}
    for e, c in p.items():
        if e[var] > 0:
            k = list(e); k[var] -= 1; r[tuple(k)] = r.get(tuple(k), 0) + c * e[var]
    return r

def verify_domain(P, tables, wmin, batch=150000, start=None):
    """Interval branch and bound of (cert) on the ordered admissible domain.

    On a box with centre c and half-widths r, with Q = 1000 (omega(u) +
    omega(v) + omega(t)) - P, Taylor's theorem gives
        Q(x) >= Q(c) - sum_v |dQ/dx_v (c)| r_v
                     - (1/2) sum_{v,w} sup_box |d^2 Q/dx_v dx_w| r_v r_w ,
    the second derivatives of the omega terms bounded by M2 and those of P
    enclosed by interval evaluation on the box.  A variable whose range is
    not inside [1/3, 1/2] has its omega term replaced by 0 (a lower bound),
    with derivatives 0.  Q(c) is bounded below through the tables."""
    us, olow, ohigh, d1lo, d1hi, du, M2 = tables
    olow = down(SCALE * olow); ohigh = up(SCALE * ohigh); d1lo = down(SCALE * d1lo); d1hi = up(SCALE * d1hi)
    M2 = up(SCALE * M2); dM = up(M2 * du)
    Pp = Poly(P); D = [Poly(derivative(P, v)) for v in range(3)]
    H = {(v, w): Poly(derivative(derivative(P, v), w)) for v in range(3) for w in range(v, 3)}
    det = Poly({(0, 0, 0): Fr(1), (1, 1, 1): Fr(2), (2, 0, 0): Fr(-1), (0, 2, 0): Fr(-1), (0, 0, 2): Fr(-1)})
    lo = np.array([[-1.0], [-1.0], [-1.0]]); hi = np.array([[0.5], [0.5], [0.5]])
    if start is not None: lo = np.array(start[0]).reshape(3, 1); hi = np.array(start[1]).reshape(3, 1)
    t0 = time.time(); level = 0; n_done = 0; n_out = 0; worst = np.inf
    while lo.shape[1] > 0:
        n = lo.shape[1]; keep_lo = []; keep_hi = []
        for s in range(0, n, batch):
            L, Hh = lo[:, s:s + batch], hi[:, s:s + batch]
            alive = (L[0] <= Hh[1]) & (L[1] <= Hh[2])
            dlo, dhi = det.eval((L, Hh))
            alive &= dhi >= 0
            n_out += int(np.sum(~alive))
            c = (L + Hh) / 2; r = np.maximum(up(Hh - c), up(c - L))      # covers [L, Hh] although c is rounded
            # the omega terms at the centre, and their first derivatives
            q0 = np.zeros(c.shape[1]); q0hi = np.zeros(c.shape[1]); grad_lo = np.zeros_like(c); grad_hi = np.zeros_like(c); m2 = np.zeros_like(c)
            for v in range(3):
                inside = L[v] >= THIRD
                idx = np.clip(np.searchsorted(us, c[v], side='right') - 1, 0, len(us) - 1)
                q0 = down(q0 + np.where(inside, olow[idx], 0.0))
                q0hi = up(q0hi + np.where(c[v] >= THIRD, ohigh[np.minimum(idx + 1, len(us) - 1)], 0.0))
                grad_lo[v] = np.where(inside, down(d1lo[idx] - dM), 0.0)
                grad_hi[v] = np.where(inside, up(d1hi[idx] + dM), 0.0)
                m2[v] = np.where(inside, M2, 0.0)
            # P and its gradient at the centre, its Hessian on the box
            plo, phi = Pp.eval((c, c))
            q0 = down(q0 - phi); q0hi = up(q0hi - plo)
            # a centre inside the domain at which (cert) fails is a counterexample
            dclo, _ = det.eval((c, c))
            bad = alive & (dclo >= 0) & (c[0] <= c[1]) & (c[1] <= c[2]) & (q0hi < 0)
            if np.any(bad):
                i = int(np.argmax(bad))
                print(f"  FAILED: (cert) is negative at the admissible triple ({c[0][i]:.9f}, {c[1][i]:.9f}, {c[2][i]:.9f}): "
                      f"at most {q0hi[i] / SCALE:.3e}")
                return False
            first = np.zeros(c.shape[1]); contrib = np.zeros_like(c)
            g_abs = np.zeros_like(c)
            for v in range(3):
                glo, ghi = D[v].eval((c, c))
                dq_lo = down(grad_lo[v] - ghi); dq_hi = up(grad_hi[v] - glo)
                g_abs[v] = np.maximum(np.abs(dq_lo), np.abs(dq_hi))
                first = up(first + up(g_abs[v] * r[v]))
                contrib[v] = up(g_abs[v] * r[v])
            second = np.zeros(c.shape[1])
            for (v, w), Hp in H.items():
                hlo, hhi = Hp.eval((L, Hh))
                habs = np.maximum(np.abs(hlo), np.abs(hhi))
                if v == w:
                    term = up(up(habs + m2[v]) * up(r[v] * r[v]))
                    second = up(second + term); contrib[v] = up(contrib[v] + term)
                else:
                    term = up(up(2 * habs) * up(r[v] * r[w]))
                    second = up(second + term)
                    contrib[v] = up(contrib[v] + term); contrib[w] = up(contrib[w] + term)
            qlo = down(down(q0 - first) - up(second / 2))
            ok = qlo >= 0
            worst = min(worst, float(np.min(np.where(alive, qlo, np.inf))))
            n_done += int(np.sum(alive & ok))
            undecided = alive & ~ok
            if np.any(undecided):
                w_ = Hh - L
                if np.max(w_[:, undecided]) < wmin:
                    i = int(np.argmax(undecided))
                    print(f"  FAILED: undecided box of width below {wmin} at u,v,t in "
                          f"[{L[0][i]:.6f},{Hh[0][i]:.6f}] x [{L[1][i]:.6f},{Hh[1][i]:.6f}] x [{L[2][i]:.6f},{Hh[2][i]:.6f}]; "
                          f"lower bound of (cert) there {qlo[i] / SCALE:.3e}, value at the centre at least {q0[i] / SCALE:.3e}")
                    return False
                Lu, Hu = L[:, undecided], Hh[:, undecided]
                axis = np.argmax(contrib[:, undecided], axis=0)
                mid = (Lu + Hu) / 2
                A_hi = Hu.copy(); B_lo = Lu.copy()
                for v in range(3):
                    sel = axis == v
                    A_hi[v, sel] = mid[v, sel]; B_lo[v, sel] = mid[v, sel]
                keep_lo += [Lu, B_lo]; keep_hi += [A_hi, Hu]
        level += 1
        lo = np.concatenate(keep_lo, axis=1) if keep_lo else np.zeros((3, 0))
        hi = np.concatenate(keep_hi, axis=1) if keep_hi else np.zeros((3, 0))
        print(f"  level {level:2d}: {n_done} boxes verified, {n_out} outside the domain, "
              f"{lo.shape[1]} to bisect, least lower bound so far {worst / SCALE:.3e}  [{time.time() - t0:.0f}s]", flush=True)
    return True

def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    wmin = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
    Z = np.load(f"continuation_out/certificate_d{d}.npz")
    f = [Fr(float(x)) for x in Z['f']]
    F = [[[Fr(float(x)) for x in row] for row in Z[f'F{k}']] for k in range(d + 1)]
    print(f"certificate of degree {d}: {d + 1} coefficients f_k, matrices F_k of sizes {[len(M) for M in F]}")
    # 1. positivity
    f = [f[0]] + [max(x, Fr(0)) for x in f[1:]]
    for k in range(d + 1):
        if not ldl_positive(F[k]):
            print(f"  FAILED: F_{k} is not positive definite"); return
    print("  step 1: f_k >= 0 for k >= 1 and every F_k positive definite (exact LDL^T)")
    # 2. the bound
    P, F111 = build_P(d, f, F)
    B = N * (N * f[0] - sum(f)) / 2 - N * F111 / (6 * (N - 2))
    om0, om1, om2, A_star, r_star = closed_forms()
    A = A_star()
    target_hi = 8 - A.a                                   # rigorous upper bound of 8 - A_*
    Bf = B / SCALE
    print(f"  step 2: exact bound B / 1000 = {float(Bf):.9f}; 8 - A_* in [{float(8 - A.b):.12f}, {float(8 - A.a):.12f}]")
    th = float(np.nextafter(float(target_hi), np.inf))    # a float above 8 - A_*
    if not (Bf > Fr(th)):
        print("  FAILED: the bound does not exceed 8 - A_*")
        if not (len(sys.argv) > 3 and sys.argv[3] == 'continue'): return
    print(f"  the bound exceeds 8 - A_* by {float(Bf - Fr(th)):.3e}")
    # 3. the inequality (cert) on the domain
    for e, c in P.items():
        for perm in PERMS:
            assert P.get((e[perm[0]], e[perm[1]], e[perm[2]]), 0) == c, "P is not symmetric"
    print(f"  step 3: P has {len(P)} monomials, degree {max(sum(e) for e in P)}; P is symmetric")
    tables = omega_tables(om0, om1, om2, r_star)
    us, olow, ohigh, d1lo, d1hi, du, M2 = tables
    print(f"  omega tabulated at {len(us)} angles (cells of width at most {du:.2e} in u); at 60 degrees omega >= {olow[-1]:.10f}, "
          f"omega' in [{d1lo[-1]:.8f}, {d1hi[-1]:.8f}]; |omega''| <= {M2:.6f} on [1/3, 1/2]")
    ok = verify_domain(P, tables, wmin)
    if ok:
        print("PASS: (cert) holds on the admissible domain; every contact configuration of 23 directions "
              f"satisfies sum omega(gamma_ij) >= {float(Bf):.9f} > 8 - A_*.")
    else:
        print("FAILED: the certificate could not be verified.")

if __name__ == "__main__":
    main()
