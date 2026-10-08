#!/usr/bin/env python3
"""
certify_cardinality.py -- rigorous verification of a three-point certificate
for  A(4, t) <= 24,  t = 1/2 + s.

Input: cert_d<d>_t<t>.npz from cardinality_sdp.py (certificate mode): floats
f_1..f_d and symmetric F_0..F_d, taken as the exact rationals they denote.

Established, in exact rational arithmetic and in outward-rounded interval
arithmetic (the interval machinery is that of certificate_check.py in the
D4 code package, reused unchanged):

  1. f_k >= 0 and every F_k positive definite (exact LDL^T, all pivots > 0).
  2. B = 1 + f(1) + F(1,1,1), exactly.
  3. (i)  P1(u) := f(u) + 3 F(u,u,1) + 1 <= e1  on [-1, t]     (1-D branch and bound)
  4. (ii) P2(u,v,w) := F(u,v,w) <= e2  on the admissible domain
          -1 <= u <= v <= w <= t,  1 + 2uvw - u^2 - v^2 - w^2 >= 0  (3-D branch and bound;
          P2 is symmetric, so the ordered domain suffices)
  5. Conclusion: every code C on S^3 with pairwise inner products <= t satisfies
          (|C|-1)(1 - e1) - (|C|-1)(|C|-2) e2 <= B - 1 ,
     and if that fails at |C| = N then |C| <= N - 1 (N = 25 unless given).

The threshold t is the decimal the certificate file records (0.5065, say), taken
exactly; the branch and bound runs to the least double >= t, so that the domain it
covers contains [-1, t] even where the double nearest t lies below it.

Usage: python3 certify_cardinality.py cert.npz [e1] [e2] [wmin] [N]
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import numpy as np
import certificate_check as CC          # exact polynomial + interval machinery
from certificate_check import (padd, pscale, pmul, univariate, gegenbauer_S3, legendre_coeffs,
                               chebyshev, phi_poly, S_entry, at_1uu, at_111, ldl_positive,
                               Poly, derivative, down, up)


def build(d, f, F):
    """P2 = F(u,v,w) exactly; P1(u) = f(u) + 3 F(u,u,1) + 1 exactly (as a univariate list); F111."""
    LC = legendre_coeffs(d); G = gegenbauer_S3(d); TC = chebyshev(d)
    P2 = {}
    P1 = {}
    for k in range(1, d + 1):
        P1 = padd(P1, pscale(univariate(G[k], 0), f[k - 1]))
    F111 = Fr(0)
    for k in range(d + 1):
        phik = phi_poly(k, LC); n = d - k + 1
        for i in range(n):
            for j in range(n):
                if F[k][i][j] == 0:
                    continue
                s = S_entry(k, i, j, phik, TC)
                P2 = padd(P2, pscale(s, F[k][i][j]))
                P1 = padd(P1, pscale(at_1uu(s), 3 * F[k][i][j]))
                F111 += F[k][i][j] * at_111(s)
    P1 = padd(P1, {(0, 0, 0): Fr(1)})
    return P1, P2, F111


def top(t):
    """the least double >= t"""
    x = float(t)
    return x if Fr(x) >= t else float(np.nextafter(x, np.inf))


def verify_1d(P1, t, e1, wmin=1e-7):
    """P1(u) <= e1 on [-1, t] by interval branch and bound (P1 univariate in var 0)."""
    Pp = Poly(P1); D = Poly(derivative(P1, 0)); H = Poly(derivative(derivative(P1, 0), 0))
    lo = np.array([-1.0]); hi = np.array([top(t)])
    worst = -np.inf; n_ok = 0; t0 = time.time()
    while lo.size:
        L = np.stack([lo, np.zeros_like(lo), np.zeros_like(lo)]); Hh = np.stack([hi, np.zeros_like(hi), np.zeros_like(hi)])
        c = (L + Hh) / 2; r = np.maximum(up(Hh - c), up(c - L))      # covers [L, Hh] although c is rounded
        _, p_c = Pp.eval((c, c))
        glo, ghi = D.eval((c, c)); g = np.maximum(np.abs(glo), np.abs(ghi))
        hlo, hhi = H.eval((L, Hh)); h = np.maximum(np.abs(hlo), np.abs(hhi))
        upper = up(up(p_c + up(g * r[0])) + up(h * up(r[0] * r[0]) / 2))
        worst = max(worst, float(p_c.max()))
        ok = upper <= e1
        n_ok += int(ok.sum())
        und = ~ok
        if np.any(und) and np.max(hi[und] - lo[und]) < wmin:
            i = int(np.argmax(und))
            print(f"  (i) FAILED: undecided interval [{lo[i]:.8f},{hi[i]:.8f}], upper bound {upper[i]:.3e}, value at centre {p_c[i]:.3e}")
            return False
        mid = (lo[und] + hi[und]) / 2
        lo, hi = np.r_[lo[und], mid], np.r_[mid, hi[und]]
    print(f"  (i) verified: P1 <= {e1:.2e} on [-1,{top(t)!r}]; {n_ok} intervals, largest centre value {worst:.3e}  [{time.time()-t0:.0f}s]")
    return True


def verify_3d(P2, t, e2, wmin=1e-5, batch=150000):
    """P2 <= e2 on the ordered admissible domain in [-1,t]^3 by interval branch and bound."""
    Pp = Poly(P2); D = [Poly(derivative(P2, v)) for v in range(3)]
    H = {(v, w): Poly(derivative(derivative(P2, v), w)) for v in range(3) for w in range(v, 3)}
    det = Poly({(0, 0, 0): Fr(1), (1, 1, 1): Fr(2), (2, 0, 0): Fr(-1), (0, 2, 0): Fr(-1), (0, 0, 2): Fr(-1)})
    tf = top(t)
    lo = np.array([[-1.0], [-1.0], [-1.0]]); hi = np.array([[tf], [tf], [tf]])
    t0 = time.time(); level = 0; n_done = 0; n_out = 0; worst = -np.inf
    while lo.shape[1] > 0:
        n = lo.shape[1]; keep_lo = []; keep_hi = []
        for s in range(0, n, batch):
            L, Hh = lo[:, s:s + batch], hi[:, s:s + batch]
            alive = (L[0] <= Hh[1]) & (L[1] <= Hh[2])
            _, dhi = det.eval((L, Hh))
            alive &= dhi >= 0
            n_out += int(np.sum(~alive))
            c = (L + Hh) / 2; r = np.maximum(up(Hh - c), up(c - L))      # covers [L, Hh] although c is rounded
            _, p_c = Pp.eval((c, c))
            dclo, _ = det.eval((c, c))
            inside = alive & (dclo >= 0) & (c[0] <= c[1]) & (c[1] <= c[2])
            worst = max(worst, float(np.max(np.where(inside, p_c, -np.inf))))
            if worst > e2:
                i = int(np.argmax(np.where(inside, p_c, -np.inf)))
                print(f"  (ii) FAILED: P2 = {p_c[i]:.3e} > e2 at admissible ({c[0][i]:.8f},{c[1][i]:.8f},{c[2][i]:.8f})")
                return False, worst
            first = np.zeros(c.shape[1])
            contrib = np.zeros_like(c)
            for v in range(3):
                glo, ghi = D[v].eval((c, c))
                g = np.maximum(np.abs(glo), np.abs(ghi))
                term = up(g * r[v]); first = up(first + term); contrib[v] = term
            second = np.zeros(c.shape[1])
            for (v, w), Hp in H.items():
                hlo, hhi = Hp.eval((L, Hh))
                habs = np.maximum(np.abs(hlo), np.abs(hhi))
                if v == w:
                    term = up(habs * up(r[v] * r[v])); second = up(second + term); contrib[v] = up(contrib[v] + term)
                else:
                    term = up(up(2 * habs) * up(r[v] * r[w])); second = up(second + term)
                    contrib[v] = up(contrib[v] + term); contrib[w] = up(contrib[w] + term)
            upper = up(up(p_c + first) + up(second / 2))
            ok = upper <= e2
            n_done += int(np.sum(alive & ok))
            und = alive & ~ok
            if np.any(und):
                w_ = Hh - L
                if np.max(w_[:, und]) < wmin:
                    i = int(np.argmax(und))
                    print(f"  (ii) FAILED: undecided box below width {wmin} at [{L[0][i]:.6f},{Hh[0][i]:.6f}]x[{L[1][i]:.6f},{Hh[1][i]:.6f}]x[{L[2][i]:.6f},{Hh[2][i]:.6f}]; upper {upper[i]:.3e}, centre {p_c[i]:.3e}")
                    return False, worst
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
        if level % 5 == 0 or lo.shape[1] == 0:
            print(f"  level {level:2d}: {n_done} boxes verified, {n_out} outside, {lo.shape[1]} to bisect, "
                  f"largest centre value {worst:.3e}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"  (ii) verified: P2 <= {e2:.2e} on the admissible domain in [-1,{tf!r}]^3")
    return True, worst


def main():
    path = sys.argv[1]
    def below(txt):                     # the largest double <= the decimal txt, so that "<= e" holds as printed
        x = float(txt)
        return x if Fr(x) <= Fr(txt) else float(np.nextafter(x, -np.inf))
    e1 = below(sys.argv[2]) if len(sys.argv) > 2 else below('1e-6')
    e2 = below(sys.argv[3]) if len(sys.argv) > 3 else below('1e-4')
    wmin = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 25      # the number of points to exclude
    Z = np.load(path)
    d = len(Z['f']); t = Fr(repr(float(Z['t'])))          # the recorded decimal, exactly
    f = [max(Fr(float(x)), Fr(0)) for x in Z['f']]
    F = [[[Fr(float(x)) for x in row] for row in Z[f'F{k}']] for k in range(d + 1)]
    for k in range(d + 1):     # symmetrise exactly
        n = len(F[k])
        for i in range(n):
            for j in range(i + 1, n):
                m = (F[k][i][j] + F[k][j][i]) / 2; F[k][i][j] = F[k][j][i] = m
    print(f"certificate degree {d}, t = {t} = 1/2 + {t - Fr(1, 2)} exactly; the boxes run to {top(t)!r}")
    t0 = time.time()
    print("1. positivity:")
    for k in range(d + 1):
        assert ldl_positive(F[k]), f"F_{k} not positive definite"
    print(f"   f_k >= 0, F_0..F_{d} positive definite (exact LDL^T)  [{time.time()-t0:.0f}s]")
    P1, P2, F111 = build(d, f, F)
    B = 1 + sum(f) + F111
    print(f"2. B = 1 + f(1) + F(1,1,1) = {float(B):.9f}  (exact rational; {len(P2)} monomials in P2)  [{time.time()-t0:.0f}s]")
    print("3. constraint (i):")
    ok1 = verify_1d(P1, t, e1)
    print("4. constraint (ii):")
    ok2, _ = verify_3d(P2, t, e2, wmin)
    if not (ok1 and ok2):
        print("FAILED"); sys.exit(1)
    # 5. conclusion at |C| = N: need (N-1)(1-e1) - (N-1)(N-2) e2 > B - 1
    lhs = Fr(N - 1) * (1 - Fr(e1)) - Fr((N - 1) * (N - 2)) * Fr(e2)   # the doubles the checks used, exactly
    print(f"5. at |C| = {N}:  {N-1}(1-e1) - {(N-1)*(N-2)} e2 = {float(lhs):.6f}  vs  B - 1 = {float(B - 1):.6f}")
    if lhs > B - 1:
        s = t - Fr(1, 2)
        print(f"PASS: A(4, {float(t)}) <= {N-1}.  Any set of points of S^3 with pairwise inner products at most 1/2 + {float(s)} has at most {N-1} points")
        print(f"      (a set of {N} or more would contain {N} points, and the bound excludes |C| = {N});")
        d_ = 2 / np.sqrt(1 - 2 * float(s))
        print(f"      equivalently, in a unit-ball packing of R^4 at most {N-1} other centres lie within distance {d_:.6f} of any centre.")
    else:
        print(f"FAILED: the bound does not exclude {N} points"); sys.exit(1)


if __name__ == '__main__':
    main()
