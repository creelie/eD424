#!/usr/bin/env python3
"""
typed_cardinality_check.py -- rigorous verification of a certificate written by
typed_cardinality_sdp.py: no code on S^3 consists of n points with pairwise
inner products at most t1 and one further point with inner product at most t2
with each of them.

Input: the .npz the programme saves (d, t1, t2, n, f, F0..Fd).  The floats are
made exact as follows, and from then on nothing is rounded:

  * f_k is replaced by max(f_k, 0) and each F_k by its eigenvalue clipping
    at 1e-10 (a positive definite matrix), both read as the exact rationals
    the doubles denote;
  * the four bounds g <= -1 + e1 on [-1, t1], g <= a2 on [-1, t2], F <= b1 on
    the admissible triples of type 111 and F <= b2 on those of type 112 are
    chosen as doubles a little above the largest values found in floating
    point (typed_cardinality_sdp.evaluate on the exact certificate), and then
    proved by the branch and bounds of certify_cardinality.py: g in interval
    arithmetic on [-1, t1] and [-1, t2], F by second-order Taylor forms on
    the ordered domain -1 <= u <= v <= w <= t1 (type 111) and on the domain
    -1 <= u <= v <= t2, -1 <= w <= t1 (type 112; F is symmetric, so u <= v
    suffices), with the Gram condition 1 + 2uvw - u^2 - v^2 - w^2 >= 0;
  * A = f(1) + F(1,1,1) exactly, and
        Z = (n+1) A + n(n-1)(e1 - 1) + 2n a2 + n(n-1)(n-2) b1 + 3n(n-1) b2
    exactly, with the doubles that the branch and bounds proved.

Z < 0 proves that no such code exists (typed_cardinality_sdp.py derives the
inequality Z >= 0 for every such code).  The thresholds t1, t2 are the
decimals recorded in the file, taken exactly; the boxes run to the least
doubles at or above them.

Usage: python3 typed_cardinality_check.py cert.npz [margin] [margin_111 margin_112]

The margin is added to the floating-point maxima to give the thresholds; the
two further margins, when given, replace it for F on the triples of type 111
and 112, whose thresholds enter Z with the factors n(n-1)(n-2) and 3n(n-1).
A larger margin leaves less of Z below 0 but lets the branch and bound close
with larger boxes.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import numpy as np
from certify_cardinality import build, top, verify_1d
from certificate_check import ldl_positive, Poly, derivative, up
import typed_cardinality_sdp as T
from three_point_sdp import leg_coeffs


def above(x):
    """the least double >= the rational or float x"""
    y = float(x)
    return y if Fr(y) >= Fr(x) else float(np.nextafter(y, np.inf))


def verify_3d(P2, tops, ordered, e2, wmin=1e-5, batch=150000):
    """P2 <= e2 on the admissible triples in the box [-1,tops[0]] x [-1,tops[1]] x [-1,tops[2]]
    with u <= v, and also v <= w if ordered, by interval branch and bound (second-order
    Taylor form about the centre of each box, as in certify_cardinality.verify_3d)."""
    Pp = Poly(P2); D = [Poly(derivative(P2, v)) for v in range(3)]
    H = {(v, w): Poly(derivative(derivative(P2, v), w)) for v in range(3) for w in range(v, 3)}
    det = Poly({(0, 0, 0): Fr(1), (1, 1, 1): Fr(2), (2, 0, 0): Fr(-1), (0, 2, 0): Fr(-1), (0, 0, 2): Fr(-1)})
    lo = np.array([[-1.0], [-1.0], [-1.0]]); hi = np.array([[tops[0]], [tops[1]], [tops[2]]])
    t0 = time.time(); level = 0; n_done = 0; n_out = 0; worst = -np.inf
    while lo.shape[1] > 0:
        n = lo.shape[1]; keep_lo = []; keep_hi = []
        for s in range(0, n, batch):
            L, Hh = lo[:, s:s + batch], hi[:, s:s + batch]
            alive = L[0] <= Hh[1]
            if ordered:
                alive &= L[1] <= Hh[2]
            _, dhi = det.eval((L, Hh))
            alive &= dhi >= 0
            n_out += int(np.sum(~alive))
            c = (L + Hh) / 2; r = np.maximum(up(Hh - c), up(c - L))      # covers [L, Hh] although c is rounded
            _, p_c = Pp.eval((c, c))
            dclo, _ = det.eval((c, c))
            inside = alive & (dclo >= 0) & (c[0] <= c[1])
            if ordered:
                inside &= c[1] <= c[2]
            worst = max(worst, float(np.max(np.where(inside, p_c, -np.inf), initial=-np.inf)))
            if worst > e2:
                i = int(np.argmax(np.where(inside, p_c, -np.inf)))
                print(f"  FAILED: F = {p_c[i]:.6e} > {e2:.6e} at the admissible triple "
                      f"({c[0][i]:.8f},{c[1][i]:.8f},{c[2][i]:.8f})")
                return False, worst, n_done
            first = np.zeros(c.shape[1]); contrib = np.zeros_like(c)
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
                    print(f"  FAILED: undecided box below width {wmin} at [{L[0][i]:.6f},{Hh[0][i]:.6f}]x"
                          f"[{L[1][i]:.6f},{Hh[1][i]:.6f}]x[{L[2][i]:.6f},{Hh[2][i]:.6f}]; upper {upper[i]:.3e}")
                    return False, worst, n_done
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
                  f"largest centre value {worst:.4e}  [{time.time() - t0:.0f}s]", flush=True)
    return True, worst, n_done


def main():
    path = sys.argv[1]
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-6
    m111 = float(sys.argv[3]) if len(sys.argv) > 3 else margin
    m112 = float(sys.argv[4]) if len(sys.argv) > 4 else margin
    Z_ = np.load(path)
    d = int(Z_['d']); n = int(Z_['n'])
    t1 = Fr(repr(float(Z_['t1']))); t2 = Fr(repr(float(Z_['t2'])))
    fv = np.maximum(np.array(Z_['f'], dtype=float), 0.0)
    Fv = []
    for k in range(d + 1):
        M = np.array(Z_[f'F{k}'], dtype=float); M = (M + M.T) / 2
        lam, V = np.linalg.eigh(M)
        M = (V * np.maximum(lam, 1e-10)) @ V.T
        Fv.append((M + M.T) / 2)
    print(f"typed certificate: degree {d}, n = {n}, t1 = {t1}, t2 = {t2} (exact); boxes to {top(t1)!r}, {top(t2)!r}")
    t0 = time.time()
    f = [Fr(float(x)) for x in fv]
    F = [[[Fr(float(x)) for x in row] for row in Fv[k]] for k in range(d + 1)]
    for k in range(d + 1):
        m = len(F[k])
        for i in range(m):
            for j in range(i + 1, m):
                a = (F[k][i][j] + F[k][j][i]) / 2; F[k][i][j] = F[k][j][i] = a
    print("1. positivity:")
    for k in range(d + 1):
        assert ldl_positive(F[k]), f"F_{k} not positive definite"
    print(f"   f_k >= 0, F_0..F_{d} positive definite (exact LDL^T)  [{time.time() - t0:.0f}s]", flush=True)
    P1, P2, F111 = build(d, f, F)                 # P1 = g + 1, P2 = F, exactly
    A = sum(f) + F111
    print(f"2. A = f(1) + F(1,1,1) = {float(A):.9f} (exact; {len(P2)} monomials in F)  [{time.time() - t0:.0f}s]", flush=True)
    # thresholds from floating point on the exact certificate (the doubles of f and F above)
    LC = [leg_coeffs(k) for k in range(d + 1)]
    R = T.Rows(d, LC)
    Zf, Af, out = T.evaluate(R, fv, Fv, float(t1), float(t2), n)
    e1 = above(out['g1'][0] + 1 + margin)             # P1 <= e1 on [-1, t1]:  g <= e1 - 1
    e1b = above(out['g2'][0] + 1 + margin)            # P1 <= e1b on [-1, t2]: g <= e1b - 1 = a2
    b1 = above(out['F1'][0] + m111)
    b2 = above(out['F2'][0] + m112)
    print(f"   floating point: Z = {Zf:.6f}; thresholds g <= {e1 - 1:.6e} on [-1,t1], g <= {e1b - 1:.6e} on [-1,t2], "
          f"F <= {b1:.6e} (111), F <= {b2:.6e} (112)", flush=True)
    Zx = (n + 1) * A + n * (n - 1) * (Fr(e1) - 1) + 2 * n * (Fr(e1b) - 1) \
        + n * (n - 1) * (n - 2) * Fr(b1) + 3 * n * (n - 1) * Fr(b2)
    print(f"   Z with these thresholds, exactly: {float(Zx):.6f}")
    if Zx >= 0:
        print("FAILED: the thresholds do not give Z < 0"); sys.exit(1)
    print("3. g on [-1, t1]:")
    if not verify_1d(P1, t1, e1):
        print("FAILED"); sys.exit(1)
    print("4. g on [-1, t2]:")
    if not verify_1d(P1, t2, e1b):
        print("FAILED"); sys.exit(1)
    print("5. F on the triples of type 111:")
    ok, w, nb = verify_3d(P2, (top(t1),) * 3, True, b1)
    if not ok:
        print("FAILED"); sys.exit(1)
    print(f"   verified on {nb} boxes")
    print("6. F on the triples of type 112:")
    ok, w, nb = verify_3d(P2, (top(t2), top(t2), top(t1)), False, b2)
    if not ok:
        print("FAILED"); sys.exit(1)
    print(f"   verified on {nb} boxes")
    print(f"PASS: Z = {float(Zx):.6f} < 0 exactly, so no {n} points of S^3 with pairwise inner products at most "
          f"{float(t1)} admit a further point with inner product at most {float(t2)} with each of them  "
          f"[{time.time() - t0:.0f}s]")


if __name__ == '__main__':
    main()
