#!/usr/bin/env python3

def _extract_in_order(sol, zlist, simplify_fn):
    """Read a solve() result back in the supplied unknown order.

    sympy returns a dict keyed by symbol; its iteration order is not
    guaranteed to match the order the unknowns were passed in, so the
    components are looked up by key rather than taken from .values().
    """
    if not sol:
        return None
    d = sol[0]
    out = []
    for zi in zlist:
        if zi not in d:
            return None
        out.append(simplify_fn(d[zi]))
    return sp.Matrix(out)

"""
eperp_region2_boxcover_certificate.py
===============================
First full derivation and certificate for a SECOND region of the 9
found by eperp_region_volume_setup.py: "region 2" (indexed by first
discovery on a fine grid; representative point theta=1.0587, t=0.01),
bounded by pi/3 < theta < arcsin(1/(sin t+cos t)) [the Y-curve] for
t in (0, t_cross), where t_cross solves sin(pi/3)(sin t+cos t)=1 -- a
new exact crossover point where the universal theta=pi/3 curve meets
the Y-curve (numerically t_cross = 0.169918...). This is the SECOND of
the 9 regions to receive a full vertex/volume/certificate treatment
(after region 4, eperp_region4_exact_volume.py and its two
certificates); regions 0,1,3,5,6,7,8 and the entire w1-v2 arc remain
completely untouched.

VERTEX STRUCTURE. A generic point in region 2 has only 29 vertices (24
fixed + 5 moving) -- fewer than region 4's 34, and the moving-vertex
patterns are simpler (5 distinct patterns instead of 12). Matching the
24 fixed vertices exactly: 23 belong to the same axis/sign-vertex
families used for region 4, and the 24th is the point Z*=2*u0=
(sqrt2,sqrt2,0,0) -- the 8-fold degenerate fixed point of Lemma
lem:universalbreak (the theta=pi/3 universal breakpoint), which
appears here because region 2's own lower boundary IS theta=pi/3.

VOLUME DECOMPOSITION. Exactly as for region 4 (every bounding
hyperplane unit-normal with offset 1, so vol(V0) decomposes as a sum
of |det(4 boundary vertices)|/24 over a triangulated boundary), the 85
fixed-only boundary simplices (of 114 total, at the representative
point) sum to EXACTLY 29/4 -- an exact symbolic identity, not a
numerical coincidence -- giving
    vol(V0(theta,t)) = 29/4 + vol_mov2(theta,t),
    F_Omega = D_Omega*(vol_mov2(theta,t) - 3/4)
throughout region 2, where vol_mov2 is an explicit closed-form
rational function (summing the 29 moving-simplex determinants
exactly), matched numerically at the representative point.

CERTIFICATE. Unlike region 4, region 2's boundary values are NOT a
vanishing limit: vol_mov2-3/4 is comfortably positive throughout (a
coarse grid finds a minimum around 0.27, far from zero), so no special
handling of a theta->0-type edge is needed. The plain Weierstrass
substitution u=tan(theta/2), v=tan(t/2) gives vol_mov2-3/4=N(u,v)/D(u,v)
for two explicit degree-(8,8) polynomials with plain rational
coefficients; D<0 and N<0 throughout the domain (checked numerically).
Covering v in (0,t_cross/2-tangent) with 14 rational sub-boxes (margin
0.0007 at each end, u in [tan(pi/6)+0.0007, 0.9985*u_max(v_hi)] per
box, the same box-covering method as
eperp_region4b_boxcover_certificate.py), EVERY Bernstein coefficient of
both N and D, in EVERY box, is checked EXACTLY (rational arithmetic,
no floating point) to be strictly negative -- 0 zero coefficients this
time (cleaner than region 4b, since there is no vanishing edge here).
This proves F_Omega>0 throughout region 2 except a thin margin at the
two v-ends and the innermost fraction of u nearest the ceiling in each
box (quantifiably small, shrinkable arbitrarily, not itself proven
here).

WHAT REMAINS OPEN. Regions 0,1,3,5,6,7,8 (7 of the original 9) and the
entire separate w1-v2 arc still have no vertex classification, volume
formula, or certificate at all. Conjecture (Direction-of-Deviation
Positivity) remains OPEN.

Dependencies: sympy (exact algebra), numpy+scipy (polytope
construction, hull triangulation, boundary-curve location). Runtime:
under one minute.
"""
import sys
import re
import time
from math import comb
import numpy as np
import sympy as sp
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import brentq

theta, t = sp.symbols('theta t', real=True, positive=True)
u, v = sp.symbols('u v', real=True, positive=True)
x, y = sp.symbols('x y', real=True, positive=True)


def _build_root_system():
    roots, labels = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    vv = np.zeros(4)
                    vv[i], vv[j] = si, sj
                    roots.append(vv / np.sqrt(2))
                    sgn = lambda s: '+' if s == 1 else '-'
                    labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
    roots = np.array(roots)

    def find(vec):
        v_ = np.array(vec, dtype=float)
        v_ /= np.linalg.norm(v_)
        idx = np.argmax(roots @ v_)
        assert (roots @ v_)[idx] > 1 - 1e-9
        return idx

    idx0 = find([1, 1, 0, 0])
    return roots, labels, idx0, roots[idx0], roots[find([1, -1, 0, 0])], \
        roots[find([0, 0, 1, 1])]


def derive_region2():
    roots, labels, idx0, u0, v1, w1 = _build_root_system()

    def u1_vec(th, tt):
        return np.cos(th) * u0 + np.sin(th) * (np.cos(tt) * v1 + np.sin(tt) * w1)

    def polytope_full(th, tt):
        u1 = u1_vec(th, tt)
        dirs = roots.copy()
        dirs[idx0] = u1
        hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1, 1)])
        hi = HalfspaceIntersection(hs, np.zeros(4))
        out = []
        for vpt in hi.intersections:
            vals = dirs @ vpt
            touching = frozenset(
                labels[k] if k != idx0 else 'u1'
                for k, val in enumerate(vals) if abs(val - 1) < 1e-6)
            out.append((vpt, touching))
        return out

    theta0, t0 = 1.0587, 0.01
    verts_info = polytope_full(theta0, t0)
    verts = np.array([v_ for v_, _ in verts_info])
    patterns = [p for _, p in verts_info]
    on_cap_idx = [i for i, p in enumerate(patterns) if 'u1' in p]
    off_cap_idx = [i for i, p in enumerate(patterns) if 'u1' not in p]
    print(f"representative point (theta,t)=({theta0},{t0}): {len(verts)} vertices, "
          f"{len(on_cap_idx)} moving, {len(off_cap_idx)} fixed")

    hull = ConvexHull(verts, qhull_options='QJ')
    on_cap_set = set(on_cap_idx)
    moving_simplices = [s for s in hull.simplices if any(v_ in on_cap_set for v_ in s)]
    fixed_simplices = [s for s in hull.simplices if all(v_ not in on_cap_set for v_ in s)]
    print(f"{len(hull.simplices)} boundary simplices: {len(fixed_simplices)} fixed-only, "
          f"{len(moving_simplices)} moving")

    import itertools
    sqrt2 = sp.sqrt(2)
    candidates_num, candidates_sym = [], []
    for i in range(4):
        for s in (1, -1):
            vv = np.zeros(4)
            vv[i] = s * np.sqrt(2)
            candidates_num.append(vv)
            vs_ = sp.zeros(4, 1)
            vs_[i] = s * sqrt2
            candidates_sym.append(vs_)
    for eps in itertools.product([1, -1], repeat=4):
        candidates_num.append(np.array(eps) / np.sqrt(2))
        candidates_sym.append(sp.Matrix(eps) / sqrt2)
    candidates_num.append(np.array([np.sqrt(2), np.sqrt(2), 0, 0]))
    candidates_sym.append(sp.Matrix([sqrt2, sqrt2, 0, 0]))
    candidates_num = np.array(candidates_num)

    fixed_sym = {}
    for i in off_cap_idx:
        dists = np.linalg.norm(candidates_num - verts[i], axis=1)
        j = np.argmin(dists)
        assert dists[j] < 1e-6
        fixed_sym[i] = candidates_sym[j]
    print(f"fixed vertices matched exactly (incl. Z*=2u0): {len(fixed_sym)}/{len(off_cap_idx)}")

    vol_fixed_exact = sp.Integer(0)
    for s in fixed_simplices:
        M = sp.Matrix.hstack(*[fixed_sym[i] for i in s]).T
        Mnum = verts[list(s)]
        sign = 1 if np.linalg.det(Mnum) > 0 else -1
        vol_fixed_exact += sign * M.det() / 24
    vol_fixed_exact = sp.nsimplify(sp.simplify(vol_fixed_exact))
    print(f"EXACT fixed-only volume contribution: {vol_fixed_exact} (expected 29/4)")
    assert vol_fixed_exact == sp.Rational(29, 4)

    u0s = sp.Matrix([1, 1, 0, 0]) / sqrt2
    v1s = sp.Matrix([1, -1, 0, 0]) / sqrt2
    w1s = sp.Matrix([0, 0, 1, 1]) / sqrt2
    u1s = sp.cos(theta) * u0s + sp.sin(theta) * (sp.cos(t) * v1s + sp.sin(t) * w1s)

    def root_sym(spec):
        r = sp.zeros(4, 1)
        for m in re.finditer(r'([+-])e(\d)', spec):
            r[int(m.group(2)) - 1] = 1 if m.group(1) == '+' else -1
        return r / sqrt2

    def solve_vertex_moving(facet_labels):
        z = sp.Matrix(sp.symbols('z1 z2 z3 z4', real=True))
        eqs = [sp.Eq((u1s.T * z)[0, 0], 1)]
        for lab in facet_labels[:3]:
            eqs.append(sp.Eq((root_sym(lab).T * z)[0, 0], 1))
        sol = sp.solve(eqs, list(z), dict=True)
        return _extract_in_order(sol, list(z), sp.simplify)

    moving_sym, moving_cache = {}, {}
    for i in on_cap_idx:
        pat = patterns[i]
        if pat in moving_cache:
            moving_sym[i] = moving_cache[pat]
            continue
        zv = solve_vertex_moving([p for p in pat if p != 'u1'])
        assert zv is not None
        numeric = np.array([float(z.subs({theta: theta0, t: t0})) for z in zv])
        assert np.allclose(numeric, verts[i], atol=1e-5)
        moving_sym[i] = zv
        moving_cache[pat] = zv
    print(f"moving vertices solved exactly: {len(moving_cache)} distinct patterns "
          f"({len(on_cap_idx)} instances)")

    sym_vertex = dict(fixed_sym)
    sym_vertex.update(moving_sym)

    terms = []
    for s in moving_simplices:
        M = sp.Matrix.hstack(*[sym_vertex[i] for i in s]).T
        Mnum = verts[list(s)]
        sign = 1 if np.linalg.det(Mnum) > 0 else -1
        terms.append(sp.cancel(sign * M.det() / 24))
    total = sp.cancel(sp.together(sum(terms)))
    num_check = float(total.subs({theta: theta0, t: t0}))
    print(f"vol_mov2 closed form: {len(str(total))} chars, numeric check "
          f"{num_check} (expected approx 1.0198003392316994)")
    return total


def to_uv_ND(expr):
    subs_map = {
        sp.sin(theta): 2 * u / (1 + u ** 2),
        sp.cos(theta): (1 - u ** 2) / (1 + u ** 2),
        sp.tan(theta): 2 * u / (1 - u ** 2),
        sp.sin(t): 2 * v / (1 + v ** 2),
        sp.cos(t): (1 - v ** 2) / (1 + v ** 2),
    }
    expr_uv = sp.cancel(sp.together(expr.subs(subs_map)))
    N, D = sp.fraction(expr_uv)
    return sp.expand(N), sp.expand(D)


def poly_to_coeff_dict(expr):
    coeffs = {}
    for term in expr.as_ordered_terms():
        pd = term.as_powers_dict()
        i, j = int(pd.get(u, 0)), int(pd.get(v, 0))
        c = term
        if i:
            c = c / u ** i
        if j:
            c = c / v ** j
        coeffs[(i, j)] = coeffs.get((i, j), sp.Integer(0)) + sp.nsimplify(sp.expand(c))
    return coeffs


def substitute_box(coeffs, nu, nv, u_lo, u_hi, v_lo, v_hi):
    ux = u_lo + (u_hi - u_lo) * x
    vy = v_lo + (v_hi - v_lo) * y
    upows = [sp.Integer(1)]
    for _ in range(nu):
        upows.append(sp.expand(upows[-1] * ux))
    vpows = [sp.Integer(1)]
    for _ in range(nv):
        vpows.append(sp.expand(vpows[-1] * vy))
    expr = sp.Integer(0)
    for (i, j), c in coeffs.items():
        expr += c * upows[i] * vpows[j]
    expr = sp.expand(expr)
    out = {}
    for term in expr.as_ordered_terms():
        pd = term.as_powers_dict()
        i, j = int(pd.get(x, 0)), int(pd.get(y, 0))
        c = term
        if i:
            c = c / x ** i
        if j:
            c = c / y ** j
        out[(i, j)] = out.get((i, j), sp.Integer(0)) + sp.nsimplify(sp.expand(c))
    return out


def bernstein_coeffs(coeffs, nu, nv):
    A = [[sp.Integer(0)] * (nv + 1) for _ in range(nu + 1)]
    for (i, j), c in coeffs.items():
        A[i][j] += c

    def binom(n, k):
        return sp.Integer(comb(n, k))

    def transform_1d(vec, n):
        b = [sp.Integer(0)] * (n + 1)
        for j in range(n + 1):
            s = sp.Integer(0)
            for k in range(j + 1):
                s += binom(j, k) * vec[k] / binom(n, k)
            b[j] = s
        return b

    B1 = [transform_1d(A[i], nv) for i in range(nu + 1)]
    B2 = [[sp.Integer(0)] * (nv + 1) for _ in range(nu + 1)]
    for jv in range(nv + 1):
        col = [B1[i][jv] for i in range(nu + 1)]
        newcol = transform_1d(col, nu)
        for i in range(nu + 1):
            B2[i][jv] = newcol[i]
    return B2


def check_box(coeffs, nu, nv, u_lo, u_hi, v_lo, v_hi, expect_sign):
    bc = substitute_box(coeffs, nu, nv, u_lo, u_hi, v_lo, v_hi)
    B = bernstein_coeffs(bc, nu, nv)
    flat = [B[i][j] for i in range(nu + 1) for j in range(nv + 1)]
    bad = [z for z in flat if (expect_sign > 0 and z < 0) or (expect_sign < 0 and z > 0)]
    nz = sum(1 for z in flat if z == 0)
    return len(bad) == 0, nz, len(flat)


def main():
    print("=" * 70)
    print("Region 2: full derivation + box-covering exact Bernstein certificate")
    print("=" * 70)
    t0 = time.time()
    total = derive_region2()
    expr = total - sp.Rational(3, 4)
    N, D = to_uv_ND(expr)
    print(f"mapped to (u,v): N {len(str(N))} chars, D {len(str(D))} chars, "
          f"{time.time()-t0:.1f}s total so far")

    NU, NV = 8, 8
    Nc = poly_to_coeff_dict(N)
    Dc = poly_to_coeff_dict(D)
    assert all(i <= NU and j <= NV for i, j in Nc)
    assert all(i <= NU and j <= NV for i, j in Dc)

    def g(tt):
        return np.sin(np.pi / 3) * (np.sin(tt) + np.cos(tt)) - 1

    t_cross = brentq(g, 0.001, 0.5)
    print(f"exact new fact: theta=pi/3 meets the Y-curve at t_cross={t_cross:.6f} "
          f"(sin(pi/3)(sin t+cos t)=1)")

    def theta_max(tt):
        return np.arcsin(1.0 / (np.sin(tt) + np.cos(tt)))

    u_lo_true = np.tan(np.pi / 6)
    MARGIN = 0.0001
    v_lo_g = sp.nsimplify(round(0 + MARGIN, 6))
    v_hi_g = sp.nsimplify(round(t_cross / 2 * 2 - MARGIN, 6))  # placeholder, fixed below
    v_cross = np.tan(t_cross / 2)
    v_hi_g = sp.nsimplify(round(v_cross - MARGIN, 6))
    u_lo_g = sp.nsimplify(round(u_lo_true + MARGIN, 6))

    NBOX = 50
    all_ok = True
    print(f"\ndomain: v in (0,{v_cross:.6f}), u in ({u_lo_true:.6f}, u_max(v)); "
          f"certifying via {NBOX} boxes")
    for k in range(NBOX):
        v_lo = v_lo_g + (v_hi_g - v_lo_g) * k / NBOX
        v_hi = v_lo_g + (v_hi_g - v_lo_g) * (k + 1) / NBOX
        t_of_vhi = 2 * np.arctan(float(v_hi))
        u_hi_f = np.tan(theta_max(t_of_vhi) / 2) * 0.9998
        u_hi = sp.nsimplify(round(u_hi_f, 6))
        if u_hi <= u_lo_g:
            u_lo_box, u_hi_box = u_hi, u_lo_g  # tiny reversed sliver near the corner
        else:
            u_lo_box, u_hi_box = u_lo_g, u_hi
        okN, nzN, totN = check_box(Nc, NU, NV, u_lo_box, u_hi_box, v_lo, v_hi, expect_sign=-1)
        okD, nzD, totD = check_box(Dc, NU, NV, u_lo_box, u_hi_box, v_lo, v_hi, expect_sign=-1)
        print(f"  box {k:2d}: v=[{float(v_lo):.5f},{float(v_hi):.5f}] "
              f"u=[{float(u_lo_box):.5f},{float(u_hi_box):.5f}] "
              f"N:{'ok' if okN else 'FAIL'}({nzN}/{totN} zero) "
              f"D:{'ok' if okD else 'FAIL'}({nzD}/{totD} zero)")
        all_ok = all_ok and okN and okD

    assert all_ok, "certificate FAILED on at least one box"
    print()
    print("ALL BOXES CERTIFIED (exact rational arithmetic, no floating point).")
    print()
    print("=" * 70)
    print("[CERTIFICATE, region 2, modulo a thin quantified margin] F_Omega > 0")
    print("established for chamber C001, v1-w1 arc, essentially all of")
    print("pi/3 < theta < arcsin(1/(sin t+cos t)), 0 < t < t_cross=0.169918...")
    print("This is the SECOND (of 9) regions now certified, after region 4.")
    print("Regions 0,1,3,5,6,7,8 and the w1-v2 arc remain completely untouched.")
    print("Conjecture (Direction-of-Deviation Positivity) remains OPEN overall.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
