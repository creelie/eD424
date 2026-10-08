import os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
def _data(_name):
    """Resolve a bundled or cached data file, wherever the script is run from."""
    _os.makedirs(_DATA_DIR, exist_ok=True)
    return _os.path.join(_DATA_DIR, _name)

#!/usr/bin/env python3
"""
bandC_boxcover_certificate.py
==============================
Attempt at the THIRD region on the w1-v2 (second) arc: the C<->W' sub-part
of the band immediately below region-top, t in (0, t_cross=0.169918),
bounded above by curve W' (theta=2*arctan(cos t)) and below by curve C
(sin(theta)(cos t + sin t) = 1, i.e. theta_C(t)=arcsin(1/(cos t+sin t))).
Adapted directly from eperp_w1v2_bandB_boxcover_certificate.py (the
already-certified middle sub-part, universal<->W'): identical vertex/
volume/box-cover machinery, only the lower boundary curve function
changes (a constant curve there, a genuinely curved one here).
"""
import sys, re, time, pickle, os, itertools
from math import comb
import numpy as np
import sympy as sp
from scipy.spatial import HalfspaceIntersection, ConvexHull

def _extract_in_order(sol, zlist, simplify_fn=None):
    """Read a solve() result back in the supplied unknown order.

    sympy returns a dict keyed by symbol; its iteration order is not
    guaranteed to match the order the unknowns were passed in, so the
    components are looked up by key rather than taken from .values().
    Returns None when the system leaves a coordinate undetermined.
    """
    if simplify_fn is None:
        simplify_fn = sp.simplify
    if not sol:
        return None
    d = sol[0]
    out = []
    for zi in zlist:
        if zi not in d:
            return None
        out.append(simplify_fn(d[zi]))
    return sp.Matrix(out)


theta, t = sp.symbols('theta t', real=True, positive=True)
u, v = sp.symbols('u v', real=True, positive=True)
x, y = sp.symbols('x y', real=True, positive=True)
sqrt2 = sp.sqrt(2)

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
        v_ = np.array(vec, dtype=float); v_ /= np.linalg.norm(v_)
        idx = np.argmax(roots @ v_)
        assert (roots @ v_)[idx] > 1 - 1e-9
        return idx
    idx0 = find([1, 1, 0, 0])
    return roots, labels, idx0, roots[idx0], roots[find([0, 0, 1, 1])], roots[find([0, 0, 1, -1])]

ROOTS, LABELS, IDX0, U0, W1, V2 = _build_root_system()

def u1_vec(th, tt):
    return np.cos(th) * U0 + np.sin(th) * (np.cos(tt) * W1 + np.sin(tt) * V2)

def polytope_full(th, tt):
    u1 = u1_vec(th, tt)
    dirs = ROOTS.copy()
    dirs[IDX0] = u1
    hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    out = []
    for vpt in hi.intersections:
        vals = dirs @ vpt
        touching = frozenset(LABELS[k] if k != IDX0 else 'u1' for k, val in enumerate(vals) if abs(val - 1) < 1e-6)
        out.append((vpt, touching))
    return out

def theta_W(tt): return 2 * np.arctan(np.cos(tt))
def theta_C(tt): return np.arcsin(1.0/(np.cos(tt)+np.sin(tt)))

T_LO, T_HI = 0.0, 0.169918

def check_stability():
    ts = np.linspace(T_LO+0.005, T_HI-0.005, 8)
    pats = []
    for tt in ts:
        th = 0.5 * (theta_C(tt) + theta_W(tt))
        info = polytope_full(th, tt)
        pat = frozenset(p for _, p in info)
        pats.append((len(info), pat))
        print(f"  t={tt:.4f} theta={th:.4f}: {len(info)} vertices")
    same = all(p == pats[0][1] for _, p in pats)
    print(f"combinatorial pattern identical across all 8 samples: {same}")
    assert same
    return pats[0][1]

def derive_band():
    t0 = 0.5 * (T_LO + T_HI)
    theta0 = 0.5 * (theta_C(t0) + theta_W(t0))
    verts_info = polytope_full(theta0, t0)
    verts = np.array([vv for vv, _ in verts_info])
    patterns = [p for _, p in verts_info]
    on_cap_idx = [i for i, p in enumerate(patterns) if 'u1' in p]
    off_cap_idx = [i for i, p in enumerate(patterns) if 'u1' not in p]
    print(f"representative point (theta,t)=({theta0:.4f},{t0:.4f}): {len(verts)} vertices, "
          f"{len(on_cap_idx)} moving, {len(off_cap_idx)} fixed")

    samples = [(tt, 0.5 * (theta_C(tt) + theta_W(tt))) for tt in np.linspace(T_LO+0.01, T_HI-0.01, 5)]
    vsets = [{p: vv for vv, p in polytope_full(th, tt) if 'u1' not in p} for tt, th in samples]
    common = set(vsets[0])
    for vs in vsets[1:]:
        common &= set(vs)
    n_diff = sum(1 for k in common if not all(np.allclose(vs[k], vsets[0][k]) for vs in vsets))
    print(f"fixed-vertex check across 5 sample points: {n_diff}/{len(common)} differ (0=all fixed)")
    assert n_diff == 0 and len(common) == len(off_cap_idx)

    hull = ConvexHull(verts, qhull_options='QJ')
    on_cap_set = set(on_cap_idx)
    moving_simplices = [s for s in hull.simplices if any(vv in on_cap_set for vv in s)]
    fixed_simplices = [s for s in hull.simplices if all(vv not in on_cap_set for vv in s)]
    print(f"{len(hull.simplices)} boundary simplices: {len(fixed_simplices)} fixed-only, {len(moving_simplices)} moving")

    candidates_num, candidates_sym = [], []
    for i in range(4):
        for s in (1, -1):
            vv = np.zeros(4); vv[i] = s * np.sqrt(2)
            candidates_num.append(vv)
            vs_ = sp.zeros(4, 1); vs_[i] = s * sqrt2
            candidates_sym.append(vs_)
    for eps in itertools.product([1, -1], repeat=4):
        candidates_num.append(np.array(eps) / np.sqrt(2))
        candidates_sym.append(sp.Matrix(eps) / sqrt2)
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    vv = np.zeros(4); vv[i], vv[j] = si * np.sqrt(2), sj * np.sqrt(2)
                    candidates_num.append(vv)
                    vs_ = sp.zeros(4, 1); vs_[i], vs_[j] = si * sqrt2, sj * sqrt2
                    candidates_sym.append(vs_)
    candidates_num = np.array(candidates_num)

    fixed_sym = {}
    for i in off_cap_idx:
        dists = np.linalg.norm(candidates_num - verts[i], axis=1)
        j = np.argmin(dists)
        assert dists[j] < 1e-6, f"unmatched fixed vertex {verts[i]}"
        fixed_sym[i] = candidates_sym[j]
    print(f"fixed vertices matched exactly: {len(fixed_sym)}/{len(off_cap_idx)}")

    vol_fixed_exact = sp.Integer(0)
    for s in fixed_simplices:
        M = sp.Matrix.hstack(*[fixed_sym[i] for i in s]).T
        Mnum = verts[list(s)]
        sign = 1 if np.linalg.det(Mnum) > 0 else -1
        vol_fixed_exact += sign * M.det() / 24
    vol_fixed_exact = sp.nsimplify(sp.simplify(vol_fixed_exact))
    print(f"EXACT fixed-only volume contribution: {vol_fixed_exact}")

    u0s = sp.Matrix([1, 1, 0, 0]) / sqrt2
    w1s = sp.Matrix([0, 0, 1, 1]) / sqrt2
    v2s_ = sp.Matrix([0, 0, 1, -1]) / sqrt2
    u1s = sp.cos(theta) * u0s + sp.sin(theta) * (sp.cos(t) * w1s + sp.sin(t) * v2s_)

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
        return _extract_in_order(sol, list(z))

    t_start = time.time()
    moving_sym, moving_cache = {}, {}
    for i in on_cap_idx:
        pat = patterns[i]
        if pat in moving_cache:
            moving_sym[i] = moving_cache[pat]; continue
        zv = solve_vertex_moving([p for p in pat if p != 'u1'])
        assert zv is not None
        numeric = np.array([float(z.subs({theta: theta0, t: t0})) for z in zv])
        assert np.allclose(numeric, verts[i], atol=1e-5)
        moving_sym[i] = zv
        moving_cache[pat] = zv
    print(f"moving vertices solved exactly: {len(moving_cache)} distinct patterns "
          f"({len(on_cap_idx)} instances), {time.time()-t_start:.1f}s", flush=True)

    sym_vertex = dict(fixed_sym); sym_vertex.update(moving_sym)

    terms = []
    for si, s in enumerate(moving_simplices):
        ts0 = time.time()
        M = sp.Matrix.hstack(*[sym_vertex[i] for i in s]).T
        Mnum = verts[list(s)]
        sign = 1 if np.linalg.det(Mnum) > 0 else -1
        term = sp.cancel(sign * M.det() / 24, extension=True)
        terms.append(term)
        dt = time.time() - ts0
        if dt > 1.0 or si % 10 == 0:
            print(f"    simplex {si}/{len(moving_simplices)}: {dt:.2f}s, term len {len(str(term))}", flush=True)

    ts_sum = time.time()
    running = sp.Integer(0)
    for i, term in enumerate(terms):
        running = running + term
        if i % 10 == 9:
            running = sp.together(running)
            print(f"    partial sum after {i+1} terms: {len(str(running))} chars, {time.time()-ts_sum:.1f}s", flush=True)
    total = sp.together(running)
    print(f"  final sum done, {time.time()-ts_sum:.1f}s", flush=True)
    num_check = float(total.subs({theta: theta0, t: t0}))
    vd = ConvexHull(verts).volume
    print(f"vol_band closed form: {len(str(total))} chars, {time.time()-t_start:.1f}s, "
          f"numeric check vol_fixed+vol_band = {float(vol_fixed_exact) + num_check} vs direct hull vol = {vd}")
    assert abs((float(vol_fixed_exact) + num_check) - vd) < 1e-8
    return total, vol_fixed_exact, terms

def _term_to_uv(term):
    subs_map = {
        sp.sin(theta): 2 * u / (1 + u ** 2), sp.cos(theta): (1 - u ** 2) / (1 + u ** 2),
        sp.tan(theta): 2 * u / (1 - u ** 2),
        sp.sin(t): 2 * v / (1 + v ** 2), sp.cos(t): (1 - v ** 2) / (1 + v ** 2),
        sp.tan(t): 2 * v / (1 - v ** 2),
    }
    term_expanded = sp.expand_trig(term)
    return sp.cancel(term_expanded.subs(subs_map), extension=True)

def to_uv_ND(terms, target, theta0_chk, t0_chk):
    import mpmath as mp
    mp.mp.dps = 200
    u0_hp = mp.tan(mp.mpf(str(theta0_chk)) / 2)
    v0_hp = mp.tan(mp.mpf(str(t0_chk)) / 2)
    def eval_theta_t(expr):
        f = sp.lambdify((theta, t), expr, modules='mpmath')
        return f(mp.mpf(str(theta0_chk)), mp.mpf(str(t0_chk)))
    def eval_uv(expr):
        f = sp.lambdify((u, v), expr, modules='mpmath')
        return f(u0_hp, v0_hp)
    TOL = mp.mpf('1e-100')
    nt = sp.nsimplify(target)
    target_hp = mp.mpf(nt.p) / mp.mpf(nt.q) if nt.is_Rational else mp.mpf(str(float(target)))
    orig_val = eval_theta_t(sum(terms)) - target_hp
    running = -sp.nsimplify(target)
    for i, term in enumerate(terms):
        ts0 = time.time()
        tuv = _term_to_uv(term)
        val = eval_uv(tuv)
        term_orig_val = eval_theta_t(term)
        assert abs(val - term_orig_val) < TOL, f"_term_to_uv FAILED on term {i}"
        running = sp.cancel(running + tuv, extension=True)
        running_val = eval_uv(running)
        terms_so_far_val = eval_theta_t(sum(terms[:i + 1])) - target_hp
        assert abs(running_val - terms_so_far_val) < TOL, f"running-total FAILED after term {i}"
        print(f"    term {i}/{len(terms)} mapped+cancelled: {time.time()-ts0:.1f}s, running len {len(str(running))}", flush=True)
    val_final = eval_uv(running)
    assert abs(val_final - orig_val) < TOL, "to_uv_ND final check FAILED"
    print(f"to_uv_ND: final check OK (value {val_final}, len {len(str(running))})", flush=True)
    N, D = sp.fraction(running)
    return sp.expand(N), sp.expand(D)

def poly_to_coeff_dict(expr):
    coeffs = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        pd = term.as_powers_dict()
        i = int(pd.get(u, 0)); j = int(pd.get(v, 0))
        c = term
        if i: c = c / u ** i
        if j: c = c / v ** j
        c = sp.expand(sp.radsimp(c))
        coeffs[(i, j)] = coeffs.get((i, j), sp.Integer(0)) + c
    return coeffs

def substitute_box(coeffs, u_lo, u_hi, v_lo, v_hi, nu, nv):
    ux = u_lo + (u_hi - u_lo) * x
    vy = v_lo + (v_hi - v_lo) * y
    upows = [sp.Integer(1)]
    for _ in range(nu): upows.append(sp.expand(upows[-1] * ux))
    vpows = [sp.Integer(1)]
    for _ in range(nv): vpows.append(sp.expand(vpows[-1] * vy))
    expr = sp.Integer(0)
    for (i, j), c in coeffs.items():
        expr += c * upows[i] * vpows[j]
    expr = sp.expand(expr)
    out = {}
    for term in sp.Add.make_args(expr):
        pd = term.as_powers_dict()
        i = int(pd.get(x, 0)); j = int(pd.get(y, 0))
        c = term
        if i: c = c / x ** i
        if j: c = c / y ** j
        out[(i, j)] = out.get((i, j), sp.Integer(0)) + sp.expand(sp.radsimp(c))
    return out

def bernstein_coeffs(coeffs, nu, nv):
    A = [[sp.Integer(0)] * (nv + 1) for _ in range(nu + 1)]
    for (i, j), c in coeffs.items(): A[i][j] += c
    def binom(n, k): return sp.Integer(comb(n, k))
    def transform_1d(vec, n):
        b = [sp.Integer(0)] * (n + 1)
        for j in range(n + 1):
            s = sp.Integer(0)
            for k in range(j + 1): s += binom(j, k) * vec[k] / binom(n, k)
            b[j] = s
        return b
    B1 = [transform_1d(A[i], nv) for i in range(nu + 1)]
    B2 = [[sp.Integer(0)] * (nv + 1) for _ in range(nu + 1)]
    for jv in range(nv + 1):
        col = [B1[i][jv] for i in range(nu + 1)]
        newcol = transform_1d(col, nu)
        for i in range(nu + 1): B2[i][jv] = newcol[i]
    return B2

def exact_sign(expr):
    import mpmath as mp
    mp.mp.dps = 100
    r = sp.expand(expr)
    f = sp.lambdify([], r, modules='mpmath')
    val = f()
    TOL = mp.mpf('1e-80')
    if val > TOL: return 1
    if val < -TOL: return -1
    return 0

def check_box(coeffs, u_lo, u_hi, v_lo, v_hi, nu, nv):
    bc = substitute_box(coeffs, u_lo, u_hi, v_lo, v_hi, nu, nv)
    B = bernstein_coeffs(bc, nu, nv)
    flat = [B[i][j] for i in range(nu + 1) for j in range(nv + 1)]
    signs = [exact_sign(c) for c in flat]
    npos = sum(1 for s in signs if s > 0)
    nneg = sum(1 for s in signs if s < 0)
    nzero = sum(1 for s in signs if s == 0)
    return (nneg == 0), (npos == 0), npos, nneg, nzero, len(flat)

def main():
    print("=" * 70)
    print("w1-v2 arc, band below region-top, C-sub-part (t in (0,t_cross))")
    print("box-covering certificate attempt")
    print("=" * 70)
    print("Step 1: combinatorial stability check")
    check_stability()
    print()

    cache_path = _data("bandC_step2_cache.pkl")
    if os.path.exists(cache_path):
        print("Step 2: loading cached exact vertex/volume derivation", flush=True)
        with open(cache_path, "rb") as f:
            total, vol_fixed, terms = pickle.load(f)
    else:
        print("Step 2: exact vertex/volume derivation")
        total, vol_fixed, terms = derive_band()
        with open(cache_path, "wb") as f:
            pickle.dump((total, vol_fixed, terms), f)
        print(f"  cached to {cache_path}", flush=True)
    target = 8 - vol_fixed
    print(f"F_Omega/D_Omega target constant: vol_band - {target}")
    print()
    print("Step 3: map to Weierstrass (u,v) PER TERM", flush=True)
    cache_path3 = _data("bandC_step3_cache.pkl")
    theta0_chk, t0_chk = 0.5 * (theta_C(0.5*(T_LO+T_HI)) + theta_W(0.5*(T_LO+T_HI))), 0.5*(T_LO+T_HI)
    if os.path.exists(cache_path3):
        print("Step 3: loading cached (u,v) mapping", flush=True)
        with open(cache_path3, "rb") as f:
            N, D = pickle.load(f)
    else:
        t0 = time.time()
        N, D = to_uv_ND(terms, target, theta0_chk, t0_chk)
        with open(cache_path3, "wb") as f:
            pickle.dump((N, D), f)
        print(f"mapped to (u,v): N {len(str(N))} chars, D {len(str(D))} chars, {time.time()-t0:.1f}s, cached", flush=True)

    Nc = poly_to_coeff_dict(N); Dc = poly_to_coeff_dict(D)
    nu = max(i for i, j in Nc); nvN = max(j for i, j in Nc)
    du = max(i for i, j in Dc); dvD = max(j for i, j in Dc)
    print(f"N degree (u,v)=({nu},{nvN})   D degree (u,v)=({du},{dvD})")
    NU, NV = max(nu, du), max(nvN, dvD)

    print()
    print("Step 4: box-covering, u in (u_C(t), u_W(t)), v=tan(t/2), t in (0,t_cross)")
    v_lo_g_raw, v_hi_g_raw = np.tan(T_LO/2), np.tan((T_HI)/2)
    MARGIN = (v_hi_g_raw - v_lo_g_raw) * 0.02
    NBOX = 40
    v_lo_global = v_lo_g_raw + MARGIN
    v_hi_global = v_hi_g_raw - MARGIN

    def u_W_of_v(vv):
        tt = 2 * np.arctan(vv)
        return np.tan(theta_W(tt) / 2)
    def u_C_of_v(vv):
        tt = 2 * np.arctan(vv)
        return np.tan(theta_C(tt) / 2)

    def clean_rational(xx):
        return sp.Rational(str(round(xx, 6)))

    all_ok = True
    n_fail = 0
    for k in range(NBOX):
        v_lo = v_lo_global + (v_hi_global - v_lo_global) * k / NBOX
        v_hi = v_lo_global + (v_hi_global - v_lo_global) * (k + 1) / NBOX
        u_hi_f = u_W_of_v(v_hi) * (1 - 0.001)
        u_lo_f = u_C_of_v(v_lo) * (1 + 0.001)
        u_lo = clean_rational(u_lo_f); u_hi = clean_rational(u_hi_f)
        v_lo_s = clean_rational(v_lo); v_hi_s = clean_rational(v_hi)
        okN_pos, okN_neg, npN, nnN, nzN, totN = check_box(Nc, u_lo, u_hi, v_lo_s, v_hi_s, NU, NV)
        okD_pos, okD_neg, npD, nnD, nzD, totD = check_box(Dc, u_lo, u_hi, v_lo_s, v_hi_s, NU, NV)
        n_status = "all>=0" if okN_pos else ("all<=0" if okN_neg else "MIXED")
        d_status = "all>=0" if okD_pos else ("all<=0" if okD_neg else "MIXED")
        same_sign = (okN_pos and okD_pos) or (okN_neg and okD_neg)
        if not same_sign:
            n_fail += 1; all_ok = False
        print(f"  box {k:2d}: v=[{float(v_lo):.5f},{float(v_hi):.5f}] u=[{float(u_lo):.5f},{float(u_hi):.5f}]  "
              f"N:{n_status}({npN}+/{nnN}-/{nzN}0 of {totN}) D:{d_status}({npD}+/{nnD}-/{nzD}0 of {totD})  "
              f"[{'OK' if same_sign else 'FAIL'}]", flush=True)

    print()
    if all_ok:
        excl_v = (2 * MARGIN) / (v_hi_g_raw - v_lo_g_raw)
        print(f"ALL {NBOX} BOXES CERTIFIED (exact rational/Q(sqrt2) arithmetic).")
        print(f"Excluded margin: ~{excl_v*100:.3f}% of the v-range at the two ends, "
              f"plus ~0.1-0.2% of u-range at each box's own floor/ceiling edges.")
    else:
        print(f"CERTIFICATE INCONCLUSIVE on {n_fail}/{NBOX} boxes with this box "
              f"resolution/margin. Reporting honestly.")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
