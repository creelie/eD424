import re, time, itertools
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

def build_root_system():
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
    return roots, labels, idx0, roots[idx0], roots[find([1, -1, 0, 0])], roots[find([0, 0, 1, 1])]

ROOTS, LABELS, IDX0, U0, V1, W1 = build_root_system()
SQRT2 = sp.sqrt(2)

def u1_vec(th, tt):
    return np.cos(th)*U0 + np.sin(th)*(np.cos(tt)*V1 + np.sin(tt)*W1)

def polytope_full(th, tt):
    u1 = u1_vec(th, tt)
    dirs = ROOTS.copy(); dirs[IDX0] = u1
    hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    out = []
    for vpt in hi.intersections:
        vals = dirs @ vpt
        touching = frozenset(LABELS[k] if k!=IDX0 else 'u1' for k,val in enumerate(vals) if abs(val-1)<1e-6)
        out.append((vpt, touching))
    return out

def theta_W(tt): return 2*np.arctan(np.sin(tt))
def theta_A(tt): return 2*np.arctan(np.cos(tt))
def theta_Y(tt):
    val = 1.0/(np.sin(tt)+np.cos(tt)); return np.arcsin(min(val,1.0))
THETA_U = np.pi/3

def candidates():
    candidates_num, candidates_sym = [], []
    for i in range(4):
        for s in (1,-1):
            vv = np.zeros(4); vv[i]=s*np.sqrt(2)
            candidates_num.append(vv)
            vs_ = sp.zeros(4,1); vs_[i]=s*SQRT2
            candidates_sym.append(vs_)
    for eps in itertools.product([1,-1], repeat=4):
        candidates_num.append(np.array(eps)/np.sqrt(2))
        candidates_sym.append(sp.Matrix(eps)/SQRT2)
    candidates_num.append(np.array([np.sqrt(2), np.sqrt(2), 0, 0]))
    candidates_sym.append(sp.Matrix([SQRT2, SQRT2, 0, 0]))
    return np.array(candidates_num), candidates_sym

CAND_NUM, CAND_SYM = candidates()

def root_sym(spec):
    r = sp.zeros(4,1)
    for m in re.finditer(r'([+-])e(\d)', spec):
        r[int(m.group(2))-1] = 1 if m.group(1)=='+' else -1
    return r/SQRT2

U0S = sp.Matrix([1,1,0,0])/SQRT2
V1S = sp.Matrix([1,-1,0,0])/SQRT2
W1S = sp.Matrix([0,0,1,1])/SQRT2
U1S = sp.cos(theta)*U0S + sp.sin(theta)*(sp.cos(t)*V1S + sp.sin(t)*W1S)

def _solve_with_labels(labels3):
    z = sp.Matrix(sp.symbols('z1 z2 z3 z4', real=True))
    eqs = [sp.Eq((U1S.T*z)[0,0], 1)]
    for lab in labels3:
        eqs.append(sp.Eq((root_sym(lab).T*z)[0,0], 1))
    sol = sp.solve(eqs, list(z), dict=True)
    return _extract_in_order(sol, list(z))


def solve_vertex_moving(facet_labels):
    """For an exactly-4-facet (u1 + 3) pattern this is unambiguous. For a
    DEGENERATE vertex where more than 3 non-u1 facets pass through the same
    point, different 3-subsets give algebraically different (but
    numerically equal) rational-function representations -- try every
    3-subset and keep the smallest one, since an arbitrary/unlucky choice
    can produce a needlessly complex expression that blows up downstream
    determinant computations."""
    if len(facet_labels) <= 3:
        return _solve_with_labels(facet_labels[:3])
    best = None
    for combo in itertools.combinations(facet_labels, 3):
        zv = _solve_with_labels(list(combo))
        if zv is None:
            continue
        size = sum(len(str(c)) for c in zv)
        if best is None or size < best[0]:
            best = (size, zv)
    return best[1] if best else None

def derive_region(theta0, t0, label, fixed_check_samples, expect_verts=None):
    """Full vertex/volume derivation for a region given a representative point."""
    verts_info = polytope_full(theta0, t0)
    verts = np.array([vv for vv,_ in verts_info])
    patterns = [p for _,p in verts_info]
    on_cap_idx = [i for i,p in enumerate(patterns) if 'u1' in p]
    off_cap_idx = [i for i,p in enumerate(patterns) if 'u1' not in p]
    print(f"[{label}] rep point theta={theta0:.4f} t={t0:.4f}: {len(verts)} verts, "
          f"{len(on_cap_idx)} moving, {len(off_cap_idx)} fixed")
    if expect_verts is not None:
        assert len(verts) == expect_verts, f"expected {expect_verts} verts, got {len(verts)}"

    vsets = [{p: vv for vv,p in polytope_full(th,tt) if 'u1' not in p} for tt,th in fixed_check_samples]
    common = set(vsets[0])
    for vs in vsets[1:]: common &= set(vs)
    n_diff = sum(1 for k in common if not all(np.allclose(vs[k], vsets[0][k]) for vs in vsets))
    print(f"[{label}] fixed-vertex check across {len(fixed_check_samples)} samples: "
          f"{n_diff}/{len(common)} differ, {len(common)}/{len(off_cap_idx)} matched to full set")
    assert n_diff == 0

    hull = ConvexHull(verts, qhull_options='QJ')
    on_cap_set = set(on_cap_idx)
    moving_simplices = [s for s in hull.simplices if any(vv in on_cap_set for vv in s)]
    fixed_simplices = [s for s in hull.simplices if all(vv not in on_cap_set for vv in s)]
    print(f"[{label}] {len(hull.simplices)} boundary simplices: {len(fixed_simplices)} fixed-only, {len(moving_simplices)} moving")

    fixed_sym = {}
    for i in off_cap_idx:
        dists = np.linalg.norm(CAND_NUM - verts[i], axis=1)
        j = np.argmin(dists)
        assert dists[j] < 1e-6, f"unmatched fixed vertex at index {i}: {verts[i]}"
        fixed_sym[i] = CAND_SYM[j]
    print(f"[{label}] fixed vertices matched exactly: {len(fixed_sym)}/{len(off_cap_idx)}")

    vol_fixed_exact = sp.Integer(0)
    for s in fixed_simplices:
        M = sp.Matrix.hstack(*[fixed_sym[i] for i in s]).T
        Mnum = verts[list(s)]
        sign = 1 if np.linalg.det(Mnum) > 0 else -1
        vol_fixed_exact += sign*M.det(method='berkowitz')/24
    vol_fixed_exact = sp.nsimplify(sp.simplify(vol_fixed_exact))
    print(f"[{label}] EXACT fixed-only volume: {vol_fixed_exact}")

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
    print(f"[{label}] moving vertices solved: {len(moving_cache)} distinct patterns "
          f"({len(on_cap_idx)} instances), {time.time()-t_start:.1f}s")

    sym_vertex = dict(fixed_sym); sym_vertex.update(moving_sym)
    terms = []
    for k, s in enumerate(moving_simplices):
        tk = time.time()
        M = sp.Matrix.hstack(*[sym_vertex[i] for i in s]).T
        Mnum = verts[list(s)]
        sign = 1 if np.linalg.det(Mnum) > 0 else -1
        det = M.det(method='berkowitz')
        tdet = time.time()
        term = sp.cancel(sign*det/24)
        terms.append(term)
        print(f"[{label}]   term {k+1}/{len(moving_simplices)}: det_time={tdet-tk:.2f}s "
              f"cancel_time={time.time()-tdet:.2f}s term_size={len(str(term))} total_t={time.time()-t_start:.1f}s",
              flush=True)
    print(f"[{label}] all {len(terms)} raw terms built, {time.time()-t_start:.1f}s", flush=True)

    # binary-tree (pairwise) reduction: combine adjacent pairs and cancel at
    # each level, instead of one long linear accumulation (which suffers
    # from repeated re-cancellation of an ever-growing common denominator)
    level = terms
    depth = 0
    while len(level) > 1:
        depth += 1
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                combined = sp.cancel(sp.together(level[i] + level[i+1]))
            else:
                combined = level[i]
            next_level.append(combined)
        sizes = [len(str(x)) for x in next_level]
        print(f"[{label}]   tree level {depth}: {len(next_level)} nodes, "
              f"max size {max(sizes)} chars, {time.time()-t_start:.1f}s", flush=True)
        level = next_level
    total = level[0]
    num_check = float(total.subs({theta: theta0, t: t0}))
    vd = ConvexHull(verts).volume
    print(f"[{label}] vol_moving closed form: {len(str(total))} chars, {time.time()-t_start:.1f}s, "
          f"numeric check {num_check} vs direct-vol_fixed={vd-float(vol_fixed_exact)}")
    assert abs((float(vol_fixed_exact)+num_check) - vd) < 1e-7
    return vol_fixed_exact, total
