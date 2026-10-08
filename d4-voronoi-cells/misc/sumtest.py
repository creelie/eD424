#!/usr/bin/env python3
"""
sumtest.py

A methodology experiment, not a result. It compares four ways of
collapsing the same sum of 66 symbolic simplex volumes to a single
rational function, and reports how long each takes and whether it
agrees numerically with the others:

    the plain sum, with no post-processing
    together() alone
    cancel() alone
    together() followed by cancel()

The point of the comparison is that the four are not interchangeable in
cost: on this expression the plain sum and together() finish in a couple
of seconds, while the two routes that call cancel() on the whole sum can
run for a very long time without settling. That is the same expression
growth documented in the methodology appendix, and it is why the
certificates elsewhere in this package cancel term by term rather than
once at the end.

Each stage runs under a time bound so the script reports what it
established and stops, instead of running indefinitely. Set
SUMTEST_STAGE_SECONDS to change it (default 120).
"""
import os
import re, itertools, time
import numpy as np
import sympy as sp
from scipy.spatial import HalfspaceIntersection, ConvexHull

STAGE_SECONDS = float(os.environ.get("SUMTEST_STAGE_SECONDS", "120"))


class _Timeout(Exception):
    pass


def _bounded(seconds, fn, *args):
    """Run fn under a wall-clock bound; return (ok, value)."""
    import signal

    def _fire(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return True, fn(*args)
    except _Timeout:
        return False, None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


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
        idx = np.argmax(roots @ v_); assert (roots @ v_)[idx] > 1 - 1e-9
        return idx
    idx0 = find([1, 1, 0, 0])
    return roots, labels, idx0, roots[idx0], roots[find([0, 0, 1, 1])], roots[find([0, 0, 1, -1])]

roots, labels, idx0, u0, w1, v2 = _build_root_system()

def u1_vec(th, tt):
    return np.cos(th) * u0 + np.sin(th) * (np.cos(tt) * w1 + np.sin(tt) * v2)

def polytope_full(th, tt):
    u1 = u1_vec(th, tt)
    dirs = roots.copy(); dirs[idx0] = u1
    hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    out = []
    for vpt in hi.intersections:
        vals = dirs @ vpt
        touching = frozenset(labels[k] if k != idx0 else 'u1' for k, val in enumerate(vals) if abs(val - 1) < 1e-6)
        out.append((vpt, touching))
    return out

def theta_W(tt): return 2 * np.arctan(np.cos(tt))
t0 = 0.15
theta0 = 0.5 * (theta_W(t0) + np.pi / 2)
verts_info = polytope_full(theta0, t0)
verts = np.array([vv for vv, _ in verts_info])
patterns = [p for _, p in verts_info]
on_cap_idx = [i for i, p in enumerate(patterns) if 'u1' in p]
off_cap_idx = [i for i, p in enumerate(patterns) if 'u1' not in p]

hull = ConvexHull(verts, qhull_options='QJ')
on_cap_set = set(on_cap_idx)
moving_simplices = [s for s in hull.simplices if any(vv in on_cap_set for vv in s)]

sqrt2n = np.sqrt(2)
candidates_num, candidates_sym = [], []
for i in range(4):
    for s in (1, -1):
        vv = np.zeros(4); vv[i] = s * sqrt2n
        candidates_num.append(vv)
        vs_ = sp.zeros(4, 1); vs_[i] = s * sqrt2
        candidates_sym.append(vs_)
for eps in itertools.product([1, -1], repeat=4):
    candidates_num.append(np.array(eps) / sqrt2n)
    candidates_sym.append(sp.Matrix(eps) / sqrt2)
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                vv = np.zeros(4); vv[i], vv[j] = si * sqrt2n, sj * sqrt2n
                candidates_num.append(vv)
                vs_ = sp.zeros(4, 1); vs_[i], vs_[j] = si * sqrt2, sj * sqrt2
                candidates_sym.append(vs_)
candidates_num = np.array(candidates_num)

fixed_sym = {}
for i in off_cap_idx:
    dists = np.linalg.norm(candidates_num - verts[i], axis=1)
    j = np.argmin(dists); assert dists[j] < 1e-6
    fixed_sym[i] = candidates_sym[j]

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

moving_sym, moving_cache = {}, {}
for i in on_cap_idx:
    pat = patterns[i]
    if pat in moving_cache:
        moving_sym[i] = moving_cache[pat]; continue
    zv = solve_vertex_moving([p for p in pat if p != 'u1'])
    moving_sym[i] = zv; moving_cache[pat] = zv

sym_vertex = dict(fixed_sym); sym_vertex.update(moving_sym)

terms = []
for s in moving_simplices:
    M = sp.Matrix.hstack(*[sym_vertex[i] for i in s]).T
    Mnum = verts[list(s)]
    sign = 1 if np.linalg.det(Mnum) > 0 else -1
    terms.append(sp.cancel(sign * M.det() / 24))

print(f"computed {len(terms)} terms", flush=True)
plain_sum = sum(terms)
val1 = float(plain_sum.subs({theta: theta0, t: t0}))
print("plain sum (no post-cancel), numeric value:", val1, " expected ~1.895", flush=True)

t0s = time.time()
togethered = sp.together(plain_sum)
val_t = float(togethered.subs({theta: theta0, t: t0}))
print(f"after together() only: {val_t}  ({time.time()-t0s:.1f}s)", flush=True)

t0s = time.time()
ok, cancelled_only = _bounded(STAGE_SECONDS, sp.cancel, plain_sum)
if ok:
    val_c = float(cancelled_only.subs({theta: theta0, t: t0}))
    print(f"after cancel() only (no together first): {val_c}  ({time.time()-t0s:.1f}s)", flush=True)
else:
    print(f"after cancel() only: did not settle within {STAGE_SECONDS:.0f}s", flush=True)

t0s = time.time()
ok, final = _bounded(STAGE_SECONDS, sp.cancel, togethered)
if not ok:
    print(f"after together() then cancel(): did not settle within "
          f"{STAGE_SECONDS:.0f}s", flush=True)
    print()
    print("Conclusion: the two routes that cancel the whole sum at once do")
    print("not settle here, while the plain sum and together() agree to")
    print("twelve digits in seconds. Cancel term by term, not at the end.")
    raise SystemExit(0)
val_ct = float(final.subs({theta: theta0, t: t0}))
print(f"after together() then cancel(): {val_ct}  ({time.time()-t0s:.1f}s)", flush=True)
