#!/usr/bin/env python3
"""
radial_case_sdp.py -- two-point radial certificates for statement (C_M) at one
count M, split into cases by how many centres lie within given radii.
radial_case_check.py proves them.

For a packing set Y of M centres in the shell 2 <= |y| < sqrt6, let N(r) be the
number of centres within r.  A case is a set of bounds lo_j <= N(r_j) <= hi_j
at rational radii r_j; every case also carries N(2.0161) <= 24, which holds for
every packing (the kissing number is stable).  Cutting the shell at the r_j
into bins, with n_b centres in bin b, the kernel inequality of thm:count31
gives, for any certificate (A_k, z, t) with K <= Pi on admissible pairs,

    U(Y) <= sum_y f(|y|) + t/2 <= sum_b n_b m_b + t/2,
    f(d) = S(d) + K(d, d, 1)/2 - z . p(d),   f <= m_b on bin b,

and a counterexample to (C) also has sum_b n_b S(e_b) >= U(Y) >= 9 pi^2/8 - 8,
e_b the lower end of bin b.  The largest value of sum_b n_b m_b over the count
vectors of the case is bounded through the dual of that linear programme:
for a, c, s >= 0 and g with

    sum_{j: bin b within r_j} (a_j - c_j) + g - s S(e_b) >= m_b   (every bin b),

sum_b n_b m_b <= sum_j (a_j hi_j - c_j lo_j) + g M - s (9 pi^2/8 - 8).  The
programme minimises that bound plus t/2, with a margin eps in the pair
inequality and in the bin bounds, and refines the sampled pairs in rounds.

A tree of cases is grown from the case with no constraint: a case whose bound
is not below the target is split at the radius and count, N(r) < c or
N(r) >= c, that best separate it, over the candidate radii below.

    python3 radial_case_sdp.py M --spec radial_certificates/case_M_spec.json

writes radial_certificates/case_M.json: the tree given by the spec (a leaf
marked residual carries no certificate) and one certificate per other leaf; a
spec with "antipodal": true also searches the pairs at u = -1, where the caps
are disjoint, between the samples, and "margins" replaces the default list of
margins;
    python3 radial_case_sdp.py M --redo SPEC k
recomputes leaf k alone, identically; and

    python3 radial_case_sdp.py M [depth] [workers]

grows a tree by trying the splits below (exploration).  With --level L
(a rational) every one of these proves U(Y) < L in place of
U(Y) < 9 pi^2/8 - 8, the multiplier s then applying to sum_b n_b S(e_b) >= L,
and the certificates go to radial_certificates/density_M.json with the level
recorded (prop:levels).
"""
import json
import math
import os
import sys
import time
from fractions import Fraction as Fr
from multiprocessing import Pool

import numpy as np
import cvxpy as cp

from truncated_search import pair, S
from radial_count_sdp import pbasis, ubasis, amax, samples, dyadic, violation, TARGET, R6, C1, C2

D, R = 12, 4
RKS = Fr(20161, 10000)                 # below 2/sqrt(1 - 0.016) = 2.016194...
DMAX = Fr(24494898, 10000000)          # above sqrt 6
CANDS = [RKS, Fr(205, 100), Fr(21, 10), Fr(215, 100), Fr(22, 10), Fr(225, 100), Fr(23, 10), Fr(235, 100), Fr(24, 10)]
EPS = 2e-5


def norm_case(M, case):
    """add N(2.0161) <= 24 and propagate the bounds along the radii."""
    case = dict(case)
    lo, hi = case.get(RKS, (0, M))
    case[RKS] = (lo, min(hi, 24))
    rs = sorted(case)
    for i, ri in enumerate(rs):
        lo, hi = case[ri]
        for rj in rs[:i]:
            lo = max(lo, case[rj][0])
        for rj in rs[i + 1:]:
            hi = min(hi, case[rj][1])
        case[ri] = (lo, hi)
    return case


def bins(case):
    rs = sorted(case)
    return rs, [Fr(2)] + rs + [DMAX]


def feasible(M, case):
    """is some count vector of the case compatible with sum_b n_b S(e_b) >= target (floating point)?"""
    case = norm_case(M, case)
    rs, edges = bins(case)
    if any(lo > hi for lo, hi in case.values()):
        return False
    nb = len(edges) - 1
    n = cp.Variable(nb, nonneg=True)
    L = np.tril(np.ones((nb, nb)))[:len(rs)]
    lo = np.array([case[x][0] for x in rs], float); hi = np.array([case[x][1] for x in rs], float)
    Sl = np.array([float(S(float(edges[b]))) for b in range(nb)])
    pr = cp.Problem(cp.Maximize(Sl @ n), [cp.sum(n) == M, L @ n >= lo, L @ n <= hi])
    pr.solve(solver='CLARABEL')
    return pr.status == 'optimal' and pr.value >= TARGET - 1e-9


def margins(pv, eps):
    """the margin of the pair inequality: eps = (near, far), near where the caps overlap (Pi > 0);
    a single number is used for both."""
    near, far = (eps, eps) if np.isscalar(eps) else eps
    return np.where(pv > 0, near, far)


def far_margin(eps):
    return eps if np.isscalar(eps) else eps[1]


def solve(M, case, P, Q, W, eps):
    case = norm_case(M, case)
    rs, edges = bins(case)
    nb, J = len(edges) - 1, len(rs)
    pv = pair(P / 2, Q / 2, W)
    A = [cp.Variable((R + 1, R + 1), symmetric=True) for _ in range(D + 1)]
    z, t, m = cp.Variable(R + 1), cp.Variable(), cp.Variable(nb)
    bp, bq, uk = pbasis(P, R), pbasis(Q, R), ubasis(W, D)
    K = sum(cp.multiply(uk[:, k], cp.sum(cp.multiply(bp @ A[k], bq), axis=1)) for k in range(D + 1))
    Z = cp.bmat([[A[0], cp.reshape(z, (R + 1, 1), order='C')], [cp.reshape(z, (1, R + 1), order='C'), cp.reshape(t, (1, 1), order='C')]])
    cons = [A[k] - 1e-9 * np.eye(R + 1) >> 0 for k in range(1, D + 1)] + [Z - 1e-9 * np.eye(R + 2) >> 0, K <= pv - margins(pv, eps)]
    for b in range(nb):
        dd = np.linspace(float(edges[b]), float(edges[b + 1]), 160)
        bd = pbasis(dd, R)
        Kd = sum(cp.sum(cp.multiply(bd @ A[k], bd), axis=1) for k in range(D + 1))
        cons.append(S(dd) + Kd / 2 - bd @ z <= m[b] - far_margin(eps))
    a, c, s, g = cp.Variable(J, nonneg=True), cp.Variable(J, nonneg=True), cp.Variable(nonneg=True), cp.Variable()
    Sl = np.array([float(S(float(edges[b]))) for b in range(nb)])
    for b in range(nb):
        inside = [j for j in range(J) if edges[b + 1] <= rs[j]]
        cons.append(sum(a[j] - c[j] for j in inside) + g - s * Sl[b] >= m[b] + far_margin(eps))
    lo = np.array([case[x][0] for x in rs], float); hi = np.array([case[x][1] for x in rs], float)
    prob = cp.Problem(cp.Minimize(hi @ a - lo @ c + M * g - TARGET * s + t / 2), cons)
    try:
        prob.solve(solver='CLARABEL')
    except Exception:
        return None
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return None
    return dict(value=prob.value, A=A, z=z, t=t, m=m, a=a, c=c, s=s, g=g, rs=rs, edges=edges)


P0, Q0, W0 = samples(19, 40)


CACHE = {}


def ckey(M, case):
    return '%d|' % M + ';'.join('%s:%d:%d' % (k, lo, hi) for k, (lo, hi) in sorted(norm_case(M, case).items()))


def value(args):
    M, case = args
    if not feasible(M, case):
        return -1.0
    sol = solve(M, case, P0, Q0, W0, 0.0)
    return float('inf') if sol is None else float(sol['value'])


def values(pool, jobs):
    """values of the cases, through the cache (radial_certificates/case_cache.json)."""
    todo = [j for j in jobs if ckey(*j) not in CACHE]
    uniq = list({ckey(*j): j for j in todo}.values())
    for j, v in zip(uniq, pool.map(value, uniq) if uniq else []):
        CACHE[ckey(*j)] = v
    if uniq:
        json.dump(CACHE, open(CACHE_FILE, 'w'))
    return [CACHE[ckey(*j)] for j in jobs]


CACHE_FILE = 'radial_certificates/case_cache.json'
LEVEL = None                           # set by --level: the rational level, as a string
OUT = 'radial_certificates/case_%d.json'


def describe(case):
    return ', '.join('%d <= N(%s) <= %d' % (case[k][0], float(k), case[k][1]) for k in sorted(case)) or 'no constraint'


def grow(M, case, depth, pool, ind=''):
    """returns the tree: {'case', 'value'} for a leaf, {'case', 'split': (r, c), 'children'} otherwise."""
    v = values(pool, [(M, case)])[0]
    tag = 'excluded by the caps' if v < 0 else ('closed' if v < TARGET - 1e-3 else 'open')
    print('%s[%s] %s %s' % (ind, describe(case), '' if v < 0 else '%.5f' % v, tag), flush=True)
    if v < TARGET - 1e-3 or depth == 0:
        return {'case': case, 'value': v}
    ncase = norm_case(M, case)
    cands = []
    for rc in CANDS:
        lo0, hi0 = ncase.get(rc, (0, M))
        for rj, (l2, h2) in ncase.items():
            if rj <= rc: lo0 = max(lo0, l2)
            if rj >= rc: hi0 = min(hi0, h2)
        for cc in range(max(lo0 + 1, 18), min(hi0, 28) + 1):
            c1 = dict(case); c1[rc] = (lo0, cc - 1)
            c2 = dict(case); c2[rc] = (cc, hi0)
            cands.append((rc, cc, c1, c2))
    vals = values(pool, [(M, c) for _, _, c1, c2 in cands for c in (c1, c2)])
    best = None
    for i, (rc, cc, c1, c2) in enumerate(cands):
        v1, v2 = vals[2 * i], vals[2 * i + 1]
        score = (-(int(v1 < TARGET - 1e-3) + int(v2 < TARGET - 1e-3)), max(v1, v2))
        if best is None or score < best[0]:
            best = (score, rc, cc, c1, c2)
    _, rc, cc, c1, c2 = best
    print('%s split: N(%s) < %d or N(%s) >= %d' % (ind, float(rc), cc, float(rc), cc), flush=True)
    return {'case': case, 'split': (rc, cc),
            'children': [grow(M, c1, depth - 1, pool, ind + '  '), grow(M, c2, depth - 1, pool, ind + '  ')]}


def leaves(tree):
    if 'children' in tree:
        return [x for ch in tree['children'] for x in leaves(ch)]
    return [tree]


def violation_antipodal(A, rng):
    """violation, together with 400000 pairs at u = -1 and just above it, where the caps
    are disjoint (Pi = 0) and the random pairs of violation are sparse."""
    viol, p_, q_, u_ = violation(A, D, R, rng)
    Av = np.array([a.value for a in A])
    n = 400000
    p = 2 + (R6 - 2) * rng.random(n); q = 2 + (R6 - 2) * rng.random(n)
    u = -1 + 2e-2 * rng.random(n) ** 2
    u[: n // 2] = -1.0
    v = np.einsum('nk,na,kab,nb->n', ubasis(u, D), pbasis(p, R), Av, pbasis(q, R)) - pair(p / 2, q / 2, u)
    return np.r_[viol, v], np.r_[p_, p], np.r_[q_, q], np.r_[u_, u]


def certify(M, case, rng, eps=EPS, antipodal=False):
    """the certificate of a leaf, with margins and refined pair samples, in exact dyadic form;
    antipodal adds the pairs of violation_antipodal to the search between the samples."""
    P, Q, W = samples(19, 40)
    converged = False
    for rnd in range(16):
        sol = solve(M, case, P, Q, W, eps)
        if sol is None:
            return None
        viol, p_, q_, u_ = violation_antipodal(sol['A'], rng) if antipodal else violation(sol['A'], D, R, rng)
        slack = viol + margins(pair(p_ / 2, q_ / 2, u_), eps) / 2
        if np.isscalar(eps):
            print('    round %d: bound %.5f, largest K - Pi between the samples %.2e' % (rnd, sol['value'], viol.max()), flush=True)
        else:
            print('    round %d: bound %.5f, largest K - Pi between the samples %.2e, against half the margin %.2e'
                  % (rnd, sol['value'], viol.max(), slack.max()), flush=True)
        if slack.max() <= 0:
            converged = True
            break
        worst = np.argsort(slack)[-6000:]
        P, Q, W = np.r_[P, p_[worst]], np.r_[Q, q_[worst]], np.r_[W, u_[worst]]
    if not converged:
        return None
    Ak = [np.array(x.value) + (2.0 ** -30) * np.eye(R + 1) * (k > 0) for k, x in enumerate(sol['A'])]
    Ak = [(x + x.T) / 2 for x in Ak]
    return {'case': [[str(k), lo, hi] for k, (lo, hi) in sorted(norm_case(M, case).items())],
            'edges': [str(e) for e in sol['edges']],
            'A': [[[str(dyadic(x[i, j])) for j in range(R + 1)] for i in range(R + 1)] for x in Ak],
            'z': [str(dyadic(v)) for v in sol['z'].value],
            'm': [str(dyadic(v + far_margin(eps) / 2)) for v in sol['m'].value],
            'a': [str(dyadic(max(v, 0.0))) for v in sol['a'].value],
            'c': [str(dyadic(max(v, 0.0))) for v in sol['c'].value],
            's': str(dyadic(max(float(sol['s'].value), 0.0))),
            'float_bound': sol['value'], 'eps': eps if np.isscalar(eps) else list(eps)}


def tree_json(tree):
    out = {'case': [[str(k), lo, hi] for k, (lo, hi) in sorted(tree['case'].items())]}
    if tree.get('residual'):
        out['residual'] = True
    if 'children' in tree:
        out['split'] = [str(tree['split'][0]), tree['split'][1]]
        out['children'] = [tree_json(ch) for ch in tree['children']]
    return out


def implied(case, M, r):
    """bounds on N(r) implied by a case, N(2.0161) <= 24 and N(dmax) = M (as radial_case_check.implied)."""
    lo, hi = 0, M
    for rj, (l, h) in case.items():
        if rj <= r:
            lo = max(lo, l)
        if rj >= r:
            hi = min(hi, h)
    if RKS >= r:
        hi = min(hi, 24)
    return lo, hi


def from_spec(M, spec, case=None):
    """a tree from a nested spec: {"split": [r, c], "children": [spec, spec]}, {} or {"residual": true}."""
    case = {} if case is None else case
    if 'split' not in spec:
        return {'case': case, 'residual': bool(spec.get('residual'))}
    r, c = Fr(spec['split'][0]), spec['split'][1]
    lo, hi = implied(case, M, r)
    c1 = dict(case); c1[r] = (lo, c - 1)
    c2 = dict(case); c2[r] = (c, hi)
    return {'case': case, 'split': (r, c),
            'children': [from_spec(M, spec['children'][0], c1), from_spec(M, spec['children'][1], c2)]}


def certify_leaf(M, case, k, antipodal=False, eps_list=None):
    """the certificate of leaf k (1-based): margins 2e-5, 5e-6, 2e-6, 1e-6 in turn, until the
    refinement converges with the bound 2e-4 below the target; the seed depends on k only.
    A spec may give its own list of margins, each a number or a pair (near, far): near
    where the caps overlap, far where they are disjoint and in the bin bounds."""
    rng = np.random.default_rng(11 + k)
    for eps in (eps_list or (2e-5, 5e-6, 2e-6, 1e-6)):
        eps = eps if np.isscalar(eps) else tuple(eps)
        cert = certify(M, case, rng, eps, antipodal)
        if cert is not None and cert['float_bound'] < TARGET - 2e-4:
            return cert
        print('    margin %s: %s' % (eps, 'not converged' if cert is None else 'bound %.5f' % cert['float_bound']), flush=True)
    return None


def main_spec(M, spec_file):
    t0 = time.time()
    spec = json.load(open(spec_file))
    tree = from_spec(M, spec)
    certs = []
    for k, x in enumerate(leaves(tree), 1):
        print('leaf %d [%s]%s' % (k, describe(x['case']), ' (residual, not certified)' if x['residual'] else ''), flush=True)
        ncase = [[str(k), lo, hi] for k, (lo, hi) in sorted(norm_case(M, x['case']).items())]
        if x['residual']:
            certs.append({'case': ncase, 'residual': True})
            continue
        if not feasible(M, x['case']):
            certs.append({'case': ncase, 'excluded': True})
            continue
        cert = certify_leaf(M, x['case'], k, bool(spec.get('antipodal')), spec.get('margins'))
        if cert is None:
            print('no certificate for this leaf'); return
        certs.append(cert)
    os.makedirs('radial_certificates', exist_ok=True)
    out = {'M': M, 'D': D, 'r': R, 'c1': str(C1), 'c2': str(C2), 'dmax': str(DMAX), 'tree': tree_json(tree), 'leaves': certs}
    if LEVEL is not None:
        out['level'] = LEVEL
    json.dump(out, open(OUT % M, 'w'), indent=1)
    print('wrote %s [%.0f s]' % (OUT % M, time.time() - t0))


def redo_leaf(M, spec_file, k):
    """recompute the certificate of leaf k (1-based) of radial_certificates/case_M.json, as main_spec does."""
    spec = json.load(open(spec_file))
    x = leaves(from_spec(M, spec))[k - 1]
    data = json.load(open(OUT % M))
    print('leaf %d [%s]' % (k, describe(x['case'])), flush=True)
    cert = certify_leaf(M, x['case'], k, bool(spec.get('antipodal')), spec.get('margins'))
    if cert is None:
        print('no certificate for this leaf'); return
    data['leaves'][k - 1] = cert
    json.dump(data, open(OUT % M, 'w'), indent=1)
    print('replaced leaf %d of %s' % (k, OUT % M))


def main():
    global TARGET, LEVEL, OUT, CACHE_FILE
    if '--level' in sys.argv:
        i = sys.argv.index('--level')
        LEVEL = str(Fr(sys.argv[i + 1]))
        TARGET = float(Fr(LEVEL))
        OUT = 'radial_certificates/density_%d.json'
        del sys.argv[i:i + 2]
        CACHE_FILE = 'radial_certificates/density_cache_%s.json' % LEVEL.replace('/', '_')
    if len(sys.argv) > 4 and sys.argv[2] == '--redo':
        return redo_leaf(int(sys.argv[1]), sys.argv[3], int(sys.argv[4]))
    if len(sys.argv) > 3 and sys.argv[2] == '--spec':
        return main_spec(int(sys.argv[1]), sys.argv[3])
    M = int(sys.argv[1])
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    t0 = time.time()
    os.makedirs('radial_certificates', exist_ok=True)
    if os.path.exists(CACHE_FILE):
        CACHE.update(json.load(open(CACHE_FILE)))
    with Pool(workers) as pool:
        tree = grow(M, {}, depth, pool)
    lv = leaves(tree)
    open_ = [x for x in lv if x['value'] >= TARGET - 1e-3]
    print('M = %d: %d leaves, %d open [%.0f s]' % (M, len(lv), len(open_), time.time() - t0), flush=True)
    if open_:
        for x in open_:
            print('  open: [%s] %.5f' % (describe(x['case']), x['value']))
        return
    rng = np.random.default_rng(11)
    certs = []
    for x in lv:
        print('  certificate for [%s]' % describe(x['case']), flush=True)
        if x['value'] < 0:
            certs.append({'case': [[str(k), lo, hi] for k, (lo, hi) in sorted(norm_case(M, x['case']).items())], 'excluded': True})
            continue
        cert = certify(M, x['case'], rng)
        if cert is None or cert['float_bound'] >= TARGET:
            print('  certificate with margins failed'); return
        certs.append(cert)
    os.makedirs('radial_certificates', exist_ok=True)
    json.dump({'M': M, 'D': D, 'r': R, 'c1': str(C1), 'c2': str(C2), 'dmax': str(DMAX), 'tree': tree_json(tree), 'leaves': certs},
              open('radial_certificates/case_%d.json' % M, 'w'), indent=1)
    print('wrote radial_certificates/case_%d.json [%.0f s]' % (M, time.time() - t0))


if __name__ == '__main__':
    main()
