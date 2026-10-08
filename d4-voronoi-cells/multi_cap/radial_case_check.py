#!/usr/bin/env python3
"""
radial_case_check.py -- proves the certificates written by radial_case_sdp.py:
every packing set of exactly M centres y with 2 <= |y| < sqrt6 has union of
caps U(Y) < 9 pi^2/8 - 8, so T(Y) > 8 (statement (C_M) at the count M).  A
certificate that records a level L proves U(Y) < L instead, by the same steps
with L in place of 9 pi^2/8 - 8 (prop:levels), and prints as well the bound
for U(Y) that the same maximum gives with no assumption on Y (s >= 0 and
S >= 0), the entry of tab:levels.

Exact rational arithmetic, Bernstein bounds and Arb ball arithmetic, as in
radial_count_check.py, whose routines it uses.  N(r) is the number of centres
within r.

1. The tree covers every packing.  Its root has no constraint; each inner node
   splits at a radius r and count c into N(r) <= c - 1 and N(r) >= c, the
   other bound of each child being at least as wide as what the node's own
   bounds imply for N(r) (N is nondecreasing in r, N(dmax) = M, and
   N(2.0161) <= 24 for every packing, by the stability of the kissing number).
   The cases of the leaves are the leaves of the tree.
2. A leaf marked excluded has sum_b n_b S(e_b) < 9 pi^2/8 - 8 for every
   integer count vector n of its case (e_b the lower end of bin b; S
   decreases, so U(Y) <= sum_y S(|y|) <= sum_b n_b S(e_b)): the largest value
   is found exactly by dynamic programming over the bins, with S(e_b) rounded
   up from Arb.
3. For every other leaf, with its bins b and bounds m_b:
   (a) the A_k are positive semidefinite and [[A_0, z], [z^T, t]] with t
       rounded up from z^T A_0^{-1} z, by exact LDL^T;
   (b) K <= Pi on the admissible domain, by the branch and bound of
       radial_count_check.py, processed in batches (pair_bb_fast.py; the same
       boxes, the same tests, and the same counts);
   (c) f(d) = S(d) + K(d, d, 1)/2 - z . p(d) <= m_b on bin b, by Bernstein
       bounds on subintervals, an interval that meets two bins needing the
       smaller bound;
   (d) with the multiplier s >= 0 of the certificate, for every integer count
       vector n of the case, sum_b n_b m_b <= sum_b n_b (m_b + s S(e_b)) - s
       (9 pi^2/8 - 8) whenever sum_b n_b S(e_b) >= 9 pi^2/8 - 8 (which a
       counterexample satisfies); the largest value of the right side is found
       exactly by dynamic programming, and it plus t/2 is below
       9 pi^2/8 - 8, in Arb.

A leaf may be marked residual: it carries no certificate, and the check then
proves that a packing of M centres with T <= 8 lies in a residual case.

Usage: python3 radial_case_check.py radial_certificates/case_M.json
"""
import json
import sys
import time
from fractions import Fraction as Fr

import numpy as np
from flint import arb

from radial_count_check import check, A_, ldl_psd, solve_exact, pmul, padd, cheb_T, cheb_U, bernstein, split, S_arb, Pi_lower

RKS = Fr(20161, 10000)


def arb_up(x):
    """a rational (a double) at least every point of the ball x."""
    v = float(x.upper())
    while not (arb(v) >= x.upper()):
        v = float(np.nextafter(v, np.inf))
    return Fr(v)


def implied(case, M, r):
    """bounds on N(r) implied by the bounds of a case (and N(2.0161) <= 24, N(dmax) = M)."""
    lo, hi = 0, M
    for rj, (l, h) in case.items():
        if rj <= r:
            lo = max(lo, l)
        if rj >= r:
            hi = min(hi, h)
    if RKS >= r:
        hi = min(hi, 24)
    return lo, hi


def parse_case(lst):
    return {Fr(k): (lo, hi) for k, lo, hi in lst}


def check_tree(node, M, out):
    case = parse_case(node['case'])
    if 'children' not in node:
        out.append(case)
        return True
    r, c = Fr(node['split'][0]), node['split'][1]
    lo, hi = implied(case, M, r)
    c1, c2 = parse_case(node['children'][0]['case']), parse_case(node['children'][1]['case'])
    ok = True
    for ch, (want_lo, want_hi) in ((c1, (None, c - 1)), (c2, (c, None))):
        rest = {k: v for k, v in ch.items() if k != r}
        pr = {k: v for k, v in case.items() if k != r}
        ok &= rest == pr and r in ch
        l2, h2 = ch[r]
        if want_hi is not None:
            ok &= h2 == want_hi and l2 <= lo
        else:
            ok &= l2 == want_lo and h2 >= hi
    return ok and check_tree(node['children'][0], M, out) and check_tree(node['children'][1], M, out)


def normalised(case, M):
    """the case with N(2.0161) <= 24 added and the bounds propagated, as radial_case_sdp.norm_case."""
    case = dict(case)
    lo, hi = case.get(RKS, (0, M))
    case[RKS] = (lo, min(hi, 24))
    rs = sorted(case)
    out = {}
    for i, ri in enumerate(rs):
        lo, hi = case[ri]
        for rj in rs[:i]:
            lo = max(lo, case[rj][0])
        for rj in rs[i + 1:]:
            hi = min(hi, case[rj][1])
        out[ri] = (lo, hi)
    return out


def dp_max(M, rs, bounds, weights):
    """max of sum_b n_b w_b over integer n >= 0 with sum n = M and lo_j <= n_0 + ... + n_j <= hi_j
    (bins b = 0..len(rs), bin j ending at rs[j]); exact rationals.  None if no vector."""
    nb = len(rs) + 1
    best = {0: Fr(0)}                       # cumulative count -> best value
    for b in range(nb):
        new = {}
        for cum, val in best.items():
            for n in range(M - cum + 1):
                c2 = cum + n
                if b < len(rs):
                    lo, hi = bounds[b]
                    if not (lo <= c2 <= hi):
                        continue
                elif c2 != M:
                    continue
                v = val + n * weights[b]
                if c2 not in new or v > new[c2]:
                    new[c2] = v
        best = new
    return best.get(M)


def pair_check(A, D, r, c1, c2, dmax, t0):
    """the branch and bound of radial_count_check.py for K <= Pi (step 2 there)."""
    alpha, beta = (4 - c1) / c2, 2 * (dmax - 2) / c2
    umax = (2 * dmax ** 2 - 4) / (2 * dmax ** 2)
    Ts = [bernstein(p, r) for p in cheb_T(alpha, beta, r)]
    Us = [bernstein([x / (k + 1) for x in p], D) for k, p in enumerate(cheb_U(Fr(-1), umax + 1, D))]
    B0 = [[[Fr(0)] * (D + 1) for _ in range(r + 1)] for _ in range(r + 1)]
    for k in range(D + 1):
        for a in range(r + 1):
            for b in range(r + 1):
                if A[k][a][b] == 0:
                    continue
                for i in range(r + 1):
                    ci = A[k][a][b] * Ts[a][i]
                    if ci == 0:
                        continue
                    for j in range(r + 1):
                        cij = ci * Ts[b][j]
                        if cij == 0:
                            continue
                        row = B0[i][j]
                        for l in range(D + 1):
                            row[l] += cij * Us[k][l]
    Bf = np.array([[[float(x) for x in row] for row in plane] for plane in B0])
    Cmax = max(abs(x) for plane in B0 for row in plane for x in row)
    u53 = 2.0 ** -53
    depth_max = 80
    err = float(Cmax) * u53 * (1 + depth_max * max(r, D)) * 1.01

    def dval(i, n):
        return 2 + (dmax - 2) * Fr(i, 2 ** n)

    def uval(i, n):
        return -1 + (umax + 1) * Fr(i, 2 ** n)

    def amaxq(d1, d2):
        return (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)

    from truncated_search import pair as pair_float
    stack = [((0, 0), (0, 0), (0, 0), Bf, 0)]
    closed = dropped = arbcalls = 0
    memo = {}
    while stack:
        (i1, n1), (i2, n2), (i3, n3), B, depth = stack.pop()
        dlo, dhi = dval(i1, n1), dval(i1 + 1, n1)
        elo, ehi = dval(i2, n2), dval(i2 + 1, n2)
        ulo = uval(i3, n3)
        if dlo > ehi or ulo > amaxq(dhi, ehi):
            dropped += 1
            continue
        kub = float(np.nextafter(float(B.max()) + err, np.inf))
        if kub <= 0:
            closed += 1
            continue
        pf = float(pair_float(float(dhi) / 2, float(ehi) / 2, float(ulo)))
        if kub <= pf - 1e-9:
            key = (dhi, ehi, ulo)
            if key not in memo:
                memo[key] = Pi_lower(dhi, ehi, ulo); arbcalls += 1
            if kub <= memo[key]:
                closed += 1
                continue
        if depth >= depth_max:
            return False, 'depth limit at d in [%s, %s], d\' in [%s, %s], u from %s' % (dlo, dhi, elo, ehi, ulo)
        w = [Fr(1, 2 ** n1), Fr(1, 2 ** n2), Fr(1, 2 ** n3) * (umax + 1) / (dmax - 2) / 3]
        ax = max(range(3), key=lambda j: w[j])
        L, Rt = split(B, ax)
        if ax == 0:
            stack += [((2 * i1, n1 + 1), (i2, n2), (i3, n3), L, depth + 1), ((2 * i1 + 1, n1 + 1), (i2, n2), (i3, n3), Rt, depth + 1)]
        elif ax == 1:
            stack += [((i1, n1), (2 * i2, n2 + 1), (i3, n3), L, depth + 1), ((i1, n1), (2 * i2 + 1, n2 + 1), (i3, n3), Rt, depth + 1)]
        else:
            stack += [((i1, n1), (i2, n2), (2 * i3, n3 + 1), L, depth + 1), ((i1, n1), (i2, n2), (2 * i3 + 1, n3 + 1), Rt, depth + 1)]
    return True, '%d boxes closed, %d dropped, %d values of Pi in Arb [%.0f s]' % (closed, dropped, arbcalls, time.time() - t0)


def pair_job(A, D, r, c1, c2, dmax):
    from pair_bb_fast import pair_check_fast
    return pair_check_fast(A, D, r, c1, c2, dmax)


def bracket_check(A, z, D, r, c1, c2, dmax, edges, m):
    """f <= m_b on each bin; bins [e_0, e_1], (e_1, e_2], ..."""
    alpha, beta = (4 - c1) / c2, 2 * (dmax - 2) / c2
    Tq = cheb_T(alpha, beta, r)
    Asum = [[sum(A[k][a][b] for k in range(D + 1)) for b in range(r + 1)] for a in range(r + 1)]
    q = [Fr(0)]
    for a in range(r + 1):
        for b in range(r + 1):
            if Asum[a][b]:
                q = padd(q, pmul(Tq[a], Tq[b]), 1, Asum[a][b] / 2)
        q = padd(q, Tq[a], 1, -z[a])
    qe = bernstein(q, 2 * r)
    qb = np.array([float(x) for x in qe])
    qmax = max(abs(x) for x in qe)
    err1 = float(qmax) * 2.0 ** -53 * (1 + 60 * 2 * r) * 1.01

    def dval(i, n):
        return 2 + (dmax - 2) * Fr(i, 2 ** n)

    stack = [(0, 0, qb, 0)]
    nint = 0
    worst = None
    while stack:
        i, n, B, depth = stack.pop()
        dlo, dhi = dval(i, n), dval(i + 1, n)
        # the bins that [dlo, dhi] meets
        meet = [b for b in range(len(edges) - 1)
                if not (dhi < edges[b] or dlo > edges[b + 1] or (b > 0 and dhi == edges[b]))]
        mb = min(m[b] for b in meet)
        bound = S_arb(dlo) + arb(float(np.nextafter(float(B.max()) + err1, np.inf)))
        if bound < A_(mb):
            nint += 1
            gap = float((A_(mb) - bound).lower())
            worst = gap if worst is None else min(worst, gap)
            continue
        if depth >= 60:
            return False, 'at d = %s' % dlo
        L, Rt = split(B, 0)
        stack += [(2 * i, n + 1, L, depth + 1), (2 * i + 1, n + 1, Rt, depth + 1)]
    return True, '%d intervals, least room %.2e' % (nint, worst)


def main():
    t0 = time.time()
    c = json.load(open(sys.argv[1]))
    M, D, r = c['M'], c['D'], c['r']
    c1, c2, dmax = Fr(c['c1']), Fr(c['c2']), Fr(c['dmax'])
    if 'level' in c:
        target = A_(Fr(c['level']))
        print('U(Y) < %s for M = %d: two-point certificates, degree %d in the angle and %d in the distance, one per case'
              % (c['level'], M, D, r))
    else:
        target = 9 * arb.pi() ** 2 / 8 - 8
        print('statement (C_M) at M = %d: two-point certificates, degree %d in the angle and %d in the distance, one per case' % (M, D, r))
    check('dmax exceeds sqrt 6', dmax ** 2 > 6)
    tl = []
    ok = check_tree(c['tree'], M, tl)
    check('the case tree covers every packing (each split is into N(r) < c and N(r) >= c)', ok and not c['tree']['case'],
          '%d leaves' % len(tl))
    lv = c['leaves']
    residual = []
    # the pair branch and bounds, the longest step, run for all leaves in parallel
    jobs = [([[[Fr(x) for x in row] for row in a] for a in leaf['A']], D, r, c1, c2, dmax)
            for leaf in lv if 'A' in leaf]
    from multiprocessing import Pool
    with Pool(min(4, max(1, len(jobs)))) as pool:
        pair_results = iter(pool.starmap(pair_job, jobs))
    check('one certificate per leaf, in order', len(lv) == len(tl) and all(normalised(x, M) == parse_case(y['case']) for x, y in zip(tl, lv)))
    for n_, leaf in enumerate(lv):
        case = parse_case(leaf['case'])
        rs = sorted(case)
        edges = [Fr(2)] + rs + [dmax]
        bounds = [case[x] for x in rs]
        Sup = [arb_up(S_arb(e)) for e in edges[:-1]]
        desc = ', '.join('%d <= N(%s) <= %d' % (lo, float(k), hi) for k, (lo, hi) in sorted(case.items()))
        print('leaf %d: %s' % (n_ + 1, desc), flush=True)
        if leaf.get('residual'):
            residual.append(desc)
            print('  [----]   residual case: no certificate; a counterexample at this count must lie here', flush=True)
            continue
        if leaf.get('excluded'):
            v = dp_max(M, rs, bounds, Sup)
            check('  excluded: every count vector has sum of caps below the target', v is None or A_(v) < target,
                  'largest %s' % ('none' if v is None else '%.6f' % float(v)))
            continue
        A = [[[Fr(x) for x in row] for row in a] for a in leaf['A']]
        z = [Fr(x) for x in leaf['z']]
        m = [Fr(x) for x in leaf['m']]
        s = Fr(leaf['s'])
        check('  edges as in the case', [Fr(e) for e in leaf['edges']] == edges and len(m) == len(edges) - 1)
        good = all(ldl_psd(A[k])[0] for k in range(1, D + 1))
        g0, piv = ldl_psd(A[0])
        w = solve_exact(A[0], z)
        tq = sum(a * b for a, b in zip(z, w))
        t = Fr(-(-tq.numerator * 2 ** 48 // tq.denominator), 2 ** 48)
        Zm = [row[:] + [z[i]] for i, row in enumerate(A[0])] + [z + [t]]
        gz, _ = ldl_psd(Zm)
        check('  A_k and [[A_0, z], [z^T, t]] positive semidefinite (exact LDL^T)', good and g0 and all(p > 0 for p in piv) and gz,
              't = %.9f' % float(t))
        okp, msg = next(pair_results)
        check('  K <= Pi on the admissible domain', okp, msg)
        okb, msg = bracket_check(A, z, D, r, c1, c2, dmax, edges, m)
        check('  f <= m_b on every bin', okb, msg)
        check('  s >= 0', s >= 0)
        wts = [mb + s * sb for mb, sb in zip(m, Sup)]
        v = dp_max(M, rs, bounds, wts)
        total = A_(v) - A_(s) * target + A_(t) / 2
        check('  largest sum_b n_b m_b over the case, plus t/2, below the target', total < target,
              '%s < %s' % (total.str(8), target.str(8)))
        if 'level' in c:
            # s >= 0 and S >= 0, so the same maximum bounds U(Y) with no assumption on Y
            print('  with no assumption on the count vector: U(Y) <= %s' % (A_(v) + A_(t) / 2).str(8), flush=True)
    if residual:
        print('PASS: a packing with exactly %d centres within sqrt 6 of a centre and T <= 8 there lies in %s: %s [%.0f s]'
              % (M, 'the residual case' if len(residual) == 1 else 'one of the residual cases', '; '.join(residual), time.time() - t0))
    elif 'level' in c:
        print('PASS: every packing set of exactly %d centres within sqrt 6 has union of caps U(Y) < %s [%.0f s]'
              % (M, c['level'], time.time() - t0))
    else:
        print('PASS: every packing with exactly %d centres within sqrt 6 of a centre has T > 8 there [%.0f s]' % (M, time.time() - t0))


if __name__ == '__main__':
    main()
