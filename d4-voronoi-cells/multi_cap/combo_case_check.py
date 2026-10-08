#!/usr/bin/env python3
"""
combo_case_check.py -- combo30_check.py for a case given in a JSON file, as written for
gap_closure/CM/combo_gen2.py: statement (C) at M centres, with the two-point kernel
labelled by distance and the three-point kernel on directions typed by distance range.

The case file gives M, the types and their distance ranges, the bins (each inside the
range of one type) and constraints [bins, lo, hi] on the number of centres in groups
of bins, each of which must be a proved count (prop:C-radial, thm:kissing-stable).
The check is that of combo30_check.py: exact positivity, the pair inequalities by
tensor Bernstein bounds, the bin brackets, the triple inequalities by Taylor forms,
and then, for every integer vector of bin counts that the constraints allow, the bound
    sum_b n_b (m_b + p_type(b)) + t/2 + sum N_st c2_st + sum N_str c3_str
in exact arithmetic, the largest of which must lie below 9 pi^2/8 - 8.

Usage: python3 combo_case_check.py case.json cert.npz d3 [margin2 margin3 marginm]

With PRECHECK=1 in the environment it stops after the thresholds: it samples and
refines every inequality as the check does, skips the branch and bounds, and prints
the bound over the count vectors that those thresholds would give.  That is floating
point and proves nothing; it tells whether a certificate is worth the full check.

With BORDER=1 the matrix [[A_0, z], [z^T, t]] is rounded to an exact positive definite
matrix as a whole, eigenvalues clipped at 0 and shifted as for the other blocks, instead
of rounding A_0 and z and setting t = z^T A_0^{-1} z rounded up; when A_0 is nearly
singular this keeps t near the solver's value.  Either way the bordered matrix is
checked positive semidefinite by exact LDL^T.

With LEVEL=<decimal> the final comparison is with that number instead of 9 pi^2/8 - 8:
the check then proves the density level U(Y) < LEVEL for every packing set of the case,
whose constraints hold whenever U(Y) >= 9 pi^2/8 - 8, so that U(Y) < max(9 pi^2/8 - 8,
LEVEL) for every packing set of M centres.

With TRIPLE_EXTRA=<kind>:<x>[,<kind>:<x>...], e.g. TRIPLE_EXTRA=FFF:1e-5, the threshold
c3 of each named triple kind is raised by x beyond the sampled and refined maximum plus
margin3.  Sampling and SLSQP can stop below a narrow peak of a triple polynomial, and the
branch and bound then fails on that kind alone; the extra room lets it close while every
other threshold stays as it was.  The final bound uses the raised thresholds.

With ONLY=<kind>[,<kind>...], e.g. ONLY=FFF, the thresholds are computed exactly as in
the full check (the sampling sequence is the same), but only the branch and bound of the
named triple kinds runs: the pair inequalities, the bins and the other triple kinds are
skipped.  It ends with a line PARTIAL and proves the named inequalities only.

With EXTRA_CASES=<case.json>[,<case.json>...] the bound with the same thresholds is
also evaluated, in exact arithmetic, over the count vectors of each listed case file
(same M, types and bins, other constraints) and compared with the target.  The pair
and triple inequalities, the bins and the positivity do not depend on the count
vectors, so a passing full check of the main case together with these comparisons
proves the listed cases as well.  The thresholds depend on the main case file only
through the triple kinds it allows (the sampling sequence is fixed), so with
PRECHECK=1 and the main case and arguments of a passing full check the run reproduces
that check's thresholds, printed for comparison, and settles the listed cases at the
cost of the sampling alone.

With REFINE_TRIPLES=<b>-<b'>-<b''>[,...] (bins numbered from 1, e.g. 1-3-7) each listed
combination of bins gets a threshold of its own: a triple of centres in the bins b, b',
b'' has inner products at most a(hi_b, hi_b'), a(hi_b, hi_b''), a(hi_b', hi_b''), so
TRIPLE3 of its kind is sampled, refined and checked by branch and bound on that smaller
set, and in the bound the triples of those bins are counted with the smaller threshold
instead of that of the kind.  These thresholds are sampled with a generator of their
own, so the other thresholds are unchanged.  TRIPLE_EXTRA takes such a combination as
well (e.g. 1-3-7:1e-6).  REFINE_PAIRS=<b>-<b'>[,...] does the same for the pair
inequality on the pairs with distances in the bins b and b', on that rectangle of
distances (sampled with a generator of their own as well).  With ONLY=NONE the branch and bounds of the kinds are skipped
and only those of the listed combinations run; the run then ends PARTIAL, and it proves
the case together with a passing full check of the same case, certificate, arguments
and options, whose thresholds it reproduces.

With PAIR_MARGIN=<kind>:<x>[,...], e.g. PAIR_MARGIN=AA:5e-6, the pair inequalities of the
named kinds (and the pairs of REFINE_PAIRS of those kinds) take the margin x in place of
margin2.  A kind whose function is nearly flat on a whole face of its domain needs many
more boxes at a small margin; this keeps its margin while the others shrink.
"""
import itertools
import json
import os
import sys
import time
from fractions import Fraction as Fr
from math import comb

import numpy as np
from flint import arb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combo30_check as C  # noqa: E402
from radial_count_check import A_, ldl_psd, solve_exact  # noqa: E402
from radial_case_check import bracket_check  # noqa: E402
from radial_count_sdp import C1, C2  # noqa: E402
from certificate_check import ldl_positive  # noqa: E402
from truncated_search import pair as pair_float  # noqa: E402

CASE = json.load(open(sys.argv[1]))
PRE = os.environ.get('PRECHECK') == '1'
BORDER = os.environ.get('BORDER') == '1'
EXTRA = {k: float(v) for k, v in (e.split(':') for e in os.environ.get('TRIPLE_EXTRA', '').split(',') if e)}
ONLY = [e for e in os.environ.get('ONLY', '').split(',') if e]
REFINE = [tuple(sorted(int(x) - 1 for x in e.split('-'))) for e in os.environ.get('REFINE_TRIPLES', '').split(',') if e]
REFINE_PAIRS = [tuple(sorted(int(x) - 1 for x in e.split('-'))) for e in os.environ.get('REFINE_PAIRS', '').split(',') if e]
PAIR_MARGIN = {k: float(v) for k, v in (e.split(':') for e in os.environ.get('PAIR_MARGIN', '').split(',') if e)}


def num(x):
    return C.DMAX if x == 'sqrt6' else Fr(str(x))


def counts():
    M, nb = CASE['M'], len(CASE['bins'])
    out = []

    def rec(prefix, left):
        if len(prefix) == nb - 1:
            v = prefix + [left]
            if all(lo <= sum(v[i] for i in bs) <= hi for bs, lo, hi in CASE['constraints']):
                out.append(tuple(v))
            return
        for k in range(left + 1):
            rec(prefix + [k], left - k)
    rec([], M)
    return out


C.TYPES = list(CASE['types'])
C.TRANGE = {k: (num(a), num(b)) for k, (a, b) in CASE['trange'].items()}
C.EDGES = [num(CASE['bins'][0][1])] + [num(b[2]) for b in CASE['bins']]
C.BINTYPE = [b[0] for b in CASE['bins']]
C.COUNTS = counts()
for i, (ty, lo, hi) in enumerate(CASE['bins']):
    assert num(lo) == C.EDGES[i] and C.TRANGE[ty][0] <= num(lo) and num(hi) <= C.TRANGE[ty][1], CASE['bins'][i]


def tcounts(nb):
    out = {s: 0 for s in C.TYPES}
    for ty, n in zip(C.BINTYPE, nb):
        out[ty] += n
    return out


C.tcounts = tcounts


def main():
    t0 = time.time()
    path, d3 = sys.argv[2], int(sys.argv[3])
    mg2 = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-5
    mg3 = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-6
    mgm = float(sys.argv[6]) if len(sys.argv) > 6 else 2e-6
    Z = np.load(path)
    rng = np.random.default_rng(7)
    if os.environ.get('LEVEL'):
        target, tname = A_(Fr(os.environ['LEVEL'])), os.environ['LEVEL']
    else:
        target, tname = 9 * arb.pi() ** 2 / 8 - 8, '9 pi^2/8 - 8'
    D2, R2 = C.D2, C.R2
    print('statement (C) at %d centres, case %s: combined certificate %s, two-point degree %d/%d, three-point degree %d; '
          '%d count vectors' % (CASE['M'], os.path.basename(sys.argv[1]), os.path.basename(path), D2, R2, d3, len(C.COUNTS)),
          flush=True)
    C.check('dmax exceeds sqrt 6', C.DMAX ** 2 > 6)
    # (a) positivity
    Af = np.array(Z['A'], float)
    if BORDER:
        # round the bordered matrix [[A_0, z], [z^T, t]] as a whole: when A_0 is nearly
        # singular, z^T A_0^{-1} z after rounding A_0 alone can exceed the solver's t
        nA = len(Af[0])
        Bd = np.zeros((nA + 1, nA + 1))
        Bd[:nA, :nA] = Af[0]; Bd[:nA, nA] = Bd[nA, :nA] = np.array(Z['z'], float); Bd[nA, nA] = float(Z['t'])
        Zm = C.psd_exact(Bd, Fr(1, 2 ** 30))
        A0 = [row[:nA] for row in Zm[:nA]]
        z = [Zm[i][nA] for i in range(nA)]
        t = Zm[nA][nA]
    else:
        A0 = C.psd_exact(Af[0], Fr(1, 2 ** 30))
        z = [C.dyad(v) for v in np.array(Z['z'], float)]
    A = [A0] + [C.psd_exact(Af[k], Fr(1, 2 ** 30)) for k in range(1, D2 + 1)]
    good = all(ldl_psd(A[k])[0] for k in range(1, D2 + 1))
    g0, piv = ldl_psd(A0)
    if not BORDER:
        wv = solve_exact(A0, z)
        tq = sum(a * b for a, b in zip(z, wv))
        t = Fr(-(-tq.numerator * 2 ** 48 // tq.denominator), 2 ** 48)
        Zm = [row[:] + [z[i]] for i, row in enumerate(A0)] + [z + [t]]
    gz, _ = ldl_psd(Zm)
    C.check('A_1..A_D and [[A_0, z], [z^T, t]] positive semidefinite (exact LDL^T)',
            good and g0 and all(p > 0 for p in piv) and gz, 't = %.9f, float t %.9f' % (float(t), float(Z['t'])))
    Bf = C.blocks_from_x3(np.array(Z['x3'], float), d3)
    Bk = {name: C.psd_exact(M) for name, M in Bf.items()}
    C.check('three-point blocks positive definite (exact LDL^T)', all(ldl_positive(M) for M in Bk.values()),
            '%d blocks' % len(Bk))
    pt = {s: C.point_term(Bk, d3, i) for i, s in enumerate(C.TYPES)}
    print('  point terms: %s' % {s: '%.6f' % float(v) for s, v in pt.items()}, flush=True)
    # (b) pairs
    Af2 = [np.array([[float(v) for v in row] for row in a]) for a in A]

    def pair_setup(i, j, box, rng_):
        """sample and refine K + PAIR3 - Pi on the pairs with distances in box"""
        Pu = C.pair3_poly(Bk, d3, i, j)
        (a0, a1), (b0, b1) = box
        n = 400000
        p = float(a0) + float(a1 - a0) * rng_.random(n); q = float(b0) + float(b1 - b0) * rng_.random(n)
        k6 = n // 6
        p[:k6] = float(a0); q[k6:2 * k6] = float(b0); p[2 * k6:3 * k6] = float(a1); q[3 * k6:4 * k6] = float(b1)
        top = (p * p + q * q - 4) / (2 * p * q)
        u = -1 + (top + 1) * rng_.random(n) ** 0.5
        u[4 * k6:5 * k6] = top[4 * k6:5 * k6] - 3e-3 * rng_.random(k6)
        gd, ge = np.meshgrid(np.linspace(float(a0), float(a1), 25), np.linspace(float(b0), float(b1), 25), indexing='ij')
        gd, ge = gd.ravel(), ge.ravel()
        gt = (gd * gd + ge * ge - 4) / (2 * gd * ge)
        sv = np.r_[0.0, np.linspace(0, 1, 80) ** 2, 1.0]
        gp = np.repeat(gd, len(sv)); gq = np.repeat(ge, len(sv))
        gu = -1 + (np.repeat(gt, len(sv)) + 1) * np.tile(1 - sv[::-1], len(gd))
        p, q, u = np.r_[p, gp], np.r_[q, gq], np.r_[u, gu]
        Kv = np.zeros(len(u))
        x = (2 * p - float(C1)) / float(C2); y = (2 * q - float(C1)) / float(C2)
        Tx = [np.ones_like(x), x]; Ty = [np.ones_like(y), y]
        for _ in range(2, R2 + 1):
            Tx.append(2 * x * Tx[-1] - Tx[-2]); Ty.append(2 * y * Ty[-1] - Ty[-2])
        Uu = [np.ones_like(u), 2 * u]
        for _ in range(2, D2 + 1):
            Uu.append(2 * u * Uu[-1] - Uu[-2])
        for k in range(D2 + 1):
            Kv += Uu[k] / (k + 1) * np.einsum('na,ab,nb->n', np.stack(Tx, 1), Af2[k], np.stack(Ty, 1))
        P3 = np.polyval([float(c) for c in Pu[::-1]], u)
        v = Kv + P3 - pair_float(p / 2, q / 2, u)
        w = np.argsort(v)[-40:]
        best = C.refine_pair(A, Pu, np.stack([p[w], q[w], u[w]], 1), box)
        return C.above(max(float(v.max()), best) + PAIR_MARGIN.get(C.TYPES[i] + C.TYPES[j], mg2)), float(v.max()), Pu

    c2 = {}
    for (s, tt) in itertools.combinations_with_replacement(C.TYPES, 2):
        i, j = C.TYPES.index(s), C.TYPES.index(tt)
        c2[(s, tt)], vmax, _ = pair_setup(i, j, (C.TRANGE[s], C.TRANGE[tt]), rng)
        print('  pair %s%s: float largest K + PAIR3 - Pi %.6e (file c2 %.6e); threshold %.6e'
              % (s, tt, vmax, float(Z['c2'][len(c2) - 1]), float(c2[(s, tt)])), flush=True)
    jobs = [(A, C.pair3_poly(Bk, d3, C.TYPES.index(s), C.TYPES.index(tt)), c2[(s, tt)], (C.TRANGE[s], C.TRANGE[tt]), s == tt)
            for (s, tt) in c2]
    labels = ['K + PAIR3_%s%s <= Pi + c2 on the admissible pairs' % st for st in c2]
    if PRE or ONLY:
        jobs, labels = [], []
    # pairs refined by bins, with a generator of their own
    c2r = {}
    prng = np.random.default_rng(13)
    for bs in REFINE_PAIRS:
        s, tt = C.BINTYPE[bs[0]], C.BINTYPE[bs[1]]
        box = ((C.EDGES[bs[0]], C.EDGES[bs[0] + 1]), (C.EDGES[bs[1]], C.EDGES[bs[1] + 1]))
        c2r[bs], vmax, Pu = pair_setup(C.TYPES.index(s), C.TYPES.index(tt), box, prng)
        name = 'bins ' + '-'.join(str(b + 1) for b in bs)
        print('  pair %s%s on %s: float largest K + PAIR3 - Pi %.6e; threshold %.6e against %.6e for the kind'
              % (s, tt, name, vmax, float(c2r[bs]), float(c2[(s, tt)])), flush=True)
        if not PRE:
            jobs.append((A, Pu, c2r[bs], box, bs[0] == bs[1]))
            labels.append('K + PAIR3_%s%s <= Pi + its threshold on the pairs of %s' % (s, tt, name))
    from multiprocessing import Pool
    if jobs:
        with Pool(min(int(os.environ.get('PAIR_PROCS', '3')), len(jobs))) as pool:
            res = pool.starmap(C.pair_box_check, jobs)
        for label, (ok, msg) in zip(labels, res):
            C.check(label, ok, msg)
    # (c) bins
    mf = np.array(Z['m'], float)
    assert len(mf) == len(C.BINTYPE)
    m = [C.above(v + mgm) for v in mf]
    if not PRE and not ONLY:
        okb, msg = bracket_check(A, z, D2, R2, C1, C2, C.DMAX, C.EDGES, m)
        C.check('f <= m_b on every bin', okb, msg)
    # (d) triples
    def triple_setup(P, T, labels, rng_, extra):
        """sample and refine TRIPLE3 = P on {u_12 <= T[0], u_13 <= T[1], u_23 <= T[2]}, and lay it
        out for the branch and bound with the symmetry of equal labels"""
        T12, T13, T23 = T
        g = C.random_gram(float(T12), float(T13), float(T23), 200000, rng_)
        ax = [np.r_[-1.0, np.linspace(-1, float(T), 40), float(T)] for T in (T12, T13, T23)]
        G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
        G = G[1 + 2 * G[:, 0] * G[:, 1] * G[:, 2] - (G ** 2).sum(1) >= 0]
        g = np.r_[g, G]
        vals = C.peval(P, g[:, 0], g[:, 1], g[:, 2])
        best = C.refine_triple(P, g[np.argsort(vals)[-40:]], (float(T12), float(T13), float(T23)))
        thr = C.above(max(float(vals.max()), best) + mg3 + extra)
        perm, sym = C.triple_layout(labels)
        tops = [None] * 3
        for src, val in enumerate((T12, T13, T23)):
            tops[perm.index(src)] = float(C.above(val))
        Pr = {}
        for e, co in P.items():
            ee = [0, 0, 0]
            for src in range(3):
                ee[perm.index(src)] += e[src]
            Pr[tuple(ee)] = Pr.get(tuple(ee), 0) + co
        return thr, Pr, tops, sym, float(vals.max()), len(g), best

    c3 = {}
    tjobs = []
    for combo in itertools.combinations_with_replacement(C.TYPES, 3):
        if all(C.Ntriple(tcounts(nb), list(combo)) == 0 for nb in C.COUNTS):
            continue
        P = C.triple3_poly(Bk, d3, combo)
        T = (C.tmax(combo[0], combo[1]), C.tmax(combo[0], combo[2]), C.tmax(combo[1], combo[2]))
        c3[combo], Pr, tops, sym, vmax, ns, best = triple_setup(P, T, combo, rng, EXTRA.get(''.join(combo), 0.0))
        print('  triple %s: float largest %.6e on %d samples; threshold %.6e%s; %d monomials, symmetry %d'
              % (''.join(combo), vmax, ns, float(c3[combo]),
                 ' (refined %.6e, extra %g)' % (best, EXTRA[''.join(combo)]) if ''.join(combo) in EXTRA else '',
                 len(P), sym), flush=True)
        if ONLY and ''.join(combo) not in ONLY:
            continue
        tjobs.append((''.join(combo), (Pr, tops, sym, float(c3[combo]), ''.join(combo))))
    # triples refined by bins, sampled with a generator of their own so that the
    # thresholds above stay those of the check without refinement
    c3r = {}
    rrng = np.random.default_rng(11)
    for bs in REFINE:
        combo = tuple(C.BINTYPE[b] for b in bs)
        assert combo in c3, bs
        hi = [C.EDGES[b + 1] for b in bs]
        T = (C.amaxq(hi[0], hi[1]), C.amaxq(hi[0], hi[2]), C.amaxq(hi[1], hi[2]))
        name = 'bins ' + '-'.join(str(b + 1) for b in bs)
        c3r[bs], Pr, tops, sym, vmax, ns, best = triple_setup(C.triple3_poly(Bk, d3, combo), T, bs, rrng,
                                                               EXTRA.get('-'.join(str(b + 1) for b in bs), 0.0))
        print('  triple %s on %s (inner products at most %.6f, %.6f, %.6f): float largest %.6e on %d samples; '
              'threshold %.6e against %.6e for the kind; symmetry %d'
              % (''.join(combo), name, *[float(x) for x in T], vmax, ns, float(c3r[bs]), float(c3[combo]), sym), flush=True)
        tjobs.append((''.join(combo) + ' on ' + name, (Pr, tops, sym, float(c3r[bs]), ''.join(combo) + ' ' + name)))
    if not PRE and tjobs:
        with Pool(min(int(os.environ.get('TRIPLE_PROCS', '3')), len(tjobs))) as pool:
            res = pool.starmap(C.triple_box_check, [a for _, a in tjobs])
        nbox = 0
        for (label, _), (ok, msg, nd) in zip(tjobs, res):
            C.check('TRIPLE3_%s <= its threshold on the admissible triples' % label if ' on ' in label
                    else 'TRIPLE3_%s <= c3 on the admissible triples' % label, ok, msg)
            nbox += nd
        print('  triple boxes in all: %d' % nbox, flush=True)

    def Nbins(nb, bs):
        out = 1
        for b in set(bs):
            out *= comb(nb[b], bs.count(b))
        return out

    def refined_gain(nb):
        return (sum(Nbins(nb, bs) * max(c3[tuple(C.BINTYPE[b] for b in bs)] - c3r[bs], 0) for bs in c3r)
                + sum(Nbins(nb, bs) * max(c2[(C.BINTYPE[bs[0]], C.BINTYPE[bs[1]])] - c2r[bs], 0) for bs in c2r))
    # the bound, over every count vector
    cache = {}
    worst, wvec = None, None
    for nb in C.COUNTS:
        tc = tcounts(nb)
        key = tuple(tc[s] for s in C.TYPES)
        if key not in cache:
            cache[key] = (t / 2 + sum(C.Npair(tc, s, tt) * c2[(s, tt)] for (s, tt) in c2)
                          + sum(C.Ntriple(tc, list(cb)) * c3[cb] for cb in c3))
        val = cache[key] + sum(nb[b] * (m[b] + pt[C.BINTYPE[b]]) for b in range(len(nb)) if nb[b])
        if c3r or c2r:
            val -= refined_gain(nb)
        if worst is None or val > worst:
            worst, wvec = val, nb
    print('  %d count vectors, %d type-count vectors; largest bound %.6f at %s' % (len(C.COUNTS), len(cache), float(worst), wvec),
          flush=True)
    for extra in [e for e in os.environ.get('EXTRA_CASES', '').split(',') if e]:
        E = json.load(open(extra))
        assert (E['M'], E['types'], E['trange'], E['bins']) == (CASE['M'], CASE['types'], CASE['trange'], CASE['bins']), extra
        main_constraints, CASE['constraints'] = CASE['constraints'], E['constraints']
        vecs = counts()
        CASE['constraints'] = main_constraints
        ew, ev = None, None
        for nb in vecs:
            tc = tcounts(nb)
            key = tuple(tc[s] for s in C.TYPES)
            if key not in cache:
                cache[key] = (t / 2 + sum(C.Npair(tc, s, tt) * c2[(s, tt)] for (s, tt) in c2)
                              + sum(C.Ntriple(tc, list(cb)) * c3[cb] for cb in c3))
            val = cache[key] + sum(nb[b] * (m[b] + pt[C.BINTYPE[b]]) for b in range(len(nb)) if nb[b])
            if c3r or c2r:
                val -= refined_gain(nb)
            if ew is None or val > ew:
                ew, ev = val, nb
        C.check('with these thresholds, largest bound over the %d count vectors of %s below %s'
                % (len(vecs), os.path.basename(extra), tname), A_(ew) < target,
                '%.6f at %s < %s' % (float(ew), ev, target.str(8)))
    if ONLY and not PRE:
        kinds = ([k for k in ONLY if k in [''.join(c) for c in c3]] + [lab for lab, _ in tjobs if ' on ' in lab]
                 + ['the pairs on bins ' + '-'.join(str(b + 1) for b in bs) for bs in c2r])
        print('PARTIAL: only the inequalities %s were checked by branch and bound; with the thresholds above the bound would be '
              '%.6f against %s = %s [%.0f s]' % (', '.join(kinds) or 'none', float(worst), tname, target.str(8), time.time() - t0))
        return
    if PRE:
        print('PRECHECK (floating point, no branch and bound): the thresholds give %.6f against %s = %s [%.0f s]'
              % (float(worst), tname, target.str(8), time.time() - t0))
        return
    C.check('largest bound over the count vectors below %s' % tname, A_(worst) < target,
            '%.6f < %s' % (float(worst), target.str(8)))
    if os.environ.get('LEVEL'):
        print('PASS: in case %s every packing set of %d centres has U(Y) <= %.6f < %s, so U(Y) < max(9 pi^2/8 - 8, %s) '
              'for every packing set of %d centres [%.0f s]'
              % (os.path.basename(sys.argv[1]), CASE['M'], float(worst), tname, tname, CASE['M'], time.time() - t0))
    else:
        print('PASS: in case %s every packing set of %d centres has U(Y) <= %.6f < 9 pi^2/8 - 8, so T(Y) > 8 [%.0f s]'
              % (os.path.basename(sys.argv[1]), CASE['M'], float(worst), time.time() - t0))


if __name__ == '__main__':
    main()
