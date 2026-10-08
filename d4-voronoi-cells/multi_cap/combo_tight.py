#!/usr/bin/env python3
"""
combo_tight.py -- where the certificates of thm:count29 and of the fifth case of
thm:count30 are tight (rem:typed-tight).

Floating point, from the certificate files and the thresholds that the checks proved
(runs/combo29_check.log, runs/combo30_check.log): the bound
    sum_b n_b (m_b + p_type(b)) + t/2 + sum N_st c2_st + sum N_str c3_str
split into its parts at the count vectors where it is largest, the count constraints
active there and the bound when each of them is relaxed by one, the spread of the
per-centre function f over the bins, and the points where the pair and triple
inequalities come closest to their thresholds.  Nothing in a proof uses these values.

Usage: python3 combo_tight.py count29|count30|level26
Logs: runs/combo29_tight.log, runs/combo30_tight.log, runs/combo26_level_tight.log (prop:levels at M = 26).
"""
import itertools
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import combo30_check as C  # noqa: E402
from radial_count_sdp import C1, C2  # noqa: E402
from truncated_search import pair as pair_float, S as S_float  # noqa: E402

WHICH = sys.argv[1]
if WHICH in ('count29', 'level26'):
    if WHICH == 'count29':
        CERT, LOG, MGM, CF = 'combo29_d8.npz', 'combo29_check.log', 1e-6, 'case29_all.json'
    else:
        CERT, LOG, MGM, CF = 'combo26_level_d8.npz', 'combo26_level_check.log', 2e-6, 'case26_all.json'
    CASE = json.load(open(os.path.join(HERE, '..', 'gap_closure', 'CM', CF)))
    num = lambda x: C.DMAX if x == 'sqrt6' else C.Fr(str(x))  # noqa: E731
    C.TYPES = list(CASE['types'])
    C.TRANGE = {k: (num(a), num(b)) for k, (a, b) in CASE['trange'].items()}
    C.EDGES = [num(CASE['bins'][0][1])] + [num(b[2]) for b in CASE['bins']]
    C.BINTYPE = [b[0] for b in CASE['bins']]
    CONS = CASE['constraints']
else:
    CERT, LOG, MGM = 'combo30_d8.npz', 'combo30_check.log', 2e-6
    CONS = None
D3 = 8
LEVEL = 3.3352 if WHICH == 'level26' else 9 * math.pi ** 2 / 8 - 8


def counts(cons, M, nb):
    out = []

    def rec(prefix, left):
        if len(prefix) == nb - 1:
            v = prefix + [left]
            if all(lo <= sum(v[i] for i in bs) <= hi for bs, lo, hi in cons):
                out.append(tuple(v))
            return
        for k in range(left + 1):
            rec(prefix + [k], left - k)
    rec([], M)
    return out


def tcounts(nb):
    out = {s: 0 for s in C.TYPES}
    for ty, n in zip(C.BINTYPE, nb):
        if ty:
            out[ty] += n
    return out


# ---------------------------------------------------------------- the proved thresholds, from the check log
txt = open(os.path.join(HERE, 'runs', LOG)).read()
t = float(re.search(r'\(t = ([0-9.]+)', txt).group(1))
pt = {k: float(v) for k, v in re.findall(r"'([ABF])': '([-0-9.]+)'", re.search(r'point terms: (\{.*\})', txt).group(1))}
c2 = {(m.group(1)[0], m.group(1)[1]): float(m.group(2)) for m in re.finditer(r'pair (\w\w): .*threshold ([-0-9.e]+)', txt)}
c3 = {tuple(m.group(1)): float(m.group(2)) for m in re.finditer(r'triple (\w\w\w): .*threshold ([-0-9.e]+)', txt)}
Z = np.load(os.path.join(HERE, 'radial_certificates', CERT))
m = list(np.array(Z['m'], float) + MGM)
for b, ty in enumerate(C.BINTYPE):                 # a bin that holds no centre has no bound of its own
    if ty is None:
        m.insert(b, 0.0)
m = np.array(m)
COUNTS = counts(CONS, CASE['M'], len(C.BINTYPE)) if CONS else C.COUNTS


def parts(nb):
    tc = tcounts(nb)
    out = {'centres m_b': sum(n * m[b] for b, n in enumerate(nb)),
           'point terms': sum(n * pt[C.BINTYPE[b]] for b, n in enumerate(nb) if n),
           't/2': t / 2}
    for st, v in c2.items():
        out['pairs ' + ''.join(st)] = C.Npair(tc, *st) * v
    for cb, v in c3.items():
        out['triples ' + ''.join(cb)] = C.Ntriple(tc, list(cb)) * v
    return out


def bound(nb):
    return sum(parts(nb).values())


print('%s: certificate %s, t/2 = %.6f, point terms %s' % (WHICH, CERT, t / 2, pt))
print('per-centre thresholds m_b by bin: %s' % ', '.join('%s %.6f' % (C.BINTYPE[b], m[b]) for b in range(len(m)) if C.BINTYPE[b]))
vals = sorted(((bound(nb), nb) for nb in COUNTS), reverse=True)
print('%d count vectors; the largest bounds (level %.6f):' % (len(COUNTS), LEVEL))
for v, nb in vals[:6]:
    tc = tcounts(nb)
    print('   %.6f  bins %s  types %s' % (v, nb, tuple(tc[s] for s in C.TYPES)))
top = vals[0][1]
print('parts of the bound at %s:' % (top,))
P = parts(top)
for k, v in P.items():
    if abs(v) > 5e-7:
        print('   %-14s %+.6f' % (k, v))
print('   %-14s %+.6f' % ('sum', sum(P.values())))
grp = {'per centre (m_b + point)': P['centres m_b'] + P['point terms'], 't/2': P['t/2'],
       'pairs': sum(v for k, v in P.items() if k.startswith('pairs')),
       'triples': sum(v for k, v in P.items() if k.startswith('triples'))}
print('   grouped: %s' % ', '.join('%s %+.4f' % kv for kv in grp.items()))
if CONS:
    act = [(i, bs, lo, hi) for i, (bs, lo, hi) in enumerate(CONS) if sum(top[j] for j in bs) in (lo, hi)]
    print('count constraints active at the maximum:')
    for i, bs, lo, hi in act:
        s = sum(top[j] for j in bs)
        side = 'lo' if s == lo else 'hi'
        cons = [list(c) for c in CONS]
        cons[i][1 if side == 'lo' else 2] += -1 if side == 'lo' else 1
        alt = counts(cons, CASE['M'], len(C.BINTYPE))
        best = max((bound(nb), nb) for nb in alt)
        print('   N(%s) = %d is its %s end %d; relaxed by one the largest bound is %.6f at %s'
              % (CASE['bins'][bs[-1]][2], s, 'lower' if side == 'lo' else 'upper', lo if side == 'lo' else hi, best[0], best[1]))
    cons = [list(c) for c in CONS if c[0] != [0, 1]]
    alt = counts(cons, CASE['M'], len(C.BINTYPE))
    print('the largest bound for each number of centres within 2.05, the other constraints kept:')
    for na in sorted({nb[0] + nb[1] for nb in alt}):
        best = max((bound(nb), nb) for nb in alt if nb[0] + nb[1] == na)
        tc = tcounts(best[1])
        print('   %2d: %.6f at types %s' % (na, best[0], tuple(tc[s] for s in C.TYPES)))
else:
    print('the five count vectors of the case, in the order N(2.05), N(2.15), N(2.25):')
    for nb in C.COUNTS:
        print('   %s  N = %d, %d, %d  bound %.6f' % (nb, nb[0], nb[0] + nb[1], nb[0] + nb[1] + nb[2], bound(nb)))

# ---------------------------------------------------------------- the per-centre function f against m_b
Af = np.array(Z['A'], float)
zf = np.array(Z['z'], float)
D2, R2 = Af.shape[0] - 1, Af.shape[1] - 1


def fvals(d):
    x = (2 * d - float(C1)) / float(C2)
    Tm = np.polynomial.chebyshev.chebvander(x, R2)
    return S_float(d) + 0.5 * np.einsum('na,ab,nb->n', Tm, Af.sum(0), Tm) - Tm @ zf


print('f against m_b on each bin (largest of f, its place, and the spread max - min of f):')
for b in range(len(m)):
    if not C.BINTYPE[b]:
        continue
    d = np.linspace(float(C.EDGES[b]), min(float(C.EDGES[b + 1]), math.sqrt(6)), 801)
    f = fvals(d)
    print('   bin %d (%s, %.4f to %.4f): max f %.6f at d = %.4f, m_b - max f %.1e, spread %.1e'
          % (b, C.BINTYPE[b], d[0], d[-1], f.max(), d[np.argmax(f)], m[b] - f.max(), f.max() - f.min()))

# ---------------------------------------------------------------- where the pair and triple inequalities are tight
Bf = C.blocks_from_x3(np.array(Z['x3'], float), D3)
Bk = {name: C.psd_exact(M) for name, M in Bf.items()}
rng = np.random.default_rng(11)
print('pair inequalities K + PAIR3 - Pi <= c2: largest value found and where (d, d\', u; a = touching inner product)')
for (s, tt), thr in c2.items():
    Pu = C.pair3_poly(Bk, D3, C.TYPES.index(s), C.TYPES.index(tt))
    pc = [float(c) for c in Pu[::-1]]
    (a0, a1), (b0, b1) = [(float(x), float(y)) for x, y in (C.TRANGE[s], C.TRANGE[tt])]
    a1, b1 = min(a1, math.sqrt(6)), min(b1, math.sqrt(6))
    gd, ge, gs = np.meshgrid(np.linspace(a0, a1, 31), np.linspace(b0, b1, 31), np.linspace(0, 1, 801), indexing='ij')
    p, q, sv = gd.ravel(), ge.ravel(), gs.ravel()
    top_u = (p * p + q * q - 4) / (2 * p * q)
    u = -1 + (top_u + 1) * sv
    v = np.array([C.kfloat(Af, pp, qq, uu) for pp, qq, uu in zip(p[:1], q[:1], u[:1])])  # warm the call
    x = (2 * p - float(C1)) / float(C2); y = (2 * q - float(C1)) / float(C2)
    Tx = np.polynomial.chebyshev.chebvander(x, R2); Ty = np.polynomial.chebyshev.chebvander(y, R2)
    Uu = [np.ones_like(u), 2 * u]
    for _ in range(2, D2 + 1):
        Uu.append(2 * u * Uu[-1] - Uu[-2])
    K = sum(Uu[k] / (k + 1) * np.einsum('na,ab,nb->n', Tx, Af[k], Ty) for k in range(D2 + 1))
    v = K + np.polyval(pc, u) - pair_float(p / 2, q / 2, u)
    order = np.argsort(v)[::-1]
    seen = []
    for i in order:
        if len(seen) == 3:
            break
        if all(abs(p[i] - a) + abs(q[i] - b) + abs(u[i] - c) > 0.08 for _, a, b, c in seen):
            seen.append((v[i], p[i], q[i], u[i]))
    print('   %s%s threshold %.6e:' % (s, tt, thr) + ''.join(
        ' %.6e at (%.4f, %.4f, %.4f; a %.4f);' % (vv, a, b, c, (a * a + b * b - 4) / (2 * a * b)) for vv, a, b, c in seen))
    near = v >= v.max() - 1e-6
    a, b = seen[0][1], seen[0][2]
    on = near & (abs(p - a) < 1e-9) & (abs(q - b) < 1e-9)
    uu = np.sort(u[on])
    runs = np.split(uu, np.where(np.diff(uu) > 0.01)[0] + 1) if len(uu) else []
    print('      within 1e-6 of the largest value on %.1f%% of the samples; at d = %.4f, d\' = %.4f for u in %s'
          % (100 * near.mean(), a, b, ', '.join('[%.3f, %.3f]' % (r[0], r[-1]) for r in runs)))

print('triple inequalities TRIPLE3 <= c3: largest value found, where (u12, u13, u23), and the Gram determinant there')
for cb, thr in c3.items():
    Pd = C.triple3_poly(Bk, D3, cb)
    T12, T13, T23 = (float(C.tmax(cb[i], cb[j])) for i, j in ((0, 1), (0, 2), (1, 2)))
    g = C.random_gram(T12, T13, T23, 20000, rng)
    ax = [np.r_[-1.0, np.linspace(-1, T, 40), T] for T in (T12, T13, T23)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    G = G[1 + 2 * G[:, 0] * G[:, 1] * G[:, 2] - (G ** 2).sum(1) >= 0]
    g = np.r_[g, G]
    vals3 = C.peval(Pd, g[:, 0], g[:, 1], g[:, 2])
    i = int(np.argmax(vals3))
    w = g[i]
    det = 1 + 2 * w[0] * w[1] * w[2] - (w ** 2).sum()
    tops = ['top' if abs(w[j] - T) < 1e-9 else ('-1' if abs(w[j] + 1) < 1e-9 else '') for j, T in enumerate((T12, T13, T23))]
    print('   %s threshold %+.6e: largest %+.6e at (%.4f, %.4f, %.4f) %s, det %.1e' % (''.join(cb), thr, vals3[i], *w, tops, det))
