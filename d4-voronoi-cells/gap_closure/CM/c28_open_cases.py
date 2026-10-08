"""
c28_open_cases.py -- the bound of prop:count28-few with the thresholds that the check at
smaller margins proved (multi_cap/runs/combo28_squeeze_check.log), on the count vectors that
the proposition leaves: 15 centres within 2.0161 with 4 beyond 2.35, 16 within 2.0161 with
1 to 4 beyond, and more.  Floating point (the check's own values for its cases are exact);
used in no proof.
"""
import sys, os, re, itertools
from math import comb
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO + '/multi_cap')
log = open(REPO + '/multi_cap/runs/combo28_squeeze_check.log').read()
c2 = {k: float(v) for k, v in re.findall(r'pair (\w\w): float .*?threshold ([\d.e+-]+)', log)}
c3 = {k: float(v) for k, v in re.findall(r'triple (\w\w\w): float .*?threshold ([\d.e+-]+)', log)}
R3 = {tuple(int(x) - 1 for x in b.split('-')): (k, float(v)) for k, b, v in
      re.findall(r'triple (\w\w\w) on bins ([\d-]+) .*?threshold ([\d.e+-]+) against', log)}
R2 = {tuple(int(x) - 1 for x in b.split('-')): (k, float(v)) for k, b, v in
      re.findall(r'pair (\w\w) on bins ([\d-]+): .*?threshold ([\d.e+-]+) against', log)}
Z = np.load(REPO + '/gap_closure/CM/c28lo13b_d8_r1.npz')
t = float(Z['t']); mb = np.array(Z['m'], float) + 1e-7
sys.argv = ['x', REPO + '/gap_closure/CM/case28_lo13.json']
import combo30_check as C
Bk = {name: C.psd_exact(M) for name, M in C.blocks_from_x3(np.array(Z['x3'], float), 8).items()}
TYPES = ['A', 'B', 'F']
pt = {s: float(C.point_term(Bk, 8, i)) for i, s in enumerate(TYPES)}
BT = ['A', 'A', 'B', 'B', 'B', 'B', 'F']


def bound(nb):
    tc = {s: sum(n for n, b in zip(nb, BT) if b == s) for s in TYPES}
    v = t / 2 + sum(n * (mb[i] + pt[BT[i]]) for i, n in enumerate(nb))
    for s, u in itertools.combinations_with_replacement(TYPES, 2):
        v += (comb(tc[s], 2) if s == u else tc[s] * tc[u]) * c2[s + u]
    for cb in itertools.combinations_with_replacement(TYPES, 3):
        cnt = 1
        for s in set(cb):
            cnt *= comb(tc[s], cb.count(s))
        v += cnt * c3[''.join(cb)]
    for bs, (kind, thr) in R3.items():
        N = 1
        for b in set(bs):
            N *= comb(nb[b], bs.count(b))
        v -= N * max(c3[kind] - thr, 0)
    for bs, (kind, thr) in R2.items():
        N = comb(nb[bs[0]], 2) if bs[0] == bs[1] else nb[bs[0]] * nb[bs[1]]
        v -= N * max(c2[kind] - thr, 0)
    return v


def vectors():
    out, lows = [], [0, 6, 21, 22, 23, 24]
    def rec(prefix, left):
        j = len(prefix)
        if j == 6:
            out.append(tuple(prefix + [left])); return
        for n in range(left + 1):
            if sum(prefix) + n < lows[j]: continue
            rec(prefix + [n], left - n)
    rec([], 28)
    return out


V = vectors()
for name, f in [('at most 14 within 2.0161 (check: 3.099739)', lambda v: v[0] <= 14),
                ('at most 15 within, at most 3 beyond 2.35 (check: 3.102197)', lambda v: v[0] <= 15 and v[6] <= 3),
                ('at most 16 within, none beyond (check: 3.102468)', lambda v: v[0] <= 16 and v[6] == 0),
                ('15 within, 4 beyond', lambda v: v[0] == 15 and v[6] == 4)] + \
        [('16 within, %d beyond' % f, lambda v, f=f: v[0] == 16 and v[6] == f) for f in (1, 2, 3, 4)] + \
        [('%d within, any' % k, lambda v, k=k: v[0] == k) for k in (17, 20, 24)]:
    X = [v for v in V if f(v)]
    b, v = max((bound(v), v) for v in X)
    print('%-58s %6d vectors, largest %.6f at %s' % (name, len(X), b, v), flush=True)
