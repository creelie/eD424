"""
c28_kscan.py -- how far the certificate of prop:count28-few reaches.  Takes the thresholds
that its exact check proved (check_c28lo13b_r2.log), the solver's bin values m_b raised by
the bin margin and the exact point terms, and evaluates the bound of eq:thirty-bound over
the count vectors with at most k centres within 2.0161, k = 13, ..., 24.  Floating point;
used in no proof (the bound at k = 13 is the check's own).
"""
import sys, os, re, json, itertools
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO + '/multi_cap')
log = open(REPO + '/gap_closure/CM/check_c28lo13b_r2.log').read()
c2 = {k: float(v) for k, v in re.findall(r'pair (\w\w): float .*?threshold ([\d.e+-]+)', log)}
c3 = {k: float(v) for k, v in re.findall(r'triple (\w\w\w): float .*?threshold ([\d.e+-]+)', log)}
Z = np.load(REPO + '/gap_closure/CM/c28lo13b_d8_r1.npz')
t = float(Z['t']); mb = np.array(Z['m'], float) + 1e-6
# point terms from the exact blocks, as the check computes them
sys.argv = ['x', REPO + '/gap_closure/CM/case28_lo13.json']
import combo30_check as C
Bf = C.blocks_from_x3(np.array(Z['x3'], float), 8)
Bk = {name: C.psd_exact(M) for name, M in Bf.items()}
TYPES = ['A', 'B', 'F']
pt = {s: float(C.point_term(Bk, 8, i)) for i, s in enumerate(TYPES)}
print('pt', pt, 't/2', t / 2)
BT = ['A', 'A', 'B', 'B', 'B', 'B', 'F']
from math import comb
def bound(nb):
    tc = {s: sum(n for n, b in zip(nb, BT) if b == s) for s in TYPES}
    v = t / 2 + sum(n * (mb[i] + pt[BT[i]]) for i, n in enumerate(nb))
    for s, u in itertools.combinations_with_replacement(TYPES, 2):
        v += (comb(tc[s], 2) if s == u else tc[s] * tc[u]) * c2[s + u]
    for cb in itertools.combinations_with_replacement(TYPES, 3):
        cnt = 1
        for s in set(cb):
            cnt *= comb(tc[s], cb.count(s))
        v += cnt * c3.get(''.join(cb), 0.0)
    return v
def vectors(kmax):
    out = []
    lows = [0, 6, 21, 22, 23, 24]
    def rec(prefix, left):
        j = len(prefix)
        if j == 6:
            out.append(tuple(prefix + [left])); return
        for n in range(left + 1):
            if j == 0 and n > kmax: break
            cum = sum(prefix) + n
            if cum < lows[j]: continue
            rec(prefix + [n], left - n)
    rec([], 28)
    return out
for k in [13, 14, 15, 16, 17, 19, 21, 24]:
    V = vectors(k)
    vals = [(bound(v), v) for v in V]
    b, v = max(vals)
    print('k=%d: %d vectors, largest %.6f at %s' % (k, len(V), b, v), flush=True)
