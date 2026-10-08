#!/usr/bin/env python3
"""
count_effect.py -- how the bound of a typed-triple certificate from combo_gen2.py moves
when the count constraints of its case change, the certificate held fixed.

Floating point, from the solver's own thresholds (the values m, t, c2, c3 saved with the
certificate, which hold on the sampled constraints only): for each variant of the case
constraints, the largest of
    sum_b n_b m_b + sum_types n_s p_s + t/2 + sum N_st c2_st + sum N_str c3_str
over the integer count vectors, and where it is attained.  Nothing in a proof uses it.

Usage: python3 count_effect.py cert.npz case.json d3 'variants'
A variant is a comma-separated list of k:lo[:hi], which replaces the constraint on the
first k bins by lo <= N <= hi; variants are separated by ';', and an empty one is the case.
Log: c28m_r3_counts.log (c28m_d10_r3.npz on case28_all.json).
"""
import itertools
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'C30'))
import typed3pt as T3

cert, casef = sys.argv[1], sys.argv[2]
d3 = int(sys.argv[3]) if len(sys.argv) > 3 else 10
Z = np.load(cert)
CASE = json.load(open(casef))
TYPES = CASE['types']; BINS = CASE['bins']; M = CASE['M']
L = T3.Layout(len(TYPES), d3)
B = T3.Builder(L)
for i, s in enumerate(TYPES):
    T3.add_point(B, np.array([i]), i)
pvec = B.matrix(len(TYPES)) @ Z['x3']
c2keys = list(itertools.combinations_with_replacement(TYPES, 2))
c3keys = list(itertools.combinations_with_replacement(TYPES, 3))
assert len(Z['c2']) == len(c2keys) and len(Z['c3']) == len(c3keys), 'a kind is missing from the certificate'
c2 = dict(zip(c2keys, Z['c2'])); c3 = dict(zip(c3keys, Z['c3']))
m, t = Z['m'], float(Z['t'])
print('t/2 = %.5f, m = %s, pvec = %s, stored bound %.5f' % (t / 2, np.round(m, 5), np.round(pvec, 6), float(Z['bound'])))

def counts(cons):
    out = []
    def rec(prefix, left):
        if len(prefix) == len(BINS) - 1:
            v = prefix + [left]
            if all(lo <= sum(v[i] for i in bs) <= hi for bs, lo, hi in cons):
                out.append(tuple(v))
            return
        for k in range(left + 1):
            rec(prefix + [k], left - k)
    rec([], M)
    return out

def value(nb):
    tc = {s: 0 for s in TYPES}
    for (ty, _, _), n in zip(BINS, nb):
        tc[ty] += n
    v = float(np.dot(nb, m)) + sum(tc[s] * pvec[i] for i, s in enumerate(TYPES)) + t / 2
    for (s, u), c in c2.items():
        v += (comb(tc[s], 2) if s == u else tc[s] * tc[u]) * c
    for cb, c in c3.items():
        n = 1
        for s in set(cb):
            n *= comb(tc[s], cb.count(s))
        v += n * c
    return v

base = CASE['constraints']
# each argument: k:lo[:hi] sets the constraint on the first k bins (k = number of bins) to lo..hi; ';'-separated variants
for variant in sys.argv[4].split(';'):
    cons = [list(c) for c in base]
    for item in variant.split(','):
        if not item:
            continue
        f = item.split(':'); k, lo = int(f[0]), int(f[1]); hi = int(f[2]) if len(f) > 2 else M
        bs = list(range(k))
        cons = [c for c in cons if c[0] != bs] + [[bs, lo, hi]]
    C = counts(cons)
    vals = np.array([value(nb) for nb in C])
    order = np.argsort(-vals)[:3]
    print('%-22s %6d vectors, largest %.5f at %s; next %s' % (variant or 'case', len(C), vals[order[0]], C[order[0]],
          ', '.join('%.5f %s' % (vals[j], C[j]) for j in order[1:])), flush=True)
