#!/usr/bin/env python3
"""
split_level.py -- the least level that levels.py finds for M centres under a
case split given as r:lo:hi tokens (lo <= N(r) <= hi).  Floating point; it
chooses the level of a certificate, it proves nothing.

    python3 split_level.py M r:lo:hi [r:lo:hi ...]
"""
import sys
import time
from fractions import Fraction as Fr
import levels as lv

M = int(sys.argv[1])
case = {}
for tok in sys.argv[2:]:
    r, lo, hi = tok.split(':')
    case[Fr(r)] = (int(lo), int(hi))
t0 = time.time()
L = lv.least_level(M, case, 3.33, 3.36, 5e-5)
print('M %d %s: level %.5f [%.0f s]' % (M, ' '.join(sys.argv[2:]), L, time.time() - t0), flush=True)
