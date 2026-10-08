#!/usr/bin/env python3
"""
compare_entries.py -- compare entries of P(S) written by the code of de Laat,
Leijenhorst and de Muinck Keizer (compute_PS, run from scratch in a clean
folder) with the same entries computed by ../zonal (psker, reduced by
verify45.py), coefficient by coefficient, in exact rationals.

Usage: python3 compare_entries.py THEIR_ZONALSTORE_4 ps.txt.reduced.pkl
"""
import os
import pickle
import re
import sys
from fractions import Fraction

theirs, ours = sys.argv[1], pickle.load(open(sys.argv[2], 'rb'))
pat = re.compile(r'pol-4-2-\[(\d+), (\d+)\]-(\d+)-(\d+)\.txt$')
same = differ = 0
for name in sorted(os.listdir(theirs)):
    m = pat.match(name)
    if not m:
        continue
    l1, l2, i, j = map(int, m.groups())
    p = {}
    for line in open(os.path.join(theirs, name)):
        f = line.split()
        if not f:
            continue
        a, _, b = f[-1].partition('//')
        c = Fraction(int(a), int(b) if b else 1)
        if c:
            p[tuple(int(x) for x in f[:-1])] = c
    q = {e: c for e, c in ours[((l1, l2), i, j)].items() if c}
    ok = p == q
    same += ok
    differ += not ok
    print('lambda=(%d,%d) k=(%d,%d): %d terms, %s' % (l1, l2, i, j, len(p), 'identical' if ok else 'DIFFERENT'))
print('%d identical, %d different' % (same, differ))
sys.exit(1 if differ else 0)
