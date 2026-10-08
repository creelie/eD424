#!/usr/bin/env python3
"""
xcheck_zonal.py -- compare the output of xcheck_zonal.jl (entries of Z_lambda
evaluated by the authors' code from our cache) with ../zonal/zonal.py at the
same inner products, in exact rationals.

Usage: python3 xcheck_zonal.py JULIA_OUTPUT ps.txt.reduced.pkl
"""
import os
import pickle
import sys
from fractions import Fraction as Q

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'zonal'))
from zonal import evaluate_zonal_matrix

ps = pickle.load(open(sys.argv[2], 'rb'))
u = [Q(1, 3), Q(-2, 7), Q(1, 5), Q(-1, 4), Q(2, 9), Q(3, 11)]
agree = total = 0
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) != 6 or not all(x.lstrip('-').isdigit() for x in f):
        continue
    l1, l2, w1, w2, num, den = map(int, f)
    ours = evaluate_zonal_matrix(ps, (l1, l2), w1, w2, u, Q(1), Q(0))
    ok = ours == Q(num, den)
    agree += ok
    total += 1
    print('lambda=(%d,%d) (%d,%d): %s' % (l1, l2, w1, w2, 'equal' if ok else 'DIFFERENT'))
print('%d of %d entries equal' % (agree, total))
sys.exit(0 if agree == total and total else 1)
