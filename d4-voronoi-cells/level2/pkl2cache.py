#!/usr/bin/env python3
"""
pkl2cache.py -- write the zonal matrices of ../zonal in the cache format of the
code of de Laat, Leijenhorst and de Muinck Keizer.

Their evaluate_zonal_matrix reads, for dimension 4, the polynomial P(S) of each
entry from cache/zonalstore/4/pol-4-2-[l1, l2]-i-j.txt: one line per monomial
in the entries (S11, S21, S12, S22) of the top-left block, the four exponents
and then the coefficient, an integer or a quotient a//b.  The reduced P(S)
that verify45.py writes next to psker's output (ps.txt.reduced.pkl) is exactly
that polynomial, with the scalar of Section 3.1 of their paper applied, so the
conversion is a change of file format and nothing else.

Usage: python3 pkl2cache.py ps.txt.reduced.pkl OUTDIR
"""
import os
import pickle
import sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
entries = pickle.load(open(src, 'rb'))
for (lam, i, j), p in entries.items():
    name = 'pol-4-2-[%d, %d]-%d-%d.txt' % (lam[0], lam[1], i, j)
    with open(os.path.join(out, name), 'w') as f:
        for ev, c in sorted(p.items()):
            if c.denominator == 1:
                cs = str(c.numerator)
            else:
                cs = '%d//%d' % (c.numerator, c.denominator)
            f.write(' '.join(map(str, ev)) + ' ' + cs + '\n')
print(len(entries), 'files written to', out)
