"""
combine4.py -- add the saved parts of a constraint and report whether it vanishes.

The four-point constraint is built in pieces, one for the zonal part and one for
each sum-of-squares block, because no single process here can hold more than a
couple of them: the coefficients are rationals with several thousand digits and
there are tens of thousands of them.  The pieces are added one file at a time,
so that only the running total and one part are ever in memory.
"""
import pickle
import sys
from fractions import Fraction as Q

import partio


def main(paths):
    acc = {}
    for p in paths:
        nin = 0
        for e, (num, den) in partio.iter_parts(p):
            nin += 1
            v = acc.get(e)
            v = Q(num, den) if v is None else v + Q(num, den)
            if v:
                acc[e] = v
            elif e in acc:
                del acc[e]
        print('%-22s %8d terms added, %8d nonzero so far'
              % (p.split('/')[-1], nin, len(acc)))
        sys.stdout.flush()
    nz = sum(1 for v in acc.values() if v)
    print()
    print('nonzero coefficients in the assembled constraint: %d' % nz)
    if nz:
        for e, v in list(acc.items())[:5]:
            if v:
                print('   %s  %s' % (e, str(v)[:120]))
    return nz == 0


if __name__ == '__main__':
    sys.set_int_max_str_digits(0)
    ok = main(sys.argv[1:])
    print('CONSTRAINT: %s' % ('HOLDS EXACTLY' if ok else 'FAILS'))
    sys.exit(0 if ok else 1)
