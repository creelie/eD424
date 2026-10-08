#!/usr/bin/env python3
"""room_table.py -- what (L) must deliver for one further centre at distance r.
The 24 closest centres lie within rho24.  The further centre z needs a hole:
<z/|z|, y_i/|y_i|> <= a(r, rho24) for every i, and cor:no-room (ii), with
a(sqrt6, rho) replaced by a(r, rho24) (its proof is unchanged), excludes it once
the 24 directions are within sqrt6 (1/sqrt2 - a(r, rho24)) of a root system in
root-sum-square.  The 24 closest have T <= 8 + S(r).  Exact formulas, floating
point evaluation.    python3 room_table.py"""
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'multi_cap'))
from truncated_search import S  # noqa: E402


def a(d1, d2):
    return (d1 * d1 + d2 * d2 - 4) / (2 * d1 * d2)


print('   r      eta = S(r)   allowed rho for rho24 = 2.25, 2.35, 2.444')
for r in (2.0161, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, math.sqrt(6)):
    print('%7.4f   %.5f      %s' % (r, float(S(r)), '  '.join(
        '%.3f' % (math.sqrt(6) * (1 / math.sqrt(2) - a(r, q))) for q in (2.25, 2.35, 2.444))))
