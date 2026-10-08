#!/usr/bin/env python3
"""
c28_typed_dirs.py m d rounds -- the typed three-point bound of ../C30/typed3pt.py on
the directions of 28 centres with the radial counts of 28, every centre at the largest
distance its count allows: m within 2.0161, max(0, 21 - m) at 2.1, the centres that
the counts 22 within 2.15, 23 within 2.2 and 24 within 2.35 still need (three when
m <= 21, 24 - m when m = 22 or 23), merged at the looser 2.35, and four at sqrt6.
A value below 0 would show that no such directions exist; floating point.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'C30'))
import typed3pt as T3

R6 = 6 ** .5
a = T3.a_
m, d, rounds = map(int, sys.argv[1:4])
ds0 = [2.0161, 2.1, 2.35, R6]
cnt0 = [m, max(0, 21 - m), min(3, 24 - max(m, 21)), 4]
ds = [x for x, c in zip(ds0, cnt0) if c > 0]
cnt = [c for c in cnt0 if c > 0]
T = [[a(x, y) for y in ds] for x in ds]
print('28: %d within 2.0161, %d at 2.1, %d at 2.35, 4 at sqrt6; T =\n%s' % (m, cnt0[1], cnt0[2], np.round(T, 5)), flush=True)
T3.solve(cnt, T, d, rounds)
