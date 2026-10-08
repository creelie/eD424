#!/usr/bin/env python3
"""
prune_resume.py -- thins the samples of a combined-kernel certificate (an .npz of
combo_direct.py) to the KEEP_P pair and KEEP_T triple samples of each kind that come
closest to binding at that certificate, and writes them to a new .npz, so that a run
at a higher degree can start from them (RESUME=<out> RESUME_NOPRUNE=1) with fewer
inequalities, hence less memory.  KEEP_BASE=1 keeps the starting pair grid as well.

Usage: KEEP_P=2500 KEEP_T=2000 KEEP_BASE=1 python3 prune_resume.py case.json cert.npz d3 out.npz
       (d3 is the degree of cert.npz)
"""
import itertools
import os
import sys

import numpy as np

case, cert, d3, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.argv = [sys.argv[0], str(d3), '1', case]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combo_gen2 as G  # noqa: E402
from combo_gen2 import T3, TYPES, COUNTS, tcounts, Ntriple  # noqa: E402

sv = np.load(cert)
psamp = {st: tuple(sv['p_' + ''.join(st)]) for st in itertools.combinations_with_replacement(TYPES, 2)}
tsamp = {cb: sv['t_' + ''.join(cb)] for cb in itertools.combinations_with_replacement(TYPES, 3)
         if 't_' + ''.join(cb) in sv}
c2keys = list(itertools.combinations_with_replacement(TYPES, 2))
c3keys = [cb for cb in itertools.combinations_with_replacement(TYPES, 3)
          if not all(Ntriple(tcounts(nb), list(cb)) == 0 for nb in COUNTS)]
before = (sum(len(v[0]) for v in psamp.values()), sum(len(v) for v in tsamp.values()))
G.prune(psamp, tsamp, sv['x3'], sv['A'], dict(zip(c2keys, sv['c2'])), dict(zip(c3keys, sv['c3'])),
        T3.Layout(len(TYPES), d3))
after = (sum(len(v[0]) for v in psamp.values()), sum(len(v) for v in tsamp.values()))
keep = {k: sv[k] for k in sv.files if not (k.startswith('p_') or k.startswith('t_'))}
np.savez(out, **keep, **{'p_' + ''.join(k): np.array(v) for k, v in psamp.items()},
         **{'t_' + ''.join(k): v for k, v in tsamp.items()})
print('pair samples %d -> %d, triple samples %d -> %d; written %s' % (before[0], after[0], before[1], after[1], out))
