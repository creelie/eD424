"""
c28_sampled_eval.py -- the certificate of prop:count28-few (c28lo13b_d8_r1.npz) with the
thresholds its solver reported on its own samples (c28lo13_d8b.log, round 1) instead of the
proved ones, evaluated over the count vectors of other cases.  This is the quantity that the
programme of combo_direct.py minimises, so it tells how much a certificate built for one of
those cases could gain over this one at degree 8.  Floating point; used in no proof.
"""
import sys, os, re
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c28_kscan.py')
exec(open(__file__).read().split('for k in [13')[0])
L = open(os.path.join(os.path.dirname(__file__), 'c28lo13_d8b.log')).read()
r1 = L[L.index('round 1:'):]
c2.update(eval(r1[r1.index('c2: ') + 4:r1.index('}', r1.index('c2: ')) + 1]))
c3.update(eval(r1[r1.index('c3: ') + 4:r1.index('}', r1.index('c3: ')) + 1]))
mb -= 1e-6
V = vectors(28)
for name, f in [('at most 13 within 2.0161', lambda v: v[0] <= 13),
                ('at most 15 within 2.0161, 4 beyond 2.35', lambda v: v[0] <= 15 and v[6] == 4),
                ('at most 16 within 2.0161', lambda v: v[0] <= 16),
                ('16 within 2.0161, 1 beyond 2.35', lambda v: v[0] == 16 and v[6] == 1)]:
    X = [v for v in V if f(v)]
    b, v = max((bound(v), v) for v in X)
    print('%-42s %6d vectors, largest %.5f at %s' % (name, len(X), b, v), flush=True)
