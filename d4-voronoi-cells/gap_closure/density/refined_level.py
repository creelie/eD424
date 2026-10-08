#!/usr/bin/env python3
"""
refined_level.py -- the value of the two-point radial programme of
multi_cap/radial_case_sdp.py at a fixed level L for one case, with the pair
samples refined in rounds as radial_case_sdp.certify does and a margin eps, at a
chosen degree D in the angle and R in the distance.  Floating point; it tells
whether a certificate at level L can exist for the case (value below L - 2e-4),
it proves nothing.

    python3 refined_level.py M D R L eps r:lo:hi [r:lo:hi ...]
"""
import os
import sys
import time
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'multi_cap'))
os.chdir(os.path.join(HERE, '..', '..', 'multi_cap'))
import radial_case_sdp as rc  # noqa: E402

M, D, R, L, eps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(Fr(sys.argv[4])), float(sys.argv[5])
case = {}
for tok in sys.argv[6:]:
    r, lo, hi = tok.split(':')
    case[Fr(r)] = (int(lo), int(hi))
rc.D, rc.R, rc.TARGET = D, R, L
rng = np.random.default_rng(12)
P, Q, W = rc.samples(19, 40)
t0 = time.time()
for rnd in range(10):
    sol = rc.solve(M, case, P, Q, W, eps)
    if sol is None:
        print('round %d: no solution' % rnd); break
    viol, p_, q_, u_ = rc.violation(sol['A'], D, R, rng)
    slack = viol + rc.margins(rc.pair(p_ / 2, q_ / 2, u_), eps) / 2
    print('M %d D %d R %d level %.5f eps %g %s round %d: value %.5f (level - value %.5f), largest K - Pi between the samples %.2e [%.0f s]'
          % (M, D, R, L, eps, ' '.join(sys.argv[6:]), rnd, sol['value'], L - sol['value'], viol.max(), time.time() - t0), flush=True)
    if slack.max() <= 0:
        print('converged'); break
    worst = np.argsort(slack)[-6000:]
    P, Q, W = np.r_[P, p_[worst]], np.r_[Q, q_[worst]], np.r_[W, u_[worst]]
