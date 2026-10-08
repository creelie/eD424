#!/usr/bin/env python3
"""
levels.py -- for M centres within sqrt6 (24 <= M <= 30), the least level L that
a two-point radial certificate proves for the union of caps, U(Y) <= L, with
the counts that hold in every packing (N(2.0161) <= 24, thm:kissing-stable)
and with optional case splits.  Floating point, sampled constraints: this
chooses the levels; multi_cap/radial_case_sdp.py M --level L writes the
certificates and multi_cap/radial_case_check.py checks them.

The programme is that of multi_cap/radial_case_sdp.py with the level
9 pi^2/8 - 8 of statement (C) replaced by L: if U(Y) > L, the count vector of
Y has sum_b n_b S(e_b) >= U(Y) > L, so a bound below L over those count
vectors proves U(Y) <= L.  The least such L is found by bisection.

    python3 levels.py M [lo hi]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'multi_cap'))
os.chdir(os.path.join(HERE, '..', '..', 'multi_cap'))
import radial_case_sdp as rc  # noqa: E402


def value(M, case, L):
    rc.TARGET = L
    sol = rc.solve(M, case, rc.P0, rc.Q0, rc.W0, 0.0)
    return float('inf') if sol is None else float(sol['value'])


def least_level(M, case, lo=3.0, hi=3.5, tol=2e-4):
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if value(M, case, mid) <= mid:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == '__main__':
    M = int(sys.argv[1])
    lo = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    hi = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
    t0 = time.time()
    L = least_level(M, {}, lo, hi)
    print('M = %d: U <= %.4f (two-point, N(2.0161) <= 24; degree %d in the angle, %d in the distance) [%.0f s]'
          % (M, L, rc.D, rc.R, time.time() - t0), flush=True)
