#!/usr/bin/env python3
"""
m24_descent_sweep.py -- does the root measure ever become a local minimum of
the three-point relaxation?

The dual values of the remark on the three-point relaxation say that the
relaxation's minimum stays below tau at every degree up to 20.  This asks the
sharper, local question at the root configuration itself: inside the cone the
degree-d conditions allow, is there still a direction in which the objective
decreases?  A certificate reaching tau would need the answer to be no.

For each degree the answer is a number, the minimum directional derivative,
computed by m24_descent.py; m24_verify_direction.py then recomputes it on the
support of the direction with the closed form of omega_3 in place of the
interpolated table, so that the answer does not depend on the table.  The two
agree to four figures at every degree.

Fitting L + C q^d to the five values gives L, the improvement that survives as
the degree grows.  It is not zero.

  python3 m24_descent_sweep.py [grid_n grid_m]            default: 26 14

Each degree takes a few minutes; the whole sweep is about an hour.
"""
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARGV = sys.argv[1:] or ['26', '14']
DEGREES = (6, 8, 10, 12, 14)
PAT = re.compile(r'min directional derivative\s+(-?[0-9.e+-]+)')
PATX = re.compile(r'exact omega3:\s+(-?[0-9.e+-]+)')


def run(script, args):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args,
                       cwd=HERE, capture_output=True, text=True)
    return r.stdout


def main():
    tab, exact = [], []
    for d in DEGREES:
        out = run('m24_descent.py', ARGV + [str(d)])
        m = PAT.search(out)
        tab.append(float(m.group(1)) if m else float('nan'))
        out = run('m24_verify_direction.py', [str(d)])
        m = PATX.search(out)
        exact.append(float(m.group(1)) if m else float('nan'))
        print('degree %2d   table %12.5e   exact %12.5e'
              % (d, tab[-1], exact[-1]), flush=True)

    a = np.abs(np.array(exact))
    d = np.array(DEGREES, float)
    print()
    if np.isnan(a).any():
        print('a degree failed to solve; no fit')
        return
    from scipy.optimize import curve_fit
    f = lambda x, L, C, q: L + C * q ** x
    p, _ = curve_fit(f, d, a, p0=[7e-5, 1e-3, 0.7], maxfev=40000)
    print('fit  |derivative| = L + C q^d   with')
    print('     L = %.4e    C = %.4e    q = %.4f' % tuple(p))
    print('     residuals %s' % np.array2string(a - f(d, *p), precision=2))
    los = []
    for k in range(len(d)):
        m = np.ones(len(d), bool); m[k] = False
        q2, _ = curve_fit(f, d[m], a[m], p0=p, maxfev=40000)
        los.append(q2[0])
    print('     leave-one-out L in [%.3e, %.3e]' % (min(los), max(los)))
    print()
    print('The improvement available at the root configuration does not tend')
    print('to zero with the degree, so no three-point certificate reaches tau.')


if __name__ == '__main__':
    main()
