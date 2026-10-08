#!/usr/bin/env python3
"""
hole_test.py: can a test polynomial exclude a further centre, given the design
defects S_1, S_2, ... of twenty-four directions?

The test is that of the paper's proposition "No room from the design defects":
p = sum_{k=0}^{deg} p_k G_k with p(t) < 0 on [-1, tau], tau = 0.614039, and
24 p_0 > sum_k |p_k| sqrt(S_k).  The linear programme below maximises
24 p_0 - sum_k |p_k| sqrt(S_k) under p(t) <= (t - tau)_+^2 on a grid of
[-1, 1] (the right side only fixes the scale); a positive value excludes a
further centre, a value of 0 means that no polynomial of that degree does.
Floating point, sampled constraints.

It is run on the root system and on the moments that diag_fc_primal.jl reads
off the stopped run of the programme with exactly 24 points at (14, 16)
(runs/fc_1416_it5_moments.txt).

Run: python hole_test.py [runs/fc_1416_it5_moments.txt]   (needs numpy, scipy)
"""
import re
import sys

import numpy as np
from scipy.optimize import linprog

TAU = 0.614039


def G(k, t):
    t = np.asarray(t, float)
    a, b = np.ones_like(t), 2 * t
    if k == 0:
        return a
    for _ in range(k - 1):
        a, b = b, 2 * t * b - a
    return b / (k + 1)


def best(S, deg, N=24):
    ts = np.linspace(-1, 1, 4001)
    nv = 2 * deg + 1  # p_0..p_deg, then bounds on |p_1|..|p_deg|
    c = np.zeros(nv)
    c[0] = -N
    for k in range(1, deg + 1):
        c[deg + k] = np.sqrt(max(S[k - 1], 0))
    A = [np.concatenate([[G(k, t) for k in range(deg + 1)], np.zeros(deg)]) for t in ts]
    b = [max(t - TAU, 0) ** 2 for t in ts]
    for k in range(1, deg + 1):
        for sg in (1, -1):
            r = np.zeros(nv)
            r[k] = sg
            r[deg + k] = -1
            A.append(r)
            b.append(0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(None, None)] * (deg + 1) + [(0, None)] * deg)
    return -res.fun


def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else "runs/fc_1416_it5_moments.txt"
    pseudo = {}
    with open(fn) as f:
        for line in f:
            m = re.match(r"S_(\d+)\s*=\s*(\S+)", line)
            if m:
                pseudo[int(m.group(1))] = float(m.group(2))
    ps = [pseudo[k] for k in sorted(pseudo)]
    # the root system: 12 pairs at -1, 96 at -1/2, 72 at 0, 96 at 1/2
    d4 = [24 + 2 * (12 * G(k, -1) + 96 * G(k, -.5) + 72 * G(k, 0) + 96 * G(k, .5))
          for k in range(1, len(ps) + 1)]
    print(" k   root system   exactly-24 run")
    for k in range(len(ps)):
        print("%2d %12.3f %14.3f" % (k + 1, d4[k], ps[k]))
    for deg in range(2, 12):
        print("degree %2d: root system %.5f, exactly-24 run %.5f" % (deg, best(d4, deg), best(ps, deg)))


if __name__ == "__main__":
    main()
