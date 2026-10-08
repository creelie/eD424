"""
ps_ref.py -- a slow reference for the step 3 integral, to check psker against.

This builds the whole product rho_A * rho_B as one dictionary, the way the
original does, and integrates it term by term.  It is only usable for small
|lambda|, which is the point: it shares no code with the C kernel beyond the
construction of the three factors, so agreement between the two is a check on
the kernel's canonical forms, hashing, parity classes and folded signs.
"""
import sys
from fractions import Fraction as Q

import ps_build as B
from o4 import int_mat


def reference(lam, k1, k2):
    A = B.a_terms(lam, k1)
    U = B.u_terms(lam, k2)
    V = B.v_terms(lam)
    acc = {}
    for ea, ca in A:
        for eu, su, cu in U:
            for ev, sv, cv in V:
                e = tuple(ea[i] + eu[i] + ev[i] for i in range(16))
                M = tuple(tuple(e[4 * i + j] for j in range(4)) for i in range(4))
                val = int_mat(4, M)
                if not val:
                    continue
                beta = tuple(su[i] + sv[i] for i in range(7))
                acc[beta] = acc.get(beta, Q(0)) + ca * cu * cv * val
    return {b: c for b, c in acc.items() if c}


def parse(path):
    out = {}
    cur = None
    for line in open(path):
        line = line.strip()
        if line.startswith('E '):
            _, l1, l2, k1, k2, deg = line.split()
            cur = ((int(l1), int(l2)), int(k1), int(k2))
            den = None
            out[cur] = {}
        elif line.startswith('D '):
            den = int(line.split()[1])
        elif line == '.':
            cur = None
        elif line and cur:
            f = line.split()
            beta = tuple(int(x) for x in f[:7])
            out[cur][beta] = Q(int(f[7]), den)
    return out


if __name__ == '__main__':
    got = parse(sys.argv[1])
    bad = 0
    for key in sorted(got):
        lam, k1, k2 = key
        want = reference(lam, k1, k2)
        if want != got[key]:
            bad += 1
            print('MISMATCH lambda=%s k1=%d k2=%d' % (list(lam), k1, k2))
            for b in sorted(set(want) | set(got[key])):
                if want.get(b, 0) != got[key].get(b, 0):
                    print('   %s  ref %s  kernel %s'
                          % (b, want.get(b, 0), got[key].get(b, 0)))
        else:
            print('ok  lambda=%-8s k1=%d k2=%d  %d coefficients'
                  % (str(list(lam)), k1, k2, len(want)))
    print('%d entries, %d mismatches' % (len(got), bad))
    sys.exit(1 if bad else 0)
